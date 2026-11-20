"""Phase I — Startup records from the Y Combinator public directory.

Source: YC's own Algolia search index (the same index the public
https://www.ycombinator.com/companies directory page queries in the
browser). The search-only API key is embedded in that public page and is
time-limited, so it is fetched fresh from the page on every run — never
hardcoded. Only companies tagged "AI" are collected: GraphOne maps the AI
ecosystem, and the index holds ~1,900 AI-tagged companies.

Fields are taken verbatim from YC: name, website, industry, batch,
team_size (used as employeeCount — YC's own published figure, never
estimated), tags, locations, stage.
"""

from __future__ import annotations

import json
import re
import urllib.parse

from ..date_utils import to_iso
from ..http_client import AsyncFetcher

YC_COMPANIES_PAGE = "https://www.ycombinator.com/companies"
ALGOLIA_URL = "https://45bwzj1sgc-dsn.algolia.net/1/indexes/*/queries"
ALGOLIA_APP_ID = "45BWZJ1SGC"
INDEX = "YCCompany_production"
SOURCE_NAME = "Y Combinator Directory"
TARGET_DEFAULT = 1100  # collect a little over 1,000 to absorb dedup losses

_ALGOLIA_OPTS_RE = re.compile(r"window\.AlgoliaOpts\s*=\s*(\{.*?\})\s*(?:;|\n)", re.DOTALL)


async def fetch_algolia_key(fetcher: AsyncFetcher) -> tuple[str, str]:
    """Fetch the public, time-limited Algolia search key from YC's page."""
    status, html = await fetcher.fetch_text(YC_COMPANIES_PAGE)
    if status != 200:
        raise RuntimeError(f"YC companies page returned HTTP {status}")
    m = _ALGOLIA_OPTS_RE.search(html)
    if not m:
        raise RuntimeError("Could not find AlgoliaOpts on the YC companies page")
    opts = json.loads(m.group(1))
    return opts["app"], opts["key"]


async def _algolia_query(
    fetcher: AsyncFetcher, app: str, key: str, params: dict
) -> dict:
    body = json.dumps(
        {"requests": [{"indexName": INDEX, "params": urllib.parse.urlencode(params)}]}
    ).encode()
    headers = {
        "Content-Type": "application/json",
        "X-Algolia-API-Key": key,
        "X-Algolia-Application-Id": app,
        "User-Agent": "GraphOnePipeline/1.0",
    }
    assert fetcher._session is not None
    async with fetcher._session.post(ALGOLIA_URL, data=body, headers=headers) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Algolia query failed: HTTP {resp.status}")
        data = await resp.json()
    return data["results"][0]


def _to_record(hit: dict, collected_at: str) -> dict:
    slug = hit.get("slug") or ""
    website = (hit.get("website") or "").strip() or None
    return {
        "schemaVersion": "1.0",
        "recordType": "STARTUP",
        "source_name": SOURCE_NAME,
        "source_url": f"https://www.ycombinator.com/companies/{slug}" if slug else None,
        "content_entityName": hit.get("name"),
        "content_data_employeeCount": hit.get("team_size"),
        "collectedAt": collected_at,
        # useful extras (kept separate from the canonical schema fields)
        "website": website,
        "industry": hit.get("industry"),
        "subindustry": hit.get("subindustry"),
        "batch": hit.get("batch"),
        "tags": "; ".join(hit.get("tags") or []),
        "locations": hit.get("all_locations"),
        "stage": hit.get("stage"),
        "one_liner": hit.get("one_liner"),
    }


TAG_FILTER = (
    '(tags:"AI" OR tags:"Machine Learning" OR '
    'tags:"Generative AI" OR tags:"LLM")'
)
ATTRIBUTES = ",".join(
    [
        "name", "slug", "website", "one_liner", "industry",
        "subindustry", "batch", "team_size", "tags",
        "all_locations", "stage",
    ]
)


async def collect(
    fetcher: AsyncFetcher, *, target: int = TARGET_DEFAULT, collected_at: str
) -> list[dict]:
    """Collect AI-tagged YC startups, paginated, deduplicated by slug.

    Algolia caps browsing at the first 1,000 hits of any single query,
    so one unfiltered query can never return more than 1,000 companies.
    The collection is therefore split by YC batch (39 batches, max ~162
    AI companies each): the batch list is read from the index's own
    facets first, then each batch is fetched separately and merged.
    """
    app, key = await fetch_algolia_key(fetcher)
    seen: set[str] = set()
    records: list[dict] = []

    # 1. Discover batches from the index facets (self-adapting list).
    facet_result = await _algolia_query(
        fetcher, app, key,
        {
            "query": "", "hitsPerPage": 0,
            "filters": TAG_FILTER,
            "facets": "batch", "maxValuesPerFacet": 100,
        },
    )
    batches = facet_result.get("facets", {}).get("batch", {})
    batch_names = sorted(batches, key=lambda b: -batches[b])
    if not batch_names:  # facet call failed/empty: fall back to one query
        batch_names = [""]

    # 2. Fetch each batch (each is far below the 1,000-hit browse cap).
    for batch in batch_names:
        if len(records) >= target:
            break
        filters = TAG_FILTER if not batch else f'{TAG_FILTER} AND batch:"{batch}"'
        page = 0
        per_page = 100
        while len(records) < target:
            result = await _algolia_query(
                fetcher, app, key,
                {
                    "query": "", "hitsPerPage": per_page, "page": page,
                    "filters": filters, "attributesToRetrieve": ATTRIBUTES,
                },
            )
            hits = result.get("hits", [])
            if not hits:
                break
            for hit in hits:
                slug = hit.get("slug") or hit.get("name")
                if not slug or slug in seen:
                    continue
                seen.add(slug)
                rec = _to_record(hit, collected_at)
                if rec["content_entityName"] and rec["source_url"]:
                    records.append(rec)
            if page >= (result.get("nbPages", 1) - 1):
                break
            page += 1
    return records
