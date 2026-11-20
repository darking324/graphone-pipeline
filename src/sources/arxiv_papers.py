"""Phase I — Research-paper records from the official arXiv API.

Source: https://export.arxiv.org/api/query (no key required).
Query: recent AI papers across cs.AI, cs.LG, cs.CL, cs.CV, cs.NE, stat.ML,
sorted by submission date (newest first). arXiv asks clients to wait 3s
between requests; we use 200-result pages with a 3.5s gap.

GitHub correlation: a paper's arXiv *comment* field sometimes names its
official code repository (e.g. "Code at https://github.com/org/repo").
Those URLs are extracted verbatim — never guessed. Papers without a
comment link get their repo from the GitHub search enricher
(see github_enrich.py), which only records repos whose README literally
cites the paper's arXiv ID.
"""

from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET

from ..date_utils import to_iso
from ..http_client import AsyncFetcher

ARXIV_API = "https://export.arxiv.org/api/query"
SOURCE_NAME = "arXiv"
TARGET_DEFAULT = 1000
CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE", "stat.ML"]
PAGE_SIZE = 200
REQUEST_GAP_SECONDS = 3.5

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
_GITHUB_RE = re.compile(
    r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", re.IGNORECASE
)


def _text(elem, tag: str) -> str | None:
    el = elem.find(tag, NS)
    return (el.text or "").strip() if el is not None and el.text else None


def _links(elem) -> dict[str, str]:
    out = {}
    for link in elem.findall("atom:link", NS):
        rel = link.get("rel") or "alternate"
        href = link.get("href") or ""
        if rel == "alternate":
            out["abs"] = href
        elif "pdf" in (link.get("title") or "").lower() or href.endswith(".pdf"):
            out["pdf"] = href
    return out


def _github_from_comment(comment: str | None) -> str | None:
    if not comment:
        return None
    m = _GITHUB_RE.search(comment)
    if not m:
        return None
    return f"https://github.com/{m.group(1)}/{m.group(2).rstrip('.')}"


def parse_feed(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    papers = []
    for entry in root.findall("atom:entry", NS):
        arxiv_id = _text(entry, "atom:id") or ""
        arxiv_id = arxiv_id.rsplit("/abs/", 1)[-1].strip()
        title = " ".join((_text(entry, "atom:title") or "").split())
        authors = [
            (a.find("atom:name", NS).text or "").strip()
            for a in entry.findall("atom:author", NS)
            if a.find("atom:name", NS) is not None
        ]
        published = _text(entry, "atom:published")
        summary = " ".join((_text(entry, "atom:summary") or "").split())
        comment = _text(entry, "arxiv:comment")
        links = _links(entry)
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "published_date": published,
                "summary": summary,
                "comment": comment,
                "paper_url": links.get("abs") or f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": links.get("pdf"),
                "github_url_from_comment": _github_from_comment(comment),
            }
        )
    return papers


def _to_record(paper: dict, collected_at: str) -> dict:
    return {
        "schemaVersion": "1.0",
        "recordType": "RESEARCH_PAPER",
        "source_name": SOURCE_NAME,
        "source_url": paper["paper_url"],
        "content_title": paper["title"],
        "content_authors": paper["authors"],
        "content_paper_url": paper["paper_url"],
        "content_github_url": paper["github_url_from_comment"],
        "content_github_stars": None,  # filled by github_enrich.py
        "content_published_date": paper["published_date"],
        "collectedAt": collected_at,
        # extras
        "arxiv_id": paper["arxiv_id"],
        "summary": paper["summary"][:1000],
        "pdf_url": paper["pdf_url"],
        "github_match_method": "arxiv_comment" if paper["github_url_from_comment"] else None,
    }


async def collect(
    fetcher: AsyncFetcher, *, target: int = TARGET_DEFAULT, collected_at: str
) -> list[dict]:
    """Fetch the newest AI papers from arXiv until `target` unique papers."""
    query = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    records: list[dict] = []
    seen: set[str] = set()
    start = 0
    while len(records) < target:
        url = (
            f"{ARXIV_API}?search_query={query}"
            f"&start={start}&max_results={PAGE_SIZE}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        status, body, _ = await fetcher.fetch(url, accept="application/atom+xml")
        if status != 200:
            raise RuntimeError(f"arXiv API returned HTTP {status}")
        batch = parse_feed(body)
        if not batch:
            break
        for paper in batch:
            if paper["arxiv_id"] in seen or not paper["title"]:
                continue
            seen.add(paper["arxiv_id"])
            records.append(_to_record(paper, collected_at))
        start += len(batch)
        if len(batch) < PAGE_SIZE:
            break
        await asyncio.sleep(REQUEST_GAP_SECONDS)
    return records
