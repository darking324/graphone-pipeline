"""Pipeline orchestration (Phases I, II, IV) with checkpointing.

Each phase writes a raw checkpoint (JSON) under data/checkpoints/ and a
final CSV under data/output/. A phase whose checkpoint is marked complete
is skipped on re-runs — collection is resumable and never re-spends
rate-limit budget on finished work.

Phase III (LLM extraction) runs inside the news/jobs collectors via
LLMOrchestrator; Phase V/VI are documented in docs/architecture.md.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone

from . import sources as _sources  # noqa: F401  (package marker)
from .date_utils import to_iso
from .entity_resolver import EntityResolver
from .http_client import AsyncFetcher
from .sources import arxiv_papers, hf_products, jobs, news, yc_startups

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(ROOT, "data", "checkpoints")
OUTPUT_DIR = os.path.join(ROOT, "data", "output")

STARTUP_COLUMNS = [
    "schemaVersion", "recordType", "source_name", "source_url",
    "content_entityName", "content_data_employeeCount", "collectedAt",
    "website", "industry", "subindustry", "batch", "tags", "locations",
    "stage", "one_liner", "canonical_entityName", "resolution_method",
]
PRODUCT_COLUMNS = [
    "schemaVersion", "recordType", "source_name", "source_url",
    "content_startupName", "content_pricingModel", "collectedAt",
    "product_name", "product_url", "short_description", "sdk", "likes",
    "lastModified", "canonical_startupName", "resolution_method",
]
PAPER_COLUMNS = [
    "schemaVersion", "recordType", "source_name", "source_url",
    "content_title", "content_authors", "content_paper_url",
    "content_github_url", "content_github_stars",
    "content_published_date", "collectedAt",
    "arxiv_id", "summary", "pdf_url", "github_match_method",
]
JOB_COLUMNS = [
    "schemaVersion", "recordType", "source_name", "source_url",
    "content_company", "content_date", "content_is_remote",
    "content_role_family", "collectedAt",
    "job_title", "job_url", "location", "salary", "tags",
    "canonical_company", "resolution_method",
]
NEWS_COLUMNS = [
    "schemaVersion", "recordType", "source_name", "source_url",
    "content_title", "content_published_date", "content_summary",
    "content_keywords", "content_full_text", "collectedAt",
]
MAPPING_COLUMNS = ["raw_name", "canonical_name", "method", "source_tab"]


def _now_iso() -> str:
    return to_iso(datetime.now(timezone.utc))


def _checkpoint_path(name: str) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"{name}.json")


def _load_checkpoint(name: str):
    path = _checkpoint_path(name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("done"):
            return data["records"]
    return None


def _save_checkpoint(name: str, records: list) -> None:
    path = _checkpoint_path(name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"done": True, "records": records}, f)
    os.replace(tmp, path)


def _write_csv(filename: str, columns: list[str], rows: list[dict]) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c) for c in columns})
    return path


def _resolve_names(rows: list[dict], raw_field: str, canon_field: str,
                   method_field: str, resolver: EntityResolver,
                   mapping: list[dict], source_tab: str) -> None:
    """Canonicalize one name column in place; append to the mapping log."""
    for row in rows:
        raw = row.get(raw_field)
        if raw:
            canon, method = resolver.resolve(str(raw))
            row[canon_field] = canon
            row[method_field] = method
            mapping.append({
                "raw_name": raw, "canonical_name": canon,
                "method": method, "source_tab": source_tab,
            })


async def run_phase(
    fetcher: AsyncFetcher, phase: str, *, collected_at: str, target: int = 1000
) -> list[dict]:
    """Run one collection phase (with checkpoint resume)."""
    cached = _load_checkpoint(phase)
    if cached is not None:
        print(f"[{phase}] checkpoint hit: {len(cached)} records")
        return cached
    if phase == "startups":
        records = await yc_startups.collect(fetcher, target=target + 100, collected_at=collected_at)
    elif phase == "products":
        records = await hf_products.collect(fetcher, target=target + 100, collected_at=collected_at)
    elif phase == "papers":
        records = await arxiv_papers.collect(fetcher, target=target, collected_at=collected_at)
    elif phase == "news":
        records = await news.collect(fetcher, collected_at=collected_at)
    elif phase == "jobs":
        records = await jobs.collect(fetcher, collected_at=collected_at)
    else:
        raise ValueError(f"unknown phase: {phase}")
    _save_checkpoint(phase, records)
    print(f"[{phase}] collected {len(records)} records")
    return records


def build_outputs(startups, products, papers, jobs_rows, news_rows) -> dict:
    """Apply entity resolution (Phase IV) and write the 6 CSVs."""
    resolver = EntityResolver()
    mapping: list[dict] = []

    _resolve_names(startups, "content_entityName", "canonical_entityName",
                   "resolution_method", resolver, mapping, "Startups")
    _resolve_names(products, "content_startupName", "canonical_startupName",
                   "resolution_method", resolver, mapping, "Products")
    _resolve_names(jobs_rows, "content_company", "canonical_company",
                   "resolution_method", resolver, mapping, "Jobs")

    # dedup mapping log (same raw -> same canonical)
    seen, mapping_dedup = set(), []
    for m in mapping:
        key = (m["raw_name"], m["canonical_name"])
        if key in seen:
            continue
        seen.add(key)
        mapping_dedup.append(m)

    # papers: authors joined for the sheet
    paper_rows = []
    for p in papers:
        row = dict(p)
        row["content_authors"] = "; ".join(p.get("content_authors") or [])
        paper_rows.append(row)

    paths = {
        "startups": _write_csv("startups.csv", STARTUP_COLUMNS, startups),
        "products": _write_csv("products.csv", PRODUCT_COLUMNS, products),
        "papers": _write_csv("papers.csv", PAPER_COLUMNS, paper_rows),
        "jobs": _write_csv("jobs.csv", JOB_COLUMNS, jobs_rows),
        "news": _write_csv("news.csv", NEWS_COLUMNS, news_rows),
        "mapping": _write_csv("entity_mapping.csv", MAPPING_COLUMNS, mapping_dedup),
    }
    counts = {
        "startups": len(startups), "products": len(products),
        "papers": len(paper_rows), "jobs": len(jobs_rows),
        "news": len(news_rows), "mapping": len(mapping_dedup),
    }
    return {"paths": paths, "counts": counts}
