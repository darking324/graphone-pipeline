#!/usr/bin/env python3
"""Validate the six output CSVs against the task requirements.

Checks: CSVs parse; minimum row counts (Startups/Products/Papers >= 1000);
every row has a source_url; required canonical fields are non-empty;
news/jobs dates fall inside the 24-hour freshness window; no duplicate
source URLs within a tab. Exit 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "data", "output")

MIN_ROWS = {"startups.csv": 1000, "products.csv": 1000, "papers.csv": 1000}


def read_csv(name: str) -> list[dict]:
    with open(os.path.join(OUTPUT_DIR, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_iso(s: str):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def main() -> int:
    errors: list[str] = []
    now = datetime.now(timezone.utc)
    counts = {}

    for name in ["startups.csv", "products.csv", "papers.csv",
                 "jobs.csv", "news.csv", "entity_mapping.csv"]:
        path = os.path.join(OUTPUT_DIR, name)
        if not os.path.exists(path):
            errors.append(f"{name}: MISSING")
            continue
        try:
            rows = read_csv(name)
        except Exception as exc:
            errors.append(f"{name}: unparseable ({exc})")
            continue
        counts[name] = len(rows)
        if name in MIN_ROWS and len(rows) < MIN_ROWS[name]:
            errors.append(f"{name}: {len(rows)} rows < minimum {MIN_ROWS[name]}")
        # source_url on every row (mapping log is the exception: it has
        # raw/canonical names, not source URLs)
        if name != "entity_mapping.csv":
            missing = sum(1 for r in rows if not (r.get("source_url") or "").strip())
            if missing:
                errors.append(f"{name}: {missing} rows missing source_url")
        # duplicate source URLs within the tab
        urls = [r.get("source_url") for r in rows if r.get("source_url")]
        if len(urls) != len(set(urls)):
            errors.append(f"{name}: {len(urls) - len(set(urls))} duplicate source_urls")

    # required canonical fields
    for name, field in [
        ("startups.csv", "content_entityName"),
        ("products.csv", "content_startupName"),
        ("papers.csv", "content_title"),
    ]:
        path = os.path.join(OUTPUT_DIR, name)
        if not os.path.exists(path):
            continue
        rows = read_csv(name)
        empty = sum(1 for r in rows if not (r.get(field) or "").strip())
        if empty:
            errors.append(f"{name}: {empty} rows with empty {field}")

    # freshness: news + jobs within 24h of their own collectedAt
    for name, date_field in [("news.csv", "content_published_date"),
                             ("jobs.csv", "content_date")]:
        path = os.path.join(OUTPUT_DIR, name)
        if not os.path.exists(path):
            continue
        rows = read_csv(name)
        stale = 0
        for r in rows:
            dt = parse_iso(r.get(date_field) or "")
            collected = parse_iso(r.get("collectedAt") or "")
            if not dt or not collected:
                stale += 1
            elif (collected - dt).total_seconds() > 24 * 3600 + 300:
                stale += 1
        if stale:
            errors.append(f"{name}: {stale} rows outside the 24h window")

    print("Row counts:")
    for name, n in counts.items():
        print(f"  {name:20s} {n}")
    if errors:
        print("\nFAILURES:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
