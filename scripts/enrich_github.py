#!/usr/bin/env python3
"""Background GitHub enrichment for research papers (resumable).

Usage:
    python scripts/enrich_github.py [--max-wait 900]

Reads data/checkpoints/papers.json, fills content_github_url /
content_github_stars / github_match_method for each paper, and rewrites
both the checkpoint and data/output/papers.csv. Safe to stop and
re-run at any time: finished papers are skipped via the checkpoint
(data/checkpoints/github_enrichment.json), so no rate-limit budget is
ever spent twice. When the GitHub rate limit is hit the script waits
for the reset (up to --max-wait seconds) and then resumes; with
--max-wait 0 it exits immediately so it can be re-run later.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import (  # noqa: E402
    CHECKPOINT_DIR, OUTPUT_DIR, PAPER_COLUMNS, _load_checkpoint,
    _save_checkpoint, _write_csv,
)
from src.sources.github_enrich import enrich_papers  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-wait", type=float, default=900.0)
    args = ap.parse_args()

    papers = _load_checkpoint("papers")
    if not papers:
        print("No papers checkpoint found — run the pipeline first.")
        return 1
    ckpt = os.path.join(CHECKPOINT_DIR, "github_enrichment.json")
    enriched = enrich_papers(papers, ckpt, max_wait_seconds=args.max_wait)
    _save_checkpoint("papers", enriched)
    rows = []
    for p in enriched:
        row = dict(p)
        row["content_authors"] = "; ".join(p.get("content_authors") or [])
        rows.append(row)
    _write_csv("papers.csv", PAPER_COLUMNS, rows)
    with_stars = sum(1 for p in enriched if p.get("content_github_stars"))
    with_url = sum(1 for p in enriched if p.get("content_github_url"))
    print(f"\nPapers: {len(enriched)} total, {with_url} with a GitHub URL, "
          f"{with_stars} with star counts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
