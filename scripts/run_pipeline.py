#!/usr/bin/env python3
"""Run the GraphOne ingestion pipeline.

Usage:
    python scripts/run_pipeline.py [--phases startups,products,papers,news,jobs]
                                   [--target 1000] [--no-output]

Phases are checkpointed under data/checkpoints/; re-running skips
completed phases. CSVs are written to data/output/ when --no-output
is not given. GitHub star enrichment is a separate, slow, resumable
step — see scripts/enrich_github.py.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.http_client import AsyncFetcher  # noqa: E402
from src.pipeline import build_outputs, run_phase, _now_iso  # noqa: E402

ALL_PHASES = ["startups", "products", "papers", "news", "jobs"]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default=",".join(ALL_PHASES))
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--no-output", action="store_true")
    args = ap.parse_args()

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    collected_at = _now_iso()
    results: dict[str, list] = {}
    async with AsyncFetcher(max_concurrency=6, min_delay_seconds=0.6) as fetcher:
        for phase in phases:
            results[phase] = await run_phase(
                fetcher, phase, collected_at=collected_at, target=args.target
            )

    if args.no_output:
        return 0

    # For phases not run in this invocation, load from checkpoints so the
    # output CSVs always reflect the full collected state.
    from src.pipeline import _load_checkpoint

    for phase in ALL_PHASES:
        if phase not in results:
            cached = _load_checkpoint(phase)
            results[phase] = cached or []
            if cached:
                print(f"[{phase}] loaded {len(cached)} records from checkpoint")

    out = build_outputs(
        results.get("startups", []),
        results.get("products", []),
        results.get("papers", []),
        results.get("jobs", []),
        results.get("news", []),
    )
    print("\nRow counts:")
    for name, count in out["counts"].items():
        print(f"  {name:10s} {count}")
    print("\nCSVs:")
    for name, path in out["paths"].items():
        print(f"  {name:10s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
