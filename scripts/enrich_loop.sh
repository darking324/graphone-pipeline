#!/bin/bash
# Resume-loop wrapper for the GitHub enricher: re-invokes it until all
# papers are enriched (rate-limit pauses just end the current invocation).
cd "$(dirname "$0")/.."
TOTAL=$(.venv/bin/python -c "import json; print(len(json.load(open('data/checkpoints/papers.json'))['records']))")
for i in $(seq 1 200); do
  .venv/bin/python scripts/enrich_github.py --max-wait 600
  DONE=$(.venv/bin/python -c "import json,os; print(len(json.load(open('data/checkpoints/github_enrichment.json'))))" 2>/dev/null || echo 0)
  echo "[loop $i] enriched $DONE / $TOTAL"
  if [ "$DONE" -ge "$TOTAL" ]; then echo "ALL DONE"; break; fi
  sleep 120
done
