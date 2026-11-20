# GraphOne / FrontierAtlas — AI Intelligence Ingestion Pipeline

**AI Engineer (Demo Task) — The AI Signal.** A scalable, fault-tolerant
data ingestion pipeline that builds an AI-intelligence graph:
**startups, products, research papers (with GitHub stars), 24-hour-fresh
AI news, and 24-hour-fresh AI jobs** — all from real public sources, all
traceable to a source URL. No hallucinated data, ever.

Submission form: https://forms.gle/3JJhukomumTFueRW7
Task doc: https://docs.google.com/document/d/1EPuMdqbXyVIxJrf4HhTM8Hv9VS8CH4yidoH7a71zqek/edit

## What was collected (run of 2026-11-20/21, UTC)

| Tab | Rows | Source |
|---|---|---|
| Startups | **1,219** | Y Combinator public directory (Algolia index) |
| Products | **1,100** | Hugging Face Hub — public Spaces API |
| Research Papers | **1,200** | arXiv API (cs.AI, cs.LG, cs.CL, cs.CV, cs.NE, stat.ML) |
| Jobs | **26** | 5 job boards — all posted within the last 24 h |
| News | **12** | 5 RSS feeds — all published within the last 24 h |
| Entity Mapping Log | **1,992** | raw → canonical name mappings from entity resolution |

Jobs and news are a **snapshot of the 24-hour window at collection
time** (2026-11-20 ~20:19 UTC). Re-run the news/jobs phases to refresh
them before submitting:

```bash
python scripts/run_pipeline.py --phases news,jobs
```

Papers carry GitHub URLs + live star counts, filled in by
`scripts/enrich_github.py` (resumable; see below) — **359 of the 1,200
papers have a verified GitHub repo, 356 of those have a live star
count** (269 with ≥1 star; 87 repos exist but have 0 stars so far).
Repos are found only via the paper's own arXiv comment (84) or a
README that literally cites the paper's arXiv ID (275); the other 841
papers have no verified repo and keep those fields empty.

## Architecture (Phases I–VI)

| Phase | What it does |
|---|---|
| **I — Bulk extraction** | Concurrent scrape of entity directories. Startups from YC's public Algolia index (batch-split to get past Algolia's 1,000-hit browse cap); products from the Hugging Face Spaces API (cursor pagination); papers from the arXiv API (3.5 s between requests, as arXiv asks). |
| **II — Signal ingestion** | 5 AI news RSS feeds (TechCrunch AI, MIT Technology Review AI, Google AI Blog, OpenAI Blog, DeepMind Blog) with full-text extraction; 5 job boards (RemoteOK, Remotive, Jobicy, Arbeitnow, Hacker News Jobs). Only items published in the last 24 h are kept; dates are normalised from ISO, RFC-2822, Unix time, human formats and relative forms ("2 hours ago"). |
| **III — LLM extraction** | `src/llm_orchestrator.py` structures raw text into the canonical schemas with a fallback chain **Gemini Flash → Groq Llama 3 → DeepSeek**, chunking (≤ ~8,000 chars) so payloads never trigger 413s, and 429 handling with exponential backoff + jitter. |
| **IV — Entity resolution** | `src/entity_resolver.py` canonicalises names ("OpenAI, Inc.", "Open AI" → "OpenAI") against a seed list of 50 known AI startups, with an alias map and difflib fuzzy matching (cutoff 0.88). Every resolution is logged in the Entity Mapping Log. |
| **V — Anti-bot & scale** | Async crawling (asyncio + aiohttp), per-host semaphores and delays, robots.txt respected, identifiable User-Agent, no CAPTCHA solving. For Cloudflare/Datadome-protected or heavily JS-rendered domains: prefer the site's public API or RSS feed; otherwise a headless browser at human-like pacing; if a challenge still appears, back off and record the source as unavailable. |
| **VI — Architecture doc** | `docs/architecture.pdf` (3 pages): scale strategy for 500k records, 413/429 handling, freshness tracking & dedup across distributed nodes, storage strategy (PostgreSQL + pgvector + graph). |

Full design: [`docs/architecture.pdf`](docs/architecture.pdf).

## No hallucinated data — the rule this pipeline is built on

- Every record stores a real `source.url` it was collected from.
- Unknown fields are left **empty/null**, never invented: no estimated
  employee counts, no guessed stars, no fabricated dates or URLs.
- A paper gets a GitHub URL only when the arXiv comment names one, or
  when a repository's README literally cites the paper's arXiv ID
  (`"<arxiv_id>" in:readme` search). Star counts come from the live
  GitHub API response for that repo.
- AI relevance for jobs is judged from **title/tags/category only** —
  matching on descriptions produced false positives (e.g. a Flutter
  engineer ad saying "We love AI").

## LLM API keys

No LLM API keys are available in this environment, and none are
hardcoded. Set them as environment variables to enable the LLM chain:

```bash
export GEMINI_API_KEY=... GROQ_API_KEY=... DEEPSEEK_API_KEY=...
```

Without keys the orchestrator **degrades gracefully to deterministic
rule-based extraction** (regex/heuristics, missing fields as null), so
the pipeline still runs end-to-end. Every extraction reports its
`method` (`llm` or `rule_based`) and provider, so you can always tell
which path produced a field.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Python 3.11+, no Docker needed.

## Running

```bash
# Full pipeline (checkpoints in data/checkpoints/; re-runs skip finished phases)
.venv/bin/python scripts/run_pipeline.py --target 1000

# Single phases
.venv/bin/python scripts/run_pipeline.py --phases news,jobs

# GitHub star enrichment for papers (slow: GitHub allows 10 search
# requests/min unauthenticated; checkpoints after every paper, so it
# resumes exactly where it stopped — just re-run it)
.venv/bin/python scripts/enrich_github.py
# or the resume loop (re-invokes until all papers are done):
./scripts/enrich_loop.sh

# Public Google Sheet (needs Google Sheets + Drive connected; see below)
.venv/bin/python scripts/create_sheet.py

# Rebuild the architecture PDF (fails if it exceeds 3 pages)
.venv/bin/python docs/build_architecture_pdf.py
```

Outputs land in `data/output/` as six CSVs — one per Sheet tab.

## Google Sheet

`scripts/create_sheet.py` creates the spreadsheet, adds the six tabs,
writes the CSVs, and shares it publicly (anyone with the link can view).

> **Note:** in the build environment the Google Sheets and Google Drive
> connectors were **not connected** (only Gmail was), so the Sheet could
> not be created from here. Connect them, then run:
>
> - Google Sheets: https://agent.meta.ai/connectors/connect/google_sheets
> - Google Drive: https://agent.meta.ai/connectors/connect/google_drive
>
> then `python scripts/create_sheet.py` and paste the printed URL into
> the submission form. The CSVs in `data/output/` are the source of
> truth until then.

## Repository layout

```
graphone-pipeline/
├── README.md
├── requirements.txt
├── docs/
│   ├── architecture.pdf            # Phase VI doc (max 3 pages)
│   └── build_architecture_pdf.py
├── scripts/
│   ├── run_pipeline.py             # CLI entry point
│   ├── enrich_github.py            # resumable GitHub star enrichment
│   ├── enrich_loop.sh              # resume-loop wrapper
│   └── create_sheet.py             # builds the public Google Sheet
├── src/
│   ├── __init__.py
│   ├── http_client.py              # async fetcher: rate limits, backoff, robots.txt
│   ├── date_utils.py               # flexible date parsing / 24h freshness
│   ├── llm_orchestrator.py         # Gemini → Groq → DeepSeek chain + rule fallback
│   ├── entity_resolver.py          # canonicalisation, 50-startup seed list
│   ├── pipeline.py                 # orchestration, checkpoints, CSV output
│   └── sources/
│       ├── yc_startups.py          # Y Combinator Algolia directory
│       ├── hf_products.py          # Hugging Face Spaces
│       ├── arxiv_papers.py         # arXiv API
│       ├── github_enrich.py        # GitHub correlation + stars
│       ├── news.py                 # 5 AI news RSS feeds
│       └── jobs.py                 # 5 AI job boards
└── data/
    ├── checkpoints/                # resumable phase state (JSON)
    ├── output/                     # the six final CSVs
    └── sheet_url.txt               # written by create_sheet.py
```

## Known limits

- GitHub's unauthenticated API allows 60 core requests/hour and
  10 search requests/minute, shared with the whole egress IP — star
  enrichment fills in over successive runs rather than failing the
  pipeline. The full 1,200-paper enrichment took ~3 hours across 7
  resume-loop iterations.
- 3 comment-linked repos returned no star count (repo deleted, renamed
  or private); their URLs are kept (they come from the arXiv comment)
  with stars left null rather than guessed.
- Papers with Code's API is dead (it redirects to Hugging Face Papers);
  paper↔repo correlation uses arXiv comments + GitHub README search
  instead.
- The LLM chain runs in deterministic rule-based mode unless the three
  API keys above are set.
