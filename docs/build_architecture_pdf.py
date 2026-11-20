#!/usr/bin/env python3
"""Build docs/architecture.pdf (max 3 pages) with reportlab.

Content: the Phase VI architecture & production design for the
GraphOne / FrontierAtlas ingestion pipeline — scale strategy (500k+),
413/429 handling, freshness tracking & dedup across distributed nodes,
and storage strategy. The script prints the final page count and fails
if it exceeds 3 pages.
"""

from __future__ import annotations

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "architecture.pdf")

ACCENT = colors.HexColor("#1a56db")
DARK = colors.HexColor("#111827")
MUTED = colors.HexColor("#4b5563")
LIGHT = colors.HexColor("#eef2ff")
RULE = colors.HexColor("#d1d5db")

S = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=19,
                            leading=22, textColor=DARK, spaceAfter=2 * mm),
    "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=9.5,
                               leading=13, textColor=MUTED, spaceAfter=4 * mm),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=11.5,
                         leading=14, textColor=ACCENT, spaceBefore=3.5 * mm,
                         spaceAfter=1.2 * mm),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10,
                         leading=12.5, textColor=DARK, spaceBefore=2 * mm,
                         spaceAfter=0.8 * mm),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.3,
                           leading=12.8, textColor=DARK, spaceAfter=1.6 * mm,
                           alignment=TA_LEFT),
    "bullet": ParagraphStyle("bullet", parent=ParagraphStyle("b", fontName="Helvetica",
                             fontSize=9.3, leading=12.8, textColor=DARK),
                             leftIndent=10, bulletIndent=3, spaceAfter=1 * mm),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.3,
                           leading=11, textColor=DARK),
    "cellh": ParagraphStyle("cellh", fontName="Helvetica-Bold", fontSize=8.3,
                            leading=11, textColor=colors.white),
    "foot": ParagraphStyle("foot", fontName="Helvetica", fontSize=8,
                           leading=10, textColor=MUTED),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(f"•&nbsp;&nbsp;{t}", S["bullet"]) for t in items]


def flow_table(headers, rows):
    data = [[P(h, "cellh") for h in headers]]
    for r in rows:
        data.append([P(c, "cell") for c in r])
    t = Table(data, colWidths=[32 * mm, 62 * mm, 62 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(15 * mm, 282 * mm,
                      "GraphOne / FrontierAtlas  —  Technical Architecture (Demo Task)")
    canvas.drawRightString(195 * mm, 282 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.line(15 * mm, 280 * mm, 195 * mm, 280 * mm)
    canvas.restoreState()


def build():
    doc = BaseDocTemplate(OUT, pagesize=A4,
                          leftMargin=15 * mm, rightMargin=15 * mm,
                          topMargin=20 * mm, bottomMargin=14 * mm,
                          title="GraphOne / FrontierAtlas — Technical Architecture",
                          author="Abhishek Kumar")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame],
                                       onPage=header_footer)])
    story = []
    story.append(P("GraphOne / FrontierAtlas", "title"))
    story.append(P("Technical architecture of the AI-intelligence ingestion pipeline — "
                   "how the demo-trial system works, how it scales to 500,000+ records, "
                   "and how it stays correct under rate limits, bot defences and "
                   "distributed crawling. AI Engineer demo task, The AI Signal.",
                   "subtitle"))
    story.append(HRFlowable(width="100%", thickness=0.6, color=ACCENT))

    story.append(P("1 &nbsp; System overview", "h1"))
    story.append(P(
        "The pipeline is a set of small, independently runnable stages sharing one "
        "canonical record model. Every record carries <b>schemaVersion</b>, "
        "<b>recordType</b>, <b>source.name</b>, <b>source.url</b> and "
        "<b>collectedAt</b> (ISO-8601), so any row can be traced back to the exact "
        "public page or API response it came from. Collection is checkpointed "
        "after every unit of work, so a crash or a rate-limit pause resumes "
        "exactly where it stopped instead of re-spending budget."))
    story.append(flow_table(
        ["Stage", "What it does", "Sources / mechanism"],
        [
            ["I — Bulk extraction", "Concurrent scrape of entity directories; 1,000 startups, 1,000 products, 1,000 papers",
             "Y Combinator public Algolia index; Hugging Face Hub API (Spaces); arXiv API"],
            ["II — Signal ingestion", "5 news + 5 job sources; only items published in the last 24 h; full-text fetch; date normalisation",
             "RSS (TechCrunch AI, MIT Tech Review, Google AI, OpenAI, DeepMind); RemoteOK, Remotive, Jobicy, Arbeitnow, HN Jobs"],
            ["III — LLM extraction", "Raw HTML/text → canonical JSON via a fallback chain; rule-based fallback when no key is set",
             "Gemini Flash → Groq Llama 3 → DeepSeek; deterministic regex/heuristics fallback"],
            ["IV — Entity resolution", "Canonicalises organisation/product names against a 50-startup seed list",
             "Normalisation + alias map + difflib fuzzy match (cutoff 0.88); full mapping log"],
            ["V — Anti-bot & scale", "Async crawling, polite concurrency, robots.txt, backoff; strategy documented in §4",
             "asyncio + aiohttp; per-host semaphores and delays; no CAPTCHA solving"],
            ["VI — Output", "6-tab public Sheet + GitHub repo (README, src/, this document)",
             "Google Sheets API; GitHub"],
        ]))
    story.append(Spacer(1, 2 * mm))
    story.append(P(
        "<b>Data flow.</b> Sources → async fetchers (rate-limited, retried) → "
        "normalisers (dates, names, schemas) → LLM/rule-based structuring → "
        "entity resolution → deduplication → CSV/Sheet + checkpoints. Nothing is "
        "written to the output that did not arrive from a real source URL."))

    story.append(P("2 &nbsp; Scale strategy — from 1,000 to 500,000+ records", "h1"))
    story.extend(bullets([
        "<b>Stateless, sharded workers.</b> Collection is split by source and by "
        "key-range (e.g. arXiv pages, Algolia pages, feed shards). Each worker "
        "owns a disjoint slice, keeps no local state except its checkpoint, and "
        "can be replicated horizontally. Scaling from 1k to 500k records means "
        "adding workers, not rewriting code.",
        "<b>Bounded concurrency, not unbounded threads.</b> Per-host semaphores "
        "(2–8 concurrent) and minimum inter-request delays keep us polite and "
        "under rate limits; a global token bucket caps total throughput. "
        "Throughput grows with worker count only up to each source's limit — "
        "the design assumes limits, it does not fight them.",
        "<b>Cursor/keyset pagination everywhere.</b> No OFFSET scans: every "
        "source is walked with cursors (arXiv <i>start</i>, Algolia pages, "
        "HF Link-header cursors, API pagination). Cost per page stays flat as "
        "the corpus grows.",
        "<b>Backpressure via queues.</b> Fetch → parse → enrich stages are "
        "connected by bounded queues (e.g. 1,000 items). A slow downstream "
        "stage throttles fetchers instead of buffering the whole corpus in "
        "memory. Memory use is O(queue), not O(corpus).",
        "<b>Idempotent writes.</b> Every record has a natural key "
        "(source URL, arXiv ID, job URL). Re-running a stage upserts instead of "
        "duplicating, so retries and restarts are safe at any scale.",
    ]))

    story.append(P("3 &nbsp; Handling 413s and 429s", "h1"))
    story.append(P("<b>413 Payload Too Large (LLM context overflow).</b> Inputs are "
                   "chunked before any LLM call: text is split on paragraph, then "
                   "sentence boundaries into ≤ ~8,000-character chunks, each sent "
                   "separately and merged afterwards (first non-empty scalar wins, "
                   "lists unioned). No request is ever built from a whole raw page. "
                   "Chunk sizes are measured in characters with a safety margin "
                   "below the smallest model context in the chain, so a longer "
                   "article simply means more chunks, never a failed call."))
    story.append(P("<b>429 Too Many Requests.</b> Every client implements exponential "
                   "backoff with full jitter (base 1 s, ×2 per attempt, capped at "
                   "60 s) and honours <i>Retry-After</i> headers. The GitHub "
                   "enricher additionally reads <i>X-RateLimit-Remaining</i> / "
                   "<i>X-RateLimit-Reset</i> and sleeps until reset instead of "
                   "retrying blindly. The LLM chain treats a 429 as a signal to "
                   "fail over to the next provider (Gemini → Groq → DeepSeek) "
                   "rather than to retry the same provider indefinitely. "
                   "Checkpoints mean a pause costs nothing: work resumes from "
                   "the last saved record."))
    story.append(P("<b>Backoff is global, not per-request.</b> A shared limiter "
                   "tracks per-host and per-provider budgets so ten workers cannot "
                   "each burn the same quota in parallel."))

    story.append(P("4 &nbsp; Freshness tracking &amp; deduplication across distributed nodes", "h1"))
    story.extend(bullets([
        "<b>One clock, one rule.</b> Freshness is evaluated against a single "
        "collection timestamp: an item is fresh iff its normalised "
        "publication date is within the last 24 hours. All dates are parsed "
        "to UTC ISO-8601 at ingest (ISO, RFC-2822, Unix time, human formats, "
        "and relative forms such as “2 hours ago”); an unparseable date is "
        "stored as null and the item is excluded from fresh sets — never "
        "guessed into freshness.",
        "<b>Deterministic dedup keys.</b> Content identity is the source URL "
        "(normalised: lower-cased host, stripped tracking params, trailing "
        "slash). News/jobs additionally hash (title + source + date-day) so "
        "the same story syndicated under two URLs still dedups. Keys are "
        "stored in a shared set (Redis SET / Postgres unique index), so any "
        "worker can test membership in O(1).",
        "<b>Exactly-once-ish via idempotent upserts.</b> Workers claim a URL "
        "by inserting its key with status <i>claimed</i>; on success the "
        "record is upserted by the same key. A crashed worker's claims "
        "expire after a TTL and are re-claimed by another node — no item is "
        "processed twice, and no item is lost.",
        "<b>Freshness is re-checked, not assumed.</b> The 24-hour window is "
        "recomputed on every run from the stored publication date, so a "
        "stale item never leaks into a later run's output.",
        "<b>Anti-bot posture.</b> Async I/O with polite per-host concurrency, "
        "robots.txt respected, identifiable User-Agent, no CAPTCHA solving "
        "and no credential stuffing. For Cloudflare/Datadome-protected or "
        "heavily JS-rendered domains the strategy is: prefer the site's "
        "public API or RSS feed; if none exists, use a headless browser "
        "(Playwright) at human-like pacing with session reuse, and if a "
        "challenge still appears, back off and record the source as "
        "unavailable rather than forcing entry.",
    ]))

    story.append(P("5 &nbsp; Storage strategy", "h1"))
    story.append(P("<b>Primary store — PostgreSQL.</b> Relational data (startups, "
                   "products, papers, jobs, news, mapping log) lives in Postgres: "
                   "the schema is stable, joins across entities are the core "
                   "query pattern (\"all products of lab X\", \"papers citing repo "
                   "Y\"), and unique constraints on the natural dedup keys give "
                   "us exactly-once writes for free. JSONB columns hold the "
                   "flexible <i>content.data</i> payloads. Postgres also stores "
                   "the crawl state (claims, checkpoints, rate-limit budgets)."))
    story.append(P("<b>Vector store — pgvector (same Postgres).</b> Embeddings of "
                   "titles/abstracts/full text are stored in pgvector alongside "
                   "the rows they describe, so similarity search (\"papers like "
                   "this one\", \"startups similar to X\") needs no second system "
                   "and stays transactionally consistent with the metadata. A "
                   "dedicated vector DB (Qdrant/pgvector-as-a-service) becomes "
                   "worth it only past ~10M vectors."))
    story.append(P("<b>Graph store — Postgres + Apache AGE, or Neo4j at scale.</b> "
                   "The intelligence graph itself (startup → product → paper → "
                   "repo → author → job) is a graph workload. At demo scale it is "
                   "modelled as Postgres adjacency tables with recursive CTEs, "
                   "which is enough for 500k nodes. Past ~5M edges, or for "
                   "multi-hop traversal queries, a native graph store (Neo4j) is "
                   "the right move — the canonical JSON schema is unchanged, only "
                   "the query layer moves."))
    story.append(P("<b>Raw content — object storage (S3/GCS).</b> Fetched HTML, PDFs "
                   "and feed snapshots are stored as immutable blobs keyed by "
                   "content hash, referenced from Postgres. This keeps the DB "
                   "small, makes re-extraction reproducible, and gives an audit "
                   "trail for every output row."))

    story.append(P("6 &nbsp; Data quality &amp; verification", "h1"))
    story.extend(bullets([
        "<b>No hallucinated data.</b> Every record traces to a legitimate "
        "source URL stored on the row. Unknown fields are null, never "
        "invented (no estimated employee counts, stars, dates or URLs). "
        "Benchmark-style numbers are only recorded when a source publishes "
        "them.",
        "<b>What this run produced.</b> 1,000+ AI startups (Y Combinator "
        "directory), 1,000+ products (Hugging Face Spaces), 1,000 research "
        "papers (arXiv, newest first) with GitHub repos found via README "
        "citation search and live star counts, plus all 24-hour-fresh AI "
        "news and jobs from 5 + 5 sources. Row counts are printed by the "
        "pipeline and listed in the README.",
        "<b>Known limits.</b> GitHub's unauthenticated API allows 60 core "
        "requests/hour and 10 search requests/minute (shared with the "
        "egress IP), so star enrichment is checkpointed and resumable — it "
        "fills in over successive runs rather than failing the pipeline. "
        "The LLM chain runs in deterministic rule-based mode unless "
        "GEMINI_API_KEY / GROQ_API_KEY / DEEPSEEK_API_KEY are set. Google "
        "Sheets/Drive were not connected in the build environment, so the "
        "Sheet is created by scripts/create_sheet.py once they are "
        "connected; the CSVs are the source of truth until then.",
    ]))
    story.append(Spacer(1, 2 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=RULE))
    story.append(P("GraphOne / FrontierAtlas — AI Engineer demo task, The AI Signal. "
                   "Built by Abhishek Kumar. Repository: "
                   "github.com/darking324/graphone-pipeline (link submitted with "
                   "the task form).", "foot"))
    doc.build(story)
    from pypdf import PdfReader  # type: ignore
    pages = len(PdfReader(OUT).pages)
    print(f"architecture.pdf: {pages} page(s)")
    return pages


if __name__ == "__main__":
    n = build()
    if n > 3:
        raise SystemExit(f"ERROR: architecture.pdf is {n} pages (max 3)")
