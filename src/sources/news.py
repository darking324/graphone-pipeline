"""Phase II — AI news ingestion from 5 RSS sources.

Sources (all public RSS feeds, no keys):
  1. TechCrunch — Artificial Intelligence
  2. MIT Technology Review — Artificial Intelligence
  3. Google AI Blog
  4. OpenAI Blog
  5. Google DeepMind Blog

Freshness rule (the task's hard requirement): only items published
within the last 24 hours are kept. Full text is fetched from the
article page (robots.txt respected) and reduced with the LLM
orchestrator's rule-based structuring (summary + keywords) — no text is
ever invented; when a page cannot be fetched the summary falls back to
the feed's own summary.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import feedparser
from bs4 import BeautifulSoup

from ..date_utils import is_within_hours, parse_date_flexible, to_iso
from ..http_client import AsyncFetcher
from ..llm_orchestrator import LLMOrchestrator

SOURCES = [
    ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/"),
    ("MIT Technology Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed/"),
    ("Google AI Blog", "https://blog.google/technology/ai/rss/"),
    ("OpenAI Blog", "https://openai.com/blog/rss.xml"),
    ("DeepMind Blog", "https://deepmind.google/blog/rss.xml"),
]
FRESHNESS_HOURS = 24
MAX_ARTICLES_PER_SOURCE = 15
FULL_TEXT_CAP = 8000


def _extract_full_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    container = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=lambda c: c and "article" in c)
        or soup.body
    )
    if container is None:
        return ""
    paras = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    text = "\n".join(p for p in paras if len(p) > 40)
    return text[:FULL_TEXT_CAP]


async def _fetch_article(fetcher: AsyncFetcher, url: str) -> str:
    try:
        status, html = await fetcher.fetch_text(url, check_robots=True)
        if status == 451:
            return ""  # robots.txt disallows
        if status != 200:
            return ""
        return _extract_full_text(html)
    except Exception:
        return ""


async def collect(
    fetcher: AsyncFetcher, *, collected_at: str, now: datetime | None = None
) -> list[dict]:
    """Collect <24h-fresh AI news with full text from the 5 RSS sources."""
    now = now or datetime.now(timezone.utc)
    orchestrator = LLMOrchestrator()
    records: list[dict] = []
    for source_name, feed_url in SOURCES:
        try:
            status, body, _ = await fetcher.fetch(feed_url)
            if status != 200:
                print(f"[news] {source_name}: feed HTTP {status}, skipped")
                continue
            feed = feedparser.parse(body)
        except Exception as exc:
            print(f"[news] {source_name}: fetch failed ({exc}), skipped")
            continue
        fresh = []
        for entry in feed.entries[:40]:
            raw_date = getattr(entry, "published", None) or getattr(entry, "updated", None)
            dt = parse_date_flexible(raw_date, now)
            if not is_within_hours(dt, FRESHNESS_HOURS, now):
                continue
            fresh.append((entry, dt))
        fresh = fresh[:MAX_ARTICLES_PER_SOURCE]
        # Full text, in order of preference:
        #  1. content:encoded in the feed itself (MIT Tech Review ships it)
        #  2. the article page, fetched only when robots.txt allows it
        #  3. the feed summary (never empty-handed)
        feed_texts: list[str] = []
        to_fetch: list[str] = []
        for entry, _ in fresh:
            content = getattr(entry, "content", None)
            text = ""
            if content:
                raw = content[0].get("value", "") if isinstance(content[0], dict) else getattr(content[0], "value", "")
                text = BeautifulSoup(raw or "", "lxml").get_text(" ", strip=True)
            feed_texts.append(text)
            to_fetch.append("" if text else getattr(entry, "link", ""))
        async def _maybe_fetch(url: str) -> str:
            return await _fetch_article(fetcher, url) if url else ""

        fetched = await asyncio.gather(*[_maybe_fetch(u) for u in to_fetch])
        texts = [ft or fx for ft, fx in zip(feed_texts, fetched)]
        for (entry, dt), full_text in zip(fresh, texts):
            title = getattr(entry, "title", "") or ""
            link = getattr(entry, "link", "") or ""
            feed_summary = getattr(entry, "summary", "") or ""
            summary_text = full_text or BeautifulSoup(feed_summary, "lxml").get_text(" ", strip=True)
            structured = orchestrator.extract(
                f"{title}\n\n{summary_text[:4000]}", "news"
            )["data"]
            records.append(
                {
                    "schemaVersion": "1.0",
                    "recordType": "NEWS",
                    "source_name": source_name,
                    "source_url": link,
                    "content_title": title,
                    "content_published_date": to_iso(dt),
                    "content_summary": (structured.get("summary") or summary_text[:500]),
                    "content_keywords": "; ".join(structured.get("keywords") or []),
                    "content_full_text": full_text,
                    "collectedAt": collected_at,
                }
            )
        print(f"[news] {source_name}: {len(fresh)} fresh items")
    return records
