"""Phase II — AI job ingestion from 5 public job boards.

Sources (all public JSON/RSS APIs, no keys):
  1. RemoteOK API            https://remoteok.com/api
  2. Remotive API            https://remotive.com/api/remote-jobs
  3. Jobicy API              https://jobicy.com/api/v2/remote-jobs
  4. Arbeitnow API           https://www.arbeitnow.com/api/job-board-api
  5. Hacker News "Who is hiring" (Algolia search_by_date, tag=job)

Freshness rule: only jobs posted within the last 24 hours are kept.
AI relevance: title/tags/description must mention AI/ML/data/LLM or a
related term. Role family and remote flag come from the LLM
orchestrator's rule-based classifier (deterministic, no guessing);
company/location come from the board's own fields.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from ..date_utils import is_within_hours, parse_date_flexible, to_iso
from ..http_client import AsyncFetcher
from ..llm_orchestrator import LLMOrchestrator

FRESHNESS_HOURS = 24
AI_KEYWORDS = re.compile(
    r"\b(ai|artificial intelligence|machine learning|\bml\b|deep learning|"
    r"llm|large language|generative ai|genai|nlp|natural language|"
    r"computer vision|data scientist|data science|data engineer|"
    r"mlops|ml ops|prompt engineer|ai engineer|"
    r"research scientist|applied scientist|ai researcher)\b",
    re.IGNORECASE,
)


def _is_ai(text: str) -> bool:
    """AI relevance from TITLE / TAGS / CATEGORY only.

    Free-text descriptions are deliberately NOT matched: they produce
    false positives (e.g. a Flutter-engineer ad saying "We love AI").
    """
    return bool(text and AI_KEYWORDS.search(text))


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "")


def _get(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, ""):
            return d[k]
    return default


def _job_record(
    *, source_name: str, source_url: str, title: str, company, posted,
    is_remote: bool, role_family: str, location, collected_at: str,
    salary=None, tags=None,
) -> dict:
    return {
        "schemaVersion": "1.0",
        "recordType": "JOB",
        "source_name": source_name,
        "source_url": source_url,
        "content_company": company,
        "content_date": to_iso(posted),
        "content_is_remote": bool(is_remote),
        "content_role_family": role_family,
        "collectedAt": collected_at,
        # extras
        "job_title": title,
        "job_url": source_url,
        "location": (location or "").rstrip(".").strip() or None,
        "salary": salary,
        "tags": "; ".join(tags or []),
    }


async def _remoteok(fetcher, orchestrator, collected_at, now) -> list[dict]:
    out = []
    # Plain feed: the ?tag=ai endpoint returns a small, stale subset.
    status, data = await fetcher.fetch_json("https://remoteok.com/api")
    if status != 200 or not isinstance(data, list):
        print(f"[jobs] RemoteOK: HTTP {status}, skipped")
        return out
    for job in data[1:]:  # first item is a legal notice
        text = f"{job.get('position','')} {' '.join(job.get('tags') or [])}"
        if not _is_ai(text):
            continue
        posted = parse_date_flexible(job.get("date"), now)
        if not is_within_hours(posted, FRESHNESS_HOURS, now):
            continue
        structured = orchestrator.extract(_strip_html(text), "job")["data"]
        out.append(_job_record(
            source_name="RemoteOK",
            source_url=job.get("url") or "https://remoteok.com/",
            title=job.get("position") or "",
            company=job.get("company"),
            posted=posted,
            is_remote=True,  # RemoteOK lists remote-only jobs
            role_family=structured.get("role_family") or "Other",
            location=job.get("location"),
            collected_at=collected_at,
            salary=(f"{job.get('salary_min','')}-{job.get('salary_max','')}"
                    if job.get("salary_min") else None),
            tags=job.get("tags") or [],
        ))
    print(f"[jobs] RemoteOK: {len(out)} fresh AI jobs")
    return out


async def _remotive(fetcher, orchestrator, collected_at, now) -> list[dict]:
    out = []
    status, data = await fetcher.fetch_json(
        "https://remotive.com/api/remote-jobs?limit=100"
    )
    if status != 200 or not isinstance(data, dict):
        print(f"[jobs] Remotive: HTTP {status}, skipped")
        return out
    for job in data.get("jobs", []):
        text = f"{job.get('title','')} {job.get('category','')} {' '.join(job.get('tags') or [])}"
        if not _is_ai(text):
            continue
        posted = parse_date_flexible(job.get("publication_date"), now)
        if not is_within_hours(posted, FRESHNESS_HOURS, now):
            continue
        structured = orchestrator.extract(text[:2000], "job")["data"]
        out.append(_job_record(
            source_name="Remotive",
            source_url=job.get("url") or "https://remotive.com/",
            title=job.get("title") or "",
            company=job.get("company_name"),
            posted=posted,
            is_remote=True,
            role_family=structured.get("role_family") or "Other",
            location=job.get("candidate_required_location"),
            collected_at=collected_at,
            salary=job.get("salary"),
            tags=job.get("tags") or [],
        ))
    print(f"[jobs] Remotive: {len(out)} fresh AI jobs")
    return out


async def _jobicy(fetcher, orchestrator, collected_at, now) -> list[dict]:
    out = []
    status, data = await fetcher.fetch_json(
        "https://jobicy.com/api/v2/remote-jobs?count=100&tag=ai"
    )
    if status != 200 or not isinstance(data, dict):
        print(f"[jobs] Jobicy: HTTP {status}, skipped")
        return out
    for job in data.get("jobs", []):
        title = _get(job, "jobTitle", "title", default="")
        company = _get(job, "companyName", "company", default=None)
        text = f"{title} {_get(job, 'jobType', default='')} {' '.join(_get(job, 'jobTags', default=[]) or [])}"
        if not _is_ai(text):
            continue
        posted = parse_date_flexible(
            _get(job, "jobPostedDate", "pubDate", "date", default=None), now
        )
        if not is_within_hours(posted, FRESHNESS_HOURS, now):
            continue
        structured = orchestrator.extract(text[:2000], "job")["data"]
        out.append(_job_record(
            source_name="Jobicy",
            source_url=_get(job, "url", "jobUrl", default="https://jobicy.com/"),
            title=title,
            company=company,
            posted=posted,
            is_remote=True,
            role_family=structured.get("role_family") or "Other",
            location=_get(job, "jobLocation", "location", default=None),
            collected_at=collected_at,
            salary=_get(job, "salary", default=None),
            tags=_get(job, "jobTags", default=[]) or [],
        ))
    print(f"[jobs] Jobicy: {len(out)} fresh AI jobs")
    return out


async def _arbeitnow(fetcher, orchestrator, collected_at, now) -> list[dict]:
    out = []
    status, data = await fetcher.fetch_json(
        "https://www.arbeitnow.com/api/job-board-api?search=AI"
    )
    if status != 200 or not isinstance(data, dict):
        print(f"[jobs] Arbeitnow: HTTP {status}, skipped")
        return out
    for job in data.get("data", []):
        # Title-only match: descriptions are long German texts that
        # mention AI tangentially ("We love AI") and caused 53 false
        # positives in the first run.
        text = job.get("title") or ""
        if not _is_ai(text):
            continue
        posted = parse_date_flexible(job.get("created_at"), now)
        if not is_within_hours(posted, FRESHNESS_HOURS, now):
            continue
        structured = orchestrator.extract(
            _strip_html(f"{text} {(job.get('description') or '')[:800]}")[:2000],
            "job",
        )["data"]
        out.append(_job_record(
            source_name="Arbeitnow",
            source_url=f"https://www.arbeitnow.com/{job.get('slug','')}",
            title=job.get("title") or "",
            company=job.get("company_name"),
            posted=posted,
            is_remote=bool(structured.get("is_remote")) or "remote" in text.lower(),
            role_family=structured.get("role_family") or "Other",
            location=job.get("location"),
            collected_at=collected_at,
            tags=job.get("job_types") or [],
        ))
    print(f"[jobs] Arbeitnow: {len(out)} fresh AI jobs")
    return out


async def _hn_jobs(fetcher, orchestrator, collected_at, now) -> list[dict]:
    out = []
    status, data = await fetcher.fetch_json(
        "https://hn.algolia.com/api/v1/search_by_date?tags=job&hitsPerPage=100"
    )
    if status != 200 or not isinstance(data, dict):
        print(f"[jobs] HN Jobs: HTTP {status}, skipped")
        return out
    for hit in data.get("hits", []):
        title = hit.get("title") or ""
        if not _is_ai(title):
            continue
        posted = parse_date_flexible(hit.get("created_at"), now)
        if not is_within_hours(posted, FRESHNESS_HOURS, now):
            continue
        structured = orchestrator.extract(title, "job")["data"]
        company = structured.get("company")
        url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
        out.append(_job_record(
            source_name="Hacker News Jobs",
            source_url=url,
            title=title,
            company=company,
            posted=posted,
            is_remote=bool(structured.get("is_remote")),
            role_family=structured.get("role_family") or "Other",
            location=structured.get("location"),
            collected_at=collected_at,
        ))
    print(f"[jobs] HN Jobs: {len(out)} fresh AI jobs")
    return out


async def collect(
    fetcher: AsyncFetcher, *, collected_at: str, now: datetime | None = None
) -> list[dict]:
    """Collect <24h-fresh AI jobs from all 5 boards, deduplicated by URL."""
    now = now or datetime.now(timezone.utc)
    orchestrator = LLMOrchestrator()
    all_jobs: list[dict] = []
    for fn in (_remoteok, _remotive, _jobicy, _arbeitnow, _hn_jobs):
        try:
            all_jobs.extend(await fn(fetcher, orchestrator, collected_at, now))
        except Exception as exc:
            print(f"[jobs] {fn.__name__} failed: {exc}")
    seen: set[str] = set()
    deduped = []
    for job in all_jobs:
        key = (job.get("job_url") or "") + "|" + (job.get("job_title") or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(job)
    return deduped
