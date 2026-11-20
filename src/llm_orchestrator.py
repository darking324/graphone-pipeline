"""llm_orchestrator.py — Multi-tier LLM extraction engine with graceful degradation.

Part of the GraphOne / FrontierAtlas data-ingestion pipeline.

Fallback chain
------------
1. Gemini  (gemini-2.0-flash) — https://generativelanguage.googleapis.com/v1beta/...
2. Groq    (llama-3.3-70b-versatile) — https://api.groq.com/openai/v1/chat/completions
3. DeepSeek (deepseek-chat) — https://api.deepseek.com/chat/completions

Providers are tried in that order. If a provider returns HTTP 429, the call is
retried with exponential backoff + jitter (base 1s, factor 2, max 3 retries),
and only after those retries are exhausted do we move on to the next provider.
If no API key is configured, or every provider fails, extraction falls back to
a deterministic rule-based extractor (regex/heuristics, stdlib only).

413-avoidance / chunking strategy
---------------------------------
LLM providers reject oversized payloads (HTTP 413) and large inputs also
waste context. No single request is allowed to carry more than ~8000
characters of source text:

* ``extract()`` first splits the input with ``_split_chunks()``.
* Splitting is done on paragraph boundaries first (blank lines), then on
  sentence boundaries (``. ! ?`` followed by whitespace) inside any paragraph
  that is still too long, and finally on hard character slices as a last
  resort — so chunks never cut a word in half unless the input contains a
  single unbroken token longer than the limit.
* Each chunk is sent as its own LLM call; the per-kind results are then
  merged with ``_merge_chunk_results()`` (first non-empty value wins for
  scalar fields, lists are unioned in order, keywords are deduplicated).
* If only one chunk exists, it is sent as-is. The rule-based fallback runs
  over the whole text at once (no network, so no 413 risk) but still reuses
  the same per-kind field contract.

429 backoff strategy
--------------------
On a 429 response the request is retried up to 3 times with delays of
1s, 2s, 4s (base 1s * factor 2**attempt) plus a small random jitter
(uniform 0-0.5s) so concurrent retries do not synchronise. After the 3rd
retry the provider is abandoned and the next provider in the chain is tried.
All other HTTP errors / transport errors abort that provider immediately
(no retry) and move to the next provider.

No hallucination guarantee
--------------------------
Neither the LLM path nor the rule-based path is allowed to invent data.
* LLM output is parsed as JSON; any field missing from the JSON is None/empty
  rather than guessed, and a JSON parse failure for a chunk yields an empty
  result for that chunk.
* The rule-based fallback only reports what it can find in the text with
  regex/heuristics; any field it cannot find is None (or False/[] where the
  schema demands a boolean/list), never a fabricated value.

Timeout: 30 seconds per HTTP request.
Dependencies: Python standard library + ``requests`` only.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CHUNK_CHARS = 8000
REQUEST_TIMEOUT_S = 30
MAX_RETRIES_429 = 3
BACKOFF_BASE_S = 1.0
BACKOFF_FACTOR = 2.0
JITTER_S = 0.5

KINDS = ("news", "job", "startup", "product", "paper")

# Fields each kind returns (rule-based defaults + LLM contract).
KIND_FIELDS: dict[str, list[str]] = {
    "news": ["title", "published_date", "source", "summary", "keywords"],
    "job": ["title", "company", "is_remote", "role_family", "location"],
    "startup": ["entity_name", "website", "industry"],
    "product": ["product_name", "startup_name", "pricing_model"],
    "paper": ["title", "authors", "published_date", "arxiv_id"],
}

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------


class _Provider:
    """One LLM provider: endpoint, model, key env var, request builder."""

    def __init__(self, name: str, model: str, endpoint: str, env_var: str):
        self.name = name
        self.model = model
        self.endpoint = endpoint
        self.env_var = env_var

    @property
    def api_key(self) -> Optional[str]:
        key = os.environ.get(self.env_var)
        return key.strip() if key and key.strip() else None

    def headers(self) -> dict[str, str]:
        if self.name == "gemini":
            # Gemini takes the key as a query param; no auth header needed.
            return {"Content-Type": "application/json"}
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def url(self) -> str:
        if self.name == "gemini":
            return f"{self.endpoint}?key={self.api_key}"
        return self.endpoint

    def build_payload(self, system_prompt: str, user_text: str) -> dict[str, Any]:
        if self.name == "gemini":
            return {
                "contents": [
                    {
                        "parts": [
                            {"text": f"{system_prompt}\n\n---\n\n{user_text}"}
                        ]
                    }
                ],
                "generationConfig": {"responseMimeType": "application/json"},
            }
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }

    def parse_response(self, resp_json: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Extract the assistant's JSON object from a provider response."""
        try:
            if self.name == "gemini":
                text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
            else:
                text = resp_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
        return _safe_json_loads(text)


PROVIDERS: list[_Provider] = [
    _Provider(
        "gemini",
        "gemini-2.0-flash",
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.0-flash:generateContent",
        "GEMINI_API_KEY",
    ),
    _Provider(
        "groq",
        "llama-3.3-70b-versatile",
        "https://api.groq.com/openai/v1/chat/completions",
        "GROQ_API_KEY",
    ),
    _Provider(
        "deepseek",
        "deepseek-chat",
        "https://api.deepseek.com/chat/completions",
        "DEEPSEEK_API_KEY",
    ),
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _safe_json_loads(text: str) -> Optional[dict[str, Any]]:
    """Parse JSON, tolerating surrounding prose / code fences."""
    if not text:
        return None
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    # Fall back to the first {...} block in the text.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))


def _parse_date_flexible(text: str) -> Optional[str]:
    """Parse a date out of free text. Returns an ISO-8601 string or None.

    Handles: ISO dates (2026-11-20, 2026-11-20T15:30:00Z), 'YYYY-MM-DD',
    'Jan 5, 2026', '5 Jan 2026', '2 hours ago', '3 days ago', 'yesterday',
    'today'. Returns a plain 'YYYY-MM-DD' date for date-only inputs and a
    full ISO-8601 timestamp for relative times. Never invents a date.
    """
    if not text:
        return None
    now = datetime.now(timezone.utc)
    today = now.date()

    # 1. ISO datetime / date, e.g. 2026-11-20 or 2026-11-20T15:30:00Z.
    m = re.search(
        r"\b(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?"
        r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b",
        text,
    )
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            if m.group(4) is not None:
                dt = datetime(
                    year, month, day,
                    int(m.group(4)), int(m.group(5)),
                    int(m.group(6) or 0), tzinfo=timezone.utc,
                )
                return dt.isoformat().replace("+00:00", "Z")
            return datetime(year, month, day).date().isoformat()
        except ValueError:
            pass

    # 2. 'Jan 5, 2026' / 'January 5, 2026'
    m = re.search(
        rf"\b({_MONTH_RE})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b",
        text, re.IGNORECASE,
    )
    if m:
        try:
            return datetime(
                int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2))
            ).date().isoformat()
        except ValueError:
            pass

    # 3. '5 Jan 2026' / '5 January 2026'
    m = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\s+(\d{{4}})\b",
        text, re.IGNORECASE,
    )
    if m:
        try:
            return datetime(
                int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1))
            ).date().isoformat()
        except ValueError:
            pass

    # 4. Relative times: '2 hours ago', '3 days ago', '45 minutes ago'.
    m = re.search(
        r"\b(\d+)\s+(second|seconds|minute|minutes|hour|hours|day|days|"
        r"week|weeks|month|months)\s+ago\b",
        text, re.IGNORECASE,
    )
    if m:
        amount = int(m.group(1))
        unit = m.group(2).lower()
        delta = {
            "second": timedelta(seconds=amount), "seconds": timedelta(seconds=amount),
            "minute": timedelta(minutes=amount), "minutes": timedelta(minutes=amount),
            "hour": timedelta(hours=amount), "hours": timedelta(hours=amount),
            "day": timedelta(days=amount), "days": timedelta(days=amount),
            "week": timedelta(weeks=amount), "weeks": timedelta(weeks=amount),
            "month": timedelta(days=30 * amount), "months": timedelta(days=30 * amount),
        }[unit]
        return (now - delta).isoformat().replace("+00:00", "Z")

    # 5. 'yesterday' / 'today'
    if re.search(r"\byesterday\b", text, re.IGNORECASE):
        return (today - timedelta(days=1)).isoformat()
    if re.search(r"\btoday\b", text, re.IGNORECASE):
        return today.isoformat()

    return None


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping the sentence-ending punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into chunks of at most ``max_chars`` characters.

    Prefers paragraph boundaries, then sentence boundaries; falls back to a
    hard character slice only for a single unbroken token. This keeps every
    LLM request small enough to avoid HTTP 413 rejections.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    def add_piece(piece: str) -> None:
        nonlocal current
        if len(current) + len(piece) + 1 <= max_chars:
            current = f"{current}\n\n{piece}" if current else piece
            return
        flush()
        if len(piece) <= max_chars:
            current = piece
            return
        # Piece itself too long: split on sentences, then hard-slice.
        for sentence in _split_sentences(piece):
            if len(sentence) <= max_chars:
                add_piece(sentence)
            else:
                for i in range(0, len(sentence), max_chars):
                    add_piece(sentence[i : i + max_chars])

    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        add_piece(paragraph)

    flush()
    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Rule-based fallback (deterministic, no hallucination)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "are",
    "was", "were", "will", "would", "can", "could", "should", "its", "their",
    "they", "them", "then", "than", "into", "about", "after", "before", "when",
    "where", "which", "while", "also", "more", "most", "some", "such", "only",
    "over", "under", "between", "through", "being", "been", "your", "yours",
    "our", "ours", "his", "her", "him", "she", "its", "not", "but", "all",
    "any", "each", "other", "there", "here", "what", "who", "how", "why",
    "new", "announced", "announces", "says", "said", "according", "today",
    "yesterday", "company", "companies",
}

_INDUSTRY_KEYWORDS: dict[str, list[str]] = {
    "AI": ["artificial intelligence", "machine learning", "ai ", " ai", "llm",
           "generative ai", "deep learning", "neural"],
    "Fintech": ["fintech", "payment", "banking", "finance", "lending",
                "insurance", "crypto", "blockchain"],
    "Healthcare": ["health", "medical", "healthcare", "biotech", "pharma",
                   "clinical", "patient", "diagnostic"],
    "SaaS": ["saas", "software as a service", "b2b software", "cloud software"],
    "E-commerce": ["e-commerce", "ecommerce", "marketplace", "retail",
                   "shopping", "d2c"],
    "Developer Tools": ["developer tool", "devtool", "sdk", "api platform",
                        "devops", "developer platform"],
    "Robotics": ["robot", "robotics", "automation hardware", "drone"],
    "Biotech": ["biotech", "genomics", "drug discovery", "life science"],
}

_ROLE_FAMILY_KEYWORDS: dict[str, list[str]] = {
    "Engineering": ["engineer", "developer", "swe", "software", "backend",
                    "frontend", "full stack", "full-stack", "devops", "sde",
                    "platform", "systems", "mobile", "android", "ios"],
    "Data": ["data", "analyst", "analytics", "scientist", "ml ", "machine learning",
             "ai ", "research scientist", "data engineer"],
    "Product": ["product", "pm ", "product manager", "product owner"],
    "Design": ["design", "designer", "ux", "ui ", "graphic"],
    "Sales": ["sales", "account executive", "business development", "bdr", "sdr"],
    "Marketing": ["marketing", "growth", "seo", "content", "social media"],
    "Operations": ["operations", "ops", "support", "success", "recruit"],
}

_KNOWN_CITIES = [
    "Bengaluru", "Bangalore", "Mumbai", "Delhi", "New Delhi", "Gurugram",
    "Gurgaon", "Noida", "Greater Noida", "Hyderabad", "Pune", "Chennai",
    "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "San Francisco",
    "New York", "London", "Berlin", "Toronto", "Singapore", "Sydney",
    "Austin", "Seattle", "Boston", "Remote",
]


def _first_non_empty_line(text: str) -> Optional[str]:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def _keywords(text: str, limit: int = 10) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
    counts = Counter(w for w in words if w not in _STOPWORDS)
    return [w for w, _ in counts.most_common(limit)]


def _rule_extract_news(text: str) -> dict[str, Any]:
    title = _first_non_empty_line(text)
    if title:
        title = title[:200]
    sentences = _split_sentences(" ".join(text.split()))
    summary = " ".join(sentences[:2])[:500] if sentences else None
    return {
        "title": title,
        "published_date": _parse_date_flexible(text),
        "source": None,
        "summary": summary,
        "keywords": _keywords(text),
    }


def _rule_extract_job(text: str) -> dict[str, Any]:
    title = _first_non_empty_line(text)
    if title:
        title = title[:200]

    company = None
    m = re.search(r"Company:\s*(.+)", text, re.IGNORECASE)
    if m:
        company = m.group(1).strip().splitlines()[0].strip() or None
    if not company:
        m = re.search(r"\bat\s+([A-Z][A-Za-z0-9&.'\- ]{1,60})", text)
        if m:
            candidate = m.group(1).strip().rstrip(".,")
            # Only accept if it looks like a name, not a sentence fragment.
            if len(candidate.split()) <= 4 and not any(
                w.lower() in _STOPWORDS for w in candidate.split()
            ):
                company = candidate

    lowered = text.lower()
    is_remote = any(
        cue in lowered for cue in ("remote", "work from home", "wfh", "work-from-home")
    )

    role_family = "Other"
    title_lower = (title or "").lower()
    # Most specific match wins: longest matching keyword decides the family
    # (e.g. 'machine learning' beats the generic 'engineer' in
    # 'Senior Machine Learning Engineer').
    best_len = 0
    for family, keywords in _ROLE_FAMILY_KEYWORDS.items():
        for kw in keywords:
            if kw in title_lower and len(kw) > best_len:
                best_len = len(kw)
                role_family = family

    location = None
    m = re.search(r"Location:\s*(.+)", text, re.IGNORECASE)
    if m:
        location = m.group(1).strip().splitlines()[0].strip() or None
    if not location:
        for city in _KNOWN_CITIES:
            if re.search(rf"\b{re.escape(city)}\b", text, re.IGNORECASE):
                location = city
                break

    return {
        "title": title,
        "company": company,
        "is_remote": is_remote,
        "role_family": role_family,
        "location": location,
    }


def _rule_extract_startup(text: str) -> dict[str, Any]:
    entity_name = _first_non_empty_line(text)
    if entity_name:
        entity_name = entity_name[:200]
    m = re.search(r"https?://[^\s)\"'<>]+", text)
    website = m.group(0).rstrip(".,") if m else None
    lowered = text.lower()
    industry = None
    for name, keywords in _INDUSTRY_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            industry = name
            break
    return {"entity_name": entity_name, "website": website, "industry": industry}


def _rule_extract_product(text: str) -> dict[str, Any]:
    product_name = _first_non_empty_line(text)
    if product_name:
        product_name = product_name[:200]
    startup_name = None
    m = re.search(r"\bby\s+([A-Z][A-Za-z0-9&.'\- ]{1,60})", text)
    if m:
        startup_name = m.group(1).strip().rstrip(".,")
    lowered = text.lower()
    pricing_model = None
    if any(cue in lowered for cue in ("free", "open source", "open-source")):
        pricing_model = "FREE"
    elif "$" in text or "pricing" in lowered:
        pricing_model = "PAID"
    return {
        "product_name": product_name,
        "startup_name": startup_name,
        "pricing_model": pricing_model,
    }


def _rule_extract_paper(text: str) -> dict[str, Any]:
    title = _first_non_empty_line(text)
    if title:
        title = title[:200]
    authors: list[str] = []
    m = re.search(r"Authors?:\s*(.+)", text, re.IGNORECASE)
    if m:
        raw = m.group(1).strip().splitlines()[0]
        authors = [a.strip() for a in re.split(r",| and ", raw) if a.strip()]
    m = re.search(r"\b(\d{4}\.\d{4,5})\b", text)
    arxiv_id = m.group(1) if m else None
    return {
        "title": title,
        "authors": authors,
        "published_date": _parse_date_flexible(text),
        "arxiv_id": arxiv_id,
    }


_RULE_EXTRACTORS = {
    "news": _rule_extract_news,
    "job": _rule_extract_job,
    "startup": _rule_extract_startup,
    "product": _rule_extract_product,
    "paper": _rule_extract_paper,
}


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

_SYSTEM_PROMPTS = {
    "news": (
        "You extract structured data from AI news text. Reply with ONLY a JSON "
        "object with keys: title (string or null), published_date (ISO-8601 "
        "string or null), source (string or null), summary (string or null), "
        "keywords (array of up to 10 strings). Never invent values; use null "
        "for anything not present in the text."
    ),
    "job": (
        "You extract structured data from a job posting. Reply with ONLY a JSON "
        "object with keys: title (string or null), company (string or null), "
        "is_remote (boolean), role_family (one of Engineering, Data, Product, "
        "Design, Sales, Marketing, Operations, Other), location (string or "
        "null). Never invent values; use null for anything not present."
    ),
    "startup": (
        "You extract structured data about a startup. Reply with ONLY a JSON "
        "object with keys: entity_name (string or null), website (string or "
        "null), industry (one of AI, Fintech, Healthcare, SaaS, E-commerce, "
        "Developer Tools, Robotics, Biotech, or null). Never invent values."
    ),
    "product": (
        "You extract structured data about a product. Reply with ONLY a JSON "
        "object with keys: product_name (string or null), startup_name "
        "(string or null), pricing_model (one of FREE, PAID, or null). "
        "Never invent values."
    ),
    "paper": (
        "You extract structured data about a research paper. Reply with ONLY "
        "a JSON object with keys: title (string or null), authors (array of "
        "strings), published_date (ISO-8601 string or null), arxiv_id "
        "(string or null). Never invent values."
    ),
}


def _empty_result(kind: str) -> dict[str, Any]:
    """All-None/empty result for a kind (used when a chunk yields nothing)."""
    defaults: dict[str, Any] = {
        "title": None, "published_date": None, "source": None, "summary": None,
        "keywords": [], "company": None, "is_remote": False, "role_family": "Other",
        "location": None, "entity_name": None, "website": None, "industry": None,
        "product_name": None, "startup_name": None, "pricing_model": None,
        "authors": [], "arxiv_id": None,
    }
    return {field: defaults[field] for field in KIND_FIELDS[kind]}


def _merge_chunk_results(kind: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-chunk results: first non-empty scalar wins; lists unioned."""
    merged = _empty_result(kind)
    for result in results:
        for field in KIND_FIELDS[kind]:
            value = result.get(field)
            if isinstance(value, list):
                existing = merged[field]
                for item in value:
                    if item not in existing:
                        existing.append(item)
            elif value not in (None, "", []):
                if merged[field] in (None, "", []):
                    merged[field] = value
                # Keep the first non-empty value; later chunks do not override.
    return merged


class LLMOrchestrator:
    """Multi-tier LLM extraction with rule-based fallback.

    Reads GEMINI_API_KEY / GROQ_API_KEY / DEEPSEEK_API_KEY from the
    environment at call time (so keys set after construction still work).
    """

    def __init__(self, providers: Optional[list[_Provider]] = None):
        self.providers = providers if providers is not None else PROVIDERS

    # -- public API ------------------------------------------------------

    def extract(self, text: str, kind: str) -> dict[str, Any]:
        """Extract structured fields of ``kind`` from ``text``.

        Returns {"data": {...}, "method": "llm"|"rule_based",
        "provider": name|None, "model": model|None, "error": str|None}.
        """
        if kind not in KINDS:
            raise ValueError(f"unknown kind: {kind!r}; expected one of {KINDS}")
        if not text or not text.strip():
            return {
                "data": _empty_result(kind),
                "method": "rule_based",
                "provider": None,
                "model": None,
                "error": "empty input text",
            }

        configured = [p for p in self.providers if p.api_key]
        if not configured:
            return self._rule_based(text, kind, error="no API key configured")

        errors: list[str] = []
        for provider in configured:
            try:
                data = self._extract_with_provider(provider, text, kind)
            except Exception as exc:  # noqa: BLE001 - provider failure -> next
                errors.append(f"{provider.name}: {exc}")
                continue
            if data is not None:
                return {
                    "data": data,
                    "method": "llm",
                    "provider": provider.name,
                    "model": provider.model,
                    "error": None,
                }
            errors.append(f"{provider.name}: no usable JSON in response")

        # All providers failed -> deterministic rule-based fallback.
        return self._rule_based(text, kind, error="; ".join(errors))

    # -- internals -------------------------------------------------------

    def _rule_based(self, text: str, kind: str, error: Optional[str]) -> dict[str, Any]:
        return {
            "data": _RULE_EXTRACTORS[kind](text),
            "method": "rule_based",
            "provider": None,
            "model": None,
            "error": error,
        }

    def _extract_with_provider(
        self, provider: _Provider, text: str, kind: str
    ) -> Optional[dict[str, Any]]:
        """Run one provider over the (chunked) text. Returns None on failure."""
        chunks = _split_chunks(text)
        results: list[dict[str, Any]] = []
        for chunk in chunks:
            parsed = self._call_provider_once(provider, chunk, kind)
            if parsed is None:
                return None  # provider-level failure -> try next provider
            results.append(self._sanitize(kind, parsed))
        return _merge_chunk_results(kind, results)

    def _call_provider_once(
        self, provider: _Provider, chunk: str, kind: str
    ) -> Optional[dict[str, Any]]:
        """One HTTP call with 429 exponential backoff + jitter.

        Returns the parsed JSON dict, or None if the provider failed
        (after retries for 429s). Raises on unexpected transport errors.
        """
        payload = provider.build_payload(_SYSTEM_PROMPTS[kind], chunk)
        last_error: Optional[str] = None
        for attempt in range(MAX_RETRIES_429 + 1):
            try:
                resp = requests.post(
                    provider.url(),
                    headers=provider.headers(),
                    json=payload,
                    timeout=REQUEST_TIMEOUT_S,
                )
            except requests.RequestException as exc:
                last_error = str(exc)
                return None  # transport error: no retry, next provider
            if resp.status_code == 429:
                if attempt < MAX_RETRIES_429:
                    delay = BACKOFF_BASE_S * (BACKOFF_FACTOR ** attempt)
                    delay += random.uniform(0, JITTER_S)
                    time.sleep(delay)
                    continue
                last_error = "HTTP 429 after retries"
                return None
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}"
                return None
            parsed = provider.parse_response(resp.json())
            if parsed is None:
                last_error = "unparsable response"
                return None
            return parsed
        # Unreachable in practice; defensive.
        _ = last_error
        return None

    @staticmethod
    def _sanitize(kind: str, parsed: dict[str, Any]) -> dict[str, Any]:
        """Keep only the declared fields for the kind; fill missing with defaults."""
        empty = _empty_result(kind)
        for field in KIND_FIELDS[kind]:
            if field in parsed and parsed[field] is not None:
                empty[field] = parsed[field]
        return empty


# ---------------------------------------------------------------------------
# Self-test (rule-based path only — needs no network and no API keys)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for _key in ("GEMINI_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY"):
        os.environ.pop(_key, None)

    orch = LLMOrchestrator()

    # news
    news = orch.extract(
        "OpenAI launches GPT-5 for developers\n"
        "OpenAI announced GPT-5 on 2026-11-20. The new model is faster and "
        "cheaper than GPT-4o. Developers can access it via the API today.",
        "news",
    )
    assert news["method"] == "rule_based", news
    assert news["data"]["title"] == "OpenAI launches GPT-5 for developers"
    assert news["data"]["published_date"] == "2026-11-20", news["data"]
    assert news["data"]["source"] is None
    assert "GPT-5" in news["data"]["summary"]
    assert len(news["data"]["keywords"]) <= 10
    assert news["provider"] is None and news["model"] is None

    # job
    job = orch.extract(
        "Senior Machine Learning Engineer\n"
        "Company: Acme AI\nLocation: Bengaluru\n"
        "We are hiring a remote machine learning engineer to build LLM systems.",
        "job",
    )
    assert job["data"]["company"] == "Acme AI", job["data"]
    assert job["data"]["is_remote"] is True
    assert job["data"]["role_family"] == "Data", job["data"]
    assert job["data"]["location"] == "Bengaluru"

    # job with no company/location info: nothing invented
    job2 = orch.extract("Backend Engineer\nBuild APIs.", "job")
    assert job2["data"]["company"] is None
    assert job2["data"]["location"] is None
    assert job2["data"]["role_family"] == "Engineering"

    # startup
    startup = orch.extract(
        "Perplexity\nPerplexity is an AI-powered search engine. "
        "Visit https://www.perplexity.ai for more.",
        "startup",
    )
    assert startup["data"]["website"] == "https://www.perplexity.ai"
    assert startup["data"]["industry"] == "AI"

    # product
    product = orch.extract(
        "ChatGPT by OpenAI\nChatGPT is free to use with an optional paid plan.",
        "product",
    )
    assert product["data"]["startup_name"] == "OpenAI", product["data"]
    assert product["data"]["pricing_model"] == "FREE"

    # paper
    paper = orch.extract(
        "Attention Is All You Need\n"
        "Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar\n"
        "Published 2017-06-12. arXiv:1706.03762",
        "paper",
    )
    assert paper["data"]["arxiv_id"] == "1706.03762", paper["data"]
    assert paper["data"]["authors"] == [
        "Ashish Vaswani", "Noam Shazeer", "Niki Parmar",
    ], paper["data"]
    assert paper["data"]["published_date"] == "2017-06-12"

    # date helper checks
    assert _parse_date_flexible("posted 2 hours ago") is not None
    assert _parse_date_flexible("Jan 5, 2026") == "2026-01-05"
    assert _parse_date_flexible("5 Jan 2026") == "2026-01-05"
    assert _parse_date_flexible("no date here") is None

    # chunking: a long input still extracts (rule-based) and chunks stay small
    long_text = ("Paragraph about AI models. " * 400 + "\n\n") * 6
    chunks = _split_chunks(long_text)
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks), [len(c) for c in chunks]
    assert len(chunks) > 1
    big = orch.extract(long_text, "news")
    assert big["method"] == "rule_based"

    # invalid kind
    try:
        orch.extract("x", "bogus")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown kind")

    print("llm_orchestrator self-test OK")
