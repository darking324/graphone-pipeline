"""Phase I — Product records from Hugging Face Spaces.

Source: the public Hugging Face Hub API (no key required).
A "Space" is a deployed AI application — a real product in the AI
ecosystem, with a public URL, an author (the startup/builder), likes,
SDK and last-modified date.

Pricing model: public Spaces are free for anyone to open and use, so
pricingModel is recorded as FREE. This is a documented rule, not a
per-product lookup — Spaces do not publish a price. Gated/private
spaces are skipped.
"""

from __future__ import annotations

from ..http_client import AsyncFetcher

HF_SPACES_API = "https://huggingface.co/api/spaces"
SOURCE_NAME = "Hugging Face Spaces"
TARGET_DEFAULT = 1100


def _to_record(space: dict, collected_at: str) -> dict | None:
    sid = space.get("id")
    if not sid or space.get("private"):
        return None
    card = space.get("cardData") or {}
    name = card.get("title") or sid.split("/")[-1]
    return {
        "schemaVersion": "1.0",
        "recordType": "PRODUCT",
        "source_name": SOURCE_NAME,
        "source_url": f"https://huggingface.co/spaces/{sid}",
        "content_startupName": space.get("author"),
        "content_pricingModel": "FREE",
        "collectedAt": collected_at,
        # useful extras
        "product_name": name,
        "product_url": f"https://huggingface.co/spaces/{sid}",
        "short_description": card.get("short_description"),
        "sdk": card.get("sdk") or space.get("sdk"),
        "likes": space.get("likes"),
        "lastModified": space.get("lastModified"),
    }


import json as _json
import re as _re

_NEXT_LINK_RE = _re.compile(r'<([^>]+)>;\s*rel="next"')


def _next_cursor(headers: dict) -> str | None:
    """The HF API returns the next page as a `Link: <url>; rel="next"` header."""
    link = headers.get("Link") or headers.get("link") or ""
    m = _NEXT_LINK_RE.search(link)
    if not m:
        return None
    from urllib.parse import parse_qs, urlparse

    qs = parse_qs(urlparse(m.group(1)).query)
    vals = qs.get("cursor")
    return vals[0] if vals else None


async def collect(
    fetcher: AsyncFetcher, *, target: int = TARGET_DEFAULT, collected_at: str
) -> list[dict]:
    """Paginate the public Spaces list (Link-header cursor) until `target`."""
    records: list[dict] = []
    seen: set[str] = set()
    cursor: str | None = None
    while len(records) < target:
        url = f"{HF_SPACES_API}?limit=100&full=true"
        if cursor:
            url += f"&cursor={cursor}"
        status, body, headers = await fetcher.fetch(url, accept="application/json")
        if status != 200:
            raise RuntimeError(f"HF Spaces API returned HTTP {status}")
        try:
            data = _json.loads(body.decode("utf-8", "replace"))
        except _json.JSONDecodeError:
            raise RuntimeError("HF Spaces API returned invalid JSON")
        if not isinstance(data, list) or not data:
            break
        for space in data:
            sid = space.get("id")
            if not sid or sid in seen:
                continue
            seen.add(sid)
            rec = _to_record(space, collected_at)
            if rec:
                records.append(rec)
        nxt = _next_cursor(headers)
        if not nxt or nxt == cursor:
            break
        cursor = nxt
    return records
