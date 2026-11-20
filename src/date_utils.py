"""Date normalisation for the GraphOne pipeline (Phase II).

Every source publishes dates differently: ISO-8601, RFC-2822, Unix
timestamps, human formats ('Jan 5, 2026', '5 Jan 2026'), and relative
times ('2 hours ago', 'yesterday', 'just now'). All of them are
normalised here to timezone-aware UTC datetimes and ISO-8601 strings.
A date that cannot be parsed is returned as None — never guessed.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_RELATIVE_RE = re.compile(
    r"(?P<num>\d+|a|an)\s+"
    r"(?P<unit>second|minute|hour|day|week|month|year)s?\s+ago",
    re.IGNORECASE,
)
_HUMAN_RE = re.compile(
    r"(?P<mon>[A-Za-z]{3,9})\.?\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<year>\d{4})"
)
_HUMAN_RE2 = re.compile(
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?P<mon>[A-Za-z]{3,9})\.?,?\s+(?P<year>\d{4})"
)
_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)


def _num(word: str) -> int:
    return 1 if word.lower() in ("a", "an") else int(word)


def parse_date_flexible(value, now: datetime | None = None) -> datetime | None:
    """Parse almost any date representation into an aware UTC datetime.

    Returns None when the value cannot be parsed — never a guessed date.
    """
    if value is None:
        return None
    now = now or datetime.now(timezone.utc)
    if isinstance(value, datetime):
        dt = value
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        # Unix timestamp (seconds or milliseconds)
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            return None
    s = str(value).strip()
    if not s:
        return None

    low = s.lower()
    if low in ("just now", "now", "today"):
        return now
    if low == "yesterday":
        return now - timedelta(days=1)

    m = _RELATIVE_RE.search(s)
    if m:
        n = _num(m.group("num"))
        unit = m.group("unit").lower()
        delta = {
            "second": timedelta(seconds=n),
            "minute": timedelta(minutes=n),
            "hour": timedelta(hours=n),
            "day": timedelta(days=n),
            "week": timedelta(weeks=n),
            "month": timedelta(days=30 * n),
            "year": timedelta(days=365 * n),
        }[unit]
        return now - delta

    # ISO-8601
    if _ISO_RE.match(s):
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    # RFC-2822 (RSS / email style)
    try:
        dt = parsedate_to_datetime(s)
        if dt is not None:
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        pass

    # Human formats: "Jan 5, 2026" / "5 Jan 2026"
    for rx in (_HUMAN_RE, _HUMAN_RE2):
        m = rx.search(s)
        if m:
            mon = _MONTHS.get(m.group("mon").lower()[:3])
            if mon:
                try:
                    return datetime(
                        int(m.group("year")), mon, int(m.group("day")),
                        tzinfo=timezone.utc,
                    )
                except ValueError:
                    return None

    # Bare date "2026-01-05"
    try:
        dt = datetime.strptime(s[:10], "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    return None


def to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def is_within_hours(dt: datetime | None, hours: float, now: datetime | None = None) -> bool:
    if dt is None:
        return False
    now = now or datetime.now(timezone.utc)
    return (now - dt) <= timedelta(hours=hours) and dt <= now + timedelta(minutes=5)
