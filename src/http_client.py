"""Polite async HTTP client for the GraphOne pipeline.

Rules enforced here (Phase V — anti-bot & scale thinking):
- Per-host concurrency limits and minimum delays (no hammering).
- Exponential backoff + jitter on 429 / 5xx; honour Retry-After.
- robots.txt respected for ordinary web pages (APIs are exempt — they are
  designed for programmatic access and publish their own rate limits).
- Identifiable User-Agent so operators can contact us.
"""

from __future__ import annotations

import asyncio
import random
import time
import urllib.parse
import urllib.robotparser

import aiohttp

USER_AGENT = (
    "GraphOnePipeline/1.0 (AI Engineer demo task; +https://github.com/darking324/graphone-pipeline)"
)


class RateLimiter:
    """Token-bucket-ish limiter: at most `rate_per_minute` calls, spaced out."""

    def __init__(self, rate_per_minute: float, burst: int = 1):
        self._interval = 60.0 / max(rate_per_minute, 0.01)
        self._burst = max(burst, 1)
        self._lock = asyncio.Lock()
        self._next_ok = 0.0
        self._tokens = float(self._burst)

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            # refill tokens
            elapsed = now - self._next_ok + self._interval * self._tokens
            self._tokens = min(self._burst, self._tokens + elapsed / self._interval)
            if self._tokens < 1:
                wait = (1 - self._tokens) * self._interval
                await asyncio.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1
            self._next_ok = time.monotonic()


class AsyncFetcher:
    """A shared aiohttp session with retries, backoff and per-host limits."""

    def __init__(
        self,
        *,
        max_concurrency: int = 8,
        min_delay_seconds: float = 0.5,
        timeout_seconds: int = 30,
        max_retries: int = 4,
    ):
        self._sem = asyncio.Semaphore(max_concurrency)
        self._min_delay = min_delay_seconds
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._max_retries = max_retries
        self._session: aiohttp.ClientSession | None = None
        self._last_host_call: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}

    async def __aenter__(self) -> "AsyncFetcher":
        # trust_env=True: honour the VM's egress proxy (HTTP(S)_PROXY env
        # vars). Without it, direct TLS connections fail in this sandbox.
        self._session = aiohttp.ClientSession(
            timeout=self._timeout,
            trust_env=True,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/html"},
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._session is not None:
            await self._session.close()

    # -- robots ---------------------------------------------------------
    def robots_allowed(self, url: str) -> bool:
        """Best-effort robots.txt check for ordinary web pages (not APIs)."""
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc
        if host not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                rp.set_url(f"{parsed.scheme}://{host}/robots.txt")
                rp.read()
            except Exception:
                return True  # fail open: unreachable robots.txt = allowed
            self._robots[host] = rp
        try:
            return self._robots[host].can_fetch(USER_AGENT, url)
        except Exception:
            return True

    # -- core fetch -----------------------------------------------------
    async def fetch(
        self, url: str, *, accept: str | None = None, check_robots: bool = False
    ) -> tuple[int, bytes, dict]:
        """GET with retries. Returns (status, body, headers)."""
        if check_robots and not self.robots_allowed(url):
            return 451, b"", {"X-Robots": "disallowed"}

        host = urllib.parse.urlparse(url).netloc
        headers = {}
        if accept:
            headers["Accept"] = accept

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            # polite per-host spacing
            now = time.monotonic()
            gap = self._last_host_call.get(host, 0.0) + self._min_delay - now
            if gap > 0:
                await asyncio.sleep(gap)
            async with self._sem:
                self._last_host_call[host] = time.monotonic()
                try:
                    assert self._session is not None
                    async with self._session.get(url, headers=headers) as resp:
                        body = await resp.read()
                        if resp.status == 429 or resp.status >= 500:
                            retry_after = resp.headers.get("Retry-After")
                            wait = self._backoff(attempt, retry_after)
                            if attempt < self._max_retries:
                                await asyncio.sleep(wait)
                                continue
                        return resp.status, body, dict(resp.headers)
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    last_exc = exc
                    if attempt < self._max_retries:
                        await asyncio.sleep(self._backoff(attempt))
                        continue
        raise RuntimeError(f"GET {url} failed after retries: {last_exc}")

    async def fetch_json(self, url: str, **kw) -> tuple[int, object]:
        status, body, _ = await self.fetch(url, accept="application/json", **kw)
        if status != 200:
            return status, None
        import json

        try:
            return status, json.loads(body.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return status, None

    async def fetch_text(self, url: str, **kw) -> tuple[int, str]:
        status, body, _ = await self.fetch(url, **kw)
        return status, body.decode("utf-8", "replace")

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None = None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 300.0) + random.uniform(0, 1)
            except ValueError:
                pass
        # exponential backoff with full jitter: base 1s, cap 60s
        cap = min(60.0, 2.0**attempt)
        return random.uniform(0, cap)
