"""Phase I — GitHub enrichment for research papers.

For every paper we find its official code repository and current star
count, using only verifiable evidence:

1. If the arXiv comment already names a github.com URL, that URL is used
   and the star count is fetched from the public GitHub repo API.
2. Otherwise the GitHub *repository search* API is queried for
   `"<arxiv_id>" in:readme` — a repository is only recorded when its
   README literally cites the paper's arXiv ID. The search response
   itself carries the star count, so one call gives both URL and stars.

Rate limits (unauthenticated): core API 60/hour, search API 10/minute,
shared with the whole egress IP. The enricher therefore:
- paces itself (default 7s between search calls, ~8/min),
- reads X-RateLimit-Remaining / X-RateLimit-Reset and sleeps until reset
  instead of hammering,
- checkpoints after every paper (JSON), so a stop/restart resumes exactly
  where it left off and never re-spends budget on finished papers.

No repository URL is ever guessed. A paper with no verified repo keeps
github_url empty and github_stars null (the task schema allows "if any").
"""

from __future__ import annotations

import json
import os
import time
from urllib.parse import quote

import requests

GITHUB_API = "https://api.github.com"
USER_AGENT = "GraphOnePipeline/1.0 (AI Engineer demo task)"
SEARCH_DELAY_SECONDS = 7.5
CORE_DELAY_SECONDS = 2.0


def _headers() -> dict:
    return {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}


def _rate_limit_sleep(resp: requests.Response, resource: str) -> float:
    """Seconds to wait when a rate limit is hit (0 if not rate-limited)."""
    if resp.status_code not in (403, 429):
        return 0.0
    remaining = resp.headers.get("X-RateLimit-Remaining")
    if remaining not in (None, "0"):
        return 0.0
    reset = resp.headers.get("X-RateLimit-Reset")
    if reset:
        wait = float(reset) - time.time() + 5
        return max(wait, 0.0)
    return 60.0


def _load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_checkpoint(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _search_repo_for_arxiv(arxiv_id: str) -> tuple[str | None, int | None]:
    """Search READMEs citing the arXiv ID; return (repo_url, stars)."""
    url = (
        f"{GITHUB_API}/search/repositories"
        f"?q={quote(arxiv_id)}+in:readme&per_page=3"
    )
    resp = requests.get(url, headers=_headers(), timeout=30)
    wait = _rate_limit_sleep(resp, "search")
    if wait:
        raise RateLimitHit(wait, "search")
    if resp.status_code in (403, 429):
        # Abuse/secondary limit without a reset header: back off briefly.
        raise RateLimitHit(60.0, "search")
    if resp.status_code >= 500:
        raise RateLimitHit(30.0, "search")
    if resp.status_code != 200:
        return None, None
    items = resp.json().get("items", [])
    if not items:
        return None, None
    top = items[0]
    return top.get("html_url"), top.get("stargazers_count")


def _stars_for_repo(repo_url: str) -> int | None:
    """Fetch current star count for a known github.com/owner/repo URL."""
    parts = repo_url.rstrip("/").split("github.com/")[-1].split("/")
    if len(parts) < 2:
        return None
    url = f"{GITHUB_API}/repos/{parts[0]}/{parts[1]}"
    resp = requests.get(url, headers=_headers(), timeout=30)
    wait = _rate_limit_sleep(resp, "core")
    if wait:
        raise RateLimitHit(wait, "core")
    if resp.status_code in (403, 429):
        raise RateLimitHit(60.0, "core")
    if resp.status_code >= 500:
        raise RateLimitHit(30.0, "core")
    if resp.status_code != 200:
        return None
    return resp.json().get("stargazers_count")


class RateLimitHit(Exception):
    def __init__(self, wait_seconds: float, resource: str):
        super().__init__(f"GitHub {resource} rate limit hit; wait {wait_seconds:.0f}s")
        self.wait_seconds = wait_seconds
        self.resource = resource


def enrich_papers(
    papers: list[dict],
    checkpoint_path: str,
    *,
    search_delay: float = SEARCH_DELAY_SECONDS,
    core_delay: float = CORE_DELAY_SECONDS,
    max_wait_seconds: float = 900.0,
    verbose: bool = True,
) -> list[dict]:
    """Enrich paper records with github_url/stars, resumable via checkpoint.

    The checkpoint maps arxiv_id -> {"github_url": ..., "github_stars": ...,
    "github_match_method": ...}. Returns the papers list with the three
    content_github_* fields filled in place.
    """
    checkpoint = _load_checkpoint(checkpoint_path)
    done = 0
    for paper in papers:
        arxiv_id = paper.get("arxiv_id")
        if not arxiv_id:
            continue
        if arxiv_id in checkpoint:
            hit = checkpoint[arxiv_id]
            paper["content_github_url"] = hit.get("github_url")
            paper["content_github_stars"] = hit.get("github_stars")
            paper["github_match_method"] = hit.get("github_match_method")
            done += 1
            continue
        try:
            comment_url = paper.get("content_github_url")
            if comment_url:
                stars = _stars_for_repo(comment_url)
                method = "arxiv_comment"
                time.sleep(core_delay)
                result = {
                    "github_url": comment_url,
                    "github_stars": stars,
                    "github_match_method": method,
                }
            else:
                url, stars = _search_repo_for_arxiv(arxiv_id)
                method = "github_search_readme" if url else None
                time.sleep(search_delay)
                result = {
                    "github_url": url,
                    "github_stars": stars,
                    "github_match_method": method,
                }
        except RateLimitHit as exc:
            if verbose:
                print(
                    f"[github] rate limit ({exc.resource}); "
                    f"stopping for now — resume later. "
                    f"Enriched {done}/{len(papers)} papers so far."
                )
            if exc.wait_seconds <= max_wait_seconds:
                time.sleep(exc.wait_seconds)
                return enrich_papers(
                    papers, checkpoint_path,
                    search_delay=search_delay, core_delay=core_delay,
                    max_wait_seconds=0, verbose=verbose,
                )
            break
        checkpoint[arxiv_id] = result
        _save_checkpoint(checkpoint_path, checkpoint)
        paper["content_github_url"] = result["github_url"]
        paper["content_github_stars"] = result["github_stars"]
        paper["github_match_method"] = result["github_match_method"]
        done += 1
        if verbose and done % 25 == 0:
            print(f"[github] enriched {done}/{len(papers)} papers")
    return papers
