"""entity_resolver.py — Deterministic entity resolution.

Part of the GraphOne / FrontierAtlas data-ingestion pipeline.

Resolves raw startup/company name variants to a single canonical name so the
same company is never counted twice under different spellings (e.g.
"OpenAI, Inc.", "Open AI" and "openai" all become "OpenAI").

Resolution order (``canonicalize``):
  1. ``exact_seed`` — the normalised name matches a normalised seed name.
  2. ``alias``      — the normalised name is a known alias variant.
  3. ``fuzzy``      — difflib.get_close_matches against the normalised seed
                     names, n=1, cutoff=0.88.
  4. ``identity``   — unknown name; returned as-is (stripped), with its
                     original casing preserved when it already looks like a
                     proper name, otherwise title-cased. Nothing is invented:
                     an unknown name is never mapped to a seed company.

Everything here is deterministic: same input -> same output, no randomness,
no network, no LLM. Only the Python standard library is used.
"""

from __future__ import annotations

import difflib
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Canonical seed list — exactly 50 well-known real AI startups / companies.
# ---------------------------------------------------------------------------

SEED_STARTUPS: list[str] = [
    "OpenAI",
    "Anthropic",
    "Google DeepMind",
    "Mistral AI",
    "Cohere",
    "Stability AI",
    "Hugging Face",
    "Databricks",
    "Scale AI",
    "Perplexity",
    "Runway",
    "Midjourney",
    "xAI",
    "Character.AI",
    "Inflection AI",
    "Adept",
    "AI21 Labs",
    "Aleph Alpha",
    "Groq",
    "Cerebras Systems",
    "SambaNova Systems",
    "Together AI",
    "Replicate",
    "Modal",
    "Anyscale",
    "Weights & Biases",
    "LangChain",
    "LlamaIndex",
    "Pinecone",
    "Weaviate",
    "Chroma",
    "Qdrant",
    "ElevenLabs",
    "Synthesia",
    "Descript",
    "DeepL",
    "Grammarly",
    "Jasper",
    "Copy.ai",
    "Notion",
    "DataRobot",
    "H2O.ai",
    "Vectara",
    "AssemblyAI",
    "Meta AI",
    "Meta",
    "Microsoft",
    "Fireworks AI",
    "Baseten",
    "Cognition",
]

# ---------------------------------------------------------------------------
# Aliases — common raw variants -> canonical name.
# Keys are stored in normalised form (see normalize_name); lookups always
# normalise the input first, so 'OpenAI, Inc.' and 'open ai' both resolve.
# ---------------------------------------------------------------------------

ALIASES: dict[str, str] = {
    "open ai": "OpenAI",
    "openai inc": "OpenAI",
    "openai, inc.": "OpenAI",
    "anthropic pbc": "Anthropic",
    "google deepmind": "Google DeepMind",
    "deepmind": "Google DeepMind",
    "huggingface": "Hugging Face",
    "hugging face inc": "Hugging Face",
    "meta ai": "Meta AI",
    "meta platforms": "Meta",
    "microsoft corporation": "Microsoft",
    "xai": "xAI",
    "mistral": "Mistral AI",
    "character ai": "Character.AI",
    "character.ai inc": "Character.AI",
    "stability ai": "Stability AI",
    "cohere inc": "Cohere",
    "databricks inc": "Databricks",
    "scale ai": "Scale AI",
    "perplexity ai": "Perplexity",
    "runway ml": "Runway",
    "midjourney inc": "Midjourney",
    "inflection": "Inflection AI",
    "adept ai": "Adept",
    "ai21": "AI21 Labs",
    "ai21 labs": "AI21 Labs",
    "aleph alpha gmbh": "Aleph Alpha",
    "cerebras": "Cerebras Systems",
    "sambanova": "SambaNova Systems",
    "together ai": "Together AI",
    "weights and biases": "Weights & Biases",
    "wandb": "Weights & Biases",
    "langchain inc": "LangChain",
    "llamaindex": "LlamaIndex",
    "pinecone systems": "Pinecone",
    "weaviate": "Weaviate",
    "elevenlabs": "ElevenLabs",
    "eleven labs": "ElevenLabs",
    "synthesia": "Synthesia",
    "descript inc": "Descript",
    "deepl": "DeepL",
    "deep l": "DeepL",
    "grammarly inc": "Grammarly",
    "jasper ai": "Jasper",
    "copy ai": "Copy.ai",
    "notion labs": "Notion",
    "datarobot inc": "DataRobot",
    "h2o ai": "H2O.ai",
    "vectara inc": "Vectara",
    "assemblyai": "AssemblyAI",
    "replicate inc": "Replicate",
    "modal labs": "Modal",
    "anyscale inc": "Anyscale",
    "qdrant": "Qdrant",
    "chroma": "Chroma",
    "groq inc": "Groq",
    "hugging face": "Hugging Face",
}

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_LEGAL_SUFFIXES = {
    "inc", "llc", "ltd", "corp", "corporation", "company", "co", "gmbh",
    "pte", "pvt", "srl", "sa", "ag", "bv", "nv", "plc", "llp",
}

_PUNCTUATION_RE = re.compile(r"[.,\-&']")


def normalize_name(name: str) -> str:
    """Normalise a company name for comparison.

    Lowercase, strip leading/trailing whitespace, remove legal suffixes
    (inc, inc., llc, ltd, corp, corporation, company, co, gmbh, pte, pvt,
    srl, sa, ag, bv, nv, plc, llp), remove punctuation (``. , - & '``),
    collapse multiple spaces, strip.
    """
    if not name:
        return ""
    text = name.strip().lower()
    # Remove punctuation first so 'inc.' becomes the token 'inc'.
    text = _PUNCTUATION_RE.sub(" ", text)
    tokens = text.split()
    # Strip trailing legal suffixes (repeat: 'pvt ltd' -> 'pvt' -> '').
    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


# Precomputed lookup tables (normalised -> canonical).
_NORMALIZED_SEEDS: dict[str, str] = {normalize_name(s): s for s in SEED_STARTUPS}
_NORMALIZED_SEED_NAMES: list[str] = sorted(_NORMALIZED_SEEDS)
_NORMALIZED_ALIASES: dict[str, str] = {
    normalize_name(k): v for k, v in ALIASES.items()
}


def _identity_case(raw_name: str) -> str:
    """Title-case sensibly for an unknown name.

    Preserves the original casing when it already looks like a proper name
    (contains an uppercase letter); otherwise title-cases it.
    """
    stripped = raw_name.strip()
    if any(ch.isupper() for ch in stripped):
        return stripped
    return stripped.title()


def canonicalize(raw_name: str) -> tuple[str, str]:
    """Resolve ``raw_name`` to (canonical_name, method).

    method is one of 'exact_seed', 'alias', 'fuzzy', 'identity'.
    """
    if not raw_name or not raw_name.strip():
        return ("", "identity")
    stripped = raw_name.strip()
    normalized = normalize_name(stripped)

    # 1. Exact match against a normalised seed name.
    if normalized in _NORMALIZED_SEEDS:
        return (_NORMALIZED_SEEDS[normalized], "exact_seed")

    # 2. Known alias variant.
    if normalized in _NORMALIZED_ALIASES:
        return (_NORMALIZED_ALIASES[normalized], "alias")

    # 3. Fuzzy match against normalised seed names (cutoff 0.88).
    if normalized:
        matches = difflib.get_close_matches(
            normalized, _NORMALIZED_SEED_NAMES, n=1, cutoff=0.88
        )
        if matches:
            return (_NORMALIZED_SEEDS[matches[0]], "fuzzy")

    # 4. Unknown — return as-is, never invent a mapping.
    return (_identity_case(stripped), "identity")


# ---------------------------------------------------------------------------
# Resolver class
# ---------------------------------------------------------------------------


class EntityResolver:
    """Holds the seed list + alias map; resolves names, keeps a log."""

    def __init__(
        self,
        seeds: Optional[list[str]] = None,
        aliases: Optional[dict[str, str]] = None,
    ):
        self.seeds = list(seeds) if seeds is not None else list(SEED_STARTUPS)
        self.aliases = dict(aliases) if aliases is not None else dict(ALIASES)
        self._log: list[tuple[str, str, str]] = []

    def resolve(self, name: str) -> tuple[str, str]:
        """Resolve one name -> (canonical, method); records it in the log."""
        canonical, method = canonicalize(name)
        self._log.append((name, canonical, method))
        return (canonical, method)

    def resolve_many(self, names: list[str]) -> list[tuple[str, str, str]]:
        """Resolve many names, deduplicated preserving first-seen order.

        Returns a list of (raw, canonical, method). Deterministic: same
        input list always yields the same output list.
        """
        seen: set[str] = set()
        results: list[tuple[str, str, str]] = []
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            canonical, method = self.resolve(name)
            results.append((name, canonical, method))
        return results

    def resolve_batch(self, names: list[str]) -> list[tuple[str, str, str]]:
        """Alias for resolve_many (same deterministic behaviour)."""
        return self.resolve_many(names)

    def mapping_log(self) -> list[tuple[str, str, str]]:
        """All resolutions seen so far, as (raw, canonical, method)."""
        return list(self._log)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    assert len(SEED_STARTUPS) == 50, f"expected 50 seeds, got {len(SEED_STARTUPS)}"
    assert len(set(SEED_STARTUPS)) == 50, "seed list must have no duplicates"

    assert canonicalize("OpenAI, Inc.") == ("OpenAI", "exact_seed") or \
        canonicalize("OpenAI, Inc.") == ("OpenAI", "alias")
    assert canonicalize("Open AI")[0] == "OpenAI"
    assert canonicalize("DeepMind")[0] == "Google DeepMind"
    assert canonicalize("huggingface") == ("Hugging Face", "alias")
    assert canonicalize("Some Brand New Startup XYZ") == (
        "Some Brand New Startup XYZ", "identity",
    )

    # Extra checks: normalisation, fuzzy, identity casing, resolver class.
    assert normalize_name("  OpenAI, Inc. ") == "openai"
    assert normalize_name("Aleph Alpha GmbH") == "aleph alpha"
    assert normalize_name("Weights & Biases, Inc.") == "weights biases"
    assert canonicalize("OpenAIi")[1] == "fuzzy"  # close typo -> fuzzy
    assert canonicalize("openai")[1] == "exact_seed"
    assert canonicalize("some lowercase startup") == (
        "Some Lowercase Startup", "identity",
    )
    assert canonicalize("") == ("", "identity")

    resolver = EntityResolver()
    assert resolver.resolve("Open AI") == ("OpenAI", "alias")
    batch = resolver.resolve_many(
        ["OpenAI, Inc.", "open ai", "DeepMind", "OpenAI, Inc.", "Brand New Co"]
    )
    assert len(batch) == 4, batch  # duplicate raw name deduplicated
    assert batch[0] == ("OpenAI, Inc.", "OpenAI", "exact_seed")
    assert batch[2] == ("DeepMind", "Google DeepMind", "alias")
    assert batch[3][1] == "Brand New Co" and batch[3][2] == "identity"
    # Deterministic: same input -> same output on a fresh resolver.
    resolver2 = EntityResolver()
    assert resolver2.resolve_many(
        ["OpenAI, Inc.", "open ai", "DeepMind", "OpenAI, Inc.", "Brand New Co"]
    ) == batch
    log = resolver.mapping_log()
    assert len(log) == 5, log  # 1 single resolve + 4 unique batch resolves

    print("entity_resolver self-test OK")
