"""GraphOne / FrontierAtlas data-ingestion pipeline.

Phase I   Massive one-time bulk extraction (startups, products, papers)
Phase II  High-fidelity signal ingestion (news + jobs, 24h freshness)
Phase III Multi-tier LLM extraction engine (with rule-based fallback)
Phase IV  Deterministic entity resolution
Phase V   Anti-bot & scale thinking (documented in docs/architecture.md)
Phase VI  Architecture & production design (docs/architecture.pdf)
"""

__version__ = "1.0.0"
