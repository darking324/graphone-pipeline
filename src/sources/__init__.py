"""Data sources for the GraphOne pipeline.

Each module exposes an async ``collect(fetcher, ...)`` returning a list of
records already shaped to the task's canonical schemas (see pipeline.py).
Every record carries its source URL — no record is ever created without one.
"""
