"""
core/rate_limiter.py
=======================
A simple in-memory, per-process sliding-window rate limiter — no Redis
or other external service required.

WHY IN-MEMORY, NOT REDIS
-----------------------------
Redis-backed rate limiting is the right choice for a real multi-instance
production deployment (see the trade-off explained below), but it adds
an external service dependency that would make this project harder to
run locally for development and for a viva demonstration — directly
against the project's own requirement to keep local development
convenient. An in-memory limiter needs zero setup and is fully
sufficient for a single-process MVP/prototype.

THE PRODUCTION TRADE-OFF, EXPLAINED HONESTLY
--------------------------------------------------
This limiter's state (`_hits`, a dict of per-key timestamp deques) lives
in ONE Python process's memory. That means:

  - If you run multiple uvicorn WORKERS (`--workers 4`) or multiple
    separate CONTAINER INSTANCES behind a load balancer, each one has
    its OWN independent counter. A client could get 30 requests/minute
    PER WORKER, not 30 total — the limit is not actually enforced
    globally.
  - Restarting the process resets everyone's count to zero.
  - It cannot be inspected or adjusted from outside the process.

For a real multi-instance production deployment, the standard fix is a
shared backing store (Redis, most commonly) that every instance reads
and writes atomically, so the limit is enforced across the whole fleet.
That is explicitly NOT built here — it would be exactly the kind of
"fake enterprise feature" this phase's own instructions say not to add
just for appearances. This limiter is correctly scoped to what a single-
process local prototype needs, and docs/security.md says so plainly.

HOW TO TEST IT
----------------
    cd backend
    pytest tests/test_rate_limiter.py -v

Pure logic, no I/O — deterministic, fast, offline.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class InMemoryRateLimiter:
    """A sliding-window limiter: at most `max_requests` per `window_seconds`, per key."""

    def __init__(self, max_requests: int, window_seconds: float):
        if max_requests < 1:
            raise ValueError("max_requests must be at least 1.")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, *, now: float | None = None) -> tuple[bool, float]:
        """
        Returns (allowed, retry_after_seconds). If allowed, this call
        also RECORDS the hit (so calling check() twice in a row for the
        same key counts as two requests, not a query-only peek).
        `now` is injectable for deterministic tests; defaults to
        time.monotonic().
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window_seconds:
                q.popleft()

            if len(q) >= self.max_requests:
                retry_after = max(0.0, self.window_seconds - (now - q[0]))
                return False, retry_after

            q.append(now)
            return True, 0.0

    def reset(self, key: str | None = None) -> None:
        """Clears rate-limit state for one key, or all keys if none given. Test/ops convenience only."""
        with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)
