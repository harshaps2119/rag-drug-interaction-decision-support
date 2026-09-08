"""
tests/test_rate_limiter.py
=============================
Tests for app/core/rate_limiter.py — pure logic, deterministic via
injectable `now`, no I/O, no network.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_rate_limiter.py -v
"""

import pytest

from app.core.rate_limiter import InMemoryRateLimiter


def test_allows_requests_under_the_limit():
    limiter = InMemoryRateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        allowed, _ = limiter.check("client-a")
        assert allowed is True


def test_rejects_requests_over_the_limit():
    limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)
    limiter.check("client-a")
    limiter.check("client-a")
    allowed, retry_after = limiter.check("client-a")
    assert allowed is False
    assert retry_after > 0


def test_different_keys_are_independent():
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
    allowed_a, _ = limiter.check("client-a")
    allowed_b, _ = limiter.check("client-b")
    assert allowed_a is True
    assert allowed_b is True  # different key, independent budget


def test_old_hits_expire_out_of_the_window():
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=10)
    limiter.check("client-a", now=0.0)
    # Still within the window.
    allowed, _ = limiter.check("client-a", now=5.0)
    assert allowed is False
    # Past the window -- the old hit should have expired.
    allowed, _ = limiter.check("client-a", now=11.0)
    assert allowed is True


def test_reset_single_key():
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
    limiter.check("client-a")
    limiter.reset("client-a")
    allowed, _ = limiter.check("client-a")
    assert allowed is True


def test_reset_all_keys():
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
    limiter.check("client-a")
    limiter.check("client-b")
    limiter.reset()
    assert limiter.check("client-a")[0] is True
    assert limiter.check("client-b")[0] is True


def test_invalid_max_requests_raises():
    with pytest.raises(ValueError):
        InMemoryRateLimiter(max_requests=0, window_seconds=60)


def test_invalid_window_raises():
    with pytest.raises(ValueError):
        InMemoryRateLimiter(max_requests=5, window_seconds=0)


def test_retry_after_decreases_as_window_progresses():
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=10)
    limiter.check("client-a", now=0.0)
    _, retry_after_early = limiter.check("client-a", now=1.0)
    _, retry_after_late = limiter.check("client-a", now=8.0)
    assert retry_after_late < retry_after_early
