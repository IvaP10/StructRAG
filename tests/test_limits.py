"""Rate limiter and spend ledger.

The ledger is the limit that actually bounds the bill, so its arithmetic is
worth pinning down precisely.
"""

from __future__ import annotations

from server.limits import SlidingWindowLimiter, SpendLedger

PRICES = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "text-embedding-3-large": {"input": 0.130, "output": 0.0},
}


# ── limiter ──────────────────────────────────────────────────────────────────

def test_allows_up_to_the_limit_then_blocks():
    limiter = SlidingWindowLimiter(max_events=3, window_seconds=3600)
    assert [limiter.check("u1")[0] for _ in range(3)] == [True, True, True]
    allowed, retry_after = limiter.check("u1")
    assert allowed is False
    assert retry_after > 0


def test_a_blocked_attempt_does_not_consume_a_slot():
    """Otherwise being throttled would extend the lockout indefinitely."""
    limiter = SlidingWindowLimiter(max_events=2, window_seconds=3600)
    limiter.check("u1")
    limiter.check("u1")
    for _ in range(5):
        assert limiter.check("u1")[0] is False
    assert limiter.remaining("u1") == 0
    limiter.refund("u1")
    assert limiter.check("u1")[0] is True


def test_keys_are_independent():
    limiter = SlidingWindowLimiter(max_events=1, window_seconds=3600)
    assert limiter.check("a")[0] is True
    assert limiter.check("a")[0] is False
    assert limiter.check("b")[0] is True


def test_expired_events_fall_out_of_the_window():
    limiter = SlidingWindowLimiter(max_events=1, window_seconds=0)
    assert limiter.check("a")[0] is True
    assert limiter.check("a")[0] is True  # previous event already outside the window


def test_refund_only_returns_one_slot():
    limiter = SlidingWindowLimiter(max_events=3, window_seconds=3600)
    for _ in range(3):
        limiter.check("a")
    limiter.refund("a")
    assert limiter.remaining("a") == 1


def test_evict_idle_reclaims_memory():
    limiter = SlidingWindowLimiter(max_events=5, window_seconds=3600)
    limiter.check("a")
    assert limiter.evict_idle(older_than_seconds=-1) == 1


# ── ledger ───────────────────────────────────────────────────────────────────

def test_pricing_matches_published_rates():
    ledger = SpendLedger(1.0, PRICES)
    cost = ledger.price([
        {"model": "gpt-4o-mini", "input_tokens": 1_000_000, "output_tokens": 1_000_000}
    ])
    assert cost == 0.75  # 0.150 in + 0.600 out


def test_unknown_model_is_never_free():
    """A model rename must not silently disable the spend cap."""
    ledger = SpendLedger(1.0, PRICES)
    cost = ledger.price([
        {"model": "some-unreleased-model", "input_tokens": 100_000, "output_tokens": 100_000}
    ])
    assert cost > 0


def test_empty_usage_costs_nothing():
    assert SpendLedger(1.0, PRICES).price([]) == 0.0


def test_cap_trips_once_reached():
    ledger = SpendLedger(0.10, PRICES)
    assert ledger.would_exceed() is False
    ledger.record_usd(0.10)
    assert ledger.would_exceed() is True
    assert ledger.snapshot()["exhausted"] is True
    assert ledger.snapshot()["remaining_usd"] == 0.0


def test_spend_accumulates_across_requests():
    ledger = SpendLedger(1.0, PRICES)
    for _ in range(3):
        ledger.record([{"model": "gpt-4o-mini", "input_tokens": 5_000, "output_tokens": 800}])
    snapshot = ledger.snapshot()
    assert snapshot["requests"] == 3
    assert snapshot["spent_usd"] > 0


def test_rolls_over_on_a_new_day():
    ledger = SpendLedger(0.10, PRICES)
    ledger.record_usd(0.10)
    assert ledger.would_exceed() is True
    ledger._day = "1999-01-01"          # simulate the date moving on
    assert ledger.would_exceed() is False
    assert ledger.snapshot()["spent_usd"] == 0.0


def test_realistic_query_count_fits_the_monthly_budget():
    """Sanity-check the cap against actual per-query cost.

    Guards against a pricing edit that would quietly allow ten times the
    intended spend.
    """
    ledger = SpendLedger(0.70, PRICES)
    per_query = ledger.price([
        {"model": "gpt-4o-mini", "input_tokens": 5_000, "output_tokens": 800}
    ])
    queries_per_day = 0.70 / per_query
    assert 300 < queries_per_day < 1200, f"got {queries_per_day:.0f} queries/day"
