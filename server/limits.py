"""Rate limiting and the spend ledger.

In-process rather than Redis: the app runs as one container, so counters reset
on restart and would not be shared across replicas.

Pure stdlib, so it tests without the model or database stack.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, Iterable, Tuple


class SlidingWindowLimiter:
    """Per-key request limiter over a rolling time window.

    Rolling rather than fixed buckets, which would allow a full quota at 10:59
    and another at 11:00.
    """

    def __init__(self, max_events: int, window_seconds: int):
        self.max_events = max_events
        self.window_seconds = window_seconds
        self._events: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> Deque[float]:
        window = self._events.setdefault(key, deque())
        cutoff = now - self.window_seconds
        while window and window[0] < cutoff:
            window.popleft()
        return window

    def check(self, key: str) -> Tuple[bool, int]:
        """Record an attempt. Returns (allowed, retry_after_seconds).

        A rejected attempt is not recorded, so being throttled cannot extend the
        lockout.
        """
        now = time.time()
        with self._lock:
            window = self._prune(key, now)
            if len(window) >= self.max_events:
                retry_after = int(window[0] + self.window_seconds - now) + 1
                return False, max(retry_after, 1)
            window.append(now)
            return True, 0

    def remaining(self, key: str) -> int:
        with self._lock:
            window = self._prune(key, time.time())
            return max(0, self.max_events - len(window))

    def reset(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)

    def refund(self, key: str) -> None:
        """Give back the most recent slot.

        For requests counted then rejected before any billable work. Removes only
        the newest entry, so it cannot clear an accumulated window.
        """
        with self._lock:
            window = self._events.get(key)
            if window:
                window.pop()

    def evict_idle(self, older_than_seconds: int) -> int:
        """Drop keys with no recent activity so memory does not grow forever."""
        cutoff = time.time() - older_than_seconds
        with self._lock:
            stale = [k for k, w in self._events.items() if not w or w[-1] < cutoff]
            for k in stale:
                del self._events[k]
            return len(stale)


class SpendLedger:
    """Tracks estimated USD spend against a daily cap, resetting at UTC midnight.

    Counts tokens reported by the API rather than inferring from request counts.
    Still an estimate: cached-token discounts and pricing changes are not
    modelled, so keep the cap below what you are willing to lose.
    """

    def __init__(self, daily_cap_usd: float, price_per_mtok: Dict[str, Dict[str, float]]):
        self.daily_cap_usd = daily_cap_usd
        self.price_per_mtok = price_per_mtok
        self._lock = threading.Lock()
        self._day = self._today()
        self._spent = 0.0
        self._requests = 0

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _roll_if_new_day(self) -> None:
        today = self._today()
        if today != self._day:
            self._day = today
            self._spent = 0.0
            self._requests = 0

    def price(self, usage_records: Iterable[Dict[str, object]]) -> float:
        """Cost of a set of usage records in USD.

        Unknown models are priced at the most expensive known rate, so a model
        rename cannot silently disable the cap.
        """
        total = 0.0
        fallback = max(
            (p["input"] + p["output"] for p in self.price_per_mtok.values()),
            default=1.0,
        )

        for record in usage_records:
            model = str(record.get("model", ""))
            in_tok = int(record.get("input_tokens", 0) or 0)
            out_tok = int(record.get("output_tokens", 0) or 0)
            rates = self.price_per_mtok.get(model)
            if rates is None:
                total += (in_tok + out_tok) / 1_000_000 * fallback
            else:
                total += in_tok / 1_000_000 * rates["input"]
                total += out_tok / 1_000_000 * rates["output"]
        return total

    def would_exceed(self) -> bool:
        """True when the cap is already reached, checked before doing work."""
        with self._lock:
            self._roll_if_new_day()
            return self._spent >= self.daily_cap_usd

    def record(self, usage_records: Iterable[Dict[str, object]]) -> float:
        """Add usage to today's total. Returns the amount charged."""
        amount = self.price(usage_records)
        with self._lock:
            self._roll_if_new_day()
            self._spent += amount
            self._requests += 1
        return amount

    def record_usd(self, amount: float) -> None:
        """Add a directly computed cost, e.g. an embedding batch."""
        with self._lock:
            self._roll_if_new_day()
            self._spent += amount
            self._requests += 1

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            self._roll_if_new_day()
            return {
                "day": self._day,
                "spent_usd": round(self._spent, 6),
                "cap_usd": self.daily_cap_usd,
                "remaining_usd": round(max(0.0, self.daily_cap_usd - self._spent), 6),
                "requests": self._requests,
                "exhausted": self._spent >= self.daily_cap_usd,
            }
