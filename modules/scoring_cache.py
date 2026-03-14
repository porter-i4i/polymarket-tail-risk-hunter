"""
Adaptive-TTL in-memory cache for AI scoring results.
Prevents duplicate Claude API calls within TTL window.
Max 5000 entries. Cleanup expired every 10 cycles (called from main_loop).
"""
import time
from typing import Optional

from modules.types import AdjustedScore, PricePoint


class ScoringCache:
    def __init__(self, default_ttl: int = 1800, max_entries: int = 5000):
        self._cache: dict[str, tuple[AdjustedScore, float]] = {}  # {condition_id: (score, expires_at)}
        self.default_ttl = default_ttl  # 30 min
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0

    def get(self, condition_id: str) -> Optional[AdjustedScore]:
        """Return cached score if exists and not expired, else None."""
        entry = self._cache.get(condition_id)
        if entry is None:
            self.misses += 1
            return None
        score, expires_at = entry
        if time.time() > expires_at:
            del self._cache[condition_id]
            self.misses += 1
            return None
        self.hits += 1
        return score

    def set(self, condition_id: str, score: AdjustedScore, ttl: Optional[int] = None) -> None:
        """Store score with TTL. Evicts oldest if at capacity."""
        if len(self._cache) >= self.max_entries:
            self._evict_oldest()
        self._cache[condition_id] = (score, time.time() + (ttl or self.default_ttl))

    def get_ttl(self, market: dict, price_history: list[PricePoint]) -> int:
        """
        Adaptive TTL based on market velocity.
        Fast-moving markets → shorter TTL (5 min).
        Stable markets → longer TTL (60 min).
        """
        if not price_history or len(price_history) < 2:
            return self.default_ttl

        # Calculate price velocity: max price change in last 3 data points
        recent = price_history[-3:]
        if len(recent) >= 2:
            prices = [p.price for p in recent]
            max_change = max(abs(prices[i] - prices[i - 1]) for i in range(1, len(prices)))
        else:
            max_change = 0

        # Velocity buckets
        if max_change > 0.05:       # >5pp move → very volatile
            return 300              # 5 min
        elif max_change > 0.02:     # >2pp move
            return 900              # 15 min
        elif max_change > 0.005:    # >0.5pp move
            return 1800             # 30 min
        else:
            return 3600             # 60 min (stable)

    def cleanup(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        expired = [k for k, (_, exp) in self._cache.items() if now > exp]
        for k in expired:
            del self._cache[k]
        return len(expired)

    def _evict_oldest(self) -> None:
        """Remove entry with earliest expiry."""
        if not self._cache:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
        del self._cache[oldest_key]

    def reset_cycle_stats(self) -> dict:
        """Reset per-cycle hit/miss counters and return stats for the completed cycle."""
        stats = {"hits": self.hits, "misses": self.misses, "hit_rate": self.hit_rate}
        self.hits = 0
        self.misses = 0
        return stats

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
