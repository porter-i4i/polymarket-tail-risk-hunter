"""
Market Scanner: scans Gamma API, filters candidates, returns typed ScanResult.

Uses httpx with tenacity retry. Rate-limited via gamma_limiter and clob_data_limiter.
Deduplicates against existing active positions/orders in the database.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
    retry_if_result,
    RetryError,
)

from config import RETRY_POLICIES
from modules.rate_limiter import RateLimiter
from modules.types import MarketCandidate, OrderBook, PricePoint, ScanResult

logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"

# Retry policy from config
_GAMMA_POLICY = RETRY_POLICIES["gamma_api"]


def _is_retryable_response(response: httpx.Response) -> bool:
    """Check if an HTTP response should be retried."""
    return response.status_code in _GAMMA_POLICY["retryable_status"]


class MarketScanner:
    def __init__(
        self,
        cfg: dict,
        gamma_limiter: RateLimiter,
        clob_data_limiter: RateLimiter,
        db=None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.cfg = cfg
        self.gamma_limiter = gamma_limiter
        self.clob_data_limiter = clob_data_limiter
        self.db = db
        self._client = http_client
        self._owns_client = http_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # === HELPERS ===

    @staticmethod
    def _resolve_category(market: dict) -> str:
        """Category resolution: tags[0]['label'] -> groupItemTitle -> 'Uncategorized'."""
        tags = market.get("tags")
        if tags and isinstance(tags, list) and len(tags) > 0:
            first_tag = tags[0]
            if isinstance(first_tag, dict):
                label = first_tag.get("label", "")
                if label:
                    return label
            elif isinstance(first_tag, str) and first_tag:
                return first_tag
        group_title = market.get("groupItemTitle", "")
        if group_title:
            return group_title
        return "Uncategorized"

    def _parse_market(self, market: dict) -> MarketCandidate | None:
        """Parse a single market dict into a MarketCandidate, or None on failure."""
        try:
            condition_id = market.get("conditionId") or market.get("condition_id", "")
            question = market.get("question", "")
            if not condition_id or not question:
                return None

            # Parse outcomePrices — may be a JSON string
            outcome_prices_raw = market.get("outcomePrices", "[]")
            if isinstance(outcome_prices_raw, str):
                outcome_prices = json.loads(outcome_prices_raw)
            else:
                outcome_prices = outcome_prices_raw

            if not outcome_prices or len(outcome_prices) < 2:
                return None

            yes_price = float(outcome_prices[0])
            no_price = float(outcome_prices[1])

            # Parse clobTokenIds — may be a JSON string
            clob_ids_raw = market.get("clobTokenIds", "[]")
            if isinstance(clob_ids_raw, str):
                clob_ids = json.loads(clob_ids_raw)
            else:
                clob_ids = clob_ids_raw

            if not clob_ids or len(clob_ids) < 2:
                return None

            yes_token_id = str(clob_ids[0])
            no_token_id = str(clob_ids[1])

            volume_24h = float(market.get("volume24hr", 0) or 0)
            liquidity = float(market.get("liquidity", 0) or 0)
            end_date = market.get("endDate", "") or ""

            # Tags list
            raw_tags = market.get("tags", [])
            if isinstance(raw_tags, list):
                tag_labels = []
                for t in raw_tags:
                    if isinstance(t, dict):
                        tag_labels.append(t.get("label", ""))
                    elif isinstance(t, str):
                        tag_labels.append(t)
            else:
                tag_labels = []

            category = self._resolve_category(market)

            # Market age
            created_at = market.get("createdAt", "")
            if created_at:
                try:
                    created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    age_hours = (now - created_dt).total_seconds() / 3600.0
                except (ValueError, TypeError):
                    age_hours = 0.0
            else:
                age_hours = 0.0

            return MarketCandidate(
                condition_id=condition_id,
                question=question,
                category=category,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                yes_price=yes_price,
                no_price=no_price,
                volume_24h=volume_24h,
                liquidity=liquidity,
                end_date=end_date,
                tags=tag_labels,
                market_age_hours=max(0.0, age_hours),
            )
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            logger.debug("Failed to parse market: %s", e)
            return None

    def _filter_candidates(self, markets: list[dict]) -> list[MarketCandidate]:
        """Filter raw market dicts through price/volume/liquidity thresholds."""
        max_price = self.cfg.get("MAX_PRICE_THRESHOLD", 0.05)
        min_volume = self.cfg.get("MIN_VOLUME_24H", 100)
        min_liquidity = self.cfg.get("MIN_LIQUIDITY", 500)

        candidates = []
        for market in markets:
            candidate = self._parse_market(market)
            if candidate is None:
                continue
            if candidate.yes_price > max_price:
                continue
            if candidate.volume_24h < min_volume:
                continue
            if candidate.liquidity < min_liquidity:
                continue
            candidates.append(candidate)
        return candidates

    def _apply_dedup(self, candidates: list[MarketCandidate]) -> list[MarketCandidate]:
        """Remove candidates where we already have an active position/order."""
        if self.db is None:
            return candidates
        return [c for c in candidates if not self.db.has_active_position(c.condition_id)]

    async def _fetch_page(self, offset: int, limit: int = 100) -> httpx.Response:
        """Fetch a single page from the Gamma API with rate limiting and retry."""
        await self.gamma_limiter.acquire()
        client = await self._get_client()

        @retry(
            stop=stop_after_attempt(_GAMMA_POLICY["max_attempts"]),
            wait=wait_exponential(
                multiplier=_GAMMA_POLICY["initial_wait_sec"],
                max=_GAMMA_POLICY["max_wait_sec"],
                exp_base=_GAMMA_POLICY["exponential_base"],
            ),
            retry=retry_if_exception_type((
                httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError,
            )),
            reraise=True,
        )
        async def _do_fetch() -> httpx.Response:
            resp = await client.get(
                f"{GAMMA_API_BASE}/markets",
                params={"offset": offset, "limit": limit, "active": "true"},
            )
            if resp.status_code == 429:
                await asyncio.sleep(30)
                raise httpx.ReadTimeout("Rate limited, retrying after 30s wait")
            if resp.status_code in _GAMMA_POLICY["retryable_status"]:
                raise httpx.ReadTimeout(f"Retryable status {resp.status_code}")
            return resp

        return await _do_fetch()

    async def scan_all_markets(self) -> ScanResult:
        """Scan all active markets from the Gamma API."""
        start = time.monotonic()
        all_markets: list[dict] = []
        errors: list[str] = []
        consecutive_failures = 0
        offset = 0
        limit = 100

        while True:
            try:
                resp = await self._fetch_page(offset, limit)
                resp.raise_for_status()
                page_data = resp.json()

                if not isinstance(page_data, list):
                    page_data = page_data.get("data", []) if isinstance(page_data, dict) else []

                if not page_data:
                    break

                all_markets.extend(page_data)
                consecutive_failures = 0
                offset += limit

            except Exception as e:
                consecutive_failures += 1
                errors.append(f"Page fetch failed at offset {offset}: {e}")
                if consecutive_failures >= 3:
                    errors.append("Aborting scan: 3 consecutive page failures")
                    break
                offset += limit

        total_scanned = len(all_markets)
        candidates = self._filter_candidates(all_markets)
        candidates = self._apply_dedup(candidates)
        duration = time.monotonic() - start

        return ScanResult(
            candidates=candidates,
            total_scanned=total_scanned,
            total_filtered=total_scanned - len(candidates),
            scan_duration_seconds=round(duration, 3),
            errors=errors,
        )

    async def get_market_by_condition(self, condition_id: str) -> dict | None:
        """Fetch a single market by condition ID."""
        try:
            await self.gamma_limiter.acquire()
            client = await self._get_client()
            resp = await client.get(
                f"{GAMMA_API_BASE}/markets",
                params={"conditionId": condition_id},
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            return None
        except Exception as e:
            logger.warning("Failed to fetch market %s: %s", condition_id, e)
            return None

    async def get_order_book(self, token_id: str) -> OrderBook | None:
        """Fetch order book for a token. Returns OrderBook or None on failure."""
        try:
            await self.clob_data_limiter.acquire()
            client = await self._get_client()
            resp = await client.get(
                f"https://clob.polymarket.com/book",
                params={"token_id": token_id},
            )
            resp.raise_for_status()
            data = resp.json()

            bids_raw = data.get("bids", [])
            asks_raw = data.get("asks", [])

            bids = [(float(b["price"]), float(b["size"])) for b in bids_raw[:10]]
            asks = [(float(a["price"]), float(a["size"])) for a in asks_raw[:10]]

            # Sort: bids descending by price, asks ascending by price
            bids.sort(key=lambda x: x[0], reverse=True)
            asks.sort(key=lambda x: x[0])

            if bids and asks:
                spread = asks[0][0] - bids[0][0]
                midpoint = (asks[0][0] + bids[0][0]) / 2
            elif bids:
                spread = 0.0
                midpoint = bids[0][0]
            elif asks:
                spread = 0.0
                midpoint = asks[0][0]
            else:
                return None

            return OrderBook(bids=bids, asks=asks, spread=spread, midpoint=midpoint)
        except Exception as e:
            logger.warning("Failed to get order book for token %s: %s", token_id, e)
            return None

    async def get_price_history(self, token_id: str) -> list[PricePoint]:
        """Fetch recent price history for a token. Returns list of PricePoint."""
        try:
            await self.clob_data_limiter.acquire()
            client = await self._get_client()
            resp = await client.get(
                f"https://clob.polymarket.com/prices-history",
                params={"market": token_id, "interval": "1h", "fidelity": 60},
            )
            resp.raise_for_status()
            data = resp.json()

            history = data.get("history", [])
            points = []
            for point in history[-20:]:
                points.append(PricePoint(
                    timestamp=str(point.get("t", "")),
                    price=float(point.get("p", 0)),
                    volume=float(point.get("v", 0)),
                ))
            return points
        except Exception as e:
            logger.warning("Failed to get price history for token %s: %s", token_id, e)
            return []

    async def get_current_price(self, token_id: str) -> float | None:
        """Fetch the current price for a token via CLOB data endpoint.

        Returns the midpoint price or None on failure.
        """
        try:
            await self.clob_data_limiter.acquire()
            client = await self._get_client()
            resp = await client.get(
                f"{GAMMA_API_BASE}/markets",
                params={"clobTokenIds": token_id},
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                market = data[0]
                outcome_prices_raw = market.get("outcomePrices", "[]")
                if isinstance(outcome_prices_raw, str):
                    prices = json.loads(outcome_prices_raw)
                else:
                    prices = outcome_prices_raw
                if prices and len(prices) >= 1:
                    return float(prices[0])
            return None
        except Exception as e:
            logger.warning("Failed to get price for token %s: %s", token_id, e)
            return None
