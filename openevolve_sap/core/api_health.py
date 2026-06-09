"""RouterAI API health checks and billing circuit breaker."""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

_BILLING_MARKERS = (
    "402",
    "недостаточно средств",
    "insufficient balance",
    "insufficient funds",
    "payment required",
)


def is_billing_error(exc_or_text: BaseException | str) -> bool:
    text = str(exc_or_text).lower()
    return any(marker in text for marker in _BILLING_MARKERS)


def check_routerai_api(
    api_key: str,
    *,
    api_base: Optional[str] = None,
    model: str = "qwen/qwen3.7-max",
) -> None:
    """
    Pre-flight RouterAI call. Raises RuntimeError on billing/auth failures.

    Uses a minimal chat completion to verify the key works before GPU eval starts.
    """
    if not api_key.strip():
        raise RuntimeError(
            "ROUTERAI_API_KEY is not set. Set it in .env before starting evolution."
        )

    base_url = (api_base or os.getenv("ROUTERAI_BASE_URL", "https://routerai.ru/api/v1")).rstrip("/")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            timeout=30,
        )
        if not response.choices:
            raise RuntimeError("RouterAI pre-flight: empty response")
        logger.info("RouterAI pre-flight OK (model=%s)", model)
    except Exception as exc:
        if is_billing_error(exc):
            raise RuntimeError(
                "RouterAI billing error during pre-flight check. "
                "Top up your balance at https://routerai.ru before running evolution. "
                f"Detail: {exc}"
            ) from exc
        raise RuntimeError(f"RouterAI pre-flight failed: {exc}") from exc


class BillingCircuitBreaker:
    """
    Pause evolution after consecutive billing (HTTP 402) failures.

    Prevents wasting GPU cycles when the API key has no balance.
    """

    def __init__(
        self,
        *,
        fail_threshold: Optional[int] = None,
        pause_seconds: Optional[float] = None,
    ) -> None:
        self.fail_threshold = fail_threshold or max(
            1, int(os.getenv("SAP_BILLING_FAIL_THRESHOLD", "3"))
        )
        self.pause_seconds = pause_seconds or float(
            os.getenv("SAP_BILLING_PAUSE_SEC", "120")
        )
        self._consecutive_failures = 0
        self._paused_until = 0.0
        self._total_billing_failures = 0

    @property
    def is_open(self) -> bool:
        return time.monotonic() < self._paused_until

    @property
    def total_billing_failures(self) -> int:
        return self._total_billing_failures

    def record_success(self) -> None:
        if self._consecutive_failures:
            logger.info("SAP billing circuit: reset after successful iteration")
        self._consecutive_failures = 0

    def record_billing_failure(self, detail: str) -> float:
        """Record a 402-style failure. Returns seconds to wait (0 if no pause triggered)."""
        self._consecutive_failures += 1
        self._total_billing_failures += 1
        logger.warning(
            "SAP billing failure %d/%d: %s",
            self._consecutive_failures,
            self.fail_threshold,
            detail[:200],
        )
        if self._consecutive_failures >= self.fail_threshold:
            self._paused_until = time.monotonic() + self.pause_seconds
            self._consecutive_failures = 0
            logger.error(
                "SAP BILLING ALERT: pausing evolution for %.0fs after %d billing failures. "
                "Top up RouterAI balance: https://routerai.ru",
                self.pause_seconds,
                self.fail_threshold,
            )
            return self.pause_seconds
        return 0.0

    async def wait_if_open(self) -> None:
        """Async sleep while circuit is open."""
        import asyncio

        while self.is_open:
            remaining = self._paused_until - time.monotonic()
            if remaining <= 0:
                break
            logger.warning(
                "SAP billing circuit OPEN: waiting %.0fs for balance recovery",
                remaining,
            )
            await asyncio.sleep(min(remaining, 10.0))
