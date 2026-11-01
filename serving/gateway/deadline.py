from __future__ import annotations

import time


class Deadline:
    """Wall-clock request lifetime. timeout_s <= 0 disables."""

    def __init__(self, timeout_s: float) -> None:
        self.enabled = timeout_s > 0
        self.timeout_s = timeout_s
        self._deadline = time.monotonic() + timeout_s if self.enabled else None

    def remaining_s(self) -> float | None:
        if not self.enabled or self._deadline is None:
            return None
        return max(0.0, self._deadline - time.monotonic())

    def expired(self) -> bool:
        remaining = self.remaining_s()
        return remaining is not None and remaining <= 0.0

    def effective_timeout_s(self, upstream_timeout_s: float) -> float:
        """httpx timeout for this hop: min(upstream, remaining deadline)."""
        if not self.enabled:
            return upstream_timeout_s
        remaining = self.remaining_s()
        assert remaining is not None
        return min(upstream_timeout_s, remaining)
