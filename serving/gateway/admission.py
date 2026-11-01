from __future__ import annotations

import asyncio


class AdmissionController:
    """Per-process in-flight limiter. try_acquire never waits (no queue)."""

    def __init__(self, max_in_flight: int) -> None:
        # max_in_flight <= 0 disables the limit (baseline / unlimited).
        self.max_in_flight = max_in_flight
        self.enabled = max_in_flight > 0
        self._in_flight = 0
        self._lock = asyncio.Lock()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self.enabled and self._in_flight >= self.max_in_flight:
                return False
            self._in_flight += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            if self._in_flight > 0:
                self._in_flight -= 1

    def snapshot(self) -> dict[str, int | bool]:
        return {
            "admission_enabled": self.enabled,
            "max_in_flight": self.max_in_flight,
            "in_flight": self._in_flight,
        }
