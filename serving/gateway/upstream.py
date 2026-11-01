from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from serving.gateway.config import Settings


class UpstreamError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True)
class UpstreamResult:
    data: dict[str, Any]
    duration_ms: float


class VLLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Default timeout is a fallback; per-call timeouts override.
        self._client = httpx.AsyncClient(
            base_url=settings.vllm_base_url,
            headers={"Authorization": f"Bearer {settings.modal_proxy_token}"},
            timeout=httpx.Timeout(settings.upstream_timeout_s),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> tuple[int, str]:
        try:
            response = await self._client.get("/health", timeout=30.0)
            return response.status_code, response.text
        except httpx.RequestError as exc:
            raise UpstreamError(503, f"upstream unreachable: {exc}") from exc

    def _timeout(self, timeout_s: float | None) -> httpx.Timeout:
        total = (
            self.settings.upstream_timeout_s if timeout_s is None else float(timeout_s)
        )
        return httpx.Timeout(connect=30.0, read=total, write=30.0, pool=30.0)

    async def chat_completions(
        self,
        payload: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> UpstreamResult:
        body = {**payload, "stream": False}
        t0 = time.perf_counter()
        try:
            response = await self._client.post(
                "/v1/chat/completions",
                json=body,
                timeout=self._timeout(timeout_s),
            )
        except httpx.TimeoutException as exc:
            raise UpstreamError(504, "upstream timeout") from exc
        except httpx.RequestError as exc:
            raise UpstreamError(503, f"upstream unreachable: {exc}") from exc
        duration_ms = (time.perf_counter() - t0) * 1000

        if response.status_code >= 400:
            raise UpstreamError(
                response.status_code,
                response.text[:500] or response.reason_phrase,
            )
        return UpstreamResult(data=response.json(), duration_ms=round(duration_ms, 1))

    async def stream_chat_completions(
        self,
        payload: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> tuple[httpx.Response, float]:
        """Open an upstream SSE stream. Caller must aclose the response."""
        body = {**payload, "stream": True}
        t0 = time.perf_counter()
        try:
            request = self._client.build_request(
                "POST",
                "/v1/chat/completions",
                json=body,
                headers={
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                },
                timeout=self._timeout(timeout_s),
            )
            response = await self._client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            raise UpstreamError(504, "upstream timeout") from exc
        except httpx.RequestError as exc:
            raise UpstreamError(503, f"upstream unreachable: {exc}") from exc

        open_ms = round((time.perf_counter() - t0) * 1000, 1)
        if response.status_code >= 400:
            detail = (await response.aread())[:500].decode(errors="replace")
            await response.aclose()
            raise UpstreamError(response.status_code, detail or response.reason_phrase)
        return response, open_ms
