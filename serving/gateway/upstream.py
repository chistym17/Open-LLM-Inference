from __future__ import annotations

from typing import Any

import httpx

from serving.gateway.config import Settings


class UpstreamError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class VLLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.vllm_base_url,
            headers={"Authorization": f"Bearer {settings.modal_proxy_token}"},
            timeout=httpx.Timeout(settings.upstream_timeout_s),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> tuple[int, str]:
        try:
            response = await self._client.get("/health")
            return response.status_code, response.text
        except httpx.RequestError as exc:
            raise UpstreamError(503, f"upstream unreachable: {exc}") from exc

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post("/v1/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise UpstreamError(504, "upstream timeout") from exc
        except httpx.RequestError as exc:
            raise UpstreamError(503, f"upstream unreachable: {exc}") from exc

        if response.status_code >= 400:
            raise UpstreamError(
                response.status_code,
                response.text[:500] or response.reason_phrase,
            )
        return response.json()
