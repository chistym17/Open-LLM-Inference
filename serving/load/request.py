from __future__ import annotations

import json
import time
from typing import Any

import httpx


def _delta_content(data: str) -> str | None:
    if not data or data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
        delta = (payload["choices"][0].get("delta") or {})
        content = delta.get("content")
        return str(content) if content else None
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None


async def run_chat(
    client: httpx.AsyncClient,
    gateway: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
    request_index: int,
    phase: str,
) -> dict[str, Any]:
    stream = bool(payload.get("stream"))
    t0 = time.perf_counter()
    base = {
        "index": request_index,
        "phase": phase,
        "stream": stream,
        "ok": False,
        "status": 0,
        "latency_ms": None,
        "ttft_ms": None,
        "completion_tokens": None,
        "prompt_tokens": None,
        "content_chunks": 0,
        "error": None,
    }

    try:
        if not stream:
            response = await client.post(
                f"{gateway}/v1/chat/completions",
                json=payload,
                timeout=timeout_s,
            )
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            body: dict[str, Any]
            try:
                body = response.json()
            except json.JSONDecodeError:
                body = {}
            usage = body.get("usage") or {}
            base.update(
                {
                    "ok": response.is_success,
                    "status": response.status_code,
                    "latency_ms": latency_ms,
                    "ttft_ms": latency_ms if response.is_success else None,
                    "completion_tokens": usage.get("completion_tokens"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "content_chunks": 1 if response.is_success else 0,
                    "error": None if response.is_success else str(body)[:200],
                }
            )
            return base

        headers = {"Accept": "text/event-stream", "X-Stream": "true"}
        ttft_ms = None
        chunks = 0
        async with client.stream(
            "POST",
            f"{gateway}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout_s,
        ) as response:
            status = response.status_code
            if status >= 400:
                detail = (await response.aread())[:200].decode(errors="replace")
                base.update(
                    {
                        "ok": False,
                        "status": status,
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "error": detail,
                    }
                )
                return base

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                content = _delta_content(line[5:].strip())
                if not content:
                    continue
                chunks += 1
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - t0) * 1000, 1)

        base.update(
            {
                "ok": True,
                "status": status,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "ttft_ms": ttft_ms,
                "content_chunks": chunks,
            }
        )
        return base
    except httpx.TimeoutException:
        base.update(
            {
                "ok": False,
                "status": 0,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "error": "timeout",
            }
        )
        return base
    except httpx.HTTPError as exc:
        base.update(
            {
                "ok": False,
                "status": 0,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "error": str(exc)[:200],
            }
        )
        return base
