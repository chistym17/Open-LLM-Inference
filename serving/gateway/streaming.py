from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import httpx
from starlette.requests import Request

from serving.gateway.deadline import Deadline
from serving.gateway.logging_util import log_event


def _line_has_content(raw_line: str) -> bool:
    line = raw_line.strip()
    if not line:
        return False
    if line.startswith("data:"):
        data = line[5:].strip()
    else:
        return False
    if not data or data == "[DONE]":
        return False
    try:
        payload = json.loads(data)
        delta = (payload["choices"][0].get("delta") or {})
        return bool(delta.get("content"))
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError):
        return False


async def proxy_chat_stream(
    *,
    upstream: httpx.Response,
    request: Request,
    request_id: str,
    open_ms: float,
    deadline: Deadline | None = None,
) -> AsyncIterator[bytes]:
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    line_count = 0
    content_chunks = 0
    disconnected = False
    deadline_hit = False
    error_class = "ok"
    status = 200

    try:
        async for line in upstream.aiter_lines():
            if await request.is_disconnected():
                disconnected = True
                error_class = "client_disconnected"
                break

            if deadline is not None and deadline.expired():
                deadline_hit = True
                error_class = "request_deadline"
                status = 504
                # Best-effort signal; response headers already sent as 200 SSE.
                yield b'data: {"error":"request_deadline"}\n\n'
                break

            line_count += 1
            if _line_has_content(line):
                content_chunks += 1
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - t0) * 1000, 1)

            # Re-emit as SSE line; blank line separates events.
            yield (line + "\n").encode()
    except httpx.HTTPError:
        error_class = "upstream_error"
        status = 502
        raise
    finally:
        # Always close the upstream HTTP stream so the connection is released.
        # This does not prove vLLM stopped GPU work — measure that separately.
        await upstream.aclose()
        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        log_event(
            "stream_finished",
            request_id=request_id,
            status=status,
            error_class=error_class,
            stream=True,
            client_disconnected=disconnected,
            request_deadline_hit=deadline_hit,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            upstream_open_ms=open_ms,
            upstream_content_type=upstream.headers.get("content-type"),
            lines=line_count,
            content_chunks=content_chunks,
        )
