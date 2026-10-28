from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from serving.gateway.config import settings
from serving.gateway.errors import map_upstream_error
from serving.gateway.logging_util import log_event
from serving.gateway.request_id import HEADER, resolve_request_id
from serving.gateway.schemas import ChatRequest, ChatResponse
from serving.gateway.streaming import proxy_chat_stream
from serving.gateway.upstream import UpstreamError, VLLMClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = VLLMClient(settings)
    app.state.vllm = client
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(title="open-llm-gateway", version="0.3.0", lifespan=lifespan)


@app.middleware("http")
async def attach_request_id(request: Request, call_next: Callable) -> Response:
    request_id = resolve_request_id(request)
    request.state.request_id = request_id
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers[HEADER] = request_id
    response.headers["X-Gateway-Ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return response


def _input_size(messages: list[Any]) -> int:
    return sum(len(message.content) for message in messages)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _validate_chat(body: ChatRequest, request_id: str) -> int:
    if _input_size(body.messages) > settings.max_input_chars:
        log_event(
            "request_finished",
            request_id=request_id,
            status=413,
            error_class="input_too_large",
        )
        raise HTTPException(
            status_code=413,
            detail=f"input exceeds {settings.max_input_chars} characters",
            headers={"X-Error-Class": "input_too_large"},
        )

    max_tokens = body.max_tokens or 64
    if max_tokens < 1 or max_tokens > settings.max_tokens:
        log_event(
            "request_finished",
            request_id=request_id,
            status=400,
            error_class="validation_error",
        )
        raise HTTPException(
            status_code=400,
            detail=f"max_tokens must be between 1 and {settings.max_tokens}",
            headers={"X-Error-Class": "validation_error"},
        )
    return max_tokens


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    request_id = _request_id(request)
    log_event(
        "request_finished",
        request_id=request_id,
        path=request.url.path,
        status=422,
        error_class="validation_error",
    )
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "request_id": request_id},
        headers={HEADER: request_id, "X-Error-Class": "validation_error"},
    )


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    client: VLLMClient = request.app.state.vllm
    try:
        status, _ = await client.health()
        upstream_ok = status == 200
    except UpstreamError:
        upstream_ok = False
    return {
        "status": "ok" if upstream_ok else "degraded",
        "gateway": "ok",
        "upstream": "ok" if upstream_ok else "down",
        "model": settings.model_name,
        "request_id": _request_id(request),
    }


def _wants_stream(body: ChatRequest, request: Request) -> bool:
    header = request.headers.get("x-stream", "").strip().lower()
    if header in {"1", "true", "yes"}:
        return True
    if header in {"0", "false", "no"}:
        return False
    return bool(body.stream)


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatRequest, request: Request):
    request_id = _request_id(request)
    t0 = time.perf_counter()
    max_tokens = _validate_chat(body, request_id)
    want_stream = _wants_stream(body, request)

    payload = {
        "model": body.model or settings.model_name,
        "messages": [message.model_dump() for message in body.messages],
        "max_tokens": max_tokens,
        "temperature": body.temperature,
    }

    client: VLLMClient = request.app.state.vllm
    log_event(
        "chat_received",
        request_id=request_id,
        stream=want_stream,
        body_stream=body.stream,
        max_tokens=max_tokens,
    )

    if want_stream:
        try:
            upstream, open_ms = await client.stream_chat_completions(payload)
        except UpstreamError as exc:
            http_exc = map_upstream_error(exc)
            log_event(
                "stream_finished",
                request_id=request_id,
                status=http_exc.status_code,
                error_class=http_exc.headers.get("X-Error-Class", "upstream_error"),
                stream=True,
                upstream_status=exc.status_code,
            )
            raise http_exc from exc

        return StreamingResponse(
            proxy_chat_stream(
                upstream=upstream,
                request=request,
                request_id=request_id,
                open_ms=open_ms,
            ),
            media_type="text/event-stream",
            headers={
                HEADER: request_id,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Gateway-Mode": "stream",
            },
        )

    try:
        result = await client.chat_completions(payload)
    except UpstreamError as exc:
        http_exc = map_upstream_error(exc)
        log_event(
            "request_finished",
            request_id=request_id,
            status=http_exc.status_code,
            error_class=http_exc.headers.get("X-Error-Class", "upstream_error"),
            upstream_status=exc.status_code,
            gateway_ms=round((time.perf_counter() - t0) * 1000, 1),
            stream=False,
        )
        raise http_exc from exc

    gateway_ms = round((time.perf_counter() - t0) * 1000, 1)
    data = result.data
    usage = data.get("usage") or {}

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        log_event(
            "request_finished",
            request_id=request_id,
            status=502,
            error_class="malformed_upstream",
            gateway_ms=gateway_ms,
            upstream_ms=result.duration_ms,
            stream=False,
        )
        raise HTTPException(
            status_code=502,
            detail="malformed upstream response",
            headers={"X-Error-Class": "malformed_upstream"},
        ) from exc

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    log_event(
        "request_finished",
        request_id=request_id,
        status=200,
        error_class="ok",
        gateway_ms=gateway_ms,
        upstream_ms=result.duration_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=data.get("model", settings.model_name),
        stream=False,
    )

    chat = ChatResponse(
        request_id=request_id,
        id=str(data.get("id", "")),
        model=str(data.get("model", settings.model_name)),
        content=content,
        usage=usage,
        gateway_ms=gateway_ms,
        upstream_ms=result.duration_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return JSONResponse(
        content=chat.model_dump(),
        headers={HEADER: request_id, "X-Gateway-Mode": "json"},
    )
