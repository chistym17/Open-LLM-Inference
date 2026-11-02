from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from serving.gateway.admission import AdmissionController
from serving.gateway.config import settings
from serving.gateway.deadline import Deadline
from serving.gateway.errors import map_upstream_error, request_deadline_error
from serving.gateway.logging_util import log_event
from serving.gateway.metrics import record_request, render, set_in_flight
from serving.gateway.request_id import HEADER, resolve_request_id
from serving.gateway.schemas import ChatRequest, ChatResponse
from serving.gateway.streaming import proxy_chat_stream
from serving.gateway.upstream import UpstreamError, VLLMClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = VLLMClient(settings)
    admission = AdmissionController(settings.max_in_flight)
    app.state.vllm = client
    app.state.admission = admission
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(title="open-llm-gateway", version="0.7.0", lifespan=lifespan)


@app.middleware("http")
async def attach_request_id(request: Request, call_next: Callable) -> Response:
    request_id = resolve_request_id(request)
    request.state.request_id = request_id
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers[HEADER] = request_id
    response.headers["X-Gateway-Ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    admission: AdmissionController = request.app.state.admission
    set_in_flight(admission.in_flight)
    response.headers["X-In-Flight"] = str(admission.in_flight)
    response.headers["X-Max-In-Flight"] = str(admission.max_in_flight)
    return response


async def _release(admission: AdmissionController) -> None:
    await admission.release()
    set_in_flight(admission.in_flight)


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
        record_request(status=413, error_class="input_too_large", stream=False)
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
        record_request(status=400, error_class="validation_error", stream=False)
        raise HTTPException(
            status_code=400,
            detail=f"max_tokens must be between 1 and {settings.max_tokens}",
            headers={"X-Error-Class": "validation_error"},
        )
    return max_tokens


def _reject_overload(request_id: str, admission: AdmissionController) -> None:
    log_event(
        "request_finished",
        request_id=request_id,
        status=429,
        error_class="admission_rejected",
        in_flight=admission.in_flight,
        max_in_flight=admission.max_in_flight,
    )
    set_in_flight(admission.in_flight)
    record_request(status=429, error_class="admission_rejected", stream=False)
    raise HTTPException(
        status_code=429,
        detail="gateway at capacity",
        headers={
            "X-Error-Class": "admission_rejected",
            "Retry-After": str(settings.admission_retry_after_s),
            "X-In-Flight": str(admission.in_flight),
            "X-Max-In-Flight": str(admission.max_in_flight),
        },
    )


def _hop_timeout_s(deadline: Deadline) -> float:
    """Timeout for FastAPI → vLLM for this request."""
    return deadline.effective_timeout_s(settings.upstream_timeout_s)


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
    record_request(status=422, error_class="validation_error", stream=False)
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "request_id": request_id},
        headers={HEADER: request_id, "X-Error-Class": "validation_error"},
    )


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    client: VLLMClient = request.app.state.vllm
    admission: AdmissionController = request.app.state.admission
    upstream_ok = await _upstream_ok(client)
    set_in_flight(admission.in_flight)
    return {
        "status": "ok" if upstream_ok else "degraded",
        "live": True,
        "ready": upstream_ok,
        "gateway": "ok",
        "upstream": "ok" if upstream_ok else "down",
        "model": settings.model_name,
        "request_id": _request_id(request),
        "upstream_timeout_s": settings.upstream_timeout_s,
        "request_deadline_s": settings.request_deadline_s,
        **admission.snapshot(),
    }


def _wants_stream(body: ChatRequest, request: Request) -> bool:
    header = request.headers.get("x-stream", "").strip().lower()
    if header in {"1", "true", "yes"}:
        return True
    if header in {"0", "false", "no"}:
        return False
    return bool(body.stream)


async def _stream_with_admission(
    *,
    admission: AdmissionController,
    upstream,
    request: Request,
    request_id: str,
    open_ms: float,
    deadline: Deadline,
) -> AsyncIterator[bytes]:
    try:
        async for chunk in proxy_chat_stream(
            upstream=upstream,
            request=request,
            request_id=request_id,
            open_ms=open_ms,
            deadline=deadline,
        ):
            yield chunk
    finally:
        await _release(admission)


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatRequest, request: Request):
    request_id = _request_id(request)
    t0 = time.perf_counter()
    deadline = Deadline(settings.request_deadline_s)
    max_tokens = _validate_chat(body, request_id)
    want_stream = _wants_stream(body, request)
    admission: AdmissionController = request.app.state.admission

    payload = {
        "model": body.model or settings.model_name,
        "messages": [message.model_dump() for message in body.messages],
        "max_tokens": max_tokens,
        "temperature": body.temperature,
    }
    if body.min_tokens is not None:
        payload["min_tokens"] = body.min_tokens

    client: VLLMClient = request.app.state.vllm
    log_event(
        "chat_received",
        request_id=request_id,
        stream=want_stream,
        body_stream=body.stream,
        max_tokens=max_tokens,
        in_flight=admission.in_flight,
        max_in_flight=admission.max_in_flight,
        upstream_timeout_s=settings.upstream_timeout_s,
        request_deadline_s=settings.request_deadline_s,
    )

    if not await admission.try_acquire():
        _reject_overload(request_id, admission)

    if deadline.expired():
        await _release(admission)
        gateway_ms = round((time.perf_counter() - t0) * 1000, 1)
        log_event(
            "request_finished",
            request_id=request_id,
            status=504,
            error_class="request_deadline",
            stream=want_stream,
            gateway_ms=gateway_ms,
        )
        record_request(
            status=504,
            error_class="request_deadline",
            stream=want_stream,
            latency_ms=gateway_ms,
        )
        raise request_deadline_error()

    hop_timeout = _hop_timeout_s(deadline)
    if hop_timeout <= 0:
        await _release(admission)
        gateway_ms = round((time.perf_counter() - t0) * 1000, 1)
        log_event(
            "request_finished",
            request_id=request_id,
            status=504,
            error_class="request_deadline",
            stream=want_stream,
            gateway_ms=gateway_ms,
        )
        record_request(
            status=504,
            error_class="request_deadline",
            stream=want_stream,
            latency_ms=gateway_ms,
        )
        raise request_deadline_error()

    if want_stream:
        try:
            upstream, open_ms = await client.stream_chat_completions(
                payload, timeout_s=hop_timeout
            )
        except UpstreamError as exc:
            await _release(admission)
            # If overall deadline was the tighter bound, classify as deadline.
            if (
                exc.status_code == 504
                and deadline.enabled
                and hop_timeout < settings.upstream_timeout_s
            ):
                http_exc = request_deadline_error()
                error_class = "request_deadline"
            else:
                http_exc = map_upstream_error(exc)
                error_class = http_exc.headers.get("X-Error-Class", "upstream_error")
            gateway_ms = round((time.perf_counter() - t0) * 1000, 1)
            log_event(
                "stream_finished",
                request_id=request_id,
                status=http_exc.status_code,
                error_class=error_class,
                stream=True,
                upstream_status=exc.status_code,
            )
            record_request(
                status=http_exc.status_code,
                error_class=error_class,
                stream=True,
                latency_ms=gateway_ms,
            )
            raise http_exc from exc
        except Exception:
            await _release(admission)
            raise

        return StreamingResponse(
            _stream_with_admission(
                admission=admission,
                upstream=upstream,
                request=request,
                request_id=request_id,
                open_ms=open_ms,
                deadline=deadline,
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
        result = await client.chat_completions(payload, timeout_s=hop_timeout)
    except UpstreamError as exc:
        await _release(admission)
        if (
            exc.status_code == 504
            and deadline.enabled
            and hop_timeout < settings.upstream_timeout_s
        ):
            http_exc = request_deadline_error()
            error_class = "request_deadline"
        else:
            http_exc = map_upstream_error(exc)
            error_class = http_exc.headers.get("X-Error-Class", "upstream_error")
        gateway_ms = round((time.perf_counter() - t0) * 1000, 1)
        log_event(
            "request_finished",
            request_id=request_id,
            status=http_exc.status_code,
            error_class=error_class,
            upstream_status=exc.status_code,
            gateway_ms=gateway_ms,
            stream=False,
        )
        record_request(
            status=http_exc.status_code,
            error_class=error_class,
            stream=False,
            latency_ms=gateway_ms,
        )
        raise http_exc from exc
    except Exception:
        await _release(admission)
        raise

    try:
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
            record_request(
                status=502,
                error_class="malformed_upstream",
                stream=False,
                latency_ms=gateway_ms,
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
            in_flight=admission.in_flight,
        )
        record_request(
            status=200,
            error_class="ok",
            stream=False,
            latency_ms=gateway_ms,
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
    finally:
        await _release(admission)


async def _upstream_ok(client: VLLMClient) -> bool:
    try:
        status, _ = await client.health()
        return status == 200
    except UpstreamError:
        return False


@app.get("/health/live")
async def health_live() -> dict[str, Any]:
    """Process is up. Does not call vLLM."""
    return {"status": "ok", "live": True}


@app.get("/health/ready")
async def health_ready(request: Request) -> JSONResponse:
    """Ready only when vLLM is reachable. In-flight at the cap is still ready."""
    client: VLLMClient = request.app.state.vllm
    admission: AdmissionController = request.app.state.admission
    upstream_ok = await _upstream_ok(client)
    set_in_flight(admission.in_flight)
    body = {
        "ready": upstream_ok,
        "upstream": "ok" if upstream_ok else "down",
        "model": settings.model_name,
        **admission.snapshot(),
    }
    return JSONResponse(status_code=200 if upstream_ok else 503, content=body)


@app.get("/metrics")
async def metrics(request: Request) -> Response:
    admission: AdmissionController = request.app.state.admission
    set_in_flight(admission.in_flight)
    return Response(content=render(), media_type="text/plain; version=0.0.4; charset=utf-8")
