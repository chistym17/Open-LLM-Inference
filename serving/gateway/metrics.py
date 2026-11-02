"""Gateway Prometheus metrics. One process, no external Prometheus server."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, generate_latest

REQUESTS = Counter(
    "gateway_requests_total",
    "Chat completions by HTTP status and error class.",
    ["status", "error_class", "stream"],
)
LATENCY_MS = Histogram(
    "gateway_request_latency_ms",
    "Gateway-observed chat latency in milliseconds.",
    ["stream"],
    buckets=(100, 250, 500, 1000, 1500, 2000, 3000, 5000, 10000, 30000, 120000),
)
IN_FLIGHT = Gauge(
    "gateway_in_flight",
    "Admitted requests currently in flight.",
)
UPSTREAM_FAILURES = Counter(
    "gateway_upstream_failures_total",
    "Failures talking to vLLM, by error class.",
    ["error_class"],
)
CLIENT_DISCONNECTS = Counter(
    "gateway_client_disconnects_total",
    "Streaming requests where the client disconnected.",
)
ADMISSION_REJECTED = Counter(
    "gateway_admission_rejected_total",
    "Requests rejected because the in-flight cap was full.",
)

_UPSTREAM_CLASSES = {
    "upstream_timeout",
    "upstream_error",
    "malformed_upstream",
    "request_deadline",
}


def set_in_flight(value: int) -> None:
    IN_FLIGHT.set(value)


def record_request(
    *,
    status: int,
    error_class: str,
    stream: bool,
    latency_ms: float | None = None,
) -> None:
    stream_label = "true" if stream else "false"
    REQUESTS.labels(
        status=str(status),
        error_class=error_class,
        stream=stream_label,
    ).inc()
    if latency_ms is not None and latency_ms >= 0:
        LATENCY_MS.labels(stream=stream_label).observe(latency_ms)
    if error_class in _UPSTREAM_CLASSES:
        UPSTREAM_FAILURES.labels(error_class=error_class).inc()
    if error_class == "admission_rejected":
        ADMISSION_REJECTED.inc()
    if error_class == "client_disconnected":
        CLIENT_DISCONNECTS.inc()


def render() -> bytes:
    return generate_latest()
