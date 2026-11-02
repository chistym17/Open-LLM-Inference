"""Stage 7 probes. Compare gateway /metrics with a known workload and with vLLM /metrics.

One experiment per run. Restart the gateway with a bad VLLM_BASE_URL before upstream_down.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DEFAULT_OUT = ROOT / "results" / "v2" / "phase-7"
DEFAULT_GATEWAY = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8081").rstrip("/")
PROMPT = "Reply with one short sentence about metrics."


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 7 observability probes")
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["idle", "known_load", "overload", "upstream_down", "cancel"],
    )
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY)
    parser.add_argument("--requests", type=int, default=8, help="known_load count")
    parser.add_argument("--concurrency", type=int, default=16, help="overload fan-out")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--client-timeout-s", type=float, default=120.0)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--out-name", default="")
    return parser.parse_args()


def parse_samples(text: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            left, value_s = line.rsplit(" ", 1)
            value = float(value_s)
        except ValueError:
            continue
        labels: dict[str, str] = {}
        if "{" in left and left.endswith("}"):
            name, rest = left.split("{", 1)
            for key, val in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"', rest):
                labels[key] = val
        else:
            name = left
        samples.append({"name": name, "labels": labels, "value": value})
    return samples


def metric_value(samples: list[dict[str, Any]], name: str, **labels: str) -> float:
    total = 0.0
    for sample in samples:
        if sample["name"] != name:
            continue
        if all(sample["labels"].get(key) == val for key, val in labels.items()):
            total += float(sample["value"])
    return total


def matching_lines(text: str, needle: str, limit: int = 12) -> list[str]:
    return [line for line in text.splitlines() if needle in line and not line.startswith("#")][:limit]


async def _get_text(client: httpx.AsyncClient, url: str, **kwargs: Any) -> tuple[int, str]:
    response = await client.get(url, **kwargs)
    return response.status_code, response.text


async def gateway_snapshot(client: httpx.AsyncClient, gateway: str) -> dict[str, Any]:
    live_status, live_body = await _get_text(client, f"{gateway}/health/live", timeout=10.0)
    ready = await client.get(f"{gateway}/health/ready", timeout=30.0)
    metrics_status, metrics_body = await _get_text(client, f"{gateway}/metrics", timeout=10.0)
    samples = parse_samples(metrics_body) if metrics_status == 200 else []
    return {
        "live_status": live_status,
        "live": json.loads(live_body) if live_status == 200 else live_body[:200],
        "ready_status": ready.status_code,
        "ready": ready.json() if ready.headers.get("content-type", "").startswith("application/json") else ready.text[:200],
        "metrics_status": metrics_status,
        "requests_ok": metric_value(
            samples, "gateway_requests_total", status="200", error_class="ok", stream="false"
        ),
        "requests_429": metric_value(
            samples, "gateway_requests_total", status="429", error_class="admission_rejected", stream="false"
        ),
        "admission_rejected": metric_value(samples, "gateway_admission_rejected_total"),
        "client_disconnects": metric_value(samples, "gateway_client_disconnects_total"),
        "in_flight": metric_value(samples, "gateway_in_flight"),
        "latency_sum_ms": metric_value(
            samples, "gateway_request_latency_ms_sum", stream="false"
        ),
        "latency_count": metric_value(
            samples, "gateway_request_latency_ms_count", stream="false"
        ),
    }


async def vllm_snapshot(client: httpx.AsyncClient) -> dict[str, Any]:
    base = os.environ.get("VLLM_BASE_URL", "").rstrip("/")
    token = os.environ.get("MODAL_PROXY_TOKEN", "").strip()
    if not base or not token:
        return {"ok": False, "error": "VLLM_BASE_URL or MODAL_PROXY_TOKEN missing"}
    response = await client.get(
        f"{base}/metrics",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    )
    text = response.text
    samples = parse_samples(text) if response.status_code == 200 else []
    return {
        "ok": response.status_code == 200,
        "status": response.status_code,
        "running": metric_value(samples, "vllm:num_requests_running"),
        "abort": metric_value(
            samples, "vllm:request_success_total", finished_reason="abort"
        ),
        "running_lines": matching_lines(text, "num_requests_running"),
        "abort_lines": matching_lines(text, 'finished_reason="abort"'),
        "error": None if response.status_code == 200 else text[:200],
    }


def _chat_body(max_tokens: int, *, stream: bool) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }


async def _json_chat(
    client: httpx.AsyncClient, gateway: str, max_tokens: int, timeout_s: float
) -> dict[str, Any]:
    t0 = time.perf_counter()
    row: dict[str, Any] = {"ok": False, "status": 0, "latency_ms": None, "error_class": None}
    try:
        response = await client.post(
            f"{gateway}/v1/chat/completions",
            json=_chat_body(max_tokens, stream=False),
            timeout=timeout_s,
        )
        row["status"] = response.status_code
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        row["error_class"] = response.headers.get("X-Error-Class")
        row["ok"] = response.status_code == 200
    except Exception as exc:  # noqa: BLE001
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        row["error"] = str(exc)
    return row


async def _abort_stream(
    client: httpx.AsyncClient,
    gateway: str,
    max_tokens: int,
    timeout_s: float,
    on_first_chunk=None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    row: dict[str, Any] = {"aborted": False, "content_chunks": 0, "status": 0, "latency_ms": None}
    long_body = _chat_body(max_tokens, stream=True)
    long_body["min_tokens"] = max_tokens
    long_body["messages"] = [
        {
            "role": "user",
            "content": "Write a long essay. Do not stop early. Keep going until you run out of tokens.",
        }
    ]
    try:
        async with client.stream(
            "POST",
            f"{gateway}/v1/chat/completions",
            json=long_body,
            headers={"Accept": "text/event-stream", "X-Stream": "1"},
            timeout=timeout_s,
        ) as response:
            row["status"] = response.status_code
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                    delta = payload["choices"][0].get("delta") or {}
                    if delta.get("content"):
                        row["content_chunks"] += 1
                        if row["content_chunks"] == 1 and on_first_chunk is not None:
                            await on_first_chunk()
                        if row["content_chunks"] >= 1:
                            row["aborted"] = True
                            break
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
    except Exception as exc:  # noqa: BLE001
        row["error"] = str(exc)
    row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return row


def _delta(after: dict[str, Any], before: dict[str, Any], key: str) -> float:
    return float(after.get(key) or 0) - float(before.get(key) or 0)


async def exp_idle(args: argparse.Namespace) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        gateway = await gateway_snapshot(client, args.gateway)
        upstream = await vllm_snapshot(client)
    return {
        "experiment": "idle",
        "gateway": gateway,
        "vllm": upstream,
        "pass": (
            gateway["live_status"] == 200
            and gateway["ready_status"] == 200
            and gateway["metrics_status"] == 200
            and bool(upstream.get("ok"))
        ),
    }


async def exp_known_load(args: argparse.Namespace) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        before = await gateway_snapshot(client, args.gateway)
        rows = []
        for _ in range(args.requests):
            rows.append(
                await _json_chat(client, args.gateway, args.max_tokens, args.client_timeout_s)
            )
        after = await gateway_snapshot(client, args.gateway)
    ok_n = sum(1 for row in rows if row.get("ok"))
    latencies = [float(row["latency_ms"]) for row in rows if row.get("ok") and row.get("latency_ms") is not None]
    client_p50 = statistics.median(latencies) if latencies else None
    count_delta = _delta(after, before, "latency_count")
    mean_delta = None
    if count_delta > 0:
        mean_delta = _delta(after, before, "latency_sum_ms") / count_delta
    close = False
    if client_p50 and mean_delta is not None:
        close = abs(mean_delta - client_p50) <= max(750.0, 0.35 * client_p50)
    return {
        "experiment": "known_load",
        "requests": args.requests,
        "client_ok_n": ok_n,
        "client_p50_ms": client_p50,
        "gateway_ok_delta": _delta(after, before, "requests_ok"),
        "gateway_latency_mean_ms": round(mean_delta, 1) if mean_delta is not None else None,
        "latency_close": close,
        "in_flight_after": after.get("in_flight"),
        "pass": ok_n == args.requests and _delta(after, before, "requests_ok") == args.requests and close and int(after.get("in_flight") or 0) == 0,
    }


async def exp_overload(args: argparse.Namespace) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        before = await gateway_snapshot(client, args.gateway)
        rows = await asyncio.gather(
            *[
                _json_chat(client, args.gateway, args.max_tokens, args.client_timeout_s)
                for _ in range(args.concurrency)
            ]
        )
        await asyncio.sleep(0.3)
        after = await gateway_snapshot(client, args.gateway)
    ok_n = sum(1 for row in rows if row.get("ok"))
    rejected_n = sum(1 for row in rows if row.get("status") == 429)
    return {
        "experiment": "overload",
        "concurrency": args.concurrency,
        "client_ok_n": ok_n,
        "client_429_n": rejected_n,
        "gateway_ok_delta": _delta(after, before, "requests_ok"),
        "gateway_429_delta": _delta(after, before, "requests_429"),
        "admission_rejected_delta": _delta(after, before, "admission_rejected"),
        "in_flight_after": after.get("in_flight"),
        "pass": (
            rejected_n > 0
            and _delta(after, before, "requests_ok") == ok_n
            and _delta(after, before, "admission_rejected") == rejected_n
            and int(after.get("in_flight") or 0) == 0
        ),
    }


async def exp_upstream_down(args: argparse.Namespace) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        gateway = await gateway_snapshot(client, args.gateway)
    ready = gateway.get("ready") if isinstance(gateway.get("ready"), dict) else {}
    return {
        "experiment": "upstream_down",
        "expect": "live 200, ready 503. Start gateway with a bad VLLM_BASE_URL.",
        "live_status": gateway.get("live_status"),
        "ready_status": gateway.get("ready_status"),
        "ready": ready,
        "pass": gateway.get("live_status") == 200 and gateway.get("ready_status") == 503 and ready.get("ready") is False,
    }


async def exp_cancel(args: argparse.Namespace) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        before_g = await gateway_snapshot(client, args.gateway)
        before_v = await vllm_snapshot(client)
        mid_v: dict[str, Any] = {}

        async def sample_while_running() -> None:
            mid_v.update(await vllm_snapshot(client))

        aborted = await _abort_stream(
            client,
            args.gateway,
            max(args.max_tokens, 128),
            args.client_timeout_s,
            on_first_chunk=sample_while_running,
        )
        await asyncio.sleep(1.0)
        after_g = await gateway_snapshot(client, args.gateway)
        after_v = await vllm_snapshot(client)
    disconnect_delta = _delta(after_g, before_g, "client_disconnects")
    abort_delta = _delta(after_v, before_v, "abort")
    running_after = after_v.get("running")
    return {
        "experiment": "cancel",
        "abort_request": aborted,
        "gateway_disconnect_delta": disconnect_delta,
        "vllm_abort_delta": abort_delta,
        "vllm_running_before": before_v.get("running"),
        "vllm_running_during": mid_v.get("running"),
        "vllm_running_after": running_after,
        "vllm_abort_lines": after_v.get("abort_lines"),
        "vllm_running_lines": after_v.get("running_lines"),
        "vllm_scrape_ok": bool(before_v.get("ok") and after_v.get("ok")),
        "pass": (
            bool(aborted.get("aborted"))
            and disconnect_delta >= 1
            and abort_delta >= 1
            and bool(before_v.get("ok") and after_v.get("ok"))
            and float(running_after or 0) == 0.0
            and int(after_g.get("in_flight") or 0) == 0
        ),
    }


EXPERIMENTS = {
    "idle": exp_idle,
    "known_load": exp_known_load,
    "overload": exp_overload,
    "upstream_down": exp_upstream_down,
    "cancel": exp_cancel,
}


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        report = asyncio.run(EXPERIMENTS[args.experiment](args))
    except Exception as exc:  # noqa: BLE001
        print(f"probe failed: {exc}", file=sys.stderr)
        return 1
    report["environment"] = {
        "model": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
        "gpu": "NVIDIA T4 (Modal)",
        "gateway": args.gateway,
        "note": "vLLM abort counter is the engine signal. Gateway disconnect alone does not prove it.",
    }
    stem = args.out_name.strip() or args.experiment
    out_path = out_dir / f"{stem}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"experiment": report["experiment"], "pass": report["pass"]}, indent=2))
    print(f"wrote {out_path}")
    return 0 if report.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
