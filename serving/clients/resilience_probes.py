"""Stage 6 resilience probes against the local gateway.

One experiment per run. Restart the gateway with the env noted for that
experiment. Does not auto-retry generations.

Gateway stop ≠ GPU stop: gpu_cancel_proxy only measures a latency proxy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "v2" / "phase-6"
DEFAULT_GATEWAY = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8081").rstrip("/")
PROMPT = "Write a long detailed essay about distributed systems and GPU inference."


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 6 timeout/cancel probes")
    p.add_argument(
        "--experiment",
        required=True,
        choices=[
            "upstream_timeout",
            "request_deadline",
            "client_disconnect",
            "gpu_cancel_proxy",
            "slot_recovery",
        ],
    )
    p.add_argument("--gateway", default=DEFAULT_GATEWAY)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument(
        "--abort-after-chunks",
        type=int,
        default=1,
        help="client_disconnect: abort after N content chunks",
    )
    p.add_argument(
        "--client-timeout-s",
        type=float,
        default=60.0,
        help="client-side httpx timeout (not gateway deadline)",
    )
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--out-name", default="")
    return p.parse_args()


async def _health(client: httpx.AsyncClient, gateway: str) -> dict[str, Any]:
    r = await client.get(f"{gateway}/health", timeout=30.0)
    r.raise_for_status()
    return r.json()


def _payload(max_tokens: int, *, stream: bool) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": stream,
    }


async def _json_chat(
    client: httpx.AsyncClient,
    gateway: str,
    max_tokens: int,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    row: dict[str, Any] = {
        "mode": "json",
        "ok": False,
        "status": 0,
        "latency_ms": None,
        "error_class": None,
        "error": None,
        "in_flight_header": None,
    }
    try:
        r = await client.post(
            f"{gateway}/v1/chat/completions",
            json=_payload(max_tokens, stream=False),
            timeout=timeout_s,
        )
        row["status"] = r.status_code
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        row["error_class"] = r.headers.get("X-Error-Class")
        row["in_flight_header"] = r.headers.get("X-In-Flight")
        row["ok"] = r.status_code == 200
        if r.status_code != 200:
            row["error"] = r.text[:300]
        else:
            data = r.json()
            row["completion_tokens"] = (data.get("usage") or {}).get(
                "completion_tokens"
            )
    except httpx.TimeoutException:
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        row["error"] = "client_timeout"
        row["error_class"] = "client_timeout"
    except Exception as exc:  # noqa: BLE001
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        row["error"] = str(exc)
    return row


async def _stream_chat(
    client: httpx.AsyncClient,
    gateway: str,
    max_tokens: int,
    *,
    timeout_s: float,
    abort_after_chunks: int | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    row: dict[str, Any] = {
        "mode": "stream",
        "ok": False,
        "status": 0,
        "latency_ms": None,
        "ttft_ms": None,
        "content_chunks": 0,
        "aborted": False,
        "deadline_event": False,
        "error_class": None,
        "error": None,
        "in_flight_header": None,
    }
    try:
        async with client.stream(
            "POST",
            f"{gateway}/v1/chat/completions",
            json=_payload(max_tokens, stream=True),
            headers={"Accept": "text/event-stream", "X-Stream": "1"},
            timeout=timeout_s,
        ) as response:
            row["status"] = response.status_code
            row["error_class"] = response.headers.get("X-Error-Class")
            row["in_flight_header"] = response.headers.get("X-In-Flight")
            if response.status_code != 200:
                body = (await response.aread())[:300]
                row["error"] = body.decode(errors="replace")
                row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                return row

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if '"error":"request_deadline"' in data:
                    row["deadline_event"] = True
                    break
                try:
                    payload = json.loads(data)
                    delta = (payload["choices"][0].get("delta") or {})
                    if delta.get("content"):
                        row["content_chunks"] += 1
                        if row["ttft_ms"] is None:
                            row["ttft_ms"] = round(
                                (time.perf_counter() - t0) * 1000, 1
                            )
                        if (
                            abort_after_chunks is not None
                            and row["content_chunks"] >= abort_after_chunks
                        ):
                            row["aborted"] = True
                            break
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue

            row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            row["ok"] = (
                not row["aborted"]
                and not row["deadline_event"]
                and row["content_chunks"] > 0
            )
    except httpx.TimeoutException:
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        row["error"] = "client_timeout"
        row["error_class"] = "client_timeout"
    except Exception as exc:  # noqa: BLE001
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        row["error"] = str(exc)
    return row


async def exp_upstream_timeout(args: argparse.Namespace) -> dict[str, Any]:
    """Expect 504 upstream_timeout when gateway hop timeout is short."""
    async with httpx.AsyncClient() as client:
        health = await _health(client, args.gateway)
        row = await _json_chat(
            client,
            args.gateway,
            args.max_tokens,
            timeout_s=args.client_timeout_s,
        )
        health_after = await _health(client, args.gateway)
    return {
        "experiment": "upstream_timeout",
        "expect": "status 504, X-Error-Class=upstream_timeout; in_flight returns to 0",
        "gateway_env_hint": "GATEWAY_UPSTREAM_TIMEOUT_S=3 GATEWAY_REQUEST_DEADLINE_S=0",
        "health_before": health,
        "health_after": health_after,
        "result": row,
        "pass": (
            row.get("status") == 504
            and row.get("error_class") == "upstream_timeout"
            and int(health_after.get("in_flight", -1)) == 0
        ),
    }


async def exp_request_deadline(args: argparse.Namespace) -> dict[str, Any]:
    """Expect 504 request_deadline when overall deadline is short."""
    async with httpx.AsyncClient() as client:
        health = await _health(client, args.gateway)
        row = await _json_chat(
            client,
            args.gateway,
            args.max_tokens,
            timeout_s=args.client_timeout_s,
        )
        health_after = await _health(client, args.gateway)
    return {
        "experiment": "request_deadline",
        "expect": "status 504, X-Error-Class=request_deadline; in_flight returns to 0",
        "gateway_env_hint": (
            "GATEWAY_REQUEST_DEADLINE_S=3 GATEWAY_UPSTREAM_TIMEOUT_S=120"
        ),
        "health_before": health,
        "health_after": health_after,
        "result": row,
        "pass": (
            row.get("status") == 504
            and row.get("error_class") == "request_deadline"
            and int(health_after.get("in_flight", -1)) == 0
        ),
    }


async def exp_client_disconnect(args: argparse.Namespace) -> dict[str, Any]:
    """Abort mid-SSE; gateway should release the admission slot."""
    async with httpx.AsyncClient() as client:
        health = await _health(client, args.gateway)
        row = await _stream_chat(
            client,
            args.gateway,
            args.max_tokens,
            timeout_s=args.client_timeout_s,
            abort_after_chunks=args.abort_after_chunks,
        )
        # Brief pause so gateway finally/release can run.
        await asyncio.sleep(0.5)
        health_after = await _health(client, args.gateway)
    return {
        "experiment": "client_disconnect",
        "expect": (
            "aborted=true after content; gateway in_flight→0. "
            "Proves client abort + slot release, not GPU cancel."
        ),
        "gateway_env_hint": "normal defaults (deadline off or large)",
        "health_before": health,
        "health_after": health_after,
        "result": row,
        "pass": (
            bool(row.get("aborted"))
            and int(row.get("content_chunks") or 0) >= 1
            and int(health_after.get("in_flight", -1)) == 0
        ),
    }


async def exp_gpu_cancel_proxy(args: argparse.Namespace) -> dict[str, Any]:
    """Latency proxy only: abort a long stream, then time a short follow-up.

    If abort freed GPU quickly, follow-up latency should be near a warm baseline.
    If GPU work continued, follow-up may be slower. Not definitive proof.
    """
    async with httpx.AsyncClient() as client:
        health = await _health(client, args.gateway)
        baseline = await _json_chat(
            client, args.gateway, max_tokens=16, timeout_s=args.client_timeout_s
        )
        long_abort = await _stream_chat(
            client,
            args.gateway,
            args.max_tokens,
            timeout_s=args.client_timeout_s,
            abort_after_chunks=1,
        )
        await asyncio.sleep(0.2)
        followup = await _json_chat(
            client, args.gateway, max_tokens=16, timeout_s=args.client_timeout_s
        )
        health_after = await _health(client, args.gateway)

    base_ms = baseline.get("latency_ms") or 0
    follow_ms = followup.get("latency_ms") or 0
    ratio = (follow_ms / base_ms) if base_ms else None
    return {
        "experiment": "gpu_cancel_proxy",
        "expect": (
            "follow-up latency near warm baseline after abort. "
            "Caveat: gateway close ≠ proven GPU stop."
        ),
        "gateway_env_hint": "normal defaults",
        "health_before": health,
        "health_after": health_after,
        "baseline_short": baseline,
        "long_abort": long_abort,
        "followup_short": followup,
        "followup_over_baseline": round(ratio, 3) if ratio is not None else None,
        "pass": (
            bool(baseline.get("ok"))
            and bool(long_abort.get("aborted"))
            and bool(followup.get("ok"))
            and int(health_after.get("in_flight", -1)) == 0
            and ratio is not None
            and ratio < 2.5
        ),
    }


async def exp_slot_recovery(args: argparse.Namespace) -> dict[str, Any]:
    """Fill admission, force a timeout path if possible, then accept new work.

    Prefer running with a short upstream timeout so in-flight slots clear via 504.
    With normal timeouts this still checks that after disconnect/abort, capacity
    returns (uses abort + immediate burst).
    """
    async with httpx.AsyncClient() as client:
        health = await _health(client, args.gateway)
        max_in = int(health.get("max_in_flight") or 6)

        # Start one long stream and abort to free a slot path; then burst.
        abort = await _stream_chat(
            client,
            args.gateway,
            args.max_tokens,
            timeout_s=args.client_timeout_s,
            abort_after_chunks=1,
        )
        await asyncio.sleep(0.5)
        mid = await _health(client, args.gateway)

        burst = await asyncio.gather(
            *[
                _json_chat(
                    client, args.gateway, max_tokens=16, timeout_s=args.client_timeout_s
                )
                for _ in range(max(1, max_in))
            ]
        )
        after = await _health(client, args.gateway)

    ok_n = sum(1 for r in burst if r.get("ok"))
    rej = sum(1 for r in burst if r.get("status") == 429)
    return {
        "experiment": "slot_recovery",
        "expect": (
            "after abort, in_flight→0 and a burst of size max_in_flight can succeed "
            "(0×429 if serialized enough; at least some oks)."
        ),
        "gateway_env_hint": "GATEWAY_MAX_IN_FLIGHT=6 (Phase 5 default)",
        "health_before": health,
        "health_after_abort": mid,
        "health_after_burst": after,
        "abort": abort,
        "burst": burst,
        "burst_ok_n": ok_n,
        "burst_429_n": rej,
        "pass": (
            bool(abort.get("aborted"))
            and int(mid.get("in_flight", -1)) == 0
            and ok_n >= 1
            and int(after.get("in_flight", -1)) == 0
        ),
    }


EXPERIMENTS = {
    "upstream_timeout": exp_upstream_timeout,
    "request_deadline": exp_request_deadline,
    "client_disconnect": exp_client_disconnect,
    "gpu_cancel_proxy": exp_gpu_cancel_proxy,
    "slot_recovery": exp_slot_recovery,
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
        "max_tokens": args.max_tokens,
        "client_timeout_s": args.client_timeout_s,
        "note": (
            "Gateway timeout proves the gateway stopped waiting. "
            "GPU cancellation is only suggested by gpu_cancel_proxy, not proven."
        ),
    }

    stem = args.out_name.strip() or args.experiment
    out_path = out_dir / f"{stem}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("experiment", "pass", "expect") if k in report}, indent=2))
    print(f"wrote {out_path}")
    return 0 if report.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
