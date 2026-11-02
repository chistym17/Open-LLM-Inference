"""Stage 8 capacity and cost session.

One container lifecycle:
  cold start -> light warm load -> sustained warm load -> idle until scale-down.

Dollars come from the Modal billing report for this app when the report moves.
Otherwise the script estimates from measured uptime and the T4 hourly rate,
plus the CPU+memory overhead ratio already on the bill.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DEFAULT_OUT = ROOT / "results" / "v2" / "phase-8"
DEFAULT_GATEWAY = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8081").rstrip("/")
APP_NAME = "open-llm-v2-vllm"
MODAL_BIN = os.environ.get(
    "MODAL_BIN", str(ROOT / "venv" / "bin" / "modal")
)
PROMPT = "Reply with one short sentence about capacity."
# Observed on the Nov 2 bill when the pre-session T4 line is still zero.
DEFAULT_OVERHEAD_RATIO = (0.00204867 + 0.00339580) / 0.11504994


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 8 capacity and cost session")
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY)
    parser.add_argument("--light-requests", type=int, default=5)
    parser.add_argument("--sustained-requests", type=int, default=20)
    parser.add_argument("--sustained-concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--cold-timeout-s", type=float, default=600.0)
    parser.add_argument("--idle-timeout-s", type=float, default=480.0)
    parser.add_argument("--bill-wait-s", type=float, default=180.0)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--out-name", default="session")
    return parser.parse_args()


def _modal_json(args: list[str]) -> Any:
    completed = subprocess.run(
        [MODAL_BIN, *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _app_id() -> str:
    apps = _modal_json(["app", "list", "--json"])
    for app in apps:
        if app.get("description") == APP_NAME:
            return str(app["app_id"])
    raise RuntimeError(f"Modal app {APP_NAME} not found")


def _billing() -> dict[str, float]:
    rows = _modal_json(
        ["billing", "report", "--for", "today", "--show-resources", "--json"]
    )
    totals = {"T4": 0.0, "CPU": 0.0, "Memory": 0.0, "Network Egress": 0.0, "other": 0.0}
    for row in rows:
        if row.get("description") != APP_NAME:
            continue
        resource = str(row.get("resource") or "other")
        cost = float(row["cost"])
        if resource in totals:
            totals[resource] += cost
        else:
            totals["other"] += cost
    totals["total"] = round(
        totals["T4"] + totals["CPU"] + totals["Memory"] + totals["Network Egress"] + totals["other"],
        8,
    )
    for key in list(totals):
        if key != "total":
            totals[key] = round(totals[key], 8)
    return totals


def _rates() -> dict[str, float]:
    data = _modal_json(["billing", "rates", "--json"])
    return {
        "t4_usd_per_hour": float(data["gpu_hour_cost_t4"]),
        "cpu_usd_per_hour": float(data["cpu_hour_cost"]),
        "mem_gib_usd_per_hour": float(data["mem_gib_hour_cost"]),
    }


def _containers(app_id: str) -> int:
    rows = _modal_json(["container", "list", "--app-id", app_id, "--json"])
    return len(rows)


def _overhead_ratio(bill: dict[str, float]) -> float:
    if bill["T4"] <= 0:
        return DEFAULT_OVERHEAD_RATIO
    return (bill["CPU"] + bill["Memory"]) / bill["T4"]


def _p50(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.median(values), 1)


async def _ensure_gateway(gateway: str) -> None:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{gateway}/health/live", timeout=3.0)
            if response.status_code == 200:
                return
    except httpx.HTTPError:
        pass
    env = os.environ.copy()
    site = str(ROOT / "venv" / "lib" / "python3.12" / "site-packages")
    env["PYTHONPATH"] = site + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.Popen(
        [
            "/usr/bin/python3.12",
            "-m",
            "uvicorn",
            "serving.gateway.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8081",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(f"{gateway}/health/live", timeout=2.0)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.4)
    raise RuntimeError("gateway did not become live")


async def _wait_ready(client: httpx.AsyncClient, gateway: str, timeout_s: float) -> float:
    started = time.perf_counter()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            response = await client.get(f"{gateway}/health/ready", timeout=30.0)
            if response.status_code == 200 and response.json().get("ready") is True:
                return round(time.perf_counter() - started, 1)
        except httpx.HTTPError:
            pass
        await asyncio.sleep(2.0)
    raise TimeoutError("vLLM did not become ready")


def _payload(max_tokens: int) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }


async def _one(
    client: httpx.AsyncClient, gateway: str, max_tokens: int
) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {"ok": False, "status": 0, "latency_ms": None, "completion_tokens": 0}
    try:
        response = await client.post(
            f"{gateway}/v1/chat/completions",
            json=_payload(max_tokens),
            timeout=180.0,
        )
        row["status"] = response.status_code
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        row["ok"] = response.status_code == 200
        if row["ok"]:
            body = response.json()
            usage = body.get("usage") or {}
            row["completion_tokens"] = int(
                body.get("completion_tokens") or usage.get("completion_tokens") or 0
            )
        else:
            row["error_class"] = response.headers.get("X-Error-Class")
    except Exception as exc:  # noqa: BLE001
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        row["error"] = str(exc)
    return row


def _summarize(rows: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    ok = [row for row in rows if row.get("ok")]
    latencies = [float(row["latency_ms"]) for row in ok if row.get("latency_ms") is not None]
    tokens = sum(int(row.get("completion_tokens") or 0) for row in ok)
    return {
        "n": len(rows),
        "ok_n": len(ok),
        "failed_n": len(rows) - len(ok),
        "wall_s": round(wall_s, 1),
        "p50_ms": _p50(latencies),
        "completion_tokens": tokens,
        "output_tok_s": round(tokens / wall_s, 2) if wall_s > 0 else None,
    }


async def _light(
    client: httpx.AsyncClient, gateway: str, n: int, max_tokens: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    rows = []
    for _ in range(n):
        rows.append(await _one(client, gateway, max_tokens))
    return _summarize(rows, time.perf_counter() - started), rows


async def _sustained(
    client: httpx.AsyncClient,
    gateway: str,
    n: int,
    concurrency: int,
    max_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    sem = asyncio.Semaphore(concurrency)

    async def bounded() -> dict[str, Any]:
        async with sem:
            return await _one(client, gateway, max_tokens)

    rows = list(await asyncio.gather(*[bounded() for _ in range(n)]))
    summary = _summarize(rows, time.perf_counter() - started)
    summary["concurrency"] = concurrency
    return summary, rows


def _wait_until_idle(app_id: str, timeout_s: float) -> dict[str, Any]:
    started = time.perf_counter()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _containers(app_id) == 0:
            return {"idle_s": round(time.perf_counter() - started, 1), "scaled_to_zero": True}
        time.sleep(10)
    return {
        "idle_s": round(time.perf_counter() - started, 1),
        "scaled_to_zero": False,
    }


def _wait_for_bill(before: dict[str, float], timeout_s: float) -> dict[str, float]:
    deadline = time.monotonic() + timeout_s
    latest = before
    while time.monotonic() < deadline:
        time.sleep(20)
        latest = _billing()
        if latest["total"] > before["total"] + 0.00001:
            return latest
    return latest


def _allocate(
    segments: dict[str, float],
    session_usd: float,
) -> dict[str, Any]:
    total_s = sum(segments.values())
    allocated = {}
    for name, seconds in segments.items():
        share = (seconds / total_s) if total_s else 0.0
        allocated[name] = {
            "seconds": round(seconds, 1),
            "usd": round(session_usd * share, 6),
        }
    return {"total_seconds": round(total_s, 1), "segments": allocated}


async def _session(args: argparse.Namespace) -> dict[str, Any]:
    app_id = _app_id()
    rates = _rates()
    before_containers = _containers(app_id)
    bill_before = _billing()
    overhead = _overhead_ratio(bill_before)
    all_in_hour = rates["t4_usd_per_hour"] * (1.0 + overhead)

    await _ensure_gateway(args.gateway)
    async with httpx.AsyncClient() as client:
        cold_s = await _wait_ready(client, args.gateway, args.cold_timeout_s)
        containers_when_ready = _containers(app_id)
        light, _light_rows = await _light(
            client, args.gateway, args.light_requests, args.max_tokens
        )
        sustained, _sustained_rows = await _sustained(
            client,
            args.gateway,
            args.sustained_requests,
            args.sustained_concurrency,
            args.max_tokens,
        )

    idle = _wait_until_idle(app_id, args.idle_timeout_s)
    bill_after = _wait_for_bill(bill_before, args.bill_wait_s)
    billed_delta = round(bill_after["total"] - bill_before["total"], 8)
    resource_delta = {
        key: round(bill_after[key] - bill_before[key], 8)
        for key in ("T4", "CPU", "Memory", "Network Egress", "other")
    }

    segments = {
        "cold_start": cold_s,
        "light": light["wall_s"],
        "sustained": sustained["wall_s"],
        "idle": idle["idle_s"],
    }
    estimate_usd = round(sum(segments.values()) / 3600.0 * all_in_hour, 6)
    session_usd = billed_delta if billed_delta > 0 else estimate_usd
    cost_source = "modal_billing_report" if billed_delta > 0 else "uptime_estimate"
    allocation = _allocate(segments, session_usd)

    ok_n = light["ok_n"] + sustained["ok_n"]
    tokens = light["completion_tokens"] + sustained["completion_tokens"]
    active_s = cold_s + light["wall_s"] + sustained["wall_s"]
    active_usd = round(session_usd * (active_s / sum(segments.values())), 6) if sum(segments.values()) else 0.0

    sustained_rps = (
        sustained["ok_n"] / sustained["wall_s"] if sustained["wall_s"] else 0.0
    )
    always_on_requests = sustained_rps * 3600.0
    return {
        "experiment": "session",
        "app": APP_NAME,
        "rates": rates,
        "overhead_ratio_cpu_mem_over_t4": round(overhead, 4),
        "all_in_usd_per_gpu_hour": round(all_in_hour, 6),
        "containers_before": before_containers,
        "containers_when_ready": containers_when_ready,
        "cold_start_s": cold_s,
        "light": light,
        "sustained": sustained,
        "idle": idle,
        "billing_before": bill_before,
        "billing_after": bill_after,
        "billing_delta_usd": resource_delta | {"total": billed_delta},
        "cost_source": cost_source,
        "session_usd": round(session_usd, 6),
        "estimate_usd": estimate_usd,
        "allocation": allocation,
        "ok_n": ok_n,
        "completion_tokens": tokens,
        "usd_per_successful_request_including_idle": round(session_usd / ok_n, 6) if ok_n else None,
        "usd_per_output_token_including_idle": round(session_usd / tokens, 8) if tokens else None,
        "usd_per_successful_request_active_only": round(active_usd / ok_n, 6) if ok_n else None,
        "always_on_counterfactual": {
            "usd_per_hour": round(all_in_hour, 6),
            "sustained_ok_per_s": round(sustained_rps, 4),
            "requests_per_hour_if_kept_this_busy": round(always_on_requests, 1),
            "usd_per_successful_request_if_hour_stays_this_busy": (
                round(all_in_hour / always_on_requests, 6) if always_on_requests else None
            ),
            "note": (
                "Always-on price is one GPU-hour at the T4 rate plus the measured "
                "CPU and memory overhead. It is not an extra hour of spend."
            ),
        },
        "counting": {
            "billed_delta": "Modal report for today, app open-llm-v2-vllm, T4+CPU+Memory+egress, after minus before.",
            "estimate": "measured seconds / 3600 * T4 hourly rate * (1 + (CPU+Memory)/T4 from the bill).",
            "per_request": "session dollars / successful light and sustained requests. Cold start and idle are included in the on-demand figure.",
            "per_token": "same session dollars / completion_tokens returned by the gateway. Prompt tokens are not in the denominator.",
            "idle": "seconds from the last sustained response until Modal reports zero containers for this app.",
        },
    }


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        report = asyncio.run(_session(args))
    except Exception as exc:  # noqa: BLE001
        print(f"capacity session failed: {exc}", file=sys.stderr)
        return 1
    report["environment"] = {
        "model": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
        "gpu": "NVIDIA T4 (Modal)",
        "scaledown_window_s": 300,
        "gateway": args.gateway,
    }
    out_path = out_dir / f"{args.out_name}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "cost_source": report["cost_source"],
        "session_usd": report["session_usd"],
        "cold_start_s": report["cold_start_s"],
        "idle_s": report["idle"]["idle_s"],
        "ok_n": report["ok_n"],
        "usd_per_successful_request_including_idle": report["usd_per_successful_request_including_idle"],
    }, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
