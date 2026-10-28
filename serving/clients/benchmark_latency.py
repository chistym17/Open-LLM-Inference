"""Stage 2: compare direct Modal vLLM vs FastAPI gateway latency.

Fair protocol:
  1) Warm up both paths (excluded from p50/p95).
  2) Run interleaved measured trials: direct, gateway, direct, gateway, ...

Both paths use persistent httpx clients (keep-alive) so TLS/connect cost
is not mistaken for gateway overhead.

Requires a reachable upstream and a running gateway.
Writes results/v2/phase-2/latency_compare.json (gitignored).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from serving.clients.stats import summarize

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

OUT = ROOT / "results" / "v2" / "phase-2"
PROMPT = "Say hello in one short sentence."
DEFAULT_GATEWAY = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8081").rstrip("/")


def _payload(max_tokens: int) -> dict:
    return {
        "model": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }


def _post(
    client: httpx.Client,
    url: str,
    payload: dict,
    timeout: float = 300,
) -> dict:
    t0 = time.perf_counter()
    try:
        response = client.post(url, json=payload, timeout=timeout)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = {"detail": response.text[:300]}
        return {
            "ok": response.is_success,
            "status": response.status_code,
            "latency_ms": latency_ms,
            "body": body,
        }
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "status": 0,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "body": {"detail": str(exc)},
        }


def _direct_once(client: httpx.Client, base: str, max_tokens: int) -> dict:
    row = _post(client, f"{base}/v1/chat/completions", _payload(max_tokens))
    usage = (row.get("body") or {}).get("usage") or {}
    return {
        "path": "direct",
        "ok": row["ok"],
        "status": row["status"],
        "client_ms": row["latency_ms"],
        "gateway_ms": None,
        "upstream_ms": row["latency_ms"] if row["ok"] else None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "request_id": None,
    }


def _gateway_once(client: httpx.Client, gateway: str, max_tokens: int) -> dict:
    row = _post(client, f"{gateway}/v1/chat/completions", _payload(max_tokens))
    body = row.get("body") or {}
    usage = body.get("usage") or {}
    return {
        "path": "gateway",
        "ok": row["ok"],
        "status": row["status"],
        "client_ms": row["latency_ms"],
        "gateway_ms": body.get("gateway_ms"),
        "upstream_ms": body.get("upstream_ms"),
        "prompt_tokens": body.get("prompt_tokens", usage.get("prompt_tokens")),
        "completion_tokens": body.get(
            "completion_tokens", usage.get("completion_tokens")
        ),
        "request_id": body.get("request_id"),
    }


def _timer_summary(rows: list[dict], key: str) -> dict[str, float] | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return summarize(values) if values else None


def _path_summary(rows: list[dict]) -> dict:
    warm = [r for r in rows if r["phase"] == "warmup"]
    measured = [r for r in rows if r["phase"] == "measured"]
    summary = {
        "warmup": {
            "client_ms": [float(r["client_ms"]) for r in warm],
            "client": _timer_summary(warm, "client_ms"),
            "note": "Excluded from steady-state p50/p95.",
        },
        "measured": {
            "client": _timer_summary(measured, "client_ms"),
        },
    }
    upstream = _timer_summary(measured, "upstream_ms")
    if upstream:
        summary["measured"]["upstream"] = upstream
    gateway = _timer_summary(measured, "gateway_ms")
    if gateway:
        summary["measured"]["gateway"] = gateway
    return summary


def _require_ok(name: str, row: dict) -> None:
    if not row["ok"]:
        raise RuntimeError(f"{name} failed: status={row['status']} body={row}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 fair latency comparison")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY)
    args = parser.parse_args()

    base = os.environ.get("VLLM_BASE_URL", "").rstrip("/")
    token = os.environ.get("MODAL_PROXY_TOKEN", "").strip()
    if not base or not token:
        raise RuntimeError("VLLM_BASE_URL and MODAL_PROXY_TOKEN required")

    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print(
        f"warmup both paths ({args.warmup} each), "
        f"then {args.runs} interleaved measured pairs "
        "(persistent httpx keep-alive on both paths)",
        flush=True,
    )

    with httpx.Client(
        headers={"Authorization": f"Bearer {token}"},
        timeout=300.0,
    ) as direct_client, httpx.Client(timeout=300.0) as gateway_client:
        for i in range(args.warmup):
            row = _direct_once(direct_client, base, args.max_tokens)
            row.update({"phase": "warmup", "index": i})
            _require_ok("direct warmup", row)
            rows.append(row)
            print(f"warmup direct[{i}] {row['client_ms']} ms", flush=True)

        for i in range(args.warmup):
            row = _gateway_once(gateway_client, args.gateway, args.max_tokens)
            row.update({"phase": "warmup", "index": i})
            _require_ok("gateway warmup", row)
            rows.append(row)
            print(f"warmup gateway[{i}] {row['client_ms']} ms", flush=True)

        for i in range(args.runs):
            direct = _direct_once(direct_client, base, args.max_tokens)
            direct.update({"phase": "measured", "index": i, "pair": i})
            _require_ok(f"direct measured[{i}]", direct)
            rows.append(direct)

            gateway = _gateway_once(gateway_client, args.gateway, args.max_tokens)
            gateway.update({"phase": "measured", "index": i, "pair": i})
            _require_ok(f"gateway measured[{i}]", gateway)
            rows.append(gateway)

            print(
                f"measured[{i}] direct={direct['client_ms']} ms "
                f"gateway={gateway['client_ms']} ms",
                flush=True,
            )

    direct_rows = [r for r in rows if r["path"] == "direct"]
    gateway_rows = [r for r in rows if r["path"] == "gateway"]
    direct_summary = _path_summary(direct_rows)
    gateway_summary = _path_summary(gateway_rows)

    d_meas = direct_summary["measured"]["client"]
    g_meas = gateway_summary["measured"]["client"]
    overhead = {
        "mean_ms": round(g_meas["mean_ms"] - d_meas["mean_ms"], 1),
        "p50_ms": round(g_meas["p50_ms"] - d_meas["p50_ms"], 1),
        "p95_ms": round(g_meas["p95_ms"] - d_meas["p95_ms"], 1),
        "note": (
            "Measured-only after warming both paths. "
            "Trials interleaved; both paths use persistent httpx keep-alive. "
            "Expect gateway >= direct by a small amount."
        ),
    }

    report = {
        "question": "How much latency does the FastAPI gateway add vs direct vLLM?",
        "environment": {
            "model": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
            "gpu": "NVIDIA T4 (Modal)",
            "max_tokens": args.max_tokens,
            "prompt": PROMPT,
            "warmup_per_path": args.warmup,
            "measured_pairs": args.runs,
            "protocol": (
                "1) warmup direct N, 2) warmup gateway N, "
                "3) measured interleaved pairs; persistent httpx on both paths"
            ),
            "timing_note": (
                "Warmup excluded from p50/p95. "
                "client_ms is wall time at this script. "
                "upstream_ms for direct equals client_ms. "
                "gateway upstream_ms is FastAPI→vLLM HTTP time, not pure GPU time."
            ),
        },
        "summary": {
            "direct": direct_summary,
            "gateway": gateway_summary,
            "gateway_overhead_vs_direct_client_measured_only": overhead,
        },
        "rows": rows,
    }

    out_path = OUT / "latency_compare.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"wrote": str(out_path), "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
