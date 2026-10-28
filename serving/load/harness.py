"""Stage 4 load harness CLI.

Examples:
  python -m serving.load.harness --mode closed --concurrency 4 --requests 20
  python -m serving.load.harness --mode open --rps 2 --requests 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from serving.load.report import summarize_rows
from serving.load.runners import run_closed_loop, run_open_loop, run_warmup
from serving.load.workload import build_prompt, chat_payload

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
OUT = ROOT / "results" / "v2" / "phase-4"
DEFAULT_GATEWAY = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8081").rstrip("/")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4 gateway load harness")
    parser.add_argument("--mode", choices=["closed", "open"], required=True)
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY)
    parser.add_argument("--concurrency", type=int, default=4, help="closed-loop workers")
    parser.add_argument("--rps", type=float, default=2.0, help="open-loop start rate")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--prompt-chars",
        type=int,
        default=80,
        help="approximate prompt size in characters",
    )
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--out-name", default="", help="optional filename stem")
    return parser.parse_args()


async def _amain(args: argparse.Namespace) -> dict:
    prompt = build_prompt(args.prompt_chars)

    def make_payload(_index: int, _phase: str) -> dict:
        return chat_payload(
            prompt=prompt,
            max_tokens=args.max_tokens,
            stream=args.stream,
        )

    max_conn = max(args.concurrency * 2, args.requests, 16)
    limits = httpx.Limits(max_connections=max_conn, max_keepalive_connections=max_conn)
    async with httpx.AsyncClient(timeout=args.timeout_s, limits=limits) as client:
        health = await client.get(f"{args.gateway}/health", timeout=30.0)
        if health.status_code != 200:
            raise RuntimeError(f"gateway health failed: {health.status_code}")

        print(f"warmup x{args.warmup}", flush=True)
        warmup_rows = await run_warmup(
            client,
            args.gateway,
            make_payload,
            count=args.warmup,
            timeout_s=args.timeout_s,
        )

        print(
            f"measured mode={args.mode} requests={args.requests} "
            f"concurrency={args.concurrency} rps={args.rps} stream={args.stream}",
            flush=True,
        )
        t0 = time.perf_counter()
        if args.mode == "closed":
            measured = await run_closed_loop(
                client,
                args.gateway,
                make_payload,
                concurrency=args.concurrency,
                requests=args.requests,
                timeout_s=args.timeout_s,
            )
        else:
            measured = await run_open_loop(
                client,
                args.gateway,
                make_payload,
                rps=args.rps,
                requests=args.requests,
                timeout_s=args.timeout_s,
            )
        wall_s = round(time.perf_counter() - t0, 3)
        for row in measured:
            row["run_wall_s"] = wall_s

    rows = warmup_rows + measured
    summary = summarize_rows(rows)
    report = {
        "question": "How do latency and throughput change under closed vs open load?",
        "environment": {
            "model": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
            "gpu": "NVIDIA T4 (Modal)",
            "gateway": "local FastAPI -> Modal vLLM",
            "mode": args.mode,
            "concurrency": args.concurrency if args.mode == "closed" else None,
            "rps": args.rps if args.mode == "open" else None,
            "requests": args.requests,
            "warmup": args.warmup,
            "max_tokens": args.max_tokens,
            "prompt_chars": args.prompt_chars,
            "stream": args.stream,
            "timeout_s": args.timeout_s,
            "note": (
                "Closed-loop: workers send the next request only after the previous finishes. "
                "Open-loop: request starts are scheduled at a fixed RPS. "
                "No gateway admission queue in this stage."
            ),
        },
        "summary": summary,
        "rows": rows,
    }
    return report


def main() -> int:
    args = _parse_args()
    if args.mode == "closed" and args.concurrency < 1:
        print("concurrency must be >= 1", file=sys.stderr)
        return 2
    if args.mode == "open" and args.rps <= 0:
        print("rps must be > 0", file=sys.stderr)
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        report = asyncio.run(_amain(args))
    except Exception as exc:  # noqa: BLE001
        print(f"load harness failed: {exc}", file=sys.stderr)
        return 1

    stem = args.out_name.strip()
    if not stem:
        if args.mode == "closed":
            stem = f"closed_c{args.concurrency}_n{args.requests}"
        else:
            stem = f"open_rps{args.rps:g}_n{args.requests}"
        if args.stream:
            stem += "_stream"
    out_path = OUT / f"{stem}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"wrote": str(out_path), "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
