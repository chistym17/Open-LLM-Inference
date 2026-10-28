"""Stage 3: streaming vs non-streaming latency (TTFT, total, inter-chunk).

Fair protocol: warmup both modes, then interleaved measured pairs.
Also runs one disconnect smoke (abort after first content chunk).

Writes results/v2/phase-3/streaming_compare.json (gitignored).
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

OUT = ROOT / "results" / "v2" / "phase-3"
PROMPT = "Say hello in one short sentence."
DEFAULT_GATEWAY = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8081").rstrip("/")


def _payload(max_tokens: int, stream: bool) -> dict:
    return {
        "model": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }


def _non_stream_once(client: httpx.Client, gateway: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    response = client.post(
        f"{gateway}/v1/chat/completions",
        json=_payload(max_tokens, stream=False),
        timeout=300.0,
    )
    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    body: dict = {}
    try:
        body = response.json()
    except json.JSONDecodeError:
        body = {"detail": response.text[:300]}
    return {
        "mode": "non_stream",
        "ok": response.is_success,
        "status": response.status_code,
        "ttft_ms": total_ms if response.is_success else None,
        "total_ms": total_ms,
        "content_chunks": 1 if response.is_success else 0,
        "inter_chunk_ms": [],
        "gateway_ms": body.get("gateway_ms"),
        "upstream_ms": body.get("upstream_ms"),
        "request_id": body.get("request_id"),
        "completion_tokens": (body.get("usage") or {}).get("completion_tokens"),
    }


def _extract_delta_content(data: str) -> str | None:
    if not data or data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    try:
        choice = payload["choices"][0]
    except (KeyError, IndexError, TypeError):
        return None
    delta = choice.get("delta") or {}
    content = delta.get("content")
    if content:
        return str(content)
    # some servers put text on the choice itself during stream
    text = choice.get("text")
    return str(text) if text else None


def _stream_once(
    client: httpx.Client,
    gateway: str,
    max_tokens: int,
    *,
    abort_after_content_chunks: int | None = None,
) -> dict:
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    content_times: list[float] = []
    content_chunks = 0
    text_parts: list[str] = []
    request_id = None
    aborted = False
    raw_head = ""
    content_type = None
    saw_data_prefix = False

    with client.stream(
        "POST",
        f"{gateway}/v1/chat/completions",
        json=_payload(max_tokens, stream=True),
        headers={
            "Accept": "text/event-stream",
            "X-Stream": "true",
        },
        timeout=300.0,
    ) as response:
        request_id = response.headers.get("x-request-id")
        content_type = response.headers.get("content-type")
        gateway_mode = response.headers.get("x-gateway-mode")
        if response.status_code >= 400:
            detail = response.read().decode(errors="replace")[:300]
            return {
                "mode": "stream",
                "ok": False,
                "status": response.status_code,
                "ttft_ms": None,
                "total_ms": round((time.perf_counter() - t0) * 1000, 1),
                "content_chunks": 0,
                "inter_chunk_ms": [],
                "request_id": request_id,
                "aborted": False,
                "content_type": content_type,
                "gateway_mode": gateway_mode,
                "detail": detail,
            }

        for line in response.iter_lines():
            if len(raw_head) < 400:
                raw_head += line[:400] + "\n"

            if not line:
                continue
            if line.startswith("data:"):
                saw_data_prefix = True
                payload = line[5:].strip()
            elif line.startswith("{") and not saw_data_prefix:
                # non-SSE JSON body (stream not actually proxied)
                payload = line
            else:
                continue

            content = _extract_delta_content(payload)
            if content is None and payload.startswith("{") and not saw_data_prefix:
                try:
                    body = json.loads(payload)
                    content = body["choices"][0]["message"]["content"]
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    content = None
            if not content:
                continue

            now = time.perf_counter()
            if ttft_ms is None:
                ttft_ms = round((now - t0) * 1000, 1)
            content_times.append(now)
            content_chunks += 1
            text_parts.append(content)
            if (
                abort_after_content_chunks is not None
                and content_chunks >= abort_after_content_chunks
            ):
                aborted = True
                break

    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    gaps = []
    for i in range(1, len(content_times)):
        gaps.append(round((content_times[i] - content_times[i - 1]) * 1000, 1))

    return {
        "mode": "stream",
        "ok": True,
        "status": 200,
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "content_chunks": content_chunks,
        "inter_chunk_ms": gaps,
        "request_id": request_id,
        "aborted": aborted,
        "text_preview": "".join(text_parts)[:120],
        "content_type": content_type,
        "gateway_mode": gateway_mode,
        "saw_data_prefix": saw_data_prefix,
        "raw_head": raw_head[:400],
    }

def _require_ok(name: str, row: dict) -> None:
    if not row.get("ok"):
        raise RuntimeError(f"{name} failed: {row}")


def _mode_summary(rows: list[dict], mode: str) -> dict:
    measured = [r for r in rows if r["phase"] == "measured" and r["mode"] == mode]
    warm = [r for r in rows if r["phase"] == "warmup" and r["mode"] == mode]
    out: dict = {
        "warmup": {
            "total_ms": [r["total_ms"] for r in warm],
            "ttft_ms": [r["ttft_ms"] for r in warm if r.get("ttft_ms") is not None],
        },
        "measured": {},
    }
    if measured:
        out["measured"]["total"] = summarize([float(r["total_ms"]) for r in measured])
        ttfts = [float(r["ttft_ms"]) for r in measured if r.get("ttft_ms") is not None]
        if ttfts:
            out["measured"]["ttft"] = summarize(ttfts)
        gaps = [g for r in measured for g in r.get("inter_chunk_ms") or []]
        if gaps:
            out["measured"]["inter_chunk"] = summarize([float(g) for g in gaps])
        out["measured"]["mean_content_chunks"] = round(
            sum(r["content_chunks"] for r in measured) / len(measured), 1
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 3 streaming benchmark")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    print(
        f"warmup both modes ({args.warmup} each), "
        f"then {args.runs} interleaved measured pairs",
        flush=True,
    )

    with httpx.Client(timeout=300.0) as client:
        for i in range(args.warmup):
            row = _non_stream_once(client, args.gateway, args.max_tokens)
            row.update({"phase": "warmup", "index": i})
            _require_ok("non_stream warmup", row)
            rows.append(row)
            print(f"warmup non_stream[{i}] total={row['total_ms']} ms", flush=True)

        for i in range(args.warmup):
            row = _stream_once(client, args.gateway, args.max_tokens)
            row.update({"phase": "warmup", "index": i})
            _require_ok("stream warmup", row)
            rows.append(row)
            print(
                f"warmup stream[{i}] ttft={row['ttft_ms']} total={row['total_ms']} ms",
                flush=True,
            )

        for i in range(args.runs):
            non_stream = _non_stream_once(client, args.gateway, args.max_tokens)
            non_stream.update({"phase": "measured", "index": i, "pair": i})
            _require_ok(f"non_stream measured[{i}]", non_stream)
            rows.append(non_stream)

            streamed = _stream_once(client, args.gateway, args.max_tokens)
            streamed.update({"phase": "measured", "index": i, "pair": i})
            _require_ok(f"stream measured[{i}]", streamed)
            rows.append(streamed)

            print(
                f"measured[{i}] non_stream={non_stream['total_ms']} ms "
                f"stream_ttft={streamed['ttft_ms']} stream_total={streamed['total_ms']} ms",
                flush=True,
            )

        disconnect = _stream_once(
            client,
            args.gateway,
            args.max_tokens,
            abort_after_content_chunks=1,
        )
        disconnect.update({"phase": "disconnect_smoke", "index": 0})
        rows.append(disconnect)
        print(
            f"disconnect_smoke aborted={disconnect.get('aborted')} "
            f"chunks={disconnect.get('content_chunks')} "
            f"ttft={disconnect.get('ttft_ms')} total={disconnect.get('total_ms')}",
            flush=True,
        )

    non_summary = _mode_summary(rows, "non_stream")
    stream_summary = _mode_summary(rows, "stream")
    report = {
        "question": "How does streaming change TTFT vs total latency vs non-streaming?",
        "environment": {
            "model": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
            "gpu": "NVIDIA T4 (Modal)",
            "max_tokens": args.max_tokens,
            "prompt": PROMPT,
            "warmup_per_mode": args.warmup,
            "measured_pairs": args.runs,
            "protocol": (
                "warmup non_stream, warmup stream, then interleaved measured pairs; "
                "plus one disconnect abort after first content chunk"
            ),
            "timing_note": (
                "ttft_ms is client-observed time to first content delta. "
                "For non_stream, ttft_ms equals total_ms (full body). "
                "Disconnect smoke only shows client abort; GPU cancel is Phase 6."
            ),
        },
        "summary": {
            "non_stream": non_summary,
            "stream": stream_summary,
            "disconnect_smoke": disconnect,
        },
        "rows": rows,
    }

    out_path = OUT / "streaming_compare.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"wrote": str(out_path), "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
