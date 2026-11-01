from __future__ import annotations

from typing import Any

from serving.clients.stats import summarize


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [r for r in rows if r.get("phase") == "measured"]
    warmup = [r for r in rows if r.get("phase") == "warmup"]
    ok = [r for r in measured if r.get("ok")]
    failed = [r for r in measured if not r.get("ok")]
    rejected = [r for r in measured if r.get("status") == 429]
    other_fail = [r for r in failed if r.get("status") != 429]

    latencies = [float(r["latency_ms"]) for r in ok if r.get("latency_ms") is not None]
    ttfts = [float(r["ttft_ms"]) for r in ok if r.get("ttft_ms") is not None]
    completion_tokens = [
        int(r["completion_tokens"])
        for r in ok
        if r.get("completion_tokens") is not None
    ]

    wall_s = max((float(r.get("run_wall_s") or 0) for r in measured), default=0.0)
    total_out_tokens = sum(completion_tokens)
    summary: dict[str, Any] = {
        "warmup_n": len(warmup),
        "measured_n": len(measured),
        "ok_n": len(ok),
        "failed_n": len(failed),
        "rejected_429_n": len(rejected),
        "other_failed_n": len(other_fail),
        "success_rate": round(len(ok) / len(measured), 4) if measured else 0.0,
        "reject_rate": round(len(rejected) / len(measured), 4) if measured else 0.0,
        "wall_s": wall_s,
        "output_tokens_total": total_out_tokens,
        "output_tok_s": round(total_out_tokens / wall_s, 2) if wall_s > 0 else None,
        "errors": [r.get("error") for r in failed[:20]],
    }
    if latencies:
        summary["latency_accepted"] = summarize(latencies)
        summary["latency"] = summary["latency_accepted"]
    if ttfts:
        summary["ttft"] = summarize(ttfts)
    return summary
