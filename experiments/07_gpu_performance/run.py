"""Phase 7 — GPU performance fundamentals.

Idle baseline, GPU sampling during sustained inference, steady-state
benchmarking, and explicit Transformers batch-size scaling.
"""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Explain how an LLM generates the next token."
OUTPUT_DIR = Path("results/phase-7")
SAMPLE_INTERVAL_S = 0.5
IDLE_SAMPLES = 10
IDLE_INTERVAL_S = 1.0
STEADY_WARMUP_REQUESTS = 3
STEADY_REQUESTS = 10
STEADY_NEW_TOKENS = 200
MONITORED_REQUESTS = 8
MONITORED_NEW_TOKENS = 200
BATCH_SIZES = [1, 2, 4, 8]
BATCH_NEW_TOKENS = 50
BATCH_REPEATS = 3


def read_gpu() -> dict:
    fields = (
        "utilization.gpu,utilization.memory,memory.used,memory.total,"
        "power.draw,temperature.gpu"
    )
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    parts = [p.strip() for p in result.stdout.strip().split(",")]
    return {
        "gpu_util_pct": float(parts[0]),
        "memory_util_pct": float(parts[1]),
        "memory_used_mib": float(parts[2]),
        "memory_total_mib": float(parts[3]),
        "power_w": float(parts[4]),
        "temperature_c": float(parts[5]),
    }


def gpu_identity() -> dict:
    name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    memory = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    cuda_version = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    # Prefer CUDA version line from nvidia-smi header when available.
    smi_header = subprocess.run(
        ["nvidia-smi"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    nvidia_smi_cuda = "unknown"
    for line in smi_header.splitlines():
        if "CUDA Version:" in line:
            nvidia_smi_cuda = line.split("CUDA Version:")[-1].split()[0]
            break

    capability = None
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        capability = f"{major}.{minor}"

    return {
        "gpu": name,
        "vram_mib": int(float(memory)),
        "compute_capability": capability,
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "nvidia_smi_cuda_version": nvidia_smi_cuda,
        "driver_version": cuda_version.stdout.strip(),
    }


class GpuSampler:
    def __init__(self, interval_s: float = SAMPLE_INTERVAL_S):
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return self.samples

    def _run(self) -> None:
        idx = 0
        while not self._stop.is_set():
            sample = read_gpu()
            sample["sample"] = idx
            sample["timestamp_s"] = time.time()
            self.samples.append(sample)
            idx += 1
            self._stop.wait(self.interval_s)


def summarize_numeric(values: list[float]) -> dict:
    return {
        "mean": round(statistics.fmean(values), 2) if values else None,
        "median": round(statistics.median(values), 2) if values else None,
        "min": round(min(values), 2) if values else None,
        "max": round(max(values), 2) if values else None,
    }


def generate_once(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_s = time.perf_counter() - start

    output_tokens = int(outputs.shape[-1] - input_len)
    return {
        "latency_s": latency_s,
        "input_tokens": input_len,
        "output_tokens": output_tokens,
        "output_tok_s": output_tokens / latency_s if latency_s > 0 else 0.0,
    }


def generate_batch(model, tokenizer, prompt: str, batch_size: int, max_new_tokens: int) -> dict:
    prompts = [prompt] * batch_size
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_s = time.perf_counter() - start

    # Pad-aware estimate: generated columns beyond the padded prompt length.
    output_tokens_per_seq = int(outputs.shape[-1] - input_len)
    total_output_tokens = output_tokens_per_seq * batch_size
    peak_gib = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )

    return {
        "batch_latency_s": latency_s,
        "aggregate_output_tok_s": total_output_tokens / latency_s if latency_s > 0 else 0.0,
        "approx_per_sequence_tok_s": (
            output_tokens_per_seq / latency_s if latency_s > 0 else 0.0
        ),
        "peak_allocated_gib": peak_gib,
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_idle_baseline() -> dict:
    print("=== Idle GPU baseline ===")
    samples = []
    for i in range(IDLE_SAMPLES):
        sample = read_gpu()
        sample["sample"] = i
        samples.append(sample)
        time.sleep(IDLE_INTERVAL_S)

    identity = gpu_identity()
    payload = {
        **identity,
        "sample_interval_seconds": IDLE_INTERVAL_S,
        "sample_count": len(samples),
        "idle_summary": {
            "gpu_util_pct_mean": statistics.fmean(s["gpu_util_pct"] for s in samples),
            "memory_util_pct_mean": statistics.fmean(s["memory_util_pct"] for s in samples),
            "memory_used_mib_mean": statistics.fmean(s["memory_used_mib"] for s in samples),
            "memory_total_mib": samples[0]["memory_total_mib"],
            "power_w_mean": round(statistics.fmean(s["power_w"] for s in samples), 3),
            "temperature_c_mean": round(statistics.fmean(s["temperature_c"] for s in samples), 1),
        },
        "samples": samples,
    }
    save_json(OUTPUT_DIR / "gpu_baseline.json", payload)
    print(f"Saved {OUTPUT_DIR / 'gpu_baseline.json'}")
    return payload


def run_monitored_inference(model, tokenizer) -> None:
    print("=== GPU activity during sustained inference ===")
    sampler = GpuSampler(interval_s=SAMPLE_INTERVAL_S)
    request_results = []

    sampler.start()
    try:
        for i in range(1, MONITORED_REQUESTS + 1):
            result = generate_once(
                model, tokenizer, PROMPT, max_new_tokens=MONITORED_NEW_TOKENS
            )
            row = {
                "request": i,
                "latency_s": result["latency_s"],
                "output_tokens": result["output_tokens"],
                "output_tok_s": result["output_tok_s"],
            }
            request_results.append(row)
            print(
                f"request {i}: latency={row['latency_s']:.3f}s "
                f"tok/s={row['output_tok_s']:.3f}"
            )
    finally:
        samples = sampler.stop()

    sample_rows = [
        {
            "sample": s["sample"],
            "timestamp_s": s["timestamp_s"],
            "gpu_util_pct": s["gpu_util_pct"],
            "memory_util_pct": s["memory_util_pct"],
            "memory_used_mib": s["memory_used_mib"],
            "memory_total_mib": s["memory_total_mib"],
            "power_w": s["power_w"],
            "temperature_c": s["temperature_c"],
        }
        for s in samples
    ]
    save_csv(
        OUTPUT_DIR / "gpu_samples.csv",
        sample_rows,
        [
            "sample",
            "timestamp_s",
            "gpu_util_pct",
            "memory_util_pct",
            "memory_used_mib",
            "memory_total_mib",
            "power_w",
            "temperature_c",
        ],
    )

    prompt_tokens = len(tokenizer.encode(PROMPT))
    summary = {
        "model": MODEL_ID,
        "precision": "FP16",
        "gpu": gpu_identity()["gpu"],
        "prompt_tokens": prompt_tokens,
        "requests": MONITORED_REQUESTS,
        "max_new_tokens_per_request": MONITORED_NEW_TOKENS,
        "sample_interval_seconds": SAMPLE_INTERVAL_S,
        "gpu_samples_collected": len(samples),
        "gpu_summary": {
            "gpu_util_pct": summarize_numeric([s["gpu_util_pct"] for s in samples]),
            "memory_util_pct": summarize_numeric([s["memory_util_pct"] for s in samples]),
            "memory_used_mib": summarize_numeric([s["memory_used_mib"] for s in samples]),
            "power_w": summarize_numeric([s["power_w"] for s in samples]),
            "temperature_c": summarize_numeric([s["temperature_c"] for s in samples]),
        },
        "request_results": [
            {
                "request": r["request"],
                "latency_s": r["latency_s"],
                "output_tokens": r["output_tokens"],
                "output_tok_s": r["output_tok_s"],
            }
            for r in request_results
        ],
        "notes": [
            "GPU utilization is sampled activity, not a direct memory-bandwidth measurement.",
        ],
    }
    save_json(OUTPUT_DIR / "gpu_activity_summary.json", summary)
    print(f"Saved {OUTPUT_DIR / 'gpu_samples.csv'}")
    print(f"Saved {OUTPUT_DIR / 'gpu_activity_summary.json'}")


def run_steady_state(model, tokenizer) -> None:
    print("=== Steady-state inference ===")
    for _ in range(STEADY_WARMUP_REQUESTS):
        generate_once(model, tokenizer, PROMPT, max_new_tokens=STEADY_NEW_TOKENS)

    rows = []
    for i in range(1, STEADY_REQUESTS + 1):
        result = generate_once(model, tokenizer, PROMPT, max_new_tokens=STEADY_NEW_TOKENS)
        row = {
            "request": i,
            "latency_s": result["latency_s"],
            "output_tokens": result["output_tokens"],
            "output_tok_s": result["output_tok_s"],
        }
        rows.append(row)
        print(
            f"request {i}: latency={row['latency_s']:.3f}s "
            f"tok/s={row['output_tok_s']:.3f}"
        )

    save_csv(
        OUTPUT_DIR / "steady_state.csv",
        rows,
        ["request", "latency_s", "output_tokens", "output_tok_s"],
    )
    latencies = [r["latency_s"] for r in rows]
    throughputs = [r["output_tok_s"] for r in rows]
    print(
        f"mean latency={statistics.fmean(latencies):.3f}s "
        f"mean tok/s={statistics.fmean(throughputs):.3f}"
    )
    print(f"Saved {OUTPUT_DIR / 'steady_state.csv'}")


def run_batch_scaling(model, tokenizer) -> None:
    print("=== Explicit Transformers batch-size scaling ===")
    rows = []
    for batch_size in BATCH_SIZES:
        generate_batch(
            model, tokenizer, PROMPT, batch_size=batch_size, max_new_tokens=BATCH_NEW_TOKENS
        )
        timed = [
            generate_batch(
                model,
                tokenizer,
                PROMPT,
                batch_size=batch_size,
                max_new_tokens=BATCH_NEW_TOKENS,
            )
            for _ in range(BATCH_REPEATS)
        ]
        row = {
            "batch_size": batch_size,
            "mean_batch_latency_s": round(
                statistics.fmean(t["batch_latency_s"] for t in timed), 3
            ),
            "aggregate_output_tok_s": round(
                statistics.fmean(t["aggregate_output_tok_s"] for t in timed), 3
            ),
            "approx_per_sequence_tok_s": round(
                statistics.fmean(t["approx_per_sequence_tok_s"] for t in timed), 3
            ),
            "peak_allocated_gib": round(
                statistics.fmean(t["peak_allocated_gib"] for t in timed), 3
            ),
        }
        rows.append(row)
        print(
            f"batch={batch_size}: latency={row['mean_batch_latency_s']}s "
            f"agg_tok/s={row['aggregate_output_tok_s']}"
        )

    save_csv(
        OUTPUT_DIR / "batch_scaling.csv",
        rows,
        [
            "batch_size",
            "mean_batch_latency_s",
            "aggregate_output_tok_s",
            "approx_per_sequence_tok_s",
            "peak_allocated_gib",
        ],
    )
    print(f"Saved {OUTPUT_DIR / 'batch_scaling.csv'}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for Phase 7.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_idle_baseline()

    print("=== Loading model ===")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"Model load time: {time.perf_counter() - load_start:.2f}s")
    print(f"Allocated GiB: {torch.cuda.memory_allocated() / (1024**3):.3f}")

    run_monitored_inference(model, tokenizer)
    run_steady_state(model, tokenizer)
    run_batch_scaling(model, tokenizer)

    # Keep a DataFrame print for Colab-style inspection of the latest CSV outputs.
    print("\n=== Result previews ===")
    for name in ("steady_state.csv", "batch_scaling.csv", "gpu_samples.csv"):
        path = OUTPUT_DIR / name
        if path.exists():
            print(f"\n{path}")
            print(pd.read_csv(path))


if __name__ == "__main__":
    main()
