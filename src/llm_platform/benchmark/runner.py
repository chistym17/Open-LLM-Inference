import time
from dataclasses import dataclass
from typing import Callable

import torch

from .metrics import (
    calculate_output_throughput,
    summarize_latencies,
)
from .workloads import Workload


@dataclass
class GenerationResult:
    input_tokens: int
    output_tokens: int
    latency_s: float
    peak_gpu_memory_gb: float


@dataclass
class BenchmarkResult:
    workload: str
    input_tokens: int
    output_tokens: int
    latency_mean_s: float
    latency_p50_s: float
    latency_p95_s: float
    output_tok_s: float
    peak_gpu_memory_gb: float


def run_benchmark(
    generate: Callable[[str, int, float], tuple[int, int]],
    workload: Workload,
    warmup_runs: int = 1,
    benchmark_runs: int = 5,
) -> BenchmarkResult:

    # Warmup
    for _ in range(warmup_runs):
        generate(
            workload.prompt,
            workload.max_new_tokens,
            workload.temperature,
        )

    latencies = []
    output_tokens = []
    input_token_counts = []
    peak_memory = []

    for _ in range(benchmark_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()

        input_count, output_count = generate(
            workload.prompt,
            workload.max_new_tokens,
            workload.temperature,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latency = time.perf_counter() - start

        latencies.append(latency)
        input_token_counts.append(input_count)
        output_tokens.append(output_count)

        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        else:
            memory_gb = 0.0

        peak_memory.append(memory_gb)

    latency_metrics = summarize_latencies(latencies)

    total_output_tokens = sum(output_tokens)
    total_latency = sum(latencies)

    output_tok_s = calculate_output_throughput(
        total_output_tokens,
        total_latency,
    )

    return BenchmarkResult(
        workload=workload.name,
        input_tokens=input_token_counts[0],
        output_tokens=round(sum(output_tokens) / len(output_tokens)),
        latency_mean_s=latency_metrics["latency_mean_s"],
        latency_p50_s=latency_metrics["latency_p50_s"],
        latency_p95_s=latency_metrics["latency_p95_s"],
        output_tok_s=output_tok_s,
        peak_gpu_memory_gb=max(peak_memory),
    )
