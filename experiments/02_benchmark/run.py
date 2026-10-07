import os

from llm_platform.benchmark.results import save_csv, save_json
from llm_platform.benchmark.runner import run_benchmark
from llm_platform.benchmark.workloads import (
    DEFAULT_WORKLOAD,
    CONTEXT_WORKLOADS,
)
from llm_platform.inference.transformers import TransformersEngine


def main():
    engine = TransformersEngine()

    workloads = [
        DEFAULT_WORKLOAD,
        *CONTEXT_WORKLOADS,
    ]

    results = []

    for workload in workloads:
        print(f"\n=== {workload.name} ===")

        result = run_benchmark(
            generate=engine.generate,
            workload=workload,
            warmup_runs=1,
            benchmark_runs=5,
        )

        results.append(result)

        print(f"Input tokens:       {result.input_tokens}")
        print(f"Output tokens:      {result.output_tokens}")
        print(f"Mean latency:       {result.latency_mean_s:.3f}s")
        print(f"P50 latency:        {result.latency_p50_s:.3f}s")
        print(f"P95 latency:        {result.latency_p95_s:.3f}s")
        print(f"Output throughput:  {result.output_tok_s:.2f} tok/s")
        print(f"Peak GPU memory:    {result.peak_gpu_memory_gb:.3f} GB")

    os.makedirs("results/phase-2", exist_ok=True)

    save_csv(
        results,
        "results/phase-2/benchmark.csv",
    )

    save_json(
        results,
        "results/phase-2/benchmark.json",
    )

    print("\nResults saved to:")
    print("results/phase-2/benchmark.csv")
    print("results/phase-2/benchmark.json")


if __name__ == "__main__":
    main()