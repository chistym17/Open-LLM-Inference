import os

from llm_platform.benchmark.results import save_csv, save_json
from llm_platform.benchmark.runner import run_benchmark
from llm_platform.benchmark.workloads import DEFAULT_WORKLOAD
from llm_platform.inference.vllm import VLLMEngine


def main():
    engine = VLLMEngine()

    print("Checking vLLM server...")

    before = engine.metrics()
    print(f"KV cache usage before: {before['kv_cache_usage']}%")

    result = run_benchmark(
        generate=engine.generate,
        workload=DEFAULT_WORKLOAD,
        warmup_runs=2,
        benchmark_runs=5,
    )

    after = engine.metrics()

    print("\n=== vLLM Baseline ===")
    print(f"Input tokens:       {result.input_tokens}")
    print(f"Output tokens:      {result.output_tokens}")
    print(f"Mean latency:       {result.latency_mean_s:.3f}s")
    print(f"P50 latency:        {result.latency_p50_s:.3f}s")
    print(f"P95 latency:        {result.latency_p95_s:.3f}s")
    print(f"Output throughput:  {result.output_tok_s:.2f} tok/s")
    print(f"KV cache before:    {before['kv_cache_usage']}%")
    print(f"KV cache after:     {after['kv_cache_usage']}%")

    os.makedirs("results/phase-3", exist_ok=True)

    save_csv(
        [result],
        "results/phase-3/vllm-baseline.csv",
    )

    save_json(
        [result],
        "results/phase-3/vllm-baseline.json",
    )


if __name__ == "__main__":
    main()