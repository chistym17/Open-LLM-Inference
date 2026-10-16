"""Phase 9 — FP16 vs 4-bit NF4 quantization.

Extracted from docs/phase9.ipynb.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Explain how an LLM generates text, step by step."
MAX_NEW_TOKENS = 64
OUTPUT_DIR = Path("results/phase-9")


def benchmark_model(model, tokenizer, inputs, label, load_seconds):
    model.eval()

    with torch.inference_mode():
        _ = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    times = []
    output = None

    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )

        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    generated_tokens = output.shape[1] - inputs["input_ids"].shape[1]

    return {
        "model": label,
        "load_seconds": round(load_seconds, 2),
        "model_footprint_gib": round(model.get_memory_footprint() / 1024**3, 3),
        "peak_gpu_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "mean_generation_seconds": round(sum(times) / len(times), 3),
        "generated_tokens": int(generated_tokens),
        "output_text": tokenizer.decode(output[0], skip_special_tokens=True),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for Phase 9.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("BitsAndBytes:", bnb.__version__)
    print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    print("Prompt tokens:", inputs["input_ids"].shape[1])

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        attn_implementation="sdpa",
    ).to("cuda")
    torch.cuda.synchronize()
    fp16_load_seconds = time.perf_counter() - start

    fp16_result = benchmark_model(
        model_fp16, tokenizer, inputs, "FP16", fp16_load_seconds
    )
    print(fp16_result)

    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    start = time.perf_counter()
    model_4bit = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    torch.cuda.synchronize()
    int4_load_seconds = time.perf_counter() - start

    int4_result = benchmark_model(
        model_4bit, tokenizer, inputs, "4-bit NF4", int4_load_seconds
    )
    print(int4_result)

    footprint_reduction = (
        1 - int4_result["model_footprint_gib"] / fp16_result["model_footprint_gib"]
    ) * 100
    peak_reduction = (
        1 - int4_result["peak_gpu_allocated_gib"] / fp16_result["peak_gpu_allocated_gib"]
    ) * 100
    slowdown = (
        int4_result["mean_generation_seconds"] / fp16_result["mean_generation_seconds"]
    )

    print("\n=== FP16 vs 4-bit NF4 ===")
    print("Model footprint reduction:", round(footprint_reduction, 1), "%")
    print("Peak allocation reduction:", round(peak_reduction, 1), "%")
    print("Latency slowdown factor:", round(slowdown, 2), "x")

    csv_rows = [
        {
            "config": "FP16",
            "model_footprint_gib": fp16_result["model_footprint_gib"],
            "peak_allocated_gib": fp16_result["peak_gpu_allocated_gib"],
            "mean_generation_latency_s": fp16_result["mean_generation_seconds"],
        },
        {
            "config": "4bit_NF4",
            "model_footprint_gib": int4_result["model_footprint_gib"],
            "peak_allocated_gib": int4_result["peak_gpu_allocated_gib"],
            "mean_generation_latency_s": int4_result["mean_generation_seconds"],
        },
    ]
    import csv

    with (OUTPUT_DIR / "quantization.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    payload = {
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "fp16": {
            "attention": "sdpa",
            "model_footprint_gib": fp16_result["model_footprint_gib"],
            "peak_allocated_gib": fp16_result["peak_gpu_allocated_gib"],
            "mean_generation_latency_s": fp16_result["mean_generation_seconds"],
            "load_seconds": fp16_result["load_seconds"],
        },
        "nf4": {
            "quantization": "bitsandbytes NF4",
            "double_quant": True,
            "compute_dtype": "FP16",
            "model_footprint_gib": int4_result["model_footprint_gib"],
            "peak_allocated_gib": int4_result["peak_gpu_allocated_gib"],
            "mean_generation_latency_s": int4_result["mean_generation_seconds"],
            "load_seconds": int4_result["load_seconds"],
        },
        "decoding": {
            "strategy": "greedy",
            "max_new_tokens": MAX_NEW_TOKENS,
            "timed_runs": 3,
            "warmup": True,
        },
        "deltas": {
            "model_footprint_reduction_pct": round(footprint_reduction, 1),
            "peak_allocated_reduction_pct": round(peak_reduction, 1),
            "latency_slowdown_factor": round(slowdown, 2),
        },
    }
    (OUTPUT_DIR / "quantization.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {OUTPUT_DIR / 'quantization.csv'}")
    print(f"Saved {OUTPUT_DIR / 'quantization.json'}")


if __name__ == "__main__":
    main()
