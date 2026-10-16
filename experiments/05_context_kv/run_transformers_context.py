"""Phase 5 exploratory — Transformers context-length latency/memory.

Extracted from docs/01_raw_transformers.ipynb (cell 18).
Writes results/phase-5/transformers_context_length.csv.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = Path("results/phase-5")
TARGETS = [500, 1000, 2000]
MAX_NEW_TOKENS = 100

BASE_TEXT = """
Explain how large language models work, including tokenization,
embeddings, transformer layers, attention, KV cache, logits,
sampling, and autoregressive generation.
"""


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

    rows = []
    for target_tokens in TARGETS:
        text = BASE_TEXT * ((target_tokens // 45) + 1)
        test_inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=target_tokens,
        ).to("cuda")
        actual_input_tokens = test_inputs["input_ids"].shape[1]

        with torch.inference_mode():
            _ = model.generate(**test_inputs, max_new_tokens=10, do_sample=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **test_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        output_tokens = output.shape[1] - actual_input_tokens
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3

        row = {
            "input_tokens": int(actual_input_tokens),
            "latency_s": round(elapsed, 3),
            "output_tok_s": round(output_tokens / elapsed, 2),
            "peak_gpu_gb": round(peak_memory, 3),
            "max_new_tokens": MAX_NEW_TOKENS,
            "engine": "transformers_fp16",
        }
        rows.append(row)
        print(
            f"Input: {row['input_tokens']:4d} | "
            f"Latency: {row['latency_s']:.3f}s | "
            f"Output tok/s: {row['output_tok_s']:.2f} | "
            f"Peak GPU: {row['peak_gpu_gb']:.3f} GB"
        )

    path = OUTPUT_DIR / "transformers_context_length.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
