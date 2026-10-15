"""Phase 8 — Attention and inference optimization.

Measure how prompt length affects prefill latency and peak GPU memory
for Qwen2.5-1.5B-Instruct with FP16 + PyTorch SDPA.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = Path("results/phase-8")
BASE_TEXT = (
    "LLM inference processes prompts and generates tokens using neural networks. "
)
LENGTHS = [32, 64, 128, 256, 512, 1024]
TIMED_PASSES = 3


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def run_prompt_length_experiment(model, tokenizer) -> list[dict]:
    model.set_attn_implementation("sdpa")
    model.eval()

    # Create one token sequence, then reuse its tokens to control input length.
    base_ids = tokenizer(
        BASE_TEXT,
        return_tensors="pt",
    ).input_ids[0]

    results = []

    for length in LENGTHS:
        token_ids = base_ids.repeat((length + len(base_ids) - 1) // len(base_ids))
        input_ids = token_ids[:length].unsqueeze(0).to("cuda")
        attention_mask = torch.ones_like(input_ids)

        # Warmup
        with torch.inference_mode():
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        baseline_mb = torch.cuda.memory_allocated() / 1024**2
        timings = []
        finite = True

        for _ in range(TIMED_PASSES):
            start = time.perf_counter()
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
            finite = finite and torch.isfinite(output.logits).all().item()
            del output

        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        avg_ms = sum(timings) / len(timings)

        results.append(
            {
                "prompt_tokens": length,
                "mean_prefill_ms": round(avg_ms, 2),
                "peak_memory_mb": round(peak_mb, 1),
                "extra_peak_mb": round(peak_mb - baseline_mb, 1),
                "finite_logits": finite,
            }
        )

        del input_ids, attention_mask
        print(results[-1])

    return results


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for Phase 8.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading model ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
    )

    print("=== Prompt-length experiment ===")
    results = run_prompt_length_experiment(model, tokenizer)

    print("\nAll results:")
    for row in results:
        print(row)

    fieldnames = [
        "prompt_tokens",
        "mean_prefill_ms",
        "peak_memory_mb",
        "extra_peak_mb",
        "finite_logits",
    ]
    save_csv(OUTPUT_DIR / "prompt_length.csv", results, fieldnames)
    save_json(
        OUTPUT_DIR / "prompt_length.json",
        {
            "model": MODEL_ID,
            "dtype": "FP16",
            "attention_implementation": "sdpa",
            "timed_passes": TIMED_PASSES,
            "warmup_passes": 1,
            "results": results,
        },
    )

    print(f"\nSaved {OUTPUT_DIR / 'prompt_length.csv'}")
    print(f"Saved {OUTPUT_DIR / 'prompt_length.json'}")


if __name__ == "__main__":
    main()
