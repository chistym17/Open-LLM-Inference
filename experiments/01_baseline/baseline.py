import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 100


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
    )

    model.eval()
    return tokenizer, model


def generate_and_measure(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    input_tokens = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    torch.cuda.synchronize()

    latency = time.perf_counter() - start

    output_tokens = output.shape[1] - input_tokens
    output_tok_per_sec = output_tokens / latency

    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_s": latency,
        "output_tok_s": output_tok_per_sec,
        "peak_gpu_memory_gb": peak_memory_gb,
    }


def measure_prefill(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)

    torch.cuda.synchronize()

    prefill_time = time.perf_counter() - start

    return {
        "prefill_time_s": prefill_time,
        "logits_shape": tuple(outputs.logits.shape),
        "kv_cache_layers": len(outputs.past_key_values),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this experiment.")

    print(f"Model: {MODEL_ID}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    tokenizer, model = load_model()

    prompt = "Explain how an LLM generates the next token."

    print("=== Baseline Generation ===")

    baseline = generate_and_measure(
        model,
        tokenizer,
        prompt,
    )

    for key, value in baseline.items():
        print(f"{key}: {value}")

    print()
    print("=== Prefill ===")

    prefill = measure_prefill(
        model,
        tokenizer,
        prompt,
    )

    for key, value in prefill.items():
        print(f"{key}: {value}")

    print()
    print("=== Context Scaling ===")

    contexts = [
        "Explain how an LLM generates the next token. " * 40,
        "Explain how an LLM generates the next token. " * 80,
        "Explain how an LLM generates the next token. " * 160,
    ]

    for prompt in contexts:
        result = generate_and_measure(
            model,
            tokenizer,
            prompt,
        )

        print(
            f"Input: {result['input_tokens']:4d} | "
            f"Latency: {result['latency_s']:.3f}s | "
            f"Output tok/s: {result['output_tok_s']:.2f} | "
            f"Peak GPU: {result['peak_gpu_memory_gb']:.3f} GB"
        )


if __name__ == "__main__":
    main()