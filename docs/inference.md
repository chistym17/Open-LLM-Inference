Good. Next: `docs/inference.md`.

This is the **core technical explanation** for Phase 1—not a tutorial, just the concepts needed to understand our measurements.

```bash
nano docs/inference.md
```

Paste:

````markdown
# LLM Inference

## Autoregressive Generation

A causal language model generates text one token at a time.

For an input sequence:

```text
x1, x2, ..., xn
````

the model produces logits over the vocabulary. The next token is selected from those logits and appended to the sequence.

Generation then repeats:

```text
tokens
  ↓
Transformer
  ↓
logits
  ↓
token selection
  ↓
next token
  ↓
repeat
```

For `Qwen/Qwen2.5-1.5B-Instruct`, the vocabulary contains 151,936 tokens.

---

## Prefill

Prefill processes the entire input prompt before the first generated token.

For example:

```text
Input: 500 tokens
        ↓
Transformer
        ↓
KV cache created
        ↓
first output token
```

Prefill is primarily responsible for the initial time-to-first-token (TTFT).

In the Phase 1 experiment:

```text
Input tokens: 11
Prefill time: 0.137 s
Logits shape: [1, 11, 151936]
```

The logits tensor contains one vocabulary-sized prediction for each input position.

---

## Decode

After prefill, generation enters the decode phase.

Each iteration generally processes the newly generated token while reusing the previously computed key/value states stored in the KV cache.

Conceptually:

```text
Prefill
  ↓
KV cache
  ↓
Decode token 1
  ↓
Decode token 2
  ↓
Decode token 3
  ↓
...
```

This avoids recomputing the full prompt representation for every generated token.

---

## KV Cache

Attention uses key and value tensors from previous tokens.

During autoregressive generation, these tensors can be cached and reused.

The KV cache therefore grows with sequence length.

For this model:

* Layers: 28
* Attention heads: 12
* KV heads: 2
* Head dimension: 128
* Precision: FP16

Because the model uses grouped-query attention (GQA), the 12 query heads share only 2 key/value heads.

The theoretical KV-cache storage per token is approximately:

```text
28 layers
× 2 (K + V)
× 2 KV heads
× 128 head dimension
× 2 bytes
= 28,672 bytes/token
```

This is approximately 28 KB per token.

The actual GPU memory footprint of an inference server also includes model weights, temporary tensors, CUDA/runtime allocations, and other overhead.

---

## Important Inference Metrics

### Time To First Token (TTFT)

Time from receiving a request until the first generated token is available.

TTFT is strongly affected by:

* Input/prompt length
* Prefill computation
* Scheduling delay
* Queueing
* System load

### End-to-End Latency

Total time required to complete a request.

```text
latency = time from request start to final token
```

### Output Throughput

Generated tokens per second:

```text
output_tok/s = output_tokens / generation_time
```

This is useful for comparing inference engines and configurations.

### Request Throughput

Completed requests per second:

```text
requests/s = completed_requests / elapsed_time
```

This becomes particularly important under concurrent workloads.

---

## Phase 1 Baseline

Raw Transformers inference produced:

```text
Input tokens:       11
Output tokens:      100
Prefill time:       0.137 s
Latency:            5.469 s
Output throughput:  18.29 tok/s
Peak GPU memory:    2.891 GB
```

Increasing the input context also increased peak GPU memory:

```text
397 tokens   → 2.929 GB
760 tokens   → 2.996 GB
1486 tokens  → 3.214 GB
```

This provides the baseline for later experiments involving:

* vLLM
* concurrency
* continuous batching
* context scaling
* KV-cache management
* scheduling
* quantization

````
