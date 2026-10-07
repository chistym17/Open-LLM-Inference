# Phase 5 — Context Length and KV Cache

## Objective

Measure how prompt length and concurrency affect KV-cache consumption and inference latency.

This phase connects the theoretical KV-cache memory calculation with the behavior observed in vLLM.

## Model

`Qwen/Qwen2.5-1.5B-Instruct`

Configuration:

* NVIDIA T4
* FP16
* maximum model length: 4096
* vLLM GPU memory utilization: 0.8
* output length: 100 tokens

## Theoretical KV Cache

The model uses:

* 28 transformer layers
* 2 KV heads
* 128 head dimension
* K and V tensors
* FP16 = 2 bytes/value

Therefore:

```text
28 × 2 × 2 × 128 × 2
= 28,672 bytes/token
≈ 28 KB/token
```

Approximate KV-cache requirement per sequence:

| Sequence Length |  KV Cache |
| --------------: | --------: |
|           1,024 |  ~28.0 MB |
|           2,048 |  ~56.0 MB |
|           4,096 | ~112.0 MB |
|           8,192 | ~224.0 MB |

This is the theoretical KV state only; actual GPU memory usage also includes model weights, activations, runtime buffers, CUDA workspace, and other overhead.

## Experimental Results

### Concurrency 8

| Prompt Length | Peak KV Utilization | Average Latency |
| ------------: | ------------------: | --------------: |
|          ~500 |              0.516% |          2.211s |
|        ~1,000 |              0.645% |          2.508s |
|        ~2,000 |              0.979% |          3.431s |

Longer prompts increased both KV-cache utilization and latency.

### Concurrency 32

| Prompt Length | Peak KV Utilization | Average Latency |
| ------------: | ------------------: | --------------: |
|          ~500 |              1.506% |          3.787s |
|        ~1,000 |              1.549% |          4.897s |
|        ~2,000 |              1.883% |          6.914s |

At higher concurrency, the total KV-cache requirement increases because multiple sequences maintain their own KV state simultaneously.

## Observations

### 1. Context length increases KV-cache consumption

Longer sequences require more K/V states to be retained for attention during decoding.

The experiments showed increasing vLLM KV-cache utilization as prompt length increased.

### 2. Concurrency multiplies memory requirements

Each active request requires its own KV state.

Therefore, the practical KV-cache requirement depends on both:

```text
sequence length × number of active sequences
```

This is one of the fundamental constraints on inference-server concurrency.

### 3. Longer context increases latency

At concurrency 8, increasing the prompt from ~500 to ~2,000 tokens increased average latency from approximately:

```text
2.21s → 3.43s
```

At concurrency 32:

```text
3.79s → 6.91s
```

The exact relationship is workload- and engine-dependent, but longer contexts increase the amount of work required during prefill and increase the KV state maintained during decoding.

### 4. KV-cache utilization is not raw GPU memory usage

The vLLM KV-cache percentage represents utilization of the memory pool allocated for KV cache.

It should not be interpreted as:

```text
percentage of total GPU VRAM used
```

Model weights, activations, CUDA runtime allocations, workspaces, and other memory are separate.

### 5. Theoretical capacity is not guaranteed user capacity

Theoretical KV-cache calculations provide a useful upper-bound estimate, but actual serving capacity depends on:

* available KV-cache memory
* maximum sequence length
* request concurrency
* block allocation
* scheduler behavior
* model/runtime overhead
* workload characteristics

## Engineering Takeaway

KV cache is one of the key resources governing LLM serving capacity.

For a decoder-only model, every active sequence accumulates K/V state as tokens are processed. Longer contexts therefore consume more memory, while higher concurrency multiplies the amount of state that must be retained.

This creates a direct relationship between:

```text
Context Length
      ↓
KV Cache per Request
      ↓
Total KV Cache Consumption
      ↓
Maximum Practical Concurrency
```

This phase provides the memory/resource foundation for the next phase: **scheduling under resource constraints**.

