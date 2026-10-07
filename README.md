# Open LLM Inference Platform

A hands-on inference engineering project focused on understanding and benchmarking modern open-weight LLM serving systems.

The project starts from raw Transformers inference and progressively investigates:

- Prefill and decode
- KV-cache behavior
- Batching and concurrency
- Scheduling
- Context-length scaling
- GPU performance
- Attention optimization
- Quantization
- Production inference engines

The goal is not to rebuild a production inference engine. Instead, the project follows:

```text
Understand mechanism
       ↓
Build minimal implementation where useful
       ↓
Inspect production implementation
       ↓
Benchmark real systems
       ↓
Document the results