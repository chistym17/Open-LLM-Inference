from dataclasses import dataclass


@dataclass
class Workload:
    name: str
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.0


DEFAULT_WORKLOAD = Workload(
    name="short",
    prompt="Explain how an LLM generates the next token.",
    max_new_tokens=100,
    temperature=0.0,
)


CONTEXT_WORKLOADS = [
    Workload(
        name="context_500",
        prompt="Explain how an LLM generates the next token. " * 40,
        max_new_tokens=100,
        temperature=0.0,
    ),
    Workload(
        name="context_1000",
        prompt="Explain how an LLM generates the next token. " * 80,
        max_new_tokens=100,
        temperature=0.0,
    ),
    Workload(
        name="context_2000",
        prompt="Explain how an LLM generates the next token. " * 160,
        max_new_tokens=100,
        temperature=0.0,
    ),
]
