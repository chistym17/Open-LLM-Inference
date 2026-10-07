import re

import requests


class VLLMEngine:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    ):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[int, int]:

        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self.model_id,
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
            },
            timeout=300,
        )

        response.raise_for_status()

        data = response.json()
        usage = data["usage"]

        return (
            usage["prompt_tokens"],
            usage["completion_tokens"],
        )

    def metrics(self) -> dict:
        response = requests.get(
            f"{self.base_url}/metrics",
            timeout=10,
        )

        response.raise_for_status()

        text = response.text

        def get_metric(name: str):
            match = re.search(
                rf"^vllm:{re.escape(name)}"
                r"(?:\{.*?\})?\s+([0-9.eE+-]+)$",
                text,
                re.MULTILINE,
            )

            return float(match.group(1)) if match else None

        return {
            "kv_cache_usage": get_metric("kv_cache_usage_perc"),
            "running_requests": get_metric("num_requests_running"),
            "waiting_requests": get_metric("num_requests_waiting"),
        }