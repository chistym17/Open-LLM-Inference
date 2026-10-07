import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TransformersEngine:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    ):
        self.model_id = model_id

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cuda",
        )

        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[int, int]:

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to("cuda")

        input_tokens = inputs["input_ids"].shape[1]

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }

        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        output_tokens = output.shape[1] - input_tokens

        return input_tokens, output_tokens