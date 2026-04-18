"""MedGemma loader and streaming SOAP generator (MLX backend, no Streamlit knowledge)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from clinical_ai.prompts import format_soap_messages

DEFAULT_MODEL_ID = "mlx-community/medgemma-27b-text-it-4bit"


def load_medgemma(model_id: str = DEFAULT_MODEL_ID) -> tuple[Any, Any]:
    return load(model_id)


def stream_soap(
    model: Any,
    tokenizer: Any,
    transcript: str,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> Iterator[str]:
    messages = format_soap_messages(transcript)
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    sampler = make_sampler(temp=temperature)
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        yield response.text
