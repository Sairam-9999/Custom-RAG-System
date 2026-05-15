from __future__ import annotations

from typing import Optional

import requests

from ..core.config import GENERATOR_CONFIG
from .extractive_fallback import extractive_fallback


def generate_answer(
    prompt: str,
    num_predict: Optional[int] = None,
    timeout: Optional[int] = None,
    context: Optional[str] = None,
    question: Optional[str] = None,
    query_analysis=None,
    model: Optional[str] = None,
    url: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
):
    """Generate an answer via Ollama using configurable model/endpoint/options."""
    request_url = url or GENERATOR_CONFIG.ollama_url
    model_name = model or GENERATOR_CONFIG.ollama_model
    max_tokens = num_predict if num_predict is not None else GENERATOR_CONFIG.num_predict
    request_timeout = timeout if timeout is not None else GENERATOR_CONFIG.timeout

    try:
        response = requests.post(
            request_url,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": GENERATOR_CONFIG.temperature if temperature is None else temperature,
                    "top_p": GENERATOR_CONFIG.top_p if top_p is None else top_p,
                    "num_predict": max_tokens,
                },
            },
            timeout=request_timeout,
        )

        if response.status_code != 200:
            if context and question:
                result = extractive_fallback(question, context, query_analysis)
                return result.answer
            return f"Ollama error {response.status_code}: {response.text}"

        return response.json().get("response", "").strip()

    except requests.exceptions.RequestException as e:
        if context and question:
            result = extractive_fallback(question, context, query_analysis)
            return result.answer
        return f"Ollama connection error: {e}"
