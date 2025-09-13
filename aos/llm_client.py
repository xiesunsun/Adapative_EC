import os
import time
import json
from typing import Any, Dict, List, Optional

import requests


class LLMClient:
    """
    Minimal OpenAI-compatible chat client with retry/backoff and timeout.
    Compatible with OpenAI Chat Completions API.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str],
        model: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_base: float = 1.5,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        response_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = self.endpoint + "/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                if resp.status_code >= 200 and resp.status_code < 300:
                    data = resp.json()
                    return data
                else:
                    last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            except Exception as e:
                last_err = e
            # backoff with jitter
            if attempt < self.max_retries:
                sleep_s = (self.backoff_base ** attempt) + (0.05 * attempt)
                time.sleep(sleep_s)
        raise last_err if last_err else RuntimeError("LLM request failed")

