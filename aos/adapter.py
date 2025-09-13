from typing import Any, Dict, List, Optional
import json
import os

from .llm_client import LLMClient
from .prompts import (
    build_state_prompt,
    build_decision_prompt,
    build_state_prompt_v2,
    build_decision_prompt_v2,
)
from .schema import validate_decision, ALLOWED_SELECTION, ALLOWED_CROSSOVER, ALLOWED_MUTATION


class AOSAdapter:
    """
    LLM-driven Adaptive Operator Selection adapter.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str],
        model: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        include_images: bool = False,
    ) -> None:
        self.client = LLMClient(endpoint=endpoint, api_key=api_key, model=model, timeout=timeout, max_retries=max_retries)
        self.include_images = include_images

    def summarize_state(self, meta: Dict[str, Any]) -> str:
        msgs = build_state_prompt_v2(meta)
        # Optionally attach overview image as base64
        if self.include_images:
            from .prompts import encode_image_b64
            pic = meta.get("algorithm_state_pic")
            if pic:
                b64 = encode_image_b64(pic)
                if b64:
                    msgs.append({"role": "user", "content": "[Overview Image as base64]"})
                    msgs.append({"role": "user", "content": f"name=overview.png: data:image/png;base64,{b64[:120000]}"})
        data = self.client.chat(messages=msgs, temperature=0.2, response_format="json_object")
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return ""

    def choose_operators(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        msgs = build_decision_prompt_v2(meta)
        data = self.client.chat(messages=msgs, temperature=0.1, response_format="json_object")
        content = data["choices"][0]["message"].get("content", "{}")
        try:
            raw = json.loads(content)
        except Exception:
            # Try to extract JSON substring
            try:
                start = content.find("{")
                end = content.rfind("}")
                raw = json.loads(content[start : end + 1]) if start >= 0 and end >= 0 else {}
            except Exception:
                raw = {}
        norm, warn = validate_decision(raw)
        if warn:
            norm["rationale"] = (norm.get("rationale", "") + f" warnings: {warn}").strip()
        return norm
