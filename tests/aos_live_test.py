#!/usr/bin/env python3
"""
Live smoke test for the LLM endpoint using AOS adapter.

Requires environment variables:
  - AOS_ENDPOINT (default: https://api.sunxie.xyz)
  - AOS_API_KEY  (required)
  - AOS_MODEL    (default: gpt-4o-mini)

Usage:
  python3 tests/aos_live_test.py
"""

import os
import json
import sys

sys.path.append(os.path.abspath('.'))

from aos.adapter import AOSAdapter
from aos.prompts import build_state_prompt_v2, build_decision_prompt_v2
from aos.schema import validate_decision


def main():
    endpoint = os.environ.get("AOS_ENDPOINT", "https://api.sunxie.xyz")
    api_key = os.environ.get("AOS_API_KEY")
    model = os.environ.get("AOS_MODEL", "gemini-2.5-flash-lite-preview-06-17")

    if not api_key:
        print("[SKIP] AOS_API_KEY not set. Export AOS_API_KEY to run live test.")
        return

    adapter = AOSAdapter(endpoint=endpoint, api_key=api_key, model=model, timeout=30.0, max_retries=2, include_images=False)

    # State summary test (v2 prompt)
    state_payload = {
        "current_generation": 20,
        "total_generations": 50,
        "adjust_interval": 10,
        "algorithm_configuration": {
            "selection": "tournament3",
            "crossover": "two_point",
            "mutation": "flip_bit",
            "cxpb": 0.6,
            "mutpb": 0.2,
        },
        "fitness_function": "CV_mean - alpha * (k/d)",
        "algorithm_state_pic": "ga_results/overview.png",
    }
    print("[INFO] Building summarize_state prompt (v2 template)...")
    try:
        state_msgs = build_state_prompt_v2(state_payload)
        print("\n[REQUEST] summarize_state messages:")
        for i, m in enumerate(state_msgs):
            role = m.get("role")
            content = m.get("content", "")
            print(f"--- message {i} ({role}) ---\n{content}\n")
        print("[INFO] Sending summarize_state request...")
        data = adapter.client.chat(messages=state_msgs, temperature=0.2, response_format="json_object")
        raw_content = data["choices"][0]["message"].get("content", "")
        print("\n[RESPONSE] summarize_state raw content:\n", raw_content)
    except Exception as e:
        print("[ERROR] summarize_state failed:", e)

    # Decision test (v2 prompt)
    decision_payload = {
        "dataset_size": "n=569 (breast_cancer)",
        "num_features": 30,
        "classification_model": "logistic",
        "current_iteration": 20,
        "total_iterations": 50,
        "adjust_interval": 10,
        "algorithm_configuration": {
            "selection": "tournament3",
            "crossover": "two_point",
            "mutation": "flip_bit",
            "cxpb": 0.6,
            "mutpb": 0.2,
        },
        "state_evaluation_text": "IR flat, diversity low, SR decreasing.",
        "fitness_function": "CV_mean - alpha * (k/d)",
        "selection": ["tournament", "sus", "best"],
        "crossover": ["one_point", "two_point", "uniform"],
        "mutation": ["bit_flip", "uniform_int"],
    }
    print("\n[INFO] Building choose_operators prompt (v2 template)...")
    try:
        decision_msgs = build_decision_prompt_v2(decision_payload)
        print("\n[REQUEST] choose_operators messages:")
        for i, m in enumerate(decision_msgs):
            role = m.get("role")
            content = m.get("content", "")
            print(f"--- message {i} ({role}) ---\n{content}\n")
        print("[INFO] Sending choose_operators request...")
        data2 = adapter.client.chat(messages=decision_msgs, temperature=0.1, response_format="json_object")
        raw_decision = data2["choices"][0]["message"].get("content", "{}")
        print("\n[RESPONSE] choose_operators raw content:\n", raw_decision)
        # Normalize
        try:
            parsed = json.loads(raw_decision)
        except Exception:
            start = raw_decision.find("{")
            end = raw_decision.rfind("}")
            parsed = json.loads(raw_decision[start:end+1]) if start >= 0 and end >= 0 else {}
        norm, warn = validate_decision(parsed)
        print("\n[POST] Normalized decision:\n", json.dumps(norm, indent=2))
        if warn:
            print("[POST] Warnings:", warn)
    except Exception as e:
        print("[ERROR] choose_operators failed:", e)


if __name__ == "__main__":
    main()
