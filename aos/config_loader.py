import json
import os
from typing import Any, Dict, List


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_configs(config_dir: str = "config") -> Dict[str, Any]:
    """
    Load static JSON configs for AOS prompts from a directory.
    Expected files:
      - task_info.json
      - operator_pools.json
      - algo_config.json
    Returns a dict with keys: task_info, operator_pools, algo_config.
    """
    task_path = os.path.join(config_dir, "task_info.json")
    ops_path = os.path.join(config_dir, "operator_pools.json")
    algo_path = os.path.join(config_dir, "algo_config.json")
    out: Dict[str, Any] = {}
    if os.path.exists(task_path):
        out["task_info"] = _load_json(task_path)
    if os.path.exists(ops_path):
        out["operator_pools"] = _load_json(ops_path)
    if os.path.exists(algo_path):
        out["algo_config"] = _load_json(algo_path)
    return out


def _format_operator_pool_for_prompt(pool: Dict[str, Any]) -> str:
    lines: List[str] = []
    for item in pool:
        name = item.get("name")
        desc = item.get("description", "")
        aliases = item.get("aliases", [])
        params = item.get("params_schema", {})
        lines.append(f"- {name} (aliases: {', '.join(aliases)})\n  desc: {desc}")
        if params:
            lines.append("  params:")
            for pname, spec in params.items():
                t = spec.get("type", "?")
                dflt = spec.get("default")
                rng = []
                if "min" in spec: rng.append(f"min={spec['min']}")
                if "max" in spec: rng.append(f"max={spec['max']}")
                rngs = ("; "+" ".join(rng)) if rng else ""
                pdesc = spec.get("description", "")
                lines.append(f"    - {pname}: type={t}, default={dflt}{rngs}; {pdesc}")
    return "\n".join(lines)


def build_decision_payload_from_configs(configs: Dict[str, Any], state_text: str = "", current_iteration: int = 0) -> Dict[str, Any]:
    ti = configs.get("task_info", {})
    ops = configs.get("operator_pools", {})
    ac = configs.get("algo_config", {})
    sel_txt = _format_operator_pool_for_prompt(ops.get("selection", []))
    cx_txt = _format_operator_pool_for_prompt(ops.get("crossover", []))
    mut_txt = _format_operator_pool_for_prompt(ops.get("mutation", []))
    payload = {
        "dataset_size": ti.get("dataset_size"),
        "num_features": ti.get("num_features"),
        "classification_model": ti.get("classification_model"),
        "current_iteration": current_iteration,
        "total_iterations": ac.get("ga", {}).get("generations", ac.get("total_iterations")),
        "adjust_interval": ac.get("switching", {}).get("interval", ac.get("adjust_interval")),
        "algorithm_configuration": ac.get("operators", {}).get("current", {}),
        "state_evaluation_text": state_text,
        "fitness_function": ti.get("fitness_function"),
        "selection": sel_txt,
        "crossover": cx_txt,
        "mutation": mut_txt,
    }
    return payload


def build_state_payload_from_configs(configs: Dict[str, Any], current_generation: int = 0, overview_image: str = "ga_results/overview.png") -> Dict[str, Any]:
    ti = configs.get("task_info", {})
    ac = configs.get("algo_config", {})
    payload = {
        "current_generation": current_generation,
        "total_generations": ac.get("ga", {}).get("generations", ac.get("total_iterations")),
        "adjust_interval": ac.get("switching", {}).get("interval", ac.get("adjust_interval")),
        "algorithm_configuration": ac.get("operators", {}).get("current", {}),
        "fitness_function": ti.get("fitness_function"),
        "algorithm_state_pic": overview_image or ac.get("paths", {}).get("overview_image", "ga_results/overview.png"),
    }
    return payload
