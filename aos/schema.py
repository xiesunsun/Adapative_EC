from typing import Any, Dict, Tuple
import json
import os


# Lazy-loaded operator pools and alias mapping derived from config/operator_pools.json
_OP_POOLS_CACHE: Dict[str, Any] = {}
_ALLOWED_BY_KIND: Dict[str, set] = {}
_ALIAS_TO_CANON: Dict[str, Dict[str, str]] = {}


def _load_operator_pools(config_dir: str = None) -> None:
    global _OP_POOLS_CACHE, _ALLOWED_BY_KIND, _ALIAS_TO_CANON
    if _OP_POOLS_CACHE:
        return
    base = config_dir or os.environ.get("AOS_CONFIG_DIR", "config")
    path = os.path.join(base, "operator_pools.json")
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        # Fallback to a minimal built-in set if config missing
        data = {
            "selection": [{"name": "tournament", "aliases": [], "params_schema": {"tournsize": {"type": "int", "min": 2, "max": 7, "default": 3}}}, {"name": "sus", "aliases": [], "params_schema": {}}, {"name": "best", "aliases": [], "params_schema": {}}],
            "crossover": [{"name": "one_point", "aliases": [], "params_schema": {}}, {"name": "two_point", "aliases": [], "params_schema": {}}, {"name": "uniform", "aliases": [], "params_schema": {"indpb": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5}}}],
            "mutation": [{"name": "flip_bit", "aliases": ["bit_flip"], "params_schema": {"indpb": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0}}}, {"name": "uniform_int", "aliases": [], "params_schema": {"indpb": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0}, "low": {"type": "int", "default": 0}, "up": {"type": "int", "default": 1}}}],
        }
    _OP_POOLS_CACHE = data
    # Build allowed and alias maps
    kinds = ("selection", "crossover", "mutation")
    _ALLOWED_BY_KIND = {k: set() for k in kinds}
    _ALIAS_TO_CANON = {k: {} for k in kinds}
    for kind in kinds:
        for item in data.get(kind, []):
            canon = str(item.get("name", "")).strip().lower()
            if not canon:
                continue
            _ALLOWED_BY_KIND[kind].add(canon)
            aliases = item.get("aliases", []) or []
            for a in [canon] + list(aliases):
                if not a:
                    continue
                _ALIAS_TO_CANON[kind][str(a).strip().lower()] = canon


def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _canonicalize(kind: str, name: str) -> Tuple[str, bool]:
    _load_operator_pools()
    raw = (name or "").strip().lower()
    canon = _ALIAS_TO_CANON.get(kind, {}).get(raw)
    if canon:
        return canon, True
    # Not found; return a sensible default per kind
    defaults = {"selection": "tournament", "crossover": "two_point", "mutation": "flip_bit"}
    return defaults.get(kind, raw), False


def _clamp_params(kind: str, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp and sanitize params based on operator_pools.json schema. Unknown params are ignored."""
    _load_operator_pools()
    schema_list = _OP_POOLS_CACHE.get(kind, [])
    spec = next((it for it in schema_list if str(it.get("name", "")).strip().lower() == name), None)
    if not spec:
        return {}
    out: Dict[str, Any] = {}
    p_schema = spec.get("params_schema", {}) or {}
    for p, s in p_schema.items():
        t = (s.get("type") or "").lower()
        default = s.get("default")
        val = params.get(p, default)
        try:
            if t == "int":
                v = int(val)
                if "min" in s:
                    v = max(int(s["min"]), v)
                if "max" in s:
                    v = min(int(s["max"]), v)
                out[p] = v
            elif t == "float":
                v = float(val)
                if "min" in s:
                    v = max(float(s["min"]), v)
                if "max" in s:
                    v = min(float(s["max"]), v)
                out[p] = v
            else:
                # passthrough (e.g., string like "1/n_features")
                out[p] = val
        except Exception:
            # If conversion fails, use default when possible
            out[p] = default if default is not None else None
    # Special correction for uniform_int low/up ordering
    if name == "uniform_int" and {"low", "up"}.issubset(out.keys()):
        if out["up"] < out["low"]:
            out["up"] = out["low"]
    return out


def validate_decision(dec: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Validate and normalize v2 decision JSON with alias mapping resolved from operator_pools.json.
    Returns (normalized_decision, warning_string).
    """
    _load_operator_pools()
    warn = []
    # Required top-level keys
    if not all(k in dec for k in ("cxpb", "mutpb", "Selection", "Crossover", "Mutation")):
        return ({"error": "missing required keys"}, "missing keys")

    cxpb = _clamp01(dec.get("cxpb", 0.6))
    mutpb = _clamp01(dec.get("mutpb", 0.2))

    # Extract operator blocks
    sel_block = dec.get("Selection") or {}
    cx_block = dec.get("Crossover") or {}
    mut_block = dec.get("Mutation") or {}

    raw_sel = (sel_block.get("name") or "").strip()
    raw_cx = (cx_block.get("name") or "").strip()
    raw_mut = (mut_block.get("name") or "").strip()

    sel_name, sel_ok = _canonicalize("selection", raw_sel)
    cx_name, cx_ok = _canonicalize("crossover", raw_cx)
    mut_name, mut_ok = _canonicalize("mutation", raw_mut)

    if not sel_ok:
        warn.append(f"invalid selection {raw_sel.lower()}, fallback {sel_name}")
    if not cx_ok:
        warn.append(f"invalid crossover {raw_cx.lower()}, fallback {cx_name}")
    if not mut_ok:
        warn.append(f"invalid mutation {raw_mut.lower()}, fallback {mut_name}")

    # Parameter blocks (v2 uses 'parameter')
    sel_param_in = sel_block.get("parameter", {}) or {}
    cx_param_in = cx_block.get("parameter", {}) or {}
    mut_param_in = mut_block.get("parameter", {}) or {}

    # Clamp per operator via schema
    sel_param = _clamp_params("selection", sel_name, sel_param_in)
    cx_param = _clamp_params("crossover", cx_name, cx_param_in)
    mut_param = _clamp_params("mutation", mut_name, mut_param_in)

    norm = {
        "cxpb": cxpb,
        "mutpb": mutpb,
        "Selection": {"name": sel_name, "parameter": sel_param},
        "Crossover": {"name": cx_name, "parameter": cx_param},
        "Mutation": {"name": mut_name, "parameter": mut_param},
        "rationale": dec.get("rationale", ""),
    }
    return norm, "; ".join(warn)
