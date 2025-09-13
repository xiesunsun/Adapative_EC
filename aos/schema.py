from typing import Any, Dict, Tuple


ALLOWED_SELECTION = {"tournament", "sus", "best"}
ALLOWED_CROSSOVER = {"one_point", "two_point", "uniform"}
ALLOWED_MUTATION = {"flip_bit", "uniform_int"}


def clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def validate_decision(dec: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Validate and normalize v2 decision JSON (no alias mapping). Returns (normalized_decision, warning).
    """
    warn = []
    # Required top-level keys
    if not all(k in dec for k in ("cxpb", "mutpb", "Selection", "Crossover", "Mutation")):
        return ({"error": "missing required keys"}, "missing keys")
    cxpb = clamp01(dec.get("cxpb", 0.6))
    mutpb = clamp01(dec.get("mutpb", 0.2))

    # Extract operator blocks
    sel_block = dec.get("Selection") or {}
    cx_block = dec.get("Crossover") or {}
    mut_block = dec.get("Mutation") or {}
    sel_name = (sel_block.get("name") or "").strip().lower()
    cx_name = (cx_block.get("name") or "").strip().lower()
    mut_name = (mut_block.get("name") or "").strip().lower()
    if sel_name not in ALLOWED_SELECTION:
        warn.append(f"invalid selection {sel_name}, fallback tournament")
        sel_name = "tournament"
    if cx_name not in ALLOWED_CROSSOVER:
        warn.append(f"invalid crossover {cx_name}, fallback two_point")
        cx_name = "two_point"
    if mut_name not in ALLOWED_MUTATION:
        warn.append(f"invalid mutation {mut_name}, fallback flip_bit")
        mut_name = "flip_bit"

    # Parameter blocks (v2 uses 'parameter')
    sel_param = sel_block.get("parameter", {}) or {}
    cx_param = cx_block.get("parameter", {}) or {}
    mut_param = mut_block.get("parameter", {}) or {}

    # Validate per operator
    if sel_name == "tournament":
        try:
            k = int(sel_param.get("k", 3))
        except Exception:
            k = 3
        if k < 2:
            k = 2
        if k > 7:
            k = 7
        sel_param = {"k": k}
    else:
        sel_param = {}

    def _clamp_prob(p, default):
        try:
            return clamp01(float(p))
        except Exception:
            return default

    if cx_name == "uniform":
        prob = _clamp_prob(cx_param.get("prob", 0.5), 0.5)
        cx_param = {"prob": prob}
    else:
        cx_param = {}

    if mut_name == "flip_bit":
        prob = _clamp_prob(mut_param.get("prob", 0.0), 0.0)  # 0.0 meaning use default 1/n_features at bind time if 0
        mut_param = {"prob": prob}
    elif mut_name == "uniform_int":
        prob = _clamp_prob(mut_param.get("prob", 0.0), 0.0)
        low = int(mut_param.get("low", 0))
        up = int(mut_param.get("up", 1))
        if up < low:
            up = low
        mut_param = {"prob": prob, "low": low, "up": up}
    else:
        mut_param = {}

    norm = {
        "cxpb": cxpb,
        "mutpb": mutpb,
        "Selection": {"name": sel_name, "parameter": sel_param},
        "Crossover": {"name": cx_name, "parameter": cx_param},
        "Mutation": {"name": mut_name, "parameter": mut_param},
        "cxpb": cxpb,
        "mutpb": mutpb,
        "rationale": dec.get("rationale", ""),
    }
    return norm, "; ".join(warn)
