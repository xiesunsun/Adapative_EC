from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from deap import tools


ParamSchema = Dict[str, Dict[str, Any]]


@dataclass
class OperatorSpec:
    kind: str  # "selection" | "crossover" | "mutation"
    name: str
    aliases: List[str]
    description: str
    params_schema: ParamSchema
    # binder: given normalized params and context -> (deap_func, kwargs)
    binder: Callable[[Dict[str, Any], Dict[str, Any]], Tuple[Callable[..., Any], Dict[str, Any]]]


class OperatorRegistry:
    def __init__(self, specs: List[OperatorSpec]):
        self._specs_by_kind: Dict[str, List[OperatorSpec]] = {"selection": [], "crossover": [], "mutation": []}
        self._canon_map: Dict[Tuple[str, str], OperatorSpec] = {}
        self._alias_map: Dict[Tuple[str, str], str] = {}
        for spec in specs:
            self._specs_by_kind[spec.kind].append(spec)
            key = (spec.kind, spec.name.lower())
            self._canon_map[key] = spec
            for alias in spec.aliases:
                akey = (spec.kind, str(alias).lower())
                self._alias_map[akey] = spec.name.lower()

    def list_specs(self, kind: str) -> List[OperatorSpec]:
        return list(self._specs_by_kind.get(kind, []))

    def canonicalize(self, kind: str, name: Optional[str]) -> str:
        raw = (name or "").strip().lower()
        if (kind, raw) in self._canon_map:
            return raw
        if (kind, raw) in self._alias_map:
            return self._alias_map[(kind, raw)]
        # defaults per kind
        defaults = {"selection": "tournament", "crossover": "two_point", "mutation": "flip_bit"}
        return defaults.get(kind, raw)

    def _coerce_value(self, val: Any, spec: Dict[str, Any]) -> Any:
        t = (spec.get("type") or "").lower()
        if t == "int":
            v = int(val)
            if "min" in spec:
                v = max(int(spec["min"]), v)
            if "max" in spec:
                v = min(int(spec["max"]), v)
            return v
        if t == "float":
            v = float(val)
            if "min" in spec:
                v = max(float(spec["min"]), v)
            if "max" in spec:
                v = min(float(spec["max"]), v)
            return v
        # passthrough
        return val

    def normalize_params(self, kind: str, canon_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        spec = self._canon_map.get((kind, canon_name))
        if not spec:
            return {}
        out: Dict[str, Any] = {}
        for p, ps in (spec.params_schema or {}).items():
            default = ps.get("default")
            if p in params:
                try:
                    out[p] = self._coerce_value(params[p], ps)
                except Exception:
                    out[p] = default
            else:
                out[p] = default
        # fix low/up ordering for uniform_int
        if canon_name == "uniform_int" and {"low", "up"}.issubset(out.keys()):
            if out["up"] < out["low"]:
                out["up"] = out["low"]
        return out

    def bind(self, kind: str, canon_name: str, params: Dict[str, Any], context: Dict[str, Any]) -> Tuple[Callable[..., Any], Dict[str, Any]]:
        spec = self._canon_map.get((kind, canon_name))
        if not spec:
            # sensible defaults
            if kind == "selection":
                return tools.selTournament, {"tournsize": 3}
            if kind == "crossover":
                return tools.cxTwoPoint, {}
            if kind == "mutation":
                # need n_features for default indpb
                n_features = max(int(context.get("n_features", 1)), 1)
                return tools.mutFlipBit, {"indpb": 1.0 / n_features}
        # apply binder
        norm = self.normalize_params(kind, canon_name, params or {})
        return spec.binder(norm, context)

    def export_prompt_json(self) -> Dict[str, Any]:
        def dump_kind(kind: str) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for s in self.list_specs(kind):
                out.append({
                    "name": s.name,
                    "aliases": list(s.aliases),
                    "description": s.description,
                    "params_schema": s.params_schema or {},
                })
            return out
        return {
            "selection": dump_kind("selection"),
            "crossover": dump_kind("crossover"),
            "mutation": dump_kind("mutation"),
        }


def _mk_specs() -> List[OperatorSpec]:
    specs: List[OperatorSpec] = []
    # Selection: tournament
    specs.append(
        OperatorSpec(
            kind="selection",
            name="tournament",
            aliases=["tournament3", "tournament_k"],
            description="Tournament selection with parameter tournsize (selection pressure).",
            params_schema={"tournsize": {"type": "int", "min": 2, "max": 7, "default": 3}},
            binder=lambda p, ctx: (tools.selTournament, {"tournsize": int(p.get("tournsize", 3))}),
        )
    )
    # Selection: SUS
    specs.append(
        OperatorSpec(
            kind="selection",
            name="sus",
            aliases=["stochastic_universal_sampling", "SUS"],
            description="Fitness-proportionate selection using SUS sampler.",
            params_schema={},
            binder=lambda p, ctx: (tools.selStochasticUniversalSampling, {}),
        )
    )
    # Selection: best
    specs.append(
        OperatorSpec(
            kind="selection",
            name="best",
            aliases=["selBest"],
            description="Pick the best individuals; strong exploitation; may reduce diversity.",
            params_schema={},
            binder=lambda p, ctx: (tools.selBest, {}),
        )
    )

    # Selection: random
    specs.append(
        OperatorSpec(
            kind="selection",
            name="random",
            aliases=["selRandom"],
            description="Uniform random selection; maximizes diversity (no fitness pressure).",
            params_schema={},
            binder=lambda p, ctx: (tools.selRandom, {}),
        )
    )
    # Selection: roulette (fitness-proportionate)
    specs.append(
        OperatorSpec(
            kind="selection",
            name="roulette",
            aliases=["selRoulette"],
            description="Fitness-proportionate selection (requires mostly non-negative fitness values).",
            params_schema={},
            binder=lambda p, ctx: (tools.selRoulette, {}),
        )
    )

    # Crossover: one_point
    specs.append(
        OperatorSpec(
            kind="crossover",
            name="one_point",
            aliases=["cxOnePoint", "onepoint"],
            description="One-point crossover for binary masks.",
            params_schema={},
            binder=lambda p, ctx: (tools.cxOnePoint, {}),
        )
    )
    # Crossover: two_point
    specs.append(
        OperatorSpec(
            kind="crossover",
            name="two_point",
            aliases=["cxTwoPoint", "twopoint"],
            description="Two-point crossover preserving blocks.",
            params_schema={},
            binder=lambda p, ctx: (tools.cxTwoPoint, {}),
        )
    )
    # Crossover: uniform
    def _cx_uniform_binder(p: Dict[str, Any], ctx: Dict[str, Any]):
        v = p.get("indpb", 0.5)
        try:
            val = float(v)
        except Exception:
            val = 0.5
        val = max(0.0, min(1.0, val))
        return tools.cxUniform, {"indpb": val}

    specs.append(
        OperatorSpec(
            kind="crossover",
            name="uniform",
            aliases=["cxUniform", "uniform_p"],
            description="Uniform crossover; per-bit exchange probability indpb in [0,1].",
            params_schema={"indpb": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5}},
            binder=_cx_uniform_binder,
        )
    )

    # --- Custom crossover operators for binary strings ---
    import random as _random

    def cx_k_point(ind1, ind2, k=3):
        """K-point crossover for sequences (in-place)."""
        size = min(len(ind1), len(ind2))
        if size <= 1 or k <= 0:
            return ind1, ind2
        k = min(int(k), size - 1)
        points = sorted(_random.sample(range(1, size), k))
        toggle = False
        last = 0
        for pt in points + [size]:
            if toggle:
                ind1[last:pt], ind2[last:pt] = ind2[last:pt], ind1[last:pt]
            toggle = not toggle
            last = pt
        return ind1, ind2

    def cx_and_or(ind1, ind2):
        """Bitwise AND/OR crossover: child1 = AND, child2 = OR (in-place)."""
        size = min(len(ind1), len(ind2))
        if size <= 0:
            return ind1, ind2
        a = ind1
        b = ind2
        and_child = [(int(x) & int(y)) for x, y in zip(a, b)]
        or_child = [(int(x) | int(y)) for x, y in zip(a, b)]
        ind1[:size] = and_child
        ind2[:size] = or_child
        return ind1, ind2

    def cx_hux(ind1, ind2):
        """Half Uniform Crossover (HUX): swap exactly half of differing bits."""
        diff_idx = [i for i, (x, y) in enumerate(zip(ind1, ind2)) if int(x) != int(y)]
        if not diff_idx:
            return ind1, ind2
        _random.shuffle(diff_idx)
        half = len(diff_idx) // 2
        for i in diff_idx[:half]:
            ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2

    def _cx_kpoint_binder(p: Dict[str, Any], ctx: Dict[str, Any]):
        k = int(p.get("k", 3))
        k = max(1, k)
        return cx_k_point, {"k": k}

    specs.append(
        OperatorSpec(
            kind="crossover",
            name="k_point",
            aliases=["cxKPoint"],
            description="K-point crossover with k cut points (k>=1).",
            params_schema={"k": {"type": "int", "min": 1, "max": 32, "default": 3}},
            binder=_cx_kpoint_binder,
        )
    )

    specs.append(
        OperatorSpec(
            kind="crossover",
            name="and_or",
            aliases=["cxAndOr"],
            description="Bitwise AND/OR crossover (child1=AND, child2=OR).",
            params_schema={},
            binder=lambda p, ctx: (cx_and_or, {}),
        )
    )

    specs.append(
        OperatorSpec(
            kind="crossover",
            name="hux",
            aliases=["cxHUX"],
            description="Half Uniform Crossover: swap half of differing bits.",
            params_schema={},
            binder=lambda p, ctx: (cx_hux, {}),
        )
    )

    # Mutation: flip_bit
    def _default_indpb(ctx: Dict[str, Any]) -> float:
        n_features = max(int(ctx.get("n_features", 1)), 1)
        return 1.0 / n_features

    def _mut_flip_binder(p: Dict[str, Any], ctx: Dict[str, Any]):
        v = p.get("indpb", None)
        try:
            val = float(v) if v is not None and v != "1/n_features" else _default_indpb(ctx)
        except Exception:
            val = _default_indpb(ctx)
        val = max(0.0, min(1.0, val))
        return tools.mutFlipBit, {"indpb": val}

    specs.append(
        OperatorSpec(
            kind="mutation",
            name="flip_bit",
            aliases=["bit_flip", "mutFlipBit"],
            description="Binary flip mutation; per-bit probability indpb (default 1/n_features).",
            params_schema={"indpb": {"type": "float", "min": 0.0, "max": 1.0, "default": "1/n_features"}},
            binder=_mut_flip_binder,
        )
    )
    # Mutation: uniform_int
    def _mut_uniform_int_binder(p: Dict[str, Any], ctx: Dict[str, Any]):
        low = int(p.get("low", 0))
        up = int(p.get("up", 1))
        if up < low:
            up = low
        v = p.get("indpb", None)
        try:
            indpb = float(v) if v is not None and v != "1/n_features" else _default_indpb(ctx)
        except Exception:
            indpb = _default_indpb(ctx)
        indpb = max(0.0, min(1.0, indpb))
        return tools.mutUniformInt, {"low": low, "up": up, "indpb": indpb}

    specs.append(
        OperatorSpec(
            kind="mutation",
            name="uniform_int",
            aliases=["mutUniformInt"],
            description="Uniform integer mutation for {0,1}; per-bit probability indpb.",
            params_schema={
                "indpb": {"type": "float", "min": 0.0, "max": 1.0, "default": "1/n_features"},
                "low": {"type": "int", "default": 0},
                "up": {"type": "int", "default": 1},
            },
            binder=_mut_uniform_int_binder,
        )
    )

    # --- Custom mutations for binary strings ---
    def mut_invert_segment(individual):
        size = len(individual)
        if size <= 1:
            return (individual,)
        i = _random.randrange(0, size - 1)
        j = _random.randrange(i + 1, size)
        for t in range(i, j):
            individual[t] = 1 - int(individual[t])
        return (individual,)

    def mut_k_flip(individual, k=1):
        size = len(individual)
        if size <= 0 or k <= 0:
            return (individual,)
        k = min(int(k), size)
        idxs = _random.sample(range(size), k)
        for i in idxs:
            individual[i] = 1 - int(individual[i])
        return (individual,)

    def _mut_k_flip_binder(p: Dict[str, Any], ctx: Dict[str, Any]):
        k = int(p.get("k", 1))
        k = max(1, k)
        return mut_k_flip, {"k": k}

    specs.append(
        OperatorSpec(
            kind="mutation",
            name="invert_segment",
            aliases=["mutInvertSegment"],
            description="Invert bits on a random contiguous segment.",
            params_schema={},
            binder=lambda p, ctx: (mut_invert_segment, {}),
        )
    )

    specs.append(
        OperatorSpec(
            kind="mutation",
            name="k_flip",
            aliases=["mutKFlip"],
            description="Flip exactly k random bits (k>=1).",
            params_schema={"k": {"type": "int", "min": 1, "max": 128, "default": 1}},
            binder=_mut_k_flip_binder,
        )
    )

    return specs


_DEFAULT_REGISTRY: Optional[OperatorRegistry] = None


def default_registry() -> OperatorRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = OperatorRegistry(_mk_specs())
    return _DEFAULT_REGISTRY
