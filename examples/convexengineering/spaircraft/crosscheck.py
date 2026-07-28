"""Evaluate the EDI rebuild's constraints at gpkit's converged optimum.

Cross-substitution: put the *reference* solution into the *rebuilt* model and
score every constraint. Whichever come back violated name the transcription
errors directly, with no reading or reasoning, and -- crucially -- without
needing the rebuild to solve first. This is the technique that found the
wing's material-property bug; see STATUS.md.

Run with ``python -m spaircraft.crosscheck``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pyomo.environ as pyo
from pyomo.core.base.var import IndexedVar

from .model import build

# gpkit model-path segment -> this rebuild's variable prefix.
COMPONENT = {
    "Wing": "Wing_", "WingNoStruct": "Wing_", "WingPerformance": "Wing_",
    "Fuselage": "Fuse_", "FuselagePerformance": "Fuse_",
    "VerticalTail": "VT_", "VerticalTailNoStruct": "VT_",
    "VerticalTailPerformance": "VT_",
    "HorizontalTail": "HT_", "HorizontalTailNoStruct": "HT_",
    "HorizontalTailPerformance": "HT_",
    "LandingGear": "LG_", "Engine": "Eng_", "EnginePerformance": "Eng_",
    "FlightState": "FS_", "Atmosphere": "FS_", "Altitude": "FS_",
}
# Nested inside a surface, the box gets its own second-level prefix.
BOX = "WingBox"

# Leaf-name rewrites where the rebuild's spelling is not mechanical.
LEAF = {
    "\\lambda": "lambda", "\\tau": "tau", "\\eta": "eta", "\\alpha": "alpha",
    "\\rho": "rho", "\\sigma": "sigma", "\\theta": "theta", "\\pi": "pi",
    "\\mu": "mu", "\\gamma": "gamma", "\\nu": "nu", "\\Delta": "d",
    "\\bar": "bar", "\\cos": "cos", "\\tan": "tan", "\\dot": "dot",
}


def leafname(raw: str) -> str:
    """Turn a gpkit LaTeX-ish variable name into this rebuild's spelling."""
    s = raw
    if s.endswith("[:]"):          # gpkit marks vectorized variables this way
        s = s[:-3]
    for k, v in LEAF.items():
        s = s.replace(k, v)
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    s = s.replace("(", "").replace(")", "")
    s = s.replace(",", "_").replace("/", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    # Station numbers lose their decimal point: T_t_4.1 -> T_t_41, M_2.5 -> M_25
    s = re.sub(r"(\d)\.(\d)", r"\1\2", s)
    # Specific heats: C_p_c -> Cp_c
    s = re.sub(r"^C_p_", "Cp_", s)
    # "+1" suffixes: alpha_+1 -> alpha_p1
    s = s.replace("+1", "p1").replace("_p_1", "_p1")
    # Deltas collapse onto the following symbol: d_x_... -> dx_..., d_R -> dR
    s = re.sub(r"^d_([xRP])", r"d\1", s)
    return s


def reference_point(path: Path) -> dict:
    blob = json.load(path.open())["values"]["optimalD8"]["values"]
    out = {}
    for key, val in blob.items():
        # Station numbers contain dots (A_{2.5}, T_{t_{4.1}}), and so does the
        # model path. Protect the numeric ones before splitting on the path.
        key = re.sub(r"(\d)\.(\d)", r"\1\2", key)
        parts = key.split(".")
        leaf = leafname(parts[-1])
        # A WingBox nested under a surface takes that surface's prefix plus
        # "box_"; getting this wrong silently drops every box variable.
        suffix = "box_" if BOX in parts[:-1] else ""
        prefix = ""
        for seg in reversed(parts[:-1]):
            if seg in COMPONENT:
                prefix = COMPONENT[seg]
                break
        out.setdefault(prefix + suffix + leaf, val)
    return out


def main() -> None:
    f = build()
    ref = reference_point(Path(__file__).with_name("reference.json"))

    # Push the reference point onto the rebuilt model's variables.
    matched, unmatched = 0, []
    for v in f.get_variables():
        name = v.name
        items = ([(f"{name}[{i}]", v[i]) for i in v.index_set()]
                 if isinstance(v, IndexedVar) else [(name, v)])
        for nm, vd in items:
            base = nm.split("[")[0]
            idx = int(nm.split("[")[1].rstrip("]")) if "[" in nm else None
            val = ref.get(base)
            if val is None:
                unmatched.append(base)
                continue
            if isinstance(val, list):
                val = val[idx] if idx is not None and idx < len(val) else val[0]
            if idx is not None and not isinstance(ref.get(base), list):
                pass
            try:
                vd.set_value(float(val), skip_validation=True)
                matched += 1
            except Exception:
                unmatched.append(base)

    uniq = sorted(set(unmatched))
    print(f"matched {matched} variable data objects; "
          f"{len(uniq)} distinct names unmatched")
    print("  unmatched sample:", uniq[:20])

    from edi.units.unitCorrector import unit_corrector
    fc = unit_corrector(f)
    rows = []
    n_eval = 0
    for con in fc.component_objects(pyo.Constraint, descend_into=True,
                                    active=True):
        for c in con.values():
            args = getattr(c.expr, "args", None)
            if not args or len(args) != 2:
                continue
            try:
                lhs, rhs = (float(pyo.value(a)) for a in args)
            except Exception:
                continue
            n_eval += 1
            scale = max(abs(lhs), abs(rhs), 1e-30)
            r = lhs - rhs
            if c.equality:
                r = abs(r)
            rows.append((max(r, 0.0) / scale, str(c.expr)[:120]))
    rows.sort(reverse=True)
    bad = [r for r in rows if r[0] > 1e-3]
    print(f"evaluated {n_eval} constraints; {len(bad)} violated by > 1e-3")
    for v, e in rows[:25]:
        print(f"  {v:10.3e}  {e}")


if __name__ == "__main__":
    main()
