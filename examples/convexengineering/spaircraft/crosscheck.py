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

# Names this rebuild spells differently, or hoists to a different owner.
# Left is the rebuild's name, right is the key derived from gpkit's.
ALIASES = {
    "LoD": "L_D",
    "Re_nacelle": "R_e_nacelle",
    "D_fuse": "Fuse_D_fuse",
    "L_fuse": "Fuse_L_fuse",
    "M_r_out": "HT_box_M_r_out",
    "VT_AR_vt": "VT_box_AR_vt",
    "VT_C_D_vis_vt": "VT_C_D_vis",
    "HT_box_pi_M_fac": "HT_box_pi_M-fac",
    "Eng_mbar_fan_D": "Eng_mbar_fan_D",
    "y_eng": "VT_y_eng",
}

# Leaf-name rewrites where the rebuild's spelling is not mechanical.
LEAF = {
    "\\lambda": "lambda", "\\tau": "tau", "\\eta": "eta", "\\alpha": "alpha",
    "\\rho": "rho", "\\sigma": "sigma", "\\theta": "theta", "\\pi": "pi",
    "\\mu": "mu", "\\gamma": "gamma", "\\nu": "nu",
    # "\\Delta x" -> "dx": the space must go too, or d_f (fan diameter) and
    # dx_m (a delta) become indistinguishable after underscore normalisation.
    "\\Delta ": "d", "\\Delta": "d",
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
    # \bar{c}_{ht} -> cbar_ht, \bar{m}_{fan_D} -> mbar_fan_D
    s = re.sub(r"bar([A-Za-z])", r"\1bar", s)
    # f(\lambda_w) -> f_lambda_w, tan(\phi) -> tan_phi. Restricted to the
    # greek names, so that fp1 (fuel-air ratio plus one) is left alone.
    s = re.sub(r"^(f|tan|cos)(lambda|phi|psi|gamma|theta|Lambda)",
               r"\1_\2", s)
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
    for mine, theirs in ALIASES.items():
        if theirs in out:
            out.setdefault(mine, out[theirs])
    # Variables this rebuild introduces that the source does not carry.
    # dx_vbend replaces the subtraction in B_0v == B_1v*(x_tail - x_vbend);
    # its reference value follows from the two stations. x_vbend is in feet
    # in the source, x_tail in metres.
    if "Fuse_x_tail" in out and "Fuse_x_vbend" in out:
        out.setdefault("Fuse_dx_vbend",
                       out["Fuse_x_tail"] - out["Fuse_x_vbend"] * 0.3048)
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

    from edi.preconditioner.unitCorrector import unit_corrector
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
