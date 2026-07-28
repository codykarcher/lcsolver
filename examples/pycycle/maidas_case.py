"""Run MAIDAS structure detection over pycycle's engine components.

Why this is an interesting case
-------------------------------
pycycle was written for gradient-based optimization with OpenMDAO. Nobody
writing it was thinking about geometric or signomial programming. So asking
"which of these components are secretly monomial / posynomial / signomial?"
is a genuine test of structure detection rather than a rediscovery of
structure that was deliberately put there.

That question is the York result restated. York et al. showed TASOPT is
almost entirely SP-representable despite never having been written that way;
if the same holds component-by-component for pycycle, that is independent
evidence — and pycycle is far easier to introspect than Fortran.

What this script does
---------------------
Each pycycle ``ExplicitComponent.compute()`` is transcribed here as a plain
function of scalars, keeping the arithmetic identical. MAIDAS traces each
one, lifts it to IR, and classifies it. The transcription is mechanical and
the source line is cited for every component so it can be checked.

The transcription step exists because MAIDAS traces plain callables while
OpenMDAO's ``compute(self, inputs, outputs)`` reads and writes dict-likes.
Nothing is simplified in the process — the expressions are copied verbatim.

Run: ``python maidas_case.py``
"""
from __future__ import annotations

import numpy as np

from maidas.core import analyze, classify_flat

# pycycle/constants.py
T_STDeng = 518.67       # degR
P_STDeng = 14.695951    # psi


# ---------------------------------------------------------------------------
# Transcribed components. Each docstring cites the source.
# ---------------------------------------------------------------------------

def corrected_inputs(Tt, Pt, W_in, Nmech):
    """pycycle/elements/compressor.py :: CorrectedInputsCalc.compute

    Corrected flow and corrected speed. Pure ratios and half-powers.
    """
    delta = Pt / P_STDeng
    theta = Tt / T_STDeng
    Wc = W_in * theta**0.5 / delta
    Nc = Nmech * theta**-0.5
    return Wc, Nc


def pressure_rise(PR, Pt_in):
    """pycycle/elements/compressor.py :: PressureRise.compute"""
    return PR * Pt_in


def enthalpy_rise(ideal_ht, inlet_ht, eff):
    """pycycle/elements/compressor.py :: EnthalpyRise.compute

    (ideal_ht - inlet_ht)/eff + inlet_ht — the subtraction is the point.
    """
    return (ideal_ht - inlet_ht) / eff + inlet_ht


def pressure_loss(dPqP, Pt_in):
    """pycycle/elements/duct.py :: PressureLoss.compute

    Pt_in*(1 - dPqP) — again a subtraction, this time of a fraction from 1.
    """
    return Pt_in * (1.0 - dPqP)


def eff_poly(PR, S_in, S_out, Rt):
    """pycycle/elements/compressor.py :: eff_poly_calc.compute

    Polytropic efficiency. Carries a log of a design variable *and* a
    difference in the denominator.
    """
    return Rt * np.log(PR) / (Rt * np.log(PR) + S_out - S_in)


def shaft_power(trq, Nmech):
    """pycycle/elements/shaft.py :: power from torque and speed."""
    return trq * Nmech


COMPONENTS = [
    ("CorrectedInputsCalc", corrected_inputs, (500.0, 14.0, 30.0, 1000.0)),
    ("PressureRise", pressure_rise, (3.0, 5.0)),
    ("PressureLoss", pressure_loss, (0.02, 5.0)),
    ("EnthalpyRise", enthalpy_rise, (2.0, 1.0, 0.85)),
    ("eff_poly_calc", eff_poly, (3.0, 1.0, 1.2, 0.0686)),
    ("ShaftPower", shaft_power, (100.0, 1000.0)),
]


# GP-relevant labels, most restrictive first.
LADDER = ["monomial", "posynomial", "signomial"]


def _flatten(ir):
    """Yield the leaf IR nodes of whatever shape ``analyze`` returned.

    The shape is not fixed and this is worth knowing before writing anything
    against it. ``analyze`` returns:

    * a bare IR node, for a single scalar output with no tapping;
    * a tuple, for a multiple-return with no tapping;
    * an ``IntermediateRepresentation`` (a dict) once tapping fires, whose
      ``'_'`` key holds the *original return* — itself possibly a tuple —
      alongside one entry per tapped intermediate.

    Whether tapping fires depends on whether the traced function's module has
    other user-defined functions in its globals to AST-rewrite. So the *same*
    function analyzed from a script and from inside a module can come back
    with different shapes; flattening recursively is the only safe way to
    consume it.
    """
    if hasattr(ir, "values"):
        for v in ir.values():
            yield from _flatten(v)
    elif isinstance(ir, (tuple, list)):
        for v in ir:
            yield from _flatten(v)
    else:
        yield ir


def classify_component(fn):
    """Return (labels, verdict) for one component.

    A component is classified by the *loosest* of its outputs: the whole
    component is GP-compatible only if every output is.
    """
    labels = set()
    for node in _flatten(analyze(fn)):
        labels |= set(classify_flat(node))
    for tag in LADDER:
        if tag in labels and "not_gp" not in labels:
            return labels, tag
    return labels, "not GP/SP"


def main() -> None:
    print("MAIDAS structure detection over pycycle components")
    print("=" * 66)
    print(f"{'component':24s} {'tightest form':14s}  labels")
    print("-" * 66)
    tally: dict = {}
    for name, fn, _args in COMPONENTS:
        try:
            labels, verdict = classify_component(fn)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:24s} {'TRACE FAILED':14s}  {type(exc).__name__}: {exc}")
            tally["trace failed"] = tally.get("trace failed", 0) + 1
            continue
        shown = ",".join(sorted(l for l in labels if l in
                                (*LADDER, "not_gp", "rational", "quadratic")))
        print(f"{name:24s} {verdict:14s}  {shown}")
        tally[verdict] = tally.get(verdict, 0) + 1

    print("-" * 66)
    for k in sorted(tally):
        print(f"   {tally[k]:2d}  {k}")
    print("""
Reading the result
------------------
A 'monomial' component is trivially GP-compatible and can appear as an
equality. 'signomial' means a subtraction survived, so it needs an SP
treatment or a reformulation. 'not GP/SP' means an opaque atom (a log or
exp of a design variable) that no monomial form can express — those are the
components that would need fitting, which is exactly what the York turbofan
model does.
""")


if __name__ == "__main__":
    main()
