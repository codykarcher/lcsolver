"""Machinery for validating one model case against one real aeroplane.

There are four size classes and only some of them have a real aircraft worth
checking against, but the ones that do all want the same thing: pin the
settings, state the reference figures WITH their provenance, run, and report
what agrees and what does not.

The provenance requirement is not ceremony. During development a gear chord
fraction was quoted that turned out to be invented rather than looked up, and
it survived a long time because it looked like data. Every entry in a
``REFERENCE`` dict says where it came from, and anything soft is flagged
``approximate`` so it cannot quietly become an acceptance bound.

``KNOWN_GAPS`` is for discrepancies that are understood and deliberately open.
It is NOT a place to file failures to make ``check()`` pass -- a validation
case that reports only its successes is not a validation case. If a quantity
is off and the reason is not known, it belongs in neither dict and should show
up as a failure until it is fixed or explained.
"""
from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field

#: Solver settings these cases are known to converge under.
MAX_ITERATIONS = 400
#: 1e-5 for a result; set EDI_FAST=1 while debugging for 1e-3, which reaches a
#: usable answer in ~15 iterations and 12s instead of ~85 and 32s. Measured on
#: the 737: loosening to 1e-4 buys nothing at all (83 iterations against 85),
#: because the solve passes through 7e-5 on its way to 2.7e-6 either way -- the
#: saving only appears once the tolerance is loose enough to stop it early.
#: Geometry is settled to about three figures at 1e-3; do not quote a weight
#: from a fast run.
STATIONARITY_TOL = 1e-3 if os.environ.get("EDI_FAST") else 1e-5

#: Modules that read technology or geometry knobs at import time. A process
#: that already imported them under different settings has stale constants
#: baked in, so every build refreshes them rather than trusting import order.
_RELOAD = ("components.technology", "components.turbofan.model",
           "components.wing", "aircraft")


@dataclass(frozen=True)
class Ref:
    """One reference figure, with where it came from."""
    value: float
    units: str
    source: str
    #: True where the figure is approximate and must not be used as a tight
    #: acceptance bound.
    approximate: bool = False


@dataclass(frozen=True)
class Case:
    """A validation case: which aircraft, which settings, which references."""
    key: str                       # classes.CLASSES key
    label: str
    #: Environment knobs that define the case (TECH, GEAR_BOX_FRAC, ...).
    locked: dict
    #: name -> Ref
    reference: dict
    #: name -> why this one is expected to disagree
    known_gaps: dict = field(default_factory=dict)
    #: Cruise Mach pinned to the real aircraft's, vs left free.
    lock_mach: bool = True
    arch: str = "conventional"
    notes: str = ""

    def with_free_mach(self):
        from dataclasses import replace
        return replace(self, lock_mach=False)


def build(case: Case):
    """The formulation for ``case``, with its locked settings applied."""
    import dataclasses
    for k, v in case.locked.items():
        os.environ[k] = v
    for mod in _RELOAD:
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    import classes
    import architectures
    import aircraft

    arch = architectures.ARCHS[case.arch]
    if case.lock_mach:
        arch = dataclasses.replace(arch, lock_mach=True)
    return aircraft.build(classes.CLASSES[case.key], arch, seed="reference")


def solve(case: Case):
    """Build, solve, return ``(result, values)``. Values empty if it failed."""
    from edi_compat import structure_detector, unit_corrector
    from edi.solvers.ipopt.slcp_bridge import solve_sia
    from edi.solvers.ipopt.sia import SIAOptions
    import pyomo.environ as pyo

    cm = unit_corrector(build(case))
    st = structure_detector(cm)
    opts = SIAOptions(max_iterations=MAX_ITERATIONS)
    opts.stationarity_tolerance = STATIONARITY_TOL
    # PCCP-style condensation of the p/q numerator as well as the denominator.
    # The condensed row is EASIER than the true one, so iterates may leave the
    # feasible set and the monotone-descent guarantee goes with it -- but p_hat
    # matches p in value and gradient at x_k, so the sub-problem's duals still
    # certify the ORIGINAL problem and the KKT termination test stays honest.
    # Measured on the 737: 240 iterations and 79.6 s without, 44 and 22.8 s
    # with, agreeing to five significant figures on every reported quantity.
    # The locked-Mach case does not reach tolerance in 400 iterations without
    # it and converges in 19 with it.
    opts.condense_numerator = True
    res = solve_sia(st, options=opts, presolve=False)
    if not res.converged:
        return res, {}
    for var, val in zip(st["variables"], res.x):
        var.set_value(float(val))
    return res, {v.name: pyo.value(v)
                 for v in cm.component_data_objects(pyo.Var)}


def check(case: Case, values, tol=0.05):
    """Quantities outside ``tol``, excluding known gaps and soft references."""
    out = []
    for name, ref in case.reference.items():
        if name in case.known_gaps or ref.approximate or name not in values:
            continue
        r = values[name] / ref.value
        if abs(r - 1.0) > tol:
            out.append((name, r))
    return out


def report(case: Case, values, res=None, tol=0.05):
    """The comparison table, as a string."""
    FT = 1.0 / 0.3048
    mach = "M locked" if case.lock_mach else "M FREE"
    L = [f"{case.label}  (TECH={case.locked.get('TECH', '?')}, {mach})"]
    if res is not None:
        L.append(f"  converged={res.converged}  iterations={res.iterations}"
                 f"  stationarity="
                 f"{getattr(res, 'stationarity', float('nan')):.3e}")
        if not res.converged:
            L.append(f"  status: {str(res.status)[:160]}")
            return "\n".join(L)
    L.append("")
    L.append(f"  {'quantity':14s} {'model':>11s} {'real':>10s} {'ratio':>7s}")
    for name, ref in case.reference.items():
        if name not in values:
            L.append(f"  {name:14s} {'--':>11s} {ref.value:10.1f}"
                     f"        (not in model)")
            continue
        got = values[name]
        soft = "  ~" if ref.approximate else ""
        gap = "  <- known gap" if name in case.known_gaps else ""
        L.append(f"  {name:14s} {got:11.2f} {ref.value:10.1f} "
                 f"{got / ref.value:7.3f}{soft}{gap}")

    if {"LG_x_m", "x_CG[3]", "LG_x_n"} <= values.keys():
        nl = ((values["LG_x_m"] - values["x_CG[3]"])
              / (values["LG_x_m"] - values["LG_x_n"]))
        L.append(f"\n  nose gear load {nl * 100:.2f}% of MTOW (band 8-15%)")

    segs = sorted(k for k in values if k.startswith("FS_h["))
    if segs:
        L.append("  mission profile:")
        for k in segs:
            i = int(k[5:-1])
            h, M = values[k], values.get(f"FS_M[{i}]", float('nan'))
            L.append(f"    seg {i}  h = {h:8.1f} m = {h * FT:7.0f} ft"
                     f"   M = {M:6.4f}")
    bad = check(case, values, tol)
    L.append(f"\n  within {tol:.0%}: " + ("yes" if not bad else
             "NO -> " + ", ".join(f"{n} {r:.3f}" for n, r in bad)))
    return "\n".join(L)


def run(case: Case, both_mach=True):
    """Run a case at locked Mach and, optionally, free Mach. Prints."""
    import warnings
    warnings.filterwarnings("ignore")
    cases = [case] + ([case.with_free_mach()] if both_mach else [])
    out = {}
    for c in cases:
        res, vals = solve(c)
        print(report(c, vals, res))
        print()
        out["free" if not c.lock_mach else "locked"] = (res, vals)
    if case.known_gaps:
        print("  known gaps:")
        for name, why in case.known_gaps.items():
            print(f"    {name}: {why}")
    return out
