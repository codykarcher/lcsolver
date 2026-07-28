"""Verification harness for the convexengineering model rebuilds.

Each rebuilt model lives in its own package under this directory and exposes::

    build()   -> edi.Formulation           the EDI reimplementation
    solve(f)  -> dict                      (optional) custom solve entry

Ground truth comes from the original gpkit models published at
https://github.com/convexengineering. Those are not a dependency of this
repo: each model directory carries a ``reference.json`` snapshot recorded
from a gpkit run, so verification is reproducible without gpkit installed.
Re-record with ``python -m examples.convexengineering.harness record ...``
or the per-model ``reference.py`` script.

Why snapshots rather than paper tables: the papers report rounded values
(often 3 significant figures) and, per the authors, contain typos. The
published source is the more precise and more self-consistent reference.
Where the two disagree, the discrepancy is recorded in the model's module
docstring and in DISCREPANCIES.md.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyomo.environ as pyo
from pyomo.core.base.var import IndexedVar


# ---------------------------------------------------------------------------
# Extracting a solution from an EDI formulation
# ---------------------------------------------------------------------------

def solution_dict(f) -> dict:
    """Return ``{variable_name: float}`` in each variable's declared units.

    Indexed variables are flattened to ``name[i]``. Values are taken as the
    raw Pyomo values, which are in the units the Variable was declared with.
    """
    out: dict[str, float] = {}
    for v in f.get_variables():
        if isinstance(v, IndexedVar):
            for ix in v.index_set():
                val = v[ix].value
                if val is not None:
                    out[f"{v.name}[{ix}]"] = float(val)
        else:
            if v.value is not None:
                out[v.name] = float(v.value)
    return out


def objective_value(f) -> float | None:
    """Value of the (single) active objective, or None."""
    for obj in f.component_data_objects(pyo.Objective, active=True):
        try:
            return float(pyo.value(obj))
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class Row:
    name: str
    actual: float | None
    expected: float | None
    units: str = ""
    note: str = ""

    @property
    def rel(self) -> float | None:
        if self.actual is None or self.expected is None:
            return None
        denom = max(abs(self.expected), 1e-30)
        return abs(self.actual - self.expected) / denom

    def ok(self, rtol: float) -> bool:
        r = self.rel
        return r is not None and r <= rtol


@dataclass
class Report:
    model: str
    rows: list = field(default_factory=list)
    rtol: float = 1e-3
    notes: list = field(default_factory=list)

    @property
    def compared(self):
        return [r for r in self.rows if r.rel is not None]

    @property
    def failures(self):
        return [r for r in self.compared if not r.ok(self.rtol)]

    @property
    def missing(self):
        return [r for r in self.rows if r.rel is None]

    @property
    def passed(self) -> bool:
        return bool(self.compared) and not self.failures

    def __str__(self) -> str:
        w = max([len(r.name) for r in self.rows] + [12])
        lines = [
            "",
            f"{'=' * (w + 46)}",
            f"{self.model}   (rtol = {self.rtol:g})",
            f"{'=' * (w + 46)}",
            f"{'variable'.ljust(w)}  {'rebuilt':>14}  {'reference':>14}  {'rel':>8}",
            f"{'-' * (w + 46)}",
        ]
        for r in sorted(self.rows, key=lambda x: x.name):
            a = "—" if r.actual is None else f"{r.actual:14.6g}"
            e = "—" if r.expected is None else f"{r.expected:14.6g}"
            if r.rel is None:
                rel, flag = "     —  ", "?"
            else:
                rel, flag = f"{r.rel:8.2e}", " " if r.ok(self.rtol) else "X"
            lines.append(f"{r.name.ljust(w)}  {a}  {e}  {rel} {flag} {r.note}")
        lines.append(f"{'-' * (w + 46)}")
        n_ok = len(self.compared) - len(self.failures)
        lines.append(
            f"{n_ok}/{len(self.compared)} within tolerance"
            + (f", {len(self.missing)} unmatched" if self.missing else "")
        )
        for n in self.notes:
            lines.append(f"  note: {n}")
        lines.append("PASS" if self.passed else "FAIL")
        return "\n".join(lines)


def compare(model_name: str, actual: dict, reference: dict,
            rtol: float = 1e-3, only: list | None = None,
            aliases: dict | None = None) -> Report:
    """Diff a rebuilt solution against a reference solution.

    Parameters
    ----------
    actual, reference : {name: value}
    only : restrict the comparison to these reference keys
    aliases : {reference_key: rebuilt_key} for names that differ between the
        gpkit model and the EDI rebuild (gpkit uses LaTeX-ish names).
    """
    aliases = aliases or {}
    keys = only if only is not None else sorted(reference)
    rep = Report(model=model_name, rtol=rtol)
    for k in keys:
        exp = reference.get(k)
        akey = aliases.get(k, k)
        rep.rows.append(Row(name=k, actual=actual.get(akey), expected=exp))
    return rep


# ---------------------------------------------------------------------------
# Reference snapshots
# ---------------------------------------------------------------------------

def load_reference(path: str | Path) -> dict:
    p = Path(path)
    with p.open() as fh:
        blob = json.load(fh)
    return blob["values"] if "values" in blob else blob


def save_reference(path: str | Path, values: dict, *, source: str,
                   meta: dict | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = {"source": source, "meta": meta or {}, "values": values}
    with p.open("w") as fh:
        json.dump(blob, fh, indent=2, sort_keys=True)
    print(f"wrote {p}  ({len(values)} values)")


# ---------------------------------------------------------------------------
# Solve helpers
# ---------------------------------------------------------------------------

def solve_edi(f, solver: str = "auto", **kw):
    """Solve an EDI formulation, returning (solution_dict, objective, note)."""
    from edi.solvers.solver import solve as _solve
    note = ""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _solve(f, solver=solver, **kw)
        for c in caught:
            if "status" in str(c.message) or "converge" in str(c.message):
                note = str(c.message)[:90]
    return solution_dict(f), objective_value(f), note


def feasibility(f, rtol: float = 1e-6, *, correct_units: bool = True) -> tuple:
    """(n_violated, worst_relative_violation, worst_constraint_name).

    Two traps this avoids, both of which produce wildly misleading numbers:

    1. ``pyo.value(expr)`` evaluates raw magnitudes and ignores units. On a
       unit-annotated model ``Range/V <= T_flight`` evaluates as
       ``58.7 (km*s/m) - 16.3 (hr)``, reporting a 98% violation on a
       constraint that is satisfied exactly. EDI's own ``unit_corrector``
       resolves this, so run it first.

    2. Scoring against Pyomo's canonical ``body``. Pyomo rewrites
       ``lhs <= rhs`` to ``body <= 0``, where *body is the residual itself* —
       so normalizing by ``|body|`` or ``|bound|`` discards the scale of the
       original terms. An absolute residual of 0.04 between two quantities of
       1.5e6 then reads as a 4% violation instead of 3e-8.

    So: evaluate the relational expression's own two sides via ``expr.args``
    and normalize by their magnitude.
    """
    if correct_units:
        try:
            from edi.units.unitCorrector import unit_corrector
            f = unit_corrector(f)
        except Exception:
            pass

    worst, n, where = 0.0, 0, ""
    for con in f.component_objects(pyo.Constraint, descend_into=True, active=True):
        for c in con.values():
            args = getattr(c.expr, "args", None)
            if not args or len(args) != 2:
                continue
            try:
                lhs, rhs = (float(pyo.value(a)) for a in args)
            except Exception:
                continue
            scale = max(abs(lhs), abs(rhs), 1e-30)
            # Pyomo normalizes every relational expression to lhs <= rhs (or
            # ==), so a positive lhs-rhs is the violation in both cases; for
            # an equality either sign counts.
            resid = lhs - rhs
            if c.equality:
                resid = abs(resid)
            v = max(resid, 0.0) / scale
            if v > rtol:
                n += 1
            if v > worst:
                worst, where = v, c.name
    return n, worst, where


__all__ = [
    "solution_dict", "objective_value", "compare", "Report", "Row",
    "load_reference", "save_reference", "solve_edi", "feasibility",
]
