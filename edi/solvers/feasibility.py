#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Find a feasible point, or name the constraints that prevent one.

The machinery has been here for a while, as the elastic Phase I inside the SIA
solver, and it was reachable only by building a low-level ``Problem`` by hand.
That is the wrong shape for the question it answers, which is one people ask
constantly and early: *is this model even satisfiable, and if not, what do I
have to relax?*

Two things make the elastic (L1) form the right one to expose. Its optimum is
**sparse**: constraints that can be satisfied go to zero slack and drop out, so
what remains is an approximate irreducible inconsistent subsystem -- the actual
answer -- rather than the flat field of identical residuals a min-max Phase I
leaves behind. And when it succeeds it hands back a strictly feasible point,
which is worth more than the reassurance: a hard model started from a feasible
point is a different problem from one started at the author's guesses.

So the result is *passable*. ``solve(f, start=result)`` begins from the point
this found::

    result = feasibility(f)
    if not result.feasible:
        print(result.summary())          # the rows that must be relaxed
    else:
        solve(f, start=result)           # start from the point it found
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["FeasibilityResult", "feasibility"]


@dataclass
class FeasibilityResult:
    """What the Phase I found: a point, or the rows that stop one existing."""

    #: True when every constraint is satisfied to ``feasibility_tolerance``.
    feasible: bool = False
    #: The point reached, in the order of ``structures['variables']``. Feasible
    #: when ``feasible``; the least-infeasible point found otherwise, which is
    #: still usually a better start than the author's guesses.
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Per-constraint slack. Zero where the row is satisfied.
    slacks: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: ``[(row_index, slack, [variable names])]`` for the rows that keep a
    #: positive slack, largest first. Empty when feasible.
    blocking: list = field(default_factory=list)
    iterations: int = 0
    names: list = field(default_factory=list)
    text: str = ''

    def summary(self) -> str:
        """The report, as a string. ``print(result.summary())``."""
        return self.text

    def __str__(self):
        return self.text

    def __bool__(self):
        """``if feasibility(f): ...`` reads as the question it is."""
        return bool(self.feasible)

    def apply(self, model):
        """Write the point onto ``model`` as its starting values.

        The backends take their initial point from the model's current variable
        values, so this is what makes the result passable. Returns the model.
        """
        from edi.preconditioner.presolve import _as_structures
        from edi.solvers.writeback import write_solution

        st = _as_structures(model)
        write_solution(st, {'x': np.asarray(self.x, dtype=float)}, model=model)
        return model


def feasibility(model, x0=None, options=None, top=12, presolve=True):
    """Find a feasible point for ``model``, or say what prevents one.

    Accepts a :class:`~edi.objects.formulation.Formulation` or an already
    detected structure. Runs the elastic Phase I:

    .. math::  \\min \\sum_i s_i \\quad\\text{s.t.}\\quad \\log g_i(x) \\le s_i,
               \\; s_i \\ge 0

    Returns a :class:`FeasibilityResult`. It is truthy when feasible, carries
    the point in ``.x``, and can be handed to ``solve(f, start=result)``.

    ``x0`` defaults to the model's current values -- the author's guesses
    before a solve, the previous answer after one. ``top`` caps how many
    blocking rows the report lists.

    This solves a *feasibility* problem and ignores the objective entirely. A
    point it returns satisfies the constraints; it is not optimal and is not
    claimed to be.
    """
    import pyomo.environ as pyo

    from edi.preconditioner.presolve import _as_structures
    from edi.solvers.ipopt.sia import SIAOptions, explain_infeasibility
    from edi.solvers.ipopt.slcp_bridge import (_apply_presolve, _restore,
                                               build_problem)

    st = _as_structures(model)
    if st.get('bounds') is not None:
        # The presolve form splits bounds out of the rows, and Phase I reads
        # rows. Detect afresh rather than quietly looking for a point in a
        # problem with no variable bounds.
        from edi.preconditioner.structureDetector import structure_detector
        from edi.preconditioner.unitCorrector import unit_corrector
        st = structure_detector(unit_corrector(model))

    if x0 is None:
        x0 = [float(pyo.value(v)) for v in st['variables']]
    log, n_original = None, len(st.get('variables') or [])
    if presolve:
        st, x0, log, n_original = _apply_presolve(st, x0)

    problem = build_problem(st, sp_form=True)
    x0 = np.asarray(x0, dtype=float)
    if len(x0) < problem.n:
        x0 = np.concatenate([x0, np.ones(problem.n - len(x0))])
    x0 = np.where(x0 > 0, x0, 1.0)

    options = options or SIAOptions()
    text, x1, slacks = explain_infeasibility(problem, x0[:problem.n],
                                             options=options, k=top)

    tol = options.feasibility_tolerance
    slacks = np.asarray(slacks, dtype=float)
    order = np.argsort(-slacks)
    blocking = []
    for i in order:
        if slacks[i] <= tol:
            break
        con = problem.constraints[i]
        body = getattr(con, 'body', None)
        terms = list(getattr(body, 'terms', None) or [])
        for side in ('p', 'q'):
            sub = getattr(body, side, None)
            if sub is not None:
                terms.extend(getattr(sub, 'terms', None) or [])
        involved = sorted({j for _c, a in terms
                           for j, e in enumerate(a) if e != 0.0})
        blocking.append((int(i), float(slacks[i]),
                         [problem.names[j] for j in involved
                          if j < len(problem.names)]))

    # Put the presolved-away columns back, so `.x` is indexed like the model.
    restored = _restore(type('R', (), {'x': np.asarray(x1, dtype=float)})(),
                        log, n_original)
    return FeasibilityResult(
        feasible=not blocking,
        x=np.asarray(getattr(restored, 'x', x1), dtype=float),
        slacks=slacks,
        blocking=blocking,
        iterations=0,
        names=list(problem.names or []),
        text=text,
    )
