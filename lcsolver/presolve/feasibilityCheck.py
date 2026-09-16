#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Find a feasible point, or name the constraints that prevent one.

Exposes the elastic Phase I that already lived inside the SIA solver. The L1
form is the right one: its optimum is sparse (rows that can be satisfied drop
to zero slack, leaving an approximate irreducible inconsistent subsystem),
and on success it hands back a strictly feasible point. The result is
passable::

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

    # True when every constraint is satisfied to feasibility_tolerance
    feasible: bool = False
    # the point reached, ordered like structures['variables']; the
    # least-infeasible point when not feasible (still a better start)
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # per-constraint slack, zero where the row is satisfied
    slacks: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # [(row_index, slack, [variable names])] for rows keeping a positive
    # slack, largest first; empty when feasible
    blocking: list = field(default_factory=list)
    iterations: int = 0
    names: list = field(default_factory=list)
    text: str = ''

    def summary(self, top=12) -> str:
        """The report, as a string. ``print(result.summary())``.

        Includes what a bare verdict lacks: rows examined, worst slack, and
        which variables the blocking rows share (the usual error site).
        """
        L = ['feasibility', '-----------']
        n = len(self.slacks)
        if self.feasible:
            L.append(f'  FEASIBLE: a point satisfying all {n} constraints was '
                     f'found in {self.iterations} elastic iteration'
                     f'{"s" if self.iterations != 1 else ""}.')
            L.append('  (That count is the presolved problem, which is what '
                     'was actually searched.)')
            L.append('  It is in `.x`, indexed like the model, and can be '
                     'handed to solve(f, start=result).')
            L.append('  This ignored the objective entirely: the point is '
                     'feasible, not optimal.')
            if n:
                # clamped: slacks are >= 0 by construction, a small negative
                # is interior-point round-off
                worst = max(0.0, float(max(self.slacks)))
                L.append(f'  Worst remaining slack {worst:.3e}, within '
                         f'tolerance.')
            return '\n'.join(L)

        tot = float(sum(s for _r, s, _v in self.blocking))
        L.append(f'  INFEASIBLE: {len(self.blocking)} of {n} constraints keep a '
                 f'positive slack (sum {tot:.4e}).')
        L.append('  These must be relaxed for the model to close. The elastic '
                 'form drives every row it CAN')
        L.append('  satisfy to zero slack, so what is listed is close to an '
                 'irreducible inconsistent set,')
        L.append('  not merely the worst offenders.')
        L.append('')
        L.append('  largest first:')
        for row, slack, variables in self.blocking[:top]:
            names = ', '.join(variables[:6]) or '-'
            more = '' if len(variables) <= 6 else f' (+{len(variables) - 6})'
            L.append(f'      row {row:5d}  slack {slack:.4e}   vars: {names}{more}')
        if len(self.blocking) > top:
            L.append(f'      ... and {len(self.blocking) - top} more')

        shared = None
        for _r, _s, variables in self.blocking:
            shared = set(variables) if shared is None else shared & set(variables)
        if shared:
            L.append('')
            L.append(f'  Every blocking row involves: {", ".join(sorted(shared))}')
            L.append('  -- a variable common to all of them is the usual place '
                     'the error is.')
        L.append('')
        L.append('  `.x` holds the least-infeasible point found, which is '
                 'still generally a better')
        L.append('  starting point than the declared guesses.')
        return '\n'.join(L)

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
        from lcsolver.presolve.reductions import as_structures
        from lcsolver.postsolve.writeback import write_solution

        st = as_structures(model)
        write_solution(st, {'x': np.asarray(self.x, dtype=float)}, model=model)
        return model


def feasibility(model, x0=None, options=None, top=12, presolve=True):
    """Find a feasible point for ``model``, or say what prevents one.

    Accepts a Formulation or an already detected structure. Runs the elastic
    Phase I: min sum(s_i) s.t. log g_i(x) <= s_i, s_i >= 0. Returns a
    FeasibilityResult -- truthy when feasible, point in .x, passable to
    solve(f, start=result). x0 defaults to the model's current values; top
    caps the blocking rows listed. Ignores the objective entirely: the point
    is feasible, not optimal.
    """
    import pyomo.environ as pyo

    from lcsolver.presolve.reductions import as_structures
    from lcsolver.solvers.sequential.sia import SIAOptions, explain_infeasibility
    from lcsolver.solvers.sequential.bridge import (apply_presolve, restore_presolved,
                                               build_problem)

    st = as_structures(model)
    if st.get('bounds') is not None:
        # Phase I reads rows, and this form split bounds out of them; detect
        # afresh rather than search a problem with no variable bounds
        from lcsolver.presolve.structureDetector import structure_detector
        from lcsolver.presolve.unitCorrector import unit_corrector
        st = structure_detector(unit_corrector(model))

    if x0 is None:
        x0 = [float(pyo.value(v)) for v in st['variables']]
    log, n_original = None, len(st.get('variables') or [])
    if presolve:
        st, x0, log, n_original = apply_presolve(st, x0)

    problem = build_problem(st, sp_form=True)
    x0 = np.asarray(x0, dtype=float)
    if len(x0) < problem.n:
        x0 = np.concatenate([x0, np.ones(problem.n - len(x0))])
    x0 = np.where(x0 > 0, x0, 1.0)

    options = options or SIAOptions()
    text, x1, slacks = explain_infeasibility(problem, x0[:problem.n],
                                             options=options, k=top)
    # pull the iteration count back out of explain_infeasibility's prose
    import re as _re
    _m = _re.search(r'in (\d+) iterations|after (\d+) elastic', text)
    n_iter = int(next(g for g in (_m.groups() if _m else ()) if g)) if _m else 0

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
    restored = restore_presolved(type('R', (), {'x': np.asarray(x1, dtype=float)})(),
                        log, n_original)
    return FeasibilityResult(
        feasible=not blocking,
        x=np.asarray(getattr(restored, 'x', x1), dtype=float),
        slacks=slacks,
        blocking=blocking,
        iterations=n_iter,
        names=list(problem.names or []),
        text=text,
    )
