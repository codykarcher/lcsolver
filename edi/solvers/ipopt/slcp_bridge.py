#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Run a detected GP/SP through the SLCP solver.

``edi.solvers.ipopt.slcp`` implements sequential log-convex programming over
its own :class:`~edi.solvers.ipopt.slcp.Problem` object, which nothing in EDI
built. This module is the missing adapter: it turns the row form produced by
:func:`~edi.structure.structureDetector.structure_detector` into that object,
so the same formulation can be solved either way and the two compared.

The row form is already close to what SLCP wants. Rows carry
``[constraint_index, coefficient, *exponents]``; index ``i >= 1`` is
constraint *i*'s numerator, index ``-i-1`` its denominator, and index 0 is the
objective. Each group maps as:

* numerator only, all coefficients positive -> ``Posynomial``, imposed exactly
  in log space;
* numerator and denominator -> ``PosynomialRatio``, which keeps the numerator
  exact and condenses only the denominator by the arithmetic-geometric-mean
  inequality. This is the classical signomial-program treatment and is
  strictly less lossy than linearizing the whole body;
* a single-term numerator with ``==`` -> a monomial equality, which is affine
  in log space and imposed exactly.

A group with a negative coefficient cannot be represented: SLCP's
``Posynomial`` requires positive coefficients, and the detector should already
have moved negative terms into a denominator. Such a group is reported rather
than silently dropped.
"""

import numpy as np

from edi.solvers.ipopt.slcp import (Constraint, Options, Posynomial,
                                    PosynomialRatio, Problem, Signomial,
                                    solve as _slcp_solve)


def _group(rows):
    """Group rows by constraint index, splitting numerator from denominator."""
    numerator, denominator = {}, {}
    for r in rows:
        idx = int(r[0])
        term = (float(r[1]), [float(e) for e in r[2:]])
        if idx >= 0:
            numerator.setdefault(idx, []).append(term)
        else:
            denominator.setdefault(-idx - 1, []).append(term)
    return numerator, denominator


def build_problem(structures, sp_form=True):
    """Translate a detected structure into an SLCP :class:`Problem`.

    ``sp_form`` selects how a signomial constraint ``p/q <= 1`` is handled:

    ``True`` (default)
        Build a :class:`~edi.solvers.ipopt.slcp.PosynomialRatio`, which keeps
        ``p`` exact in log space and condenses only ``q`` by the AGM
        inequality. Less approximation, and conservative.
    ``False``
        Build a plain :class:`~edi.solvers.ipopt.slcp.Signomial` -- a
        value/gradient callback over the same ratio -- which SLCP then
        linearizes whole, discarding ``p``'s log-convexity along with ``q``'s
        curvature. This is stock SLCP as the paper describes it, and is the
        setting to use when comparing against it.

    Turning it off also loses sub-problem caching for those constraints: a
    linearization moves every iteration, so there is nothing to cache.
    """
    key = ('Signomial_Program' if structures['Signomial_Program'][0]
           else 'Geometric_Program')
    if not structures[key][0]:
        raise ValueError('structure is neither a GP nor an SP')
    rows, operators = structures[key][1], structures[key][2]
    numerator, denominator = _group(rows)

    if 0 not in numerator:
        raise ValueError('no objective rows found in the detected structure')
    n = max(len(a) for terms in numerator.values() for _, a in terms)

    def pad(a):
        return list(a) + [0.0] * (n - len(a))

    def posynomial(terms, what):
        bad = [c for c, _ in terms if c <= 0]
        if bad:
            raise ValueError(
                f'{what} has non-positive coefficients {bad}; SLCP needs each '
                'posynomial part to be positive, so the detector should have '
                'moved these into a denominator')
        return Posynomial([(c, pad(a)) for c, a in terms], n)

    objective = posynomial(numerator[0], 'objective')

    constraints = []
    for idx in sorted(k for k in numerator if k != 0):
        op = operators[idx - 1] if (idx - 1) < len(operators) else '<='
        num = numerator[idx]
        den = denominator.get(idx)
        if den:
            ratio = PosynomialRatio(
                posynomial(num, f'constraint {idx} numerator'),
                posynomial(den, f'constraint {idx} denominator'), n)
            if sp_form:
                body = ratio
            else:
                # Hand the same ratio over as an opaque value/gradient
                # callback, so SLCP linearizes the whole body instead of
                # keeping the numerator exact.
                body = Signomial(
                    lambda x, r=ratio: (r(x), r.grad(x)), n)
            # An equality over a ratio is not representable: the AGM
            # condensation is one-sided, so it is only conservative for '<='.
            constraints.append(Constraint(body, '<='))
        else:
            body = posynomial(num, f'constraint {idx}')
            if op == '==' and body.is_monomial:
                constraints.append(Constraint(body, '=='))
            else:
                constraints.append(Constraint(body, '<='))

    return Problem(n, objective, constraints)


def solve_slcp(structures, x0=None, method='slcp', options=None,
               sp_form=True):
    """Solve a detected GP/SP with SLCP.

    ``x0`` is in the natural (not log) variables and must be strictly
    positive; it defaults to the current values of ``structures['variables']``.
    Returns the SLCP :class:`~edi.solvers.ipopt.slcp.Result`.
    """
    import pyomo.environ as pyo

    problem = build_problem(structures, sp_form=sp_form)
    if x0 is None:
        x0 = [float(pyo.value(v)) for v in structures['variables']]
    x0 = np.asarray(x0, dtype=float)
    if len(x0) < problem.n:
        # PCCP-style slack columns, if any, start at 1.
        x0 = np.concatenate([x0, np.ones(problem.n - len(x0))])
    x0 = np.where(x0 > 0, x0, 1.0)
    return _slcp_solve(problem, x0[:problem.n], method=method, options=options)


def solve_sia(structures, x0=None, options=None, sp_form=True):
    """Solve a detected GP/SP by sequential inner approximation.

    Same adapter as :func:`solve_slcp`, pointed at
    :func:`edi.solvers.ipopt.sia.solve_sia`. ``sp_form`` defaults to True here
    and should stay that way -- the conservative condensation is the whole
    basis of the method, and turning it off downgrades every signomial
    constraint to a linearization that then has to be globalized.
    """
    import pyomo.environ as pyo

    from edi.solvers.ipopt.sia import solve_sia as _sia_solve

    problem = build_problem(structures, sp_form=sp_form)
    if x0 is None:
        x0 = [float(pyo.value(v)) for v in structures['variables']]
    x0 = np.asarray(x0, dtype=float)
    if len(x0) < problem.n:
        x0 = np.concatenate([x0, np.ones(problem.n - len(x0))])
    x0 = np.where(x0 > 0, x0, 1.0)
    return _sia_solve(problem, x0[:problem.n], options=options)
