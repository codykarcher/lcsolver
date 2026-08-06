#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Constraints that must hold but must not bind, and the check that they didn't.

A holographic constraint is one that is not part of the design problem. It is
there to keep the problem well posed, or to mark where the model stops being a
model:

* the 1e-30..1e30 box that stops a variable running to zero or to infinity;
* the edges of the data a fit was made from;
* any limit that means "beyond here I am extrapolating".

The distinction matters because an *active* one silently invalidates the
answer. A solve that ends on the edge of a fit's validity has not found an
optimum; it has told you it wanted to go somewhere you have no data for, and
the number it returned is whatever the fit happened to extrapolate to out
there. Nothing about that looks wrong: the solve converges, the duals are
finite, the table prints. The only way to know is to have said in advance
which constraints were never supposed to bind, and then to look.

So this check runs on **every** solve rather than only when someone asks for a
diagnostic. It costs one expression evaluation per declared constraint.
"""
from __future__ import annotations

__all__ = ["holographic_report", "format_holographic"]


def _margin(body, lo, hi):
    """How much room is left, relative to the scale of the numbers involved.

    Absolute slack is meaningless across a model whose variables span thirty
    orders of magnitude, so the margin is normalised by the larger of the
    bound and the body.
    """
    out = []
    for bound, side in ((lo, '>='), (hi, '<=')):
        if bound is None:
            continue
        scale = max(abs(bound), abs(body), 1e-300)
        slack = (body - bound) if side == '>=' else (bound - body)
        out.append((side, bound, slack / scale))
    return out


def holographic_report(model, rtol=1e-6):
    """Which holographic constraints are active at the model's current point.

    Returns ``[{name, operator, bound, value, margin}]`` for the ones that are
    binding, worst (most negative margin) first. An empty list is the good
    outcome and the common one.

    Reads the values currently on the model, so it means what it says only
    after a solve has written them back -- which is why `solve` is what calls
    it.
    """
    import pyomo.environ as pyo

    names = getattr(model, '_holographic', None)
    if not names:
        return []

    found = []
    for nm in sorted(names):
        con = getattr(model, nm, None)
        if con is None:
            continue
        try:
            datas = list(con.values()) if hasattr(con, 'values') else [con]
        except Exception:
            datas = [con]
        for cd in datas:
            try:
                body = float(pyo.value(cd.body))
                lo = None if cd.lower is None else float(pyo.value(cd.lower))
                hi = None if cd.upper is None else float(pyo.value(cd.upper))
            except Exception:
                continue                      # never fail a solve over a check
            if lo is not None and hi is not None and lo == hi:
                # An equality is always binding. A holographic equality is a
                # contradiction in terms, so say so rather than list it every
                # time as though it were news.
                found.append({'name': getattr(cd, 'name', nm),
                              'operator': '==', 'bound': hi, 'value': body,
                              'margin': 0.0, 'equality': True})
                continue
            for side, bound, margin in _margin(body, lo, hi):
                if margin <= rtol:
                    found.append({'name': getattr(cd, 'name', nm),
                                  'operator': side, 'bound': bound,
                                  'value': body, 'margin': margin,
                                  'equality': False})
    found.sort(key=lambda d: d['margin'])
    return found


def format_holographic(active, total=None, k=8):
    """The report, as a string. Empty when nothing is active."""
    if not active:
        return ''
    n = len(active)
    L = ['holographic constraints', '-----------------------',
         f'  {n} of {total if total is not None else n} are ACTIVE at the '
         f'solution.',
         '  These were declared as limits that should not bind, so the answer',
         '  is sitting on one of them. It is not an optimum of the problem you',
         '  meant to pose -- it is where the solver stopped because it was not',
         '  allowed to go further.']
    for d in active[:k]:
        if d.get('equality'):
            L.append(f"    {d['name']}: declared holographic but is an "
                     f"EQUALITY, so it always binds")
            continue
        L.append(f"    {d['name']}: {d['value']:.6g} {d['operator']} "
                 f"{d['bound']:.6g}   (margin {d['margin']:+.2e})")
    if n > k:
        L.append(f'    ... and {n - k} more')
    return '\n'.join(L)
