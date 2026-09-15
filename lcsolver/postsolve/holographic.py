#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Constraints that must hold but must not bind, and the check that they didn't.

A holographic constraint is not part of the design problem: the 1e-30..1e30
box that stops a variable running away, the edges of the data a fit was made
from, any limit meaning "beyond here I am extrapolating".

An active one silently invalidates the answer: the solve converges, the duals
are finite, but the solver wanted to go where there is no data. So this check
runs on every solve; it costs one expression evaluation per declared constraint.
"""
from __future__ import annotations

__all__ = ["holographic_report", "holographic_total", "format_holographic"]


def _datas(con):
    """The constraint data objects behind a declared name.
    A vector constraint is several datas; the count and the walk must agree."""
    try:
        return list(con.values()) if hasattr(con, 'values') else [con]
    except Exception:
        return [con]


def _margin(body, lo, hi):
    """Room left, normalised by the larger of bound and body.
    Absolute slack is meaningless across 30 orders of magnitude."""
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

    Returns [{name, operator, bound, value, margin}] for the binding ones,
    worst first. Reads values off the model, so only meaningful after a
    solve writes back; solve is what calls it.
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
        for cd in _datas(con):
            try:
                body = float(pyo.value(cd.body))
                lo = None if cd.lower is None else float(pyo.value(cd.lower))
                hi = None if cd.upper is None else float(pyo.value(cd.upper))
            except Exception:
                continue                      # never fail a solve over a check
            # symbolic body so the report can say WHICH relation binds
            try:
                expr = str(cd.expr)
            except Exception:
                expr = ''
            if len(expr) > 72:
                expr = expr[:69] + '...'
            if lo is not None and hi is not None and lo == hi:
                # a holographic equality always binds; say so rather than report it as news
                found.append({'name': getattr(cd, 'name', nm),
                              'operator': '==', 'bound': hi, 'value': body,
                              'margin': 0.0, 'equality': True, 'expr': expr})
                continue
            for side, bound, margin in _margin(body, lo, hi):
                if margin <= rtol:
                    found.append({'name': getattr(cd, 'name', nm),
                                  'operator': side, 'bound': bound,
                                  'value': body, 'margin': margin,
                                  'equality': False, 'expr': expr})
    found.sort(key=lambda d: d['margin'])
    return found


def holographic_total(model):
    """How many holographic constraints the model declares.
    Counted the same way holographic_report walks them, so "n of N" is a true fraction."""
    names = getattr(model, '_holographic', None)
    if not names:
        return 0
    total = 0
    for nm in names:
        con = getattr(model, nm, None)
        if con is not None:
            total += len(_datas(con))
    return total


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
            if d.get('expr'):
                L.append(f"        {d['expr']}")
            continue
        L.append(f"    {d['name']}: {d['value']:.6g} {d['operator']} "
                 f"{d['bound']:.6g}   (margin {d['margin']:+.2e})")
        if d.get('expr'):
            L.append(f"        {d['expr']}")
    if n > k:
        L.append(f'    ... and {n - k} more')
    return '\n'.join(L)
