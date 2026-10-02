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


def _is_variable(expr):
    """True when an expression contains a decision variable (not a number, a
    unit or a Constant)."""
    try:
        return bool(expr.is_potentially_variable())
    except Exception:
        return False


def _in_units_of(value, expr, target):
    """`value`, the number pyo.value gave for `expr`, converted into the units
    of `target`. pyo.value DROPS units rather than converting them, so 2.05
    for "4.1 in / 2" must become 0.052 before it is compared with metres.
    Returns the value unchanged when either side is dimensionless or the
    units cannot be converted: never fail a solve over a check."""
    try:
        from pyomo.environ import units as u
        frm, to = u.get_units(expr), u.get_units(target)
        if frm is None or to is None:
            return value
        return float(u.convert_value(value, from_units=frm, to_units=to))
    except Exception:
        return value


def _pairs(cd):
    """The inequalities behind one constraint data, as (lesser, greater)
    expression pairs, read from the relation AS WRITTEN. pyomo also offers
    cd.body / cd.lower / cd.upper, but for a row with variables on both
    sides those are "lhs - rhs" against a bound of 0, where a relative
    margin is +-1 by roundoff sign and means nothing. Returns None when the
    relation is not a plain or ranged inequality."""
    e = getattr(cd, 'expr', None)
    args = getattr(e, 'args', None)
    name = type(e).__name__
    if args is None or 'Equality' in name:
        return None
    if len(args) == 2:
        return [(args[0], args[1])]
    if len(args) == 3:
        return [(args[0], args[1]), (args[1], args[2])]
    return None


def _check(lesser, greater):
    """(operator, bound, value, margin) for one `lesser <= greater`.

    The side holding the variables is the value; the other is the bound,
    converted into the value's units. With variables on both sides the
    lesser side is the value. The margin is the room left over the larger
    of the two, so it is relative and unit-free."""
    import pyomo.environ as pyo
    lo, hi = float(pyo.value(lesser)), float(pyo.value(greater))
    if _is_variable(lesser) or not _is_variable(greater):
        value, bound, side = lo, _in_units_of(hi, greater, lesser), '<='
        slack = bound - value
    else:
        value, bound, side = hi, _in_units_of(lo, lesser, greater), '>='
        slack = value - bound
    return side, bound, value, slack / max(abs(bound), abs(value), 1e-300)


def holographic_report(model, rtol=1e-6, vtol=1e-4):
    """Which holographic constraints are active at the model's current point.

    Returns [{name, operator, bound, value, margin, violated}] for the
    binding ones, worst first; value and bound are in the units of the
    variable side. A margin below -vtol is not "active" but VIOLATED: the
    point breaks a limit that was declared to hold, and is reported as such.
    Reads values off the model, so only meaningful after a solve writes
    back; solve is what calls it.
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
            # symbolic body so the report can say WHICH relation binds
            try:
                expr = str(cd.expr)
            except Exception:
                expr = ''
            if len(expr) > 72:
                expr = expr[:69] + '...'
            try:
                pairs = _pairs(cd)
                if pairs is not None:
                    checks = [_check(a, b) for a, b in pairs]
                else:
                    body = float(pyo.value(cd.body))
                    lo = None if cd.lower is None else float(pyo.value(cd.lower))
                    hi = None if cd.upper is None else float(pyo.value(cd.upper))
                    if lo is not None and hi is not None and lo == hi:
                        # a holographic equality always binds; say so rather than report it as news
                        found.append({'name': getattr(cd, 'name', nm),
                                      'operator': '==', 'bound': hi, 'value': body,
                                      'margin': 0.0, 'equality': True,
                                      'violated': False, 'expr': expr})
                        continue
                    checks = [(side, bound, body, margin)
                              for side, bound, margin in _margin(body, lo, hi)]
            except Exception:
                continue                      # never fail a solve over a check
            for side, bound, value, margin in checks:
                if margin <= rtol:
                    found.append({'name': getattr(cd, 'name', nm),
                                  'operator': side, 'bound': bound,
                                  'value': value, 'margin': margin,
                                  'equality': False,
                                  'violated': margin < -vtol, 'expr': expr})
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
                 f"{d['bound']:.6g}   (margin {d['margin']:+.2e})"
                 + ("   VIOLATED, not merely active" if d.get('violated') else ""))
        if d.get('expr'):
            L.append(f"        {d['expr']}")
    if n > k:
        L.append(f'    ... and {n - k} more')
    return '\n'.join(L)
