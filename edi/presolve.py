#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Structural checks on a detected formulation, before anything is solved.

Two kinds of problem show up here, and they need different tools.

**Structural** defects are visible in the sparsity pattern alone: a variable
that appears in no constraint, one that appears in only one, a "constraint"
that is really a bound, a variable nothing can push down on. These are the
classical presolve reductions (Andersen & Andersen; Achterberg et al.,
*Presolve Reductions in Mixed Integer Programming*) and they cost nothing to
find. Most of them are modelling errors -- a variable nothing determines is
usually a constraint somebody forgot to write.

**Solution-dependent** defects are not. A variable can appear in half a dozen
constraints, all of them perfectly ordinary, and still be undetermined at the
optimum because every one of those constraints goes slack there. Nothing in
the sparsity pattern says so. For those see :func:`degeneracy_report`, which
runs *after* a solve and tests the definition directly.

Measured on SPaircraft: of 29 variables the optimum does not determine, the
structural checks find 7. The other 22 need the post-solve test. Both are
worth running.

Boundedness
-----------
The check gpkit prints as "x is not upper bounded". In log space a posynomial
constraint ``sum_k c_k prod_j x_j^a_jk <= 1`` bounds ``x_j`` from **above**
through any term with ``a_jk > 0`` (pushing ``x_j`` up pushes the constraint
toward violation) and from **below** through any term with ``a_jk < 0``. An
equality bounds both ways. For a ratio ``p/q <= 1`` the numerator counts as
written and the denominator counts negated, since growing ``q`` relaxes it.

A variable bounded on only one side is not necessarily wrong -- plenty of
quantities only need a floor -- but an unbounded direction is where an
optimizer runs away, and it is worth seeing the list.

A bound counts only if it says something. EDI gives every variable a default
1e-30..1e30 box, so counting bounds naively would pronounce everything bounded
both ways and report nothing at all. But throwing out every single-variable
row goes too far the other way: a hand-written ``w >= 1`` is a real modelling
statement and genuinely does bound ``w`` below.

So the test is on the **value**, not the shape: a bound counts unless it is
vacuous (see :data:`VACUOUS_LO` / :data:`VACUOUS_HI`). "x is not upper
bounded" then means *nothing in this model holds x down except a limit chosen
to be no limit at all*, which is the statement worth printing -- and is what
gpkit's ``Bounded`` reports about the box it adds.

The objective counts. Minimising a term with a positive exponent on ``x``
pushes ``x`` down and so bounds it above, exactly as a constraint would.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field

__all__ = ["PresolveReport", "presolve_report", "degeneracy_report",
           "VACUOUS_LO", "VACUOUS_HI"]

#: A bound at or beyond these is treated as no bound at all. EDI's default box
#: is exactly 1e-30..1e30, and models routinely restate it; either way it was
#: chosen to keep the solver in positive territory, not to say anything about
#: the design, so it must not count as bounding.
VACUOUS_LO = 1e-29
VACUOUS_HI = 1e29


def _bound_from_row(coeff, expo, j, op):
    """Recover the numeric bound a single-variable monomial row imposes.

    The row is ``c * x_j**a <= 1`` (or ``== 1``), so ``x_j**a <= 1/c`` and the
    bound is ``(1/c)**(1/a)`` -- an upper bound when ``a > 0``, a lower bound
    when ``a < 0``, and both when the operator is an equality.

    Returns ``(lo, hi)``, either possibly None.
    """
    a = expo[j]
    if coeff <= 0.0 or a == 0.0:
        return None, None
    try:
        val = (1.0 / coeff) ** (1.0 / a)
    except (OverflowError, ValueError, ZeroDivisionError):
        return None, None
    if op == "==":
        return val, val
    return (None, val) if a > 0 else (val, None)


@dataclass
class PresolveReport:
    """What the structural checks found. ``print(report)`` for the summary."""

    n_variables: int = 0
    n_rows: int = 0
    names: list = field(default_factory=list)

    empty_columns: list = field(default_factory=list)
    singleton_columns: list = field(default_factory=list)
    bound_only_columns: list = field(default_factory=list)
    fixed_columns: list = field(default_factory=list)
    bounds: dict = field(default_factory=dict)
    unbounded_above: list = field(default_factory=list)
    unbounded_below: list = field(default_factory=list)
    singleton_rows: list = field(default_factory=list)
    duplicate_rows: int = 0
    row_counts: dict = field(default_factory=dict)

    @property
    def clean(self):
        return not (self.empty_columns or self.unbounded_above
                    or self.unbounded_below)

    def __str__(self):
        L = [f"presolve: {self.n_variables} variables, {self.n_rows} rows"]

        if self.singleton_rows:
            pct = 100.0 * len(self.singleton_rows) / max(self.n_rows, 1)
            L.append(f"  {len(self.singleton_rows)} rows ({pct:.0f}%) are "
                     "single-variable bounds and could be folded into "
                     "variable bounds")
        if self.duplicate_rows:
            L.append(f"  {self.duplicate_rows} rows duplicate another row's "
                     "variable pattern")

        if self.fixed_columns:
            L.append(f"  {len(self.fixed_columns)} variables are fixed by "
                     "equal bounds and could be substituted out")

        for label, names in (("appears in no constraint", self.empty_columns),
                             ("appears in only one constraint",
                              self.singleton_columns),
                             ("is held only by its own bounds",
                              self.bound_only_columns)):
            if names:
                L.append(f"  {len(names)} variables: {label}")
                for nm in names[:12]:
                    L.append(f"    {nm}")
                if len(names) > 12:
                    L.append(f"    ... and {len(names) - 12} more")

        for nm in self.unbounded_above[:12]:
            L.append(f"  {nm} is not upper bounded")
        if len(self.unbounded_above) > 12:
            L.append(f"  ... and {len(self.unbounded_above) - 12} more not "
                     "upper bounded")
        for nm in self.unbounded_below[:12]:
            L.append(f"  {nm} is not lower bounded")
        if len(self.unbounded_below) > 12:
            L.append(f"  ... and {len(self.unbounded_below) - 12} more not "
                     "lower bounded")

        if self.clean:
            L.append("  no empty columns and every variable is bounded "
                     "both ways")
        return "\n".join(L)


def nm_at(names, j):
    return names[j] if j < len(names) else f"<var {j}>"


def _rows_of(structures):
    """``(rows, operators, key)`` for whichever structure was detected."""
    for key in ("Signomial_Program", "Geometric_Program"):
        if structures.get(key, (False,))[0]:
            return structures[key][1], structures[key][2], key
    raise ValueError("presolve needs a detected GP or SP structure")


def presolve_report(structures, names=None) -> PresolveReport:
    """Run the structural checks. Nothing is solved and nothing is modified."""
    rows, operators, _ = _rows_of(structures)
    if names is None:
        names = [str(v) for v in structures.get("variables", [])]

    # Group rows by constraint, splitting numerator from denominator. Index 0
    # is the objective and is excluded -- being in the objective is not a
    # constraint, though it does mean the variable is not free.
    numer, denom = collections.defaultdict(list), collections.defaultdict(list)
    width = 0
    for r in rows:
        idx = int(r[0])
        coeff, expo = float(r[1]), [float(e) for e in r[2:]]
        width = max(width, len(expo))
        (numer if idx >= 0 else denom)[
            idx if idx >= 0 else -idx - 1].append((coeff, expo))

    n = max(width, len(names))
    con_idx = sorted(k for k in set(numer) | set(denom) if k != 0)

    rep = PresolveReport(n_variables=n, n_rows=len(con_idx),
                         names=list(names))

    in_rows = [set() for _ in range(n)]      # rows the variable appears in
    in_real = [set() for _ in range(n)]      # ... that are not plain bounds
    upper = [False] * n
    lower = [False] * n
    patterns = collections.Counter()

    # The tightest bound seen on each variable, from any source: a declared
    # box, or a single-variable row. Collected numerically so that a vacuous
    # limit can be told from a real one.
    box_lo = [None] * n
    box_hi = [None] * n

    def note_bound(j, lo, hi):
        if lo is not None:
            box_lo[j] = lo if box_lo[j] is None else max(box_lo[j], lo)
        if hi is not None:
            box_hi[j] = hi if box_hi[j] is None else min(box_hi[j], hi)

    # Bounds the detector split out (bounds_as_rows=False) never appear as
    # rows, so pick them up here.
    for j, pair in enumerate(structures.get("bounds") or []):
        if j < n and pair is not None:
            note_bound(j, pair[0], pair[1])

    # Minimising c * prod x^a pushes a positive-exponent variable down, so the
    # objective bounds it above just as a constraint would.
    for _, expo in numer.get(0, []):
        for j, e in enumerate(expo):
            if e > 1e-12:
                upper[j] = True
            elif e < -1e-12:
                lower[j] = True

    for i in con_idx:
        op = operators[i - 1] if 0 <= i - 1 < len(operators) else "<="
        # numerator counts as written; denominator counts negated, since
        # growing the denominator relaxes p/q <= 1
        signed = ([(c, expo, +1.0) for c, expo in numer.get(i, [])]
                  + [(c, expo, -1.0) for c, expo in denom.get(i, [])])
        touched = set()
        for _, expo, _s in signed:
            touched |= {j for j, e in enumerate(expo) if abs(e) > 1e-12}

        n_terms = len(signed)
        is_bound = (len(touched) == 1 and n_terms == 1 and not denom.get(i))

        if is_bound:
            # Record the value rather than the fact. Whether it bounds anything
            # is decided below, once we can see how big it is.
            j = next(iter(touched))
            coeff, expo = numer[i][0]
            note_bound(j, *_bound_from_row(coeff, expo, j, op))
            rep.singleton_rows.append(i)
        else:
            for coeff, expo, sgn in signed:
                for j, e in enumerate(expo):
                    if abs(e) <= 1e-12:
                        continue
                    if op == "==":
                        upper[j] = lower[j] = True
                    elif sgn * e > 0:
                        upper[j] = True
                    else:
                        lower[j] = True
            patterns[frozenset(touched)] += 1

        for j in touched:
            in_rows[j].add(i)
            if not is_bound:
                in_real[j].add(i)

    # A bound counts only if it is not vacuous.
    for j in range(n):
        if box_lo[j] is not None and box_lo[j] > VACUOUS_LO:
            lower[j] = True
        if box_hi[j] is not None and box_hi[j] < VACUOUS_HI:
            upper[j] = True
        if box_lo[j] is not None or box_hi[j] is not None:
            rep.bounds[nm_at(names, j)] = (box_lo[j], box_hi[j])
        if (box_lo[j] is not None and box_hi[j] is not None
                and box_hi[j] <= box_lo[j] * (1.0 + 1e-9)):
            rep.fixed_columns.append(nm_at(names, j))

    rep.duplicate_rows = sum(c - 1 for c in patterns.values() if c > 1)

    def nm(j):
        return nm_at(names, j)

    # Counted in real constraints, not bound rows -- a variable that appears
    # in one genuine constraint plus its own box is a singleton column, and
    # counting the box would hide it.
    for j in range(n):
        if not in_rows[j]:
            rep.empty_columns.append(nm(j))
        elif not in_real[j]:
            rep.bound_only_columns.append(nm(j))
        elif len(in_real[j]) == 1:
            rep.singleton_columns.append(nm(j))
        if not upper[j]:
            rep.unbounded_above.append(nm(j))
        if not lower[j]:
            rep.unbounded_below.append(nm(j))

    hist = collections.Counter(len(s) for s in in_real)
    rep.row_counts = dict(sorted(hist.items()))
    return rep


def degeneracy_report(problem, x, rel_step=0.05, obj_tol=1e-9,
                      viol_tol=1e-9, names=None):
    """Variables the solution does not determine, tested directly.

    A variable is degenerate when it can be moved in **both** directions
    without changing the objective and without worsening feasibility. That is
    the definition, so there is no heuristic to be wrong about -- which
    matters, because the obvious structural heuristic ("only bounds are active
    on it") also flags every variable pinned by a single-variable *equality*,
    and those are maximally determined rather than free.

    ``problem`` is an :class:`~edi.solvers.ipopt.slcp.Problem`; ``x`` the
    solution. Returns a list of ``(name, value)``.
    """
    import math

    import numpy as np

    x = np.asarray(x, dtype=float)
    names = list(names or [])

    def viol(xx):
        worst = -math.inf
        for c in problem.constraints:
            worst = max(worst, math.log(max(c.body(xx), 1e-300)))
        return worst

    f0, v0 = problem.objective_value(x), viol(x)
    out = []
    for j in range(problem.n):
        free = True
        for s in (rel_step, -rel_step):
            xp = x.copy()
            xp[j] *= math.exp(s)
            if (abs(problem.objective_value(xp) - f0)
                    > obj_tol * max(1.0, abs(f0))
                    or viol(xp) > max(v0, 0.0) + viol_tol):
                free = False
                break
        if free:
            out.append((names[j] if j < len(names) else f"<var {j}>",
                        float(x[j])))
    return out
