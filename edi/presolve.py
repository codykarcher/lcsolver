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

from edi.structure.detected import Term, as_detected

class InfeasibleProblem(ValueError):
    """Presolve proved the model infeasible before any solve was attempted."""


__all__ = ["PresolveReport", "presolve_report", "degeneracy_report",
           "InfeasibleProblem",
           "cancellation_report", "fold_singleton_rows",
           "reduce_columns", "restore_columns", "Removed",
           "propagate_bounds", "eliminate_monomial_equalities",
           "presolve", "PresolveLog",
           "evaluate", "equivalence_error", "assert_equivalent",
           "diagnose",
           "VACUOUS_LO", "VACUOUS_HI"]

#: A bound at or beyond these is treated as no bound at all. EDI's default box
#: is exactly 1e-30..1e30, and models routinely restate it; either way it was
#: chosen to keep the solver in positive territory, not to say anything about
#: the design, so it must not count as bounding.
VACUOUS_LO = 1e-29
VACUOUS_HI = 1e29


def _check_width(structures, n, who):
    """Rows and the variable list must describe the same number of columns.

    Both column-removing passes used to write

        [variables[j] for j in keep if j < len(variables)]

    which silently drops the tail when the rows are wider than the declared
    variable list, leaving rows and `variables` describing different problems
    and compounding on the next pass. A guard is better than a mask: if this
    ever fires, whatever produced the structure is at fault and should be
    fixed there.
    """
    have = structures.get("variables")
    if have is not None and 0 < len(have) < n:
        raise ValueError(
            f"{who}: the rows span {n} columns but only {len(have)} variables "
            "are declared. The structure is inconsistent -- some earlier pass "
            "renumbered rows and variables differently.")


def _bound_from_term(term, j, op):
    """The numeric bound a single-variable monomial term imposes on ``x_j``.

    The term is ``c * x_j**a`` compared against 1, so ``x_j**a <= 1/c`` and the
    bound is ``(1/c)**(1/a)`` -- an upper bound when ``a > 0``, a lower one when
    ``a < 0``, and both when the operator is an equality.
    """
    a = term.exponents.get(j, 0.0)
    if term.coeff <= 0.0 or a == 0.0:
        return None, None
    try:
        val = (1.0 / term.coeff) ** (1.0 / a)
    except (OverflowError, ValueError, ZeroDivisionError):
        return None, None
    if op == "==":
        return val, val
    return (None, val) if a > 0 else (val, None)


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
    output_columns: list = field(default_factory=list)
    degenerate: list = field(default_factory=list)
    cancelling: list = field(default_factory=list)
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
        if self.output_columns:
            L.append(f"  {len(self.output_columns)} variables are OUTPUT ONLY "
                     "-- computed from the design and read by nothing; they "
                     "can be post-computed instead of solved for")
            for nm in self.output_columns[:12]:
                L.append(f"    {nm}")
            if len(self.output_columns) > 12:
                L.append(f"    ... and {len(self.output_columns) - 12} more")

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

    def post_solve_text(self):
        """The checks that need a solution, if one was supplied."""
        L = []
        if self.degenerate:
            L.append(f"  {len(self.degenerate)} variables the optimum does not "
                     "determine (moving them changes neither the objective nor "
                     "feasibility):")
            for nm, val in self.degenerate[:12]:
                L.append(f"    {nm} = {val:.6g}")
            if len(self.degenerate) > 12:
                L.append(f"    ... and {len(self.degenerate) - 12} more")
        if self.cancelling:
            L.append(f"  {len(self.cancelling)} signomial terms contribute "
                     "essentially nothing to their constraint, so the quantity "
                     "they carry is disconnected:")
            for i, side, share, vs in self.cancelling[:8]:
                L.append(f"    constraint {i} ({side}): {', '.join(vs[:4])} "
                         f"at {share:.1e} of the total")
            if len(self.cancelling) > 8:
                L.append(f"    ... and {len(self.cancelling) - 8} more")
        return "\n".join(L)


def nm_at(names, j):
    return names[j] if j < len(names) else f"<var {j}>"


def _rows_of(structures):
    """``(rows, operators, key)`` for the log-space encoding.

    Signomial before geometric, which used to be duplicated here as folklore.
    :attr:`~edi.structure.detected.Detected.log_key` names it now, and names
    why: a model can satisfy several structure flags at once, so "which kind
    is this" and "which encoding are the terms in" are different questions.
    """
    from edi.structure.detected import as_detected

    st = as_detected(structures)
    key = st.log_key
    if key is None:
        raise ValueError("presolve needs a detected GP or SP structure")
    return st[key][1], st[key][2], key


def presolve_report(structures, names=None) -> PresolveReport:
    """Run the structural checks. Nothing is solved and nothing is modified."""
    rows, operators, _ = _rows_of(structures)
    st = as_detected(structures)
    if names is None:
        names = [str(v) for v in structures.get("variables", [])]

    # Index 0 is the objective and is excluded from the constraint scan --
    # being in the objective is not a constraint, though it does mean the
    # variable is not free.
    n = max([len(r) - 2 for r in rows] + [len(names)])
    con_idx = st.constraint_indices

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
    for t in st.terms(0):
        for j, e in t.exponents.items():
            if e > 1e-12:
                upper[j] = True
            elif e < -1e-12:
                lower[j] = True

    for i in con_idx:
        op = st.operator(i)
        terms = st.terms(i)
        # numerator counts as written; denominator counts negated, since
        # growing the denominator relaxes p/q <= 1
        has_den = any(t.denominator for t in terms)
        touched = set()
        for t in terms:
            touched |= t.variables

        is_bound = (len(touched) == 1 and len(terms) == 1 and not has_den)

        if is_bound:
            # Record the value rather than the fact. Whether it bounds anything
            # is decided below, once we can see how big it is.
            j = next(iter(touched))
            note_bound(j, *_bound_from_term(terms[0], j, op))
            rep.singleton_rows.append(i)
        else:
            for t in terms:
                sgn = -1.0 if t.denominator else 1.0
                for j, e in t.exponents.items():
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
        # A variable whose only rows were folded into bounds has no rows left,
        # but it is not unconstrained -- calling it "appears in no constraint"
        # is both alarming and wrong. Distinguish by whether anything
        # meaningful bounds it.
        bounded = ((box_lo[j] is not None and box_lo[j] > VACUOUS_LO)
                   or (box_hi[j] is not None and box_hi[j] < VACUOUS_HI))
        if not in_rows[j] and not bounded:
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

    # Output-only variables need the terms grouped per constraint, and the
    # bounds separated, so they are only reported when that is available.
    if structures.get("bounds") is not None:
        obj_vars = {j for t in st.terms(0) for j in t.variables}
        outs = _output_only(st, con_idx, structures["bounds"], obj_vars, n)
        rep.output_columns = [nm(j) for j, _i in outs]
    return rep


def fold_singleton_rows(structures):
    """Move single-variable rows into the bounds, and drop them.

    A row like ``x <= 3`` states a bound and nothing else, so a solver that
    takes bounds natively should be given it as one. This is the classical
    singleton-row reduction, and it is worth a lot here: SPaircraft writes
    about 2500 of its constraints this way, on top of the 2346 that come from
    variable declarations.

    Requires ``structures['bounds']`` -- run ``structure_detector`` with
    ``bounds_as_rows=False`` first, since otherwise there is nowhere to put
    them. Returns a **new** structures dict; the input is untouched.

    Bounds are intersected, never loosened: several rows bounding the same
    variable all apply, and the tightest wins. An equality row fixes the
    variable, giving an equal pair. Nothing is rounded or clipped -- SPaircraft
    needs its full 1e-30..1e30 box for the reference solution to lie inside it,
    and a reduction that "tidied" those limits would cut off the answer.
    """
    if structures.get("bounds") is None:
        raise ValueError(
            "fold_singleton_rows needs structures['bounds'] to fold into; "
            "run structure_detector with bounds_as_rows=False")

    rows, operators, key = _rows_of(structures)
    names = [str(v) for v in structures.get("variables", [])]
    numer, denom = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        idx = int(r[0])
        (numer if idx >= 0 else denom)[
            idx if idx >= 0 else -idx - 1].append(r)

    bounds = [tuple(b) if b is not None else (None, None)
              for b in structures["bounds"]]

    def tighten(j, lo, hi):
        cur_lo, cur_hi = bounds[j]
        if lo is not None:
            cur_lo = lo if cur_lo is None else max(cur_lo, lo)
        if hi is not None:
            cur_hi = hi if cur_hi is None else min(cur_hi, hi)
        # Crossed bounds are a proof of infeasibility, and the cheapest one
        # available. Saying so beats silently picking a side: `x >= 2` with
        # `x <= 1` folded naively becomes "x is fixed at 2", and the solver
        # then answers a different question than the one that was asked.
        if (cur_lo is not None and cur_hi is not None
                and cur_hi < cur_lo * (1.0 - 1e-9)):
            raise InfeasibleProblem(
                f"{nm_at(names, j)} is required to be both >= {cur_lo:g} and "
                f"<= {cur_hi:g}; the model has no feasible point")
        bounds[j] = (cur_lo, cur_hi)

    folded = set()
    for i in sorted(k for k in set(numer) | set(denom) if k != 0):
        if denom.get(i) or len(numer.get(i, [])) != 1:
            continue
        row = numer[i][0]
        expo = [float(e) for e in row[2:]]
        nz = [j for j, e in enumerate(expo) if abs(e) > 1e-12]
        if len(nz) != 1 or nz[0] >= len(bounds):
            continue
        j = nz[0]
        op = operators[i - 1] if 0 <= i - 1 < len(operators) else "<="
        lo, hi = _bound_from_row(float(row[1]), expo, j, op)
        if lo is None and hi is None:
            continue
        tighten(j, lo, hi)
        folded.add(i)

    # Renumber what survives; constraint indices must stay contiguous from 1
    # because the operator list is positional.
    keep = [i for i in sorted(set(numer) | set(denom)) if i != 0
            and i not in folded]

    st = as_detected(structures)
    n = max([len(r) - 2 for r in rows] + [len(bounds)])
    info = dict(structures.get("info") or {})
    info["N_cons_total"] = len(keep)
    info["N_cons_folded"] = len(folded)
    return st.rebuild(
        st.terms(0),
        [st.terms(i) for i in keep],
        [st.operator(i) for i in keep],
        n=n, bounds=bounds, info=info)


class Removed:
    """One variable taken out of the solve, and how to get its value back.

    Unpacks as ``(index, name, value, reason)`` so existing callers keep
    working; ``recover`` carries the extra data needed for an output-only
    variable, whose value is not known until the core solve has finished.
    """

    __slots__ = ('index', 'name', 'value', 'reason', 'recover')

    def __init__(self, index, name, value, reason, recover=None):
        self.index, self.name = index, name
        self.value, self.reason = value, reason
        self.recover = recover

    def __iter__(self):
        return iter((self.index, self.name, self.value, self.reason))

    def __repr__(self):
        return (f"Removed({self.name!r}, {self.reason}, "
                f"value={self.value!r})")


def _eval_terms(terms, x, j=None, tj=None):
    """``sum_k c_k prod_i x_i**a_ik``, optionally overriding ``log x_j``."""
    import math

    total = 0.0
    for c, a in terms:
        acc = math.log(c) if c > 0 else -math.inf
        for i, e in enumerate(a):
            if abs(e) <= 1e-12:
                continue
            lx = tj if (j is not None and i == j) else (
                math.log(x[i]) if i < len(x) and x[i] > 0 else -math.inf)
            acc += e * lx
        total += math.exp(acc) if acc > -700 else 0.0
    return total


def _solve_for(num, den, j, x, lo=1e-300, hi=1e300):
    """Solve ``num/den == 1`` for ``x_j``, holding everything else at ``x``.

    Bisection on ``log x_j``. The caller has already established that the
    constraint is monotone in ``x_j`` -- that is what made the variable
    output-only in the first place -- so a sign change is bracketed and
    bisection is both safe and enough. This runs once per variable after the
    solve, so its cost is irrelevant.
    """
    import math

    def f(t):
        n = _eval_terms(num, x, j, t)
        d = _eval_terms(den, x, j, t) if den else 1.0
        if n <= 0:
            return -math.inf
        if d <= 0:
            return math.inf
        return math.log(n) - math.log(d)

    a, b = math.log(lo), math.log(hi)
    fa, fb = f(a), f(b)
    if not (math.isfinite(fa) or math.isfinite(fb)):
        return None
    if fa == 0.0:
        return math.exp(a)
    if fb == 0.0:
        return math.exp(b)
    if (fa > 0) == (fb > 0):
        return None                    # no sign change: not recoverable here
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = f(m)
        if fm == 0.0:
            return math.exp(m)
        if (fm > 0) == (fa > 0):
            a, fa = m, fm
        else:
            b, fb = m, fm
    return math.exp(0.5 * (a + b))


def _output_only(st, con_idx, bounds, in_objective, n):
    """Variables that are computed but never fed back, peeled in rounds.

    A variable is **output-only** when it appears in exactly one constraint,
    is absent from the objective, and that constraint cannot restrict anything
    else through it -- which needs two things:

    * the constraint is monotone in the variable, so it can always be satisfied
      by moving the variable; and
    * the bound in the direction that relaxes it is vacuous, so moving it is
      actually allowed. This is the part that is easy to get wrong. Given
      ``A_tri >= f(...)`` with ``A_tri`` unbounded above, the constraint says
      nothing about ``f``; add ``A_tri <= 100`` and it suddenly forces
      ``f <= 100``, which is a real restriction on real variables.

    Peeling is iterative because removing one output variable can expose
    another behind it -- a reporting quantity computed from another reporting
    quantity. Returns ``[(j, constraint_index), ...]`` in peel order.
    """
    alive = set(con_idx)
    taken, order = {}, []
    while True:
        rows = collections.defaultdict(set)
        for i in alive:
            for t in st.terms(i):
                for j in t.variables:
                    rows[j].add(i)

        progress = False
        for j in range(n):
            if j in taken or j in in_objective or len(rows.get(j, ())) != 1:
                continue
            i = next(iter(rows[j]))
            op = st.operator(i)

            # Monotone in x_j? Numerator exponents one sign, denominator the
            # other. Mixed signs mean moving x_j can tighten and loosen, so the
            # constraint really does pin it.
            signs = set()
            for t in st.terms(i):
                e = t.exponents.get(j)
                if e is None or abs(e) <= 1e-12:
                    continue
                signs.add((e > 0) != t.denominator)
            if len(signs) != 1:
                continue
            grows_tighter = signs.pop()

            lo, hi = (bounds[j] if j < len(bounds) else (None, None)) \
                or (None, None)
            if op == "==":
                # An equality pins x_j exactly; it restricts others only via
                # x_j's own bounds, so both must be vacuous.
                free = ((lo is None or lo <= VACUOUS_LO)
                        and (hi is None or hi >= VACUOUS_HI))
            elif grows_tighter:
                # Raising x_j tightens, so the constraint is escaped downward.
                free = lo is None or lo <= VACUOUS_LO
            else:
                free = hi is None or hi >= VACUOUS_HI
            if not free:
                continue

            taken[j] = i
            order.append((j, i))
            alive.discard(i)
            progress = True
        if not progress:
            # Peel order matters: a variable peeled in round 1 may sit in the
            # constraint that defines a variable peeled in round 2, so it can
            # only be recovered once that one is known. Callers recover in
            # REVERSE of this order.
            return order


def _tighten_linear(linear, L, U, names, max_passes, min_gain):
    """Interval propagation on ``coeffs . v <= rhs`` (or ``==``).

    ``L`` and ``U`` bound ``v`` in whatever space the caller works in --
    natural variables for an LP, log variables for a GP. The arithmetic does
    not care which, which is why this is shared.

    For ``a . v <= b`` and any ``k``, isolate ``a_k v_k <= b - S`` where ``S``
    is the sum of the other terms. The binding case is ``S`` at its **minimum**,
    reached at ``L_j`` where ``a_j > 0`` and ``U_j`` where ``a_j < 0``.

    An **equality** additionally gives ``a_k v_k >= b - S`` with ``S`` at its
    **maximum**, and that is a different sum -- the opposite endpoint of every
    other variable. Reusing the minimum for both directions manufactures
    contradictions: on SPaircraft it "proved" a variable with a wide-open box
    both <= 1.6e7 and >= 2.6e-15 from two unrelated monomial equalities, and
    declared the model infeasible.

    Mutates ``L``/``U`` in place; returns the number of tightenings.
    """
    import math

    NEG, POS = -math.inf, math.inf
    tightened = 0

    def side(a_j, lo, hi, want_min):
        """Contribution of one term at whichever endpoint is asked for."""
        if want_min:
            return a_j * lo if a_j > 0 else a_j * hi
        return a_j * hi if a_j > 0 else a_j * lo

    for _pass in range(max_passes):
        changed = False
        for rhs, a, nz, eq in linear:
            # Sum of all terms at their min, and (for an equality) at their max,
            # each carrying its own count of infinite contributions.
            sums = {}
            for want_min in ((True, False) if eq else (True,)):
                tot, infs, at = 0.0, 0, -1
                for j in nz:
                    m = side(a[j], L[j], U[j], want_min)
                    if m == NEG or m == POS:
                        infs += 1
                        at = j
                        if infs > 1:
                            break
                    else:
                        tot += m
                sums[want_min] = (tot, infs, at)

            for k in nz:
                for want_min in ((True, False) if eq else (True,)):
                    tot, infs, at = sums[want_min]
                    if infs > 1 or (infs == 1 and k != at):
                        continue
                    mk = side(a[k], L[k], U[k], want_min)
                    rest = tot if (infs == 1 and k == at) else tot - mk
                    if rest == NEG or rest == POS:
                        continue
                    limit = (rhs - rest) / a[k]
                    # want_min bounds a_k v_k from ABOVE, want_max from BELOW
                    upper = (a[k] > 0) == want_min
                    if upper:
                        if limit < U[k] - min_gain:
                            U[k] = limit; tightened += 1; changed = True
                    else:
                        if limit > L[k] + min_gain:
                            L[k] = limit; tightened += 1; changed = True
                    if L[k] > U[k] + 1e-6:
                        raise InfeasibleProblem(
                            f"bound propagation drove {nm_at(names, k)} to an "
                            "empty range; the model has no feasible point")
        if not changed:
            break
    return tightened


def propagate_bounds(structures, max_passes=8, min_gain=1e-6):
    """Tighten variable bounds by interval propagation.

    Works on a linear program, a quadratic program (whose constraints are
    linear), and a geometric or signomial program. The last is the interesting
    case: a monomial ``c * prod x_j**a_j <= 1`` is **linear** once written in
    ``y = log x``, as ``a . y <= -log c``, so the ordinary LP propagation
    applies unchanged. Only the space differs, so the arithmetic is shared --
    see :func:`_tighten_linear`.

    A **posynomial** yields more than it looks like it should. Every term of
    ``sum_k c_k m_k(x) <= 1`` is strictly positive, so each separately
    satisfies ``c_k m_k(x) <= 1``. Each term is a monomial, so one posynomial
    hands over one linear implication per term for free. That is what makes
    this worth running on a GP at all: most constraints are posynomials, and a
    strictly-monomial rule would skip nearly everything.

    Ratios are left alone -- ``p <= q`` bounds neither side without a point to
    evaluate at, and being wrong here would be silent.

    Returns ``(structures, n_tightened)`` with a new bounds list; the input is
    untouched. Raises :class:`InfeasibleProblem` if a range comes out empty.
    """
    import math

    if structures.get("bounds") is None:
        raise ValueError(
            "propagate_bounds needs structures['bounds']; run "
            "structure_detector with bounds_as_rows=False first")

    names = [str(v) for v in structures.get("variables", [])]
    bounds = list(structures["bounds"])
    NEG, POS = -math.inf, math.inf

    lp = (structures.get("Linear_Program", (False,))[0]
          or structures.get("Quadratic_Program", (False,))[0])

    if lp:
        # Natural variables: rows are AG . x <= b, and x may be negative.
        parts = as_detected(structures).linear_parts()
        AG, bh = parts.A, parts.b
        if AG is None:
            return structures, 0
        operators = parts.operators
        n = max(len(bounds), max(len(r) for r in AG))
        while len(bounds) < n:
            bounds.append((None, None))
        L = [b[0] if (b and b[0] is not None) else NEG for b in bounds]
        U = [b[1] if (b and b[1] is not None) else POS for b in bounds]
        linear = []
        for i, row in enumerate(AG):
            a = [float(v) for v in row] + [0.0] * (n - len(row))
            nz = [j for j in range(n) if abs(a[j]) > 1e-12]
            if not nz:
                continue
            op = operators[i] if i < len(operators) else "<="
            linear.append((-float(bh[i]), a, nz, op == "=="))
        k = _tighten_linear(linear, L, U, names, max_passes, min_gain)
        new_bounds = [(None if L[j] == NEG else L[j],
                       None if U[j] == POS else U[j]) for j in range(n)]
    else:
        rows, operators, _key = _rows_of(structures)
        n = max([len(r) - 2 for r in rows] + [len(bounds)])
        while len(bounds) < n:
            bounds.append((None, None))
        L = [math.log(b[0]) if (b and b[0] and b[0] > 0) else NEG
             for b in bounds]
        U = [math.log(b[1]) if (b and b[1] and b[1] > 0) else POS
             for b in bounds]

        numer, denom = collections.defaultdict(list), collections.defaultdict(list)
        for r in rows:
            idx = int(r[0])
            (numer if idx >= 0 else denom)[
                idx if idx >= 0 else -idx - 1].append(r)

        linear = []
        for i in sorted(kk for kk in set(numer) | set(denom) if kk != 0):
            if denom.get(i):
                continue
            op = operators[i - 1] if 0 <= i - 1 < len(operators) else "<="
            terms = numer.get(i, [])
            # Only a single-term equality is an equality term-wise; a
            # multi-term one implies just the <= half per term.
            eq = (op == "==" and len(terms) == 1)
            for r in terms:
                c = float(r[1])
                if c <= 0:
                    continue
                a = [float(e) for e in r[2:]] + [0.0] * (n - (len(r) - 2))
                nz = [j for j in range(n) if abs(a[j]) > 1e-12]
                if nz:
                    linear.append((-math.log(c), a, nz, eq))
        k = _tighten_linear(linear, L, U, names, max_passes, min_gain)
        new_bounds = [(None if L[j] == NEG else math.exp(L[j]),
                       None if U[j] == POS else math.exp(U[j]))
                      for j in range(n)]

    out = dict(structures)
    out["bounds"] = new_bounds
    info = dict(structures.get("info") or {})
    info["N_bounds_tightened"] = k
    out["info"] = info
    return out, k


def eliminate_monomial_equalities(structures, max_fill=16, min_pivot=1e-6,
                                  max_eliminations=None):
    """Substitute out variables that a monomial equality already determines.

    A monomial equality ``c * prod x_j**a_j == 1`` is a **linear** equality in
    ``y = log x``, so it can be solved for one variable and substituted
    everywhere else -- Gaussian elimination on the exponent matrix. Solving for
    the pivot ``p`` gives

        x_p = c**(-1/a_p) * prod_{j != p} x_j**(-a_j/a_p)

    which is itself a monomial, so substituting it into any term keeps that
    term a monomial with a positive coefficient. Posynomial structure survives
    intact, which is what makes this safe on a GP: nothing becomes signomial,
    and no approximation is introduced. The result is exact.

    SPaircraft carries 752 monomial equalities among 1267 constraints, so the
    ceiling here is high.

    **Only variables whose declared bounds are vacuous are eliminated.** A real
    bound on an eliminated variable does not disappear -- it becomes a
    constraint on the survivors, and re-adding it as two monomial rows gives
    back most of what the elimination saved. Bounds *derived* by
    :func:`propagate_bounds` are a different matter, being implied by the
    constraints already, but this runs on declared bounds and does not try to
    tell them apart.

    Fill-in is the real cost. Substituting a dense pivot row into many terms
    densifies the exponent matrix, and a GP's matrix is normally very sparse.
    Pivots are chosen greedily by a Markowitz-style estimate,
    ``(row_nnz - 1) * (col_nnz - 1)``, and any pivot whose estimate exceeds
    ``max_fill`` is skipped.

    Returns ``(structures, removed)`` with ``removed`` in
    :func:`reduce_columns` form, so :func:`restore_columns` recovers the
    eliminated variables by back-substitution.
    """
    import math

    if structures.get("bounds") is None:
        raise ValueError(
            "eliminate_monomial_equalities needs structures['bounds']; run "
            "structure_detector with bounds_as_rows=False first")

    rows, operators, key = _rows_of(structures)
    names = [str(v) for v in structures.get("variables", [])]
    bounds = list(structures["bounds"])
    n = max([len(r) - 2 for r in rows] + [len(bounds)])
    while len(bounds) < n:
        bounds.append((None, None))

    # Sparse form: constraint -> list of (coeff, {j: exponent}, is_denominator)
    _check_width(structures, n, "eliminate_monomial_equalities")

    terms = collections.defaultdict(list)
    for r in rows:
        idx = int(r[0])
        i = idx if idx >= 0 else -idx - 1
        e = {j: float(v) for j, v in enumerate(r[2:]) if abs(float(v)) > 1e-12}
        terms[i].append([float(r[1]), e, idx < 0])

    def op_of(i):
        return operators[i - 1] if 0 <= i - 1 < len(operators) else "<="

    con_idx = sorted(k for k in terms if k != 0)
    alive = set(con_idx)

    def vacuous(j):
        lo, hi = (bounds[j] if j < len(bounds) else (None, None)) or (None, None)
        return ((lo is None or lo <= VACUOUS_LO)
                and (hi is None or hi >= VACUOUS_HI))

    # Column occupancy, for the Markowitz estimate.
    col = collections.defaultdict(set)
    for i in con_idx:
        for _c, e, _d in terms[i]:
            for j in e:
                col[j].add(i)

    removed, done = [], 0
    changed = True
    while changed:
        changed = False
        # Candidate monomial equalities, cheapest pivot first.
        cands = []
        for i in sorted(alive):
            if op_of(i) != "==" or len(terms[i]) != 1 or terms[i][0][2]:
                continue
            c_eq, a, _d = terms[i][0]
            if c_eq <= 0:
                continue
            for pj, ap in a.items():
                if abs(ap) < min_pivot or not vacuous(pj):
                    continue
                fill = (len(a) - 1) * (len(col[pj]) - 1)
                if fill > max_fill:
                    continue
                cands.append((fill, i, pj))
        if not cands:
            break
        cands.sort()

        used_con, used_var = set(), set()
        for fill, i, pj in cands:
            if max_eliminations is not None and done >= max_eliminations:
                break
            if i in used_con or pj in used_var or i not in alive:
                continue
            c_eq, a, _d = terms[i][0]
            ap = a.get(pj)
            if ap is None or abs(ap) < min_pivot:
                continue
            # x_p = C * prod_{j != p} x_j ** m_j
            C = c_eq ** (-1.0 / ap)
            m = {j: -v / ap for j, v in a.items() if j != pj}

            for k in list(col[pj]):
                if k == i or k not in alive:
                    continue
                for t in terms[k]:
                    ep = t[1].pop(pj, None)
                    if ep is None:
                        continue
                    t[0] *= C ** ep
                    for j, mv in m.items():
                        nv = t[1].get(j, 0.0) + ep * mv
                        if abs(nv) > 1e-12:
                            t[1][j] = nv
                            col[j].add(k)
                        else:
                            t[1].pop(j, None)
            alive.discard(i)
            col.pop(pj, None)
            used_con.add(i)
            used_var.add(pj)
            removed.append(Removed(pj, nm_at(names, pj), None, "substituted",
                                   recover=("monomial", C, dict(m))))
            done += 1
            changed = True

    if not removed:
        return structures, []

    gone = {r.index for r in removed}
    keep = [j for j in range(n) if j not in gone]
    pos = {j: t for t, j in enumerate(keep)}
    surviving = [i for i in con_idx if i in alive]

    def rewrite(i):
        """The surviving terms of constraint ``i``, renumbered."""
        out_terms = []
        for coeff, expo, den in terms[i]:
            out_terms.append(Term(
                coeff=float(coeff),
                exponents={pos[j]: v for j, v in expo.items()
                           if j in pos and abs(v) > 1e-12},
                denominator=bool(den)))
        return out_terms

    st = as_detected(structures)
    info = dict(structures.get("info") or {})
    info["N_vars_substituted"] = len(removed)
    info["N_cons_total"] = len(surviving)
    out = st.rebuild(
        rewrite(0),
        [rewrite(i) for i in surviving],
        [op_of(i) for i in surviving],
        n=len(keep),
        bounds=[bounds[j] for j in keep],
        info=info)
    if structures.get("variables"):
        out["variables"] = [structures["variables"][j] for j in keep]

    # REVERSE elimination order. A pivot's formula is captured at the moment it
    # is eliminated, and it may reference variables eliminated in a later
    # round, so those have to be known first. Recovering forwards instead of
    # backwards silently produces values off by orders of magnitude while the
    # reduced problem itself stays perfectly correct -- measured on SPaircraft
    # at a relative error of 4.3e+03 with an objective still exact to 12
    # figures.
    return out, removed[::-1]


def reduce_columns(structures, guess=None, eliminate_outputs=True):
    """Remove variables the model does not connect to anything.

    Two reductions, both exact -- the optimal objective is unchanged and the
    removed variables are given values that are feasible for the original
    problem.

    **Disconnected.** In ``min x**2 s.t. x >= 1, y >= 4`` the variable ``y``
    appears in no real constraint and in no objective term. Nothing determines
    it, nothing is affected by it, and carrying it through the solve only costs
    time. It is fixed at its tightest finite bound and dropped.

    **Fixed.** A variable whose bounds are equal is a constant wearing a
    variable's clothing. Its value is folded into the coefficient of every term
    it appears in -- ``c * v**a`` -- and the column goes.

    This is structural, which is the whole point: it happens before the solve,
    unlike :func:`degeneracy_report`, which can only report after one. The two
    do not overlap much. A degenerate variable typically sits in several real
    constraints that all happen to go slack at this particular optimum, and
    removing it would delete those constraints along with it -- they would bind
    at a different design point. Nothing here touches a variable that appears
    in a real constraint.

    Requires ``structures['bounds']``; run :func:`fold_singleton_rows` first so
    that rows which are really bounds have already been recognised as such,
    otherwise ``y >= 4`` still counts as a constraint and ``y`` is not seen as
    disconnected.

    Returns ``(reduced_structures, removed)``, where ``removed`` is a list of
    ``(original_index, name, value, reason)`` ordered by index. Feed it to
    :func:`restore_columns` to put the values back into a solution vector.
    """
    if structures.get("bounds") is None:
        raise ValueError(
            "reduce_columns needs structures['bounds']; run structure_detector "
            "with bounds_as_rows=False (and fold_singleton_rows) first")

    rows, operators, key = _rows_of(structures)
    names = [str(v) for v in structures.get("variables", [])]
    bounds = list(structures["bounds"])
    n = max([len(r) - 2 for r in rows] + [len(bounds)])

    _check_width(structures, n, "reduce_columns")

    in_objective, in_constraint = set(), set()
    for r in rows:
        target = in_objective if int(r[0]) == 0 else in_constraint
        for j, e in enumerate(r[2:]):
            if abs(float(e)) > 1e-12:
                target.add(j)

    # Group once; the output-only scan needs terms per constraint.
    numer, denom = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        idx = int(r[0])
        expo = [float(e) for e in r[2:]] + [0.0] * (n - (len(r) - 2))
        (numer if idx >= 0 else denom)[
            idx if idx >= 0 else -idx - 1].append((float(r[1]), expo))
    con_idx = sorted(k for k in set(numer) | set(denom) if k != 0)

    outputs = (_output_only(as_detected(structures), con_idx, bounds,
                            in_objective, n)
               if eliminate_outputs else [])
    out_vars = {j for j, _i in outputs}
    out_cons = {i for _j, i in outputs}

    removed = []
    for j in range(n):
        lo, hi = (bounds[j] if j < len(bounds) else (None, None)) or (None, None)

        if lo is not None and hi is not None and lo > 0:
            if hi < lo * (1.0 - 1e-9):
                raise InfeasibleProblem(
                    f"{nm_at(names, j)} is required to be both >= {lo:g} and "
                    f"<= {hi:g}; the model has no feasible point")
            if hi <= lo * (1.0 + 1e-9):
                removed.append(Removed(j, nm_at(names, j), float(lo), "fixed"))
                continue

        if j in out_vars:
            continue                    # handled below, in peel order

        if j in in_constraint or j in in_objective:
            continue
        # Nothing refers to it. Any feasible value will do, so take the
        # tightest bound that means anything; failing that, the user's guess.
        if lo is not None and lo > VACUOUS_LO:
            val = float(lo)
        elif hi is not None and hi < VACUOUS_HI:
            val = float(hi)
        elif guess is not None and j < len(guess) and guess[j] > 0:
            val = float(guess[j])
        else:
            val = 1.0
        removed.append(Removed(j, nm_at(names, j), val, "disconnected"))

    # Output-only variables carry their defining constraint rather than a
    # value, since the value is not known until the core solve has finished.
    # Recorded in REVERSE peel order, which is the order they can be evaluated.
    for j, i in reversed(outputs):
        removed.append(Removed(
            j, nm_at(names, j), None, "output",
            recover=(numer.get(i, []), denom.get(i, []), j)))

    if not removed:
        return structures, []

    drop = {r.index: r.value for r in removed if r.reason != "output"}
    gone = {r.index for r in removed}
    keep = [j for j in range(n) if j not in gone]
    pos = {j: t for t, j in enumerate(keep)}
    surviving = [i for i in con_idx if i not in out_cons]

    def rewrite(terms):
        """Fold removed constants into the coefficient, renumber the rest."""
        out_terms = []
        for t in terms:
            coeff = t.coeff
            expo = {}
            for j, e in t.exponents.items():
                if j in drop:
                    coeff *= drop[j] ** e         # a constant, into the coeff
                elif j in pos:
                    expo[pos[j]] = e              # survivor, renumbered
            out_terms.append(Term(coeff=coeff, exponents=expo,
                                  denominator=t.denominator))
        return out_terms

    st = as_detected(structures)
    info = dict(structures.get("info") or {})
    info["N_vars_removed"] = len(removed)
    info["N_vars_output"] = len(outputs)
    info["N_cons_total"] = len(surviving)
    out = st.rebuild(
        rewrite(st.terms(0)),
        [rewrite(st.terms(i)) for i in surviving],
        [st.operator(i) for i in surviving],
        n=len(keep),
        bounds=[bounds[j] if j < len(bounds) else (None, None) for j in keep],
        info=info)
    if structures.get("variables"):
        out["variables"] = [structures["variables"][j] for j in keep]
    return out, removed


def restore_columns(removed, x_reduced, n_original=None):
    """Put removed variables back into a reduced solution vector.

    Constants go straight back. **Output-only** variables are post-computed
    from the constraint that defined them, evaluated at the solved values of
    everything else -- so a caller sees a full solution vector and cannot tell
    which quantities took part in the optimization and which were worked out
    afterwards.

    ``removed`` already holds the output entries in the order they can be
    evaluated, so this walks them as given.
    """
    import math

    import numpy as np

    x_reduced = np.asarray(x_reduced, dtype=float)
    if n_original is None:
        n_original = len(x_reduced) + len(removed)

    constants = {r.index: r.value for r in removed
                 if getattr(r, 'reason', None) != 'output'
                 and getattr(r, 'recover', None) is None}
    outputs = [r for r in removed if getattr(r, 'recover', None) is not None]
    placed = set(constants) | {r.index for r in outputs}

    out = np.ones(n_original, dtype=float)
    it = iter(x_reduced)
    for j in range(n_original):
        if j in placed:
            out[j] = constants.get(j, 1.0)
        else:
            out[j] = next(it)

    for r in outputs:
        spec = r.recover
        if spec[0] == "monomial":
            # x_p = C * prod x_j ** m_j, from a monomial equality solved for p
            _tag, C, m = spec
            acc = math.log(C) if C > 0 else -math.inf
            for j, e in m.items():
                if j < len(out) and out[j] > 0:
                    acc += e * math.log(out[j])
            val = math.exp(acc) if -700 < acc < 700 else (
                0.0 if acc <= -700 else math.inf)
        else:
            num, den, j0 = spec
            val = _solve_for(num, den, j0, out)
        if val is not None:
            out[r.index] = val
            r.value = float(val)
    return out


def cancellation_report(structures, x, tol=1e-6, names=None):
    """Terms in a signomial constraint that contribute nothing at ``x``.

    EDI writes a constraint containing a subtraction as a ratio ``p/q <= 1``,
    moving the negative terms into the denominator alongside the left-hand
    side. So ``M_r*c >= A + B - C`` becomes ``(A + B) / (M_r*c + C) <= 1``, and
    the two terms in that denominator are in direct competition: whatever ``C``
    supplies, ``M_r`` need not.

    When one of them supplies essentially all of it, the other is inert. The
    constraint holds no matter what that variable does, so a quantity the
    modeller believed was being sized is in fact disconnected -- which is how a
    variable ends up parked at 1e-30 with nothing complaining.

    This is a signomial-specific failure and the LP presolve battery has no
    reason to look for it, since LP has no signomials. It is also
    solution-dependent, so it runs after a solve, like
    :func:`degeneracy_report`.

    Returns ``[(constraint_index, side, share, variables), ...]`` for each term
    whose share of its own group falls below ``tol``, worst first. ``side`` is
    ``'numerator'`` or ``'denominator'``.
    """
    import math

    import numpy as np

    x = np.asarray(x, dtype=float)
    st = as_detected(structures)
    if names is None:
        names = [str(v) for v in structures.get("variables", [])]

    out = []
    # Only constraints that actually have a denominator -- i.e. the signomial
    # ones. A small term in a plain posynomial is ordinary and not a defect.
    for i in st.constraint_indices:
        terms = st.terms(i)
        if not any(t.denominator for t in terms):
            continue
        for side, group in (("numerator",
                             [t for t in terms if not t.denominator]),
                            ("denominator",
                             [t for t in terms if t.denominator])):
            if len(group) < 2:
                continue                      # nothing to be crowded out by
            vals = [t.value(x) for t in group]
            total = sum(vals)
            if total <= 0:
                continue
            for t, v in zip(group, vals):
                share = v / total
                if share < tol:
                    out.append((i, side, share,
                                [nm_at(names, j) for j in sorted(t.variables)]))

    out.sort(key=lambda t: t[2])
    return out


def evaluate(structures, x):
    """``(objective, worst_violation)`` for a detected structure at ``x``.

    The violation is in log space for a GP or SP and natural for an LP or QP,
    matching the space each is linear in. An equality contributes the magnitude
    of its residual; an inequality only its positive part. Variable bounds
    count, since presolve turns rows into bounds and a bound violation would
    otherwise become invisible.

    Written to be independent of any solver, so it can compare two structures
    that no solver has seen.
    """
    import math

    x = list(x)
    # Dispatch exactly as propagate_bounds does, linear flag first. It has to
    # match: this compares a structure before and after a transform, and
    # reading one side in a different encoding than the transform used would
    # compare two different problems. The linear-first rule also matters for
    # correctness rather than only consistency -- a model with negative
    # variables can carry a signomial flag whose log encoding is invalid for
    # it, and preferring log there would take logs of negative numbers.
    lp = (structures.get("Linear_Program", (False,))[0]
          or structures.get("Quadratic_Program", (False,))[0])
    worst = -math.inf

    st = as_detected(structures)
    if lp:
        parts = st.linear_parts()
        c, shift, AG, bh = parts.linear, parts.shift, parts.A, parts.b
        operators = parts.operators
        obj = float(shift or 0.0) + sum(float(ci) * x[i]
                                        for i, ci in enumerate(c) if i < len(x))
        if parts.hessian is not None:
            # EDI stores the quadratic COEFFICIENT matrix, not the Hessian,
            # so the objective is x'Px + q'x + shift with no factor of a half:
            # `x**2 + y**2` gives P = I, and x'Ix = 2 at (1,1), matching the
            # Pyomo objective. cvxopt.solvers.qp minimises (1/2) x'Px + q'x,
            # and solve_QP passes `2.0*P` for exactly that reason, so the two
            # agree -- verified on `min x**2 + y**2 - 4x s.t. x + y >= 1`,
            # whose optimum is at x=2 and where a missing factor of two would
            # put it at x=4. cvxopt and IPOPT both return x=2.
            #
            # Omitting the quadratic term altogether, as this did at first,
            # reports the objective of the LP left by deleting it: 0 instead
            # of 2 on `min x**2 + y**2 s.t. x + y >= 2`.
            P = parts.hessian
            for i, row in enumerate(P):
                if i >= len(x):
                    continue
                for j, pij in enumerate(row):
                    if j < len(x):
                        obj += float(pij) * x[i] * x[j]
        for i, row in enumerate([] if AG is None else AG):
            lhs = sum(float(v) * x[j] for j, v in enumerate(row) if j < len(x))
            r = lhs + float(bh[i])
            op = operators[i] if i < len(operators) else "<="
            worst = max(worst, abs(r) if op == "==" else r)
    else:
        obj = sum(t.value(x) for t in st.terms(0) if not t.denominator)
        for i in st.constraint_indices:
            terms = st.terms(i)
            p_ = sum(t.value(x) for t in terms if not t.denominator)
            den = [t for t in terms if t.denominator]
            q_ = sum(t.value(x) for t in den) if den else 1.0
            if p_ <= 0 or q_ <= 0:
                continue
            lg = math.log(p_) - math.log(q_)
            op = st.operator(i)
            worst = max(worst, abs(lg) if op == "==" else lg)

    for j, pair in enumerate(structures.get("bounds") or []):
        if j >= len(x) or not pair:
            continue
        lo, hi = pair
        if lp:
            if lo is not None:
                worst = max(worst, lo - x[j])
            if hi is not None:
                worst = max(worst, x[j] - hi)
        elif x[j] > 0:
            if lo is not None and lo > 0:
                worst = max(worst, math.log(lo) - math.log(x[j]))
            if hi is not None and hi > 0:
                worst = max(worst, math.log(x[j]) - math.log(hi))
    return obj, worst


def equivalence_error(before, after, x, log=None, x_after=None):
    """How far a transform moved the problem, measured at a known point.

    Every transform in this module claims to preserve the problem. This is the
    claim, checked: map ``x`` through the transform and compare the objective
    and the worst violation on both sides. Returns
    ``(objective_relative_error, violation_absolute_error)``.

    ``log`` is a :class:`PresolveLog`, used to drop the columns the transform
    removed; pass ``x_after`` instead if the mapping is something else. With
    neither, ``x`` is assumed to survive unchanged.

    This is the check that caught the elimination bug -- a reduced problem
    exact to twelve figures whose recovered values were out by 4.3e+03 -- and
    it would have caught the propagation bug on sight. It costs one evaluation
    per side and works on any model, which is most of the value of a full
    integration test at a fraction of the price.
    """
    if x_after is None:
        x_after = list(x)
        if log is not None:
            for _label, removed, _counts in log.steps:
                if removed:
                    gone = {r.index for r in removed}
                    x_after = [v for j, v in enumerate(x_after)
                               if j not in gone]
    f0, v0 = evaluate(before, x)
    f1, v1 = evaluate(after, x_after)
    return (abs(f1 - f0) / max(abs(f0), 1e-300), abs(v1 - v0))


def assert_equivalent(before, after, x, log=None, x_after=None,
                      rtol=1e-8, atol=1e-8):
    """Raise unless a transform preserved the problem at ``x``."""
    df, dv = equivalence_error(before, after, x, log=log, x_after=x_after)
    if df > rtol or dv > atol:
        raise AssertionError(
            f"transform changed the problem: objective differs by {df:.3e} "
            f"(tolerance {rtol:g}), worst violation by {dv:.3e} "
            f"(tolerance {atol:g})")


class PresolveLog:
    """A record of what presolve did, and the means to undo it.

    Passes compose awkwardly on their own: each one renumbers the columns it
    leaves behind, so a ``Removed.index`` is relative to the structure as it
    stood when that pass ran, and concatenating two passes' lists silently
    mixes two index spaces. This keeps them separate and unwinds them in
    reverse, which needs no remapping at all -- each list is applied in exactly
    the space it was recorded in.

    ``print(log)`` gives a human-readable account. It is off by default: the
    reductions are exact, so most of the time there is nothing an engineer
    needs to do about them.
    """

    __slots__ = ('steps', 'names', 'infeasible')

    def __init__(self, names=None):
        self.steps = []          # (label, removed | None, counts dict)
        self.names = list(names or [])
        self.infeasible = None

    def record(self, label, removed=None, **counts):
        self.steps.append((label, removed, counts))

    @property
    def removed_variables(self):
        """Every variable taken out, innermost pass last."""
        return [r for _l, rem, _c in self.steps for r in (rem or [])]

    def restore(self, x):
        """Rebuild the full-length solution, undoing each pass in reverse."""
        for _label, removed, _counts in reversed(self.steps):
            if removed:
                x = restore_columns(removed, x)
        return x

    def __str__(self):
        if self.infeasible:
            return f"presolve: INFEASIBLE -- {self.infeasible}"
        if not self.steps:
            return "presolve: nothing to do"
        L = ["presolve:"]
        total_vars = 0
        for label, removed, counts in self.steps:
            bits = [f"{k.replace('_', ' ')} {v}"
                    for k, v in counts.items() if v]
            if removed:
                by = collections.Counter(r.reason for r in removed)
                bits.append("removed " + ", ".join(
                    f"{n} {reason}" for reason, n in sorted(by.items())))
                total_vars += len(removed)
            if bits:
                L.append(f"  {label}: " + "; ".join(bits))
        if total_vars:
            L.append(f"  {total_vars} variables removed in total; each is "
                     "recovered exactly and reported with the solution")
        return "\n".join(L)

    def detail(self, limit=40):
        """The per-variable account, for when the summary is not enough."""
        L = [str(self)]
        for label, removed, _c in self.steps:
            if not removed:
                continue
            L.append(f"  {label}:")
            for r in removed[:limit]:
                v = "" if r.value is None else f" = {r.value:.6g}"
                L.append(f"    {r.name}{v}   [{r.reason}]")
            if len(removed) > limit:
                L.append(f"    ... and {len(removed) - limit} more")
        return "\n".join(L)


def diagnose(structures, x=None, problem=None, names=None, quiet=False):
    """Every structural check, in one call, as one report.

    The individual checks are expert tools: each needs the structure detected a
    particular way and read in a particular order, which means in practice
    nobody runs them. This is the entry point that makes them the default.

    ``x`` and ``problem``, when given, enable the two checks that can only run
    after a solve -- degeneracy and signomial cancellation. Without them only
    the structural half runs, which is the half that needs nothing.

    Returns a :class:`PresolveReport` with ``degenerate`` and ``cancelling``
    filled in when a solution was supplied. ``quiet`` suppresses the printout
    and returns the report for a caller to inspect.
    """
    from edi.structure.detected import as_detected

    st = as_detected(structures)
    if st.bounds is None:
        # The interesting checks need bounds separated from rows. Fold a copy
        # rather than making the caller know that.
        try:
            st = fold_singleton_rows(_with_empty_bounds(st))
        except Exception:
            pass

    rep = presolve_report(st)
    if x is not None and problem is not None:
        try:
            rep.degenerate = degeneracy_report(problem, x, names=names)
        except Exception:
            rep.degenerate = []
        try:
            rep.cancelling = cancellation_report(st, x, names=names)
        except Exception:
            rep.cancelling = []
    if not quiet:
        text = str(rep)
        extra = rep.post_solve_text()
        print(text + ("\n" + extra if extra else ""))
    return rep


def _with_empty_bounds(structures):
    """A copy carrying an empty bounds array, so rows can be folded into it."""
    st = dict(structures)
    if st.get("bounds") is None:
        rows, _ops, key = _rows_of(st)
        width = max((len(r) - 2 for r in rows), default=0)
        n = max(width, len(st.get("variables") or []))
        st["bounds"] = [(None, None)] * n
    from edi.structure.detected import as_detected
    return as_detected(st)


def presolve(structures, fold=True, eliminate=True, propagate=False,
             reduce=True, verbose=False):
    """Run the presolve passes in an order that is safe to compose.

    The order is not a preference, it is a constraint, and two interactions
    force it:

    * **reduce before propagate.** Output-only detection requires a *vacuous*
      bound in the relaxing direction, and propagation fills exactly those in.
      Propagating first costs real reductions -- 40 variables on SPaircraft.
    * **eliminate before propagate**, for the same reason: elimination only
      takes variables whose declared bounds are vacuous.

    ``propagate`` is therefore off by default. It tightens bounds, which is
    useful as a diagnostic and for a solver that exploits them, but on
    SPaircraft it buys no time once elimination has run and it blocks other
    reductions if run early.

    Returns ``(structures, log)``. ``log.restore(x)`` rebuilds the full-length
    solution and ``print(log)`` says what happened.
    """
    log = PresolveLog([str(v) for v in structures.get("variables", [])])
    try:
        if fold:
            before = (structures.get("info") or {}).get("N_cons_total")
            structures = fold_singleton_rows(structures)
            after = structures["info"]["N_cons_total"]
            log.record("bounds", rows_folded=(before - after) if before else 0)
        if reduce:
            structures, removed = reduce_columns(structures)
            log.record("columns", removed=removed)
        if eliminate:
            structures, removed = eliminate_monomial_equalities(structures)
            log.record("monomial equalities", removed=removed)
        if propagate:
            structures, n = propagate_bounds(structures)
            log.record("bound propagation", bounds_tightened=n)
    except InfeasibleProblem as exc:
        log.infeasible = str(exc)
        raise
    if verbose:
        print(log)
    return structures, log


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
