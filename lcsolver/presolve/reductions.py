#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Structural checks on a detected formulation, before anything is solved.

Structural defects show in the sparsity pattern alone (empty/singleton
columns, bound rows, unbounded directions) -- the classical presolve
reductions. Solution-dependent defects do not: a variable can be undetermined
at the optimum because all its constraints go slack there; see
:func:`degeneracy_report`, which runs after a solve. Measured on SPaircraft:
7 of 29 undetermined variables are structural, the other 22 need the
post-solve test.

Boundedness: in log space a posynomial term bounds x_j above through
a_jk > 0 and below through a_jk < 0; an equality bounds both ways; a
denominator counts negated. The objective counts too. A bound only counts if
non-vacuous (VACUOUS_LO/HI) -- LCsolver's default 1e-30..1e30 box would
otherwise make everything look bounded.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field

from lcsolver.presolve.detected import Term, as_detected

class InfeasibleProblem(ValueError):
    """Presolve proved the model infeasible before any solve was attempted."""


__all__ = ["PresolveReport", "presolve_report", "degeneracy_report",
           "InfeasibleProblem",
           "cancellation_report", "fold_singleton_rows",
           "reduce_columns", "restore_columns", "Removed",
           "peel_output_columns",
           "propagate_bounds", "eliminate_monomial_equalities",
           "presolve", "PresolveLog",
           "evaluate", "equivalence_error", "assert_equivalent",
           "optimization_check", "floor_report", "structure_report",
           "VACUOUS_LO", "VACUOUS_HI"]

# A bound at or beyond these is no bound at all: LCsolver's default box is
# 1e-30..1e30 and models routinely restate it, so it must not count as bounding.
VACUOUS_LO = 1e-29
VACUOUS_HI = 1e29


def check_width(structures, n, who):
    """Rows and the variable list must describe the same number of columns.

    The old `j < len(variables)` mask silently dropped the tail and let the
    mismatch compound on the next pass; fail loudly at the source instead.
    """
    have = structures.get("variables")
    if have is not None and 0 < len(have) < n:
        raise ValueError(
            f"{who}: the rows span {n} columns but only {len(have)} variables "
            "are declared. The structure is inconsistent -- some earlier pass "
            "renumbered rows and variables differently.")


def bound_from_term(term, j, op):
    """Bound a single-variable term c * x_j**a imposes: (1/c)**(1/a),
    upper when a > 0, lower when a < 0, both for an equality."""
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


def bound_from_row(coeff, expo, j, op):
    """Bound a single-variable row c * x_j**a <= 1 (or ==) imposes.
    Returns (lo, hi), either possibly None."""
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
    at_floor: list = field(default_factory=list)
    # structure_report text, filled in by optimization_check. Kept as a field
    # so a caller can print the two halves separately.
    structure: str = ''
    defaulted_guesses: list = field(default_factory=list)
    cancelling: list = field(default_factory=list)
    bounds: dict = field(default_factory=dict)
    unbounded_above: list = field(default_factory=list)
    unbounded_below: list = field(default_factory=list)
    singleton_rows: list = field(default_factory=list)
    duplicate_rows: int = 0
    row_counts: dict = field(default_factory=dict)
    # rigidity_report output, filled in by optimization_check; needs no solution
    rigidity: dict = field(default_factory=dict)
    # unopposed_report output: variables nothing resists
    unopposed: list = field(default_factory=list)
    # annihilated_report output: constraint sides a constant has zeroed
    annihilated: list = field(default_factory=list)

    @property
    def clean(self):
        return not (self.empty_columns or self.unbounded_above
                    or self.unbounded_below)

    def summary(self) -> str:
        """Same text as ``str(report)``; named to match Solution.summary()."""
        return str(self)

    def __str__(self):
        # Structure first: str(report) has to be the whole report
        L = ([self.structure, ""] if self.structure else [])
        L += [f"presolve: {self.n_variables} variables, {self.n_rows} rows"]

        # First: it explains failures further down, including an empty report
        if self.annihilated:
            from lcsolver.core import codes
            L.append(f"  [{codes.ANNIHILATED_TERM}] "
                     f"{len(self.annihilated)} constraint side(s) are "
                     "IDENTICALLY ZERO at the current constant values. A "
                     "posynomial cannot be zero, so each of these rows has "
                     "stopped being one:")
            for hit in self.annihilated[:8]:
                who = ', '.join(f"{n} = {v:g}" for n, v in hit['constants'])
                because = f" -- annihilated by {who}" if who else ""
                L.append(f"    {hit['constraint']}: "
                         f"{hit.get('expression', hit['side'] + ' side')}"
                         f"{because}")
            if len(self.annihilated) > 8:
                L.append(f"    ... and {len(self.annihilated) - 8} more")
            L.append("    Consequences, all silent: the model may classify "
                     "as neither GP nor SP (which disables the rest of this "
                     "report); a variable the side was bounding is left "
                     "unbounded below in log space and the solve returns an "
                     "arbitrary value for it; or the backend aborts "
                     "somewhere unrecognisable. If the zero is intended, "
                     "give the row a small additive floor so its right side "
                     "stays positive, or omit the term entirely.")

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

        if self.defaulted_guesses:
            L.append(f"  {len(self.defaulted_guesses)} variables took a "
                     "default guess (require_guesses is off). Harmless for a "
                     "geometric program, which is solved globally in log "
                     "space; for a signomial or black-box model the starting "
                     "point decides which optimum you reach:")
            for nm in self.defaulted_guesses[:12]:
                L.append(f"    {nm}")
            if len(self.defaulted_guesses) > 12:
                L.append(f"    ... and {len(self.defaulted_guesses) - 12} more")
        if self.clean:
            L.append("  no empty columns and every variable is bounded "
                     "both ways")
        if self.rigidity:
            L.append("")
            L.append(rigidity_text(self.rigidity))
        if self.unopposed:
            L.append("")
            L.append(unopposed_text(self.unopposed))
        return "\n".join(L)

    def post_solve_text(self):
        """The checks that need a solution, if one was supplied."""
        L = []
        if self.at_floor:
            L.append(f"  {len(self.at_floor)} variables are resting on the "
                     "solver's positivity floor, which is not a constraint you "
                     "wrote -- they are pinned by the algorithm, not the model:")
            for nm, val in self.at_floor[:12]:
                L.append(f"    {nm} = {val:.3g}")
            if len(self.at_floor) > 12:
                L.append(f"    ... and {len(self.at_floor) - 12} more")
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
    if j < len(names):
        return names[j]
    return f"<var {j}>"


def operator_at(operators, i):
    """The operator of constraint i (1-based, as the rows index them);
    '<=' when the list is short."""
    if 0 <= i - 1 < len(operators):
        return operators[i - 1]
    return "<="


def has_vacuous_bounds(bounds, j):
    """True when x_j's declared box is no box at all (the 1e-30..1e30
    default, or None)."""
    if j < len(bounds) and bounds[j] is not None:
        lo, hi = bounds[j]
    else:
        lo, hi = None, None
    return ((lo is None or lo <= VACUOUS_LO)
            and (hi is None or hi >= VACUOUS_HI))


def max_matching(edges):
    """Kuhn's algorithm: match equality rows to variables they determine.

    edges[i] is the set of variables in equality row i. Returns
    (match_var, match_row): variable -> row that determines it, and row ->
    variable (-1 when unmatched). Matching size = STRUCTURAL rank, an upper
    bound on true rank: an under-determined verdict is genuine; a square
    one may still be singular.
    """
    import sys

    match_var = {}
    match_row = [-1] * len(edges)

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            owner = match_var.get(j)
            if owner is None or augment(owner, seen):
                match_var[j] = i
                match_row[i] = j
                return True
        return False

    # the augmenting search recurses once per row on a long alternating path
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 10 * len(edges) + 1000))
    try:
        for i in range(0, len(edges)):
            augment(i, set())
    finally:
        sys.setrecursionlimit(limit)
    return match_var, match_row


def log_rows_of(structures):
    """``(rows, operators, key)`` for the log-space encoding.

    log_key picks signomial before geometric -- a model can satisfy several
    structure flags at once.
    """
    from lcsolver.presolve.detected import as_detected

    from lcsolver.presolve.unitCorrector import UnitMismatch
    model = None if isinstance(structures, dict) else structures
    try:
        st = as_detected(as_structures(structures))
    except UnitMismatch as exc:
        rep = PresolveReport()
        rep.structure = units_diagnosis(exc)
        return rep
    key = st.log_key
    if key is None:
        raise ValueError("presolve needs a detected GP or SP structure")
    return st[key][1], st[key][2], key


def underdetermined_variables(structures, n) -> set:
    """Variables the equality system leaves free (Dulmage-Mendelsohn).

    Max matching of equality rows to variables, then everything reachable by
    alternating paths from UNMATCHED variables; independent of which maximum
    matching is found.
    """
    try:
        edges, rows, _n, _names = equality_graph(structures, None)
    except Exception:
        return set()
    match_var, match_row = max_matching(edges)

    var_rows = collections.defaultdict(list)
    for i, e in enumerate(edges):
        for j in e:
            var_rows[j].append(i)
    involved = set()
    for e in edges:
        involved |= e
    # unmatched variables, and a variable in NO equality is trivially
    # undetermined by them
    free = set()
    for j in range(0, n):
        if j not in involved or j not in match_var:
            free.add(j)
    seen, stack = set(free), list(free)
    while stack:
        j = stack.pop()
        for i in var_rows.get(j, []):
            u = match_row[i]
            if u >= 0 and u not in seen:
                seen.add(u)
                stack.append(u)
    return seen


def presolve_report(structures, names=None) -> PresolveReport:
    """Run the structural checks. Nothing is solved and nothing is modified."""
    rows, operators, _ = log_rows_of(structures)
    st = as_detected(structures)
    if names is None:
        names = [str(v) for v in structures.get("variables", [])]

    # Index 0 is the objective; being in it is not a constraint, so it is
    # excluded from the scan
    n = max([len(r) - 2 for r in rows] + [len(names)])
    con_idx = st.constraint_indices

    rep = PresolveReport(n_variables=n, n_rows=len(con_idx),
                         names=list(names))

    in_rows = [set() for _ in range(n)]      # rows the variable appears in
    in_real = [set() for _ in range(n)]      # ... that are not plain bounds
    upper = [False] * n
    lower = [False] * n
    patterns = collections.Counter()

    # Tightest bound seen from any source (declared box or single-variable
    # row), kept numerically so a vacuous limit can be told from a real one
    box_lo = [None] * n
    box_hi = [None] * n

    # Bounds the detector split out (bounds_as_rows=False) never appear as rows
    for j, pair in enumerate(structures.get("bounds") or []):
        if j < n and pair is not None:
            note_bound(box_lo, box_hi, j, pair[0], pair[1])

    # The objective bounds a positive-exponent variable above, as a
    # constraint would
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
            # Record the value; whether it bounds anything is decided below
            j = next(iter(touched))
            lo, hi = bound_from_term(terms[0], j, op)
            note_bound(box_lo, box_hi, j, lo, hi)
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

    # Count real constraints, not bound rows -- counting the box would hide a
    # singleton column
    for j in range(n):
        # A variable whose rows all folded into bounds is not unconstrained;
        # distinguish by whether anything meaningful bounds it
        bounded = ((box_lo[j] is not None and box_lo[j] > VACUOUS_LO)
                   or (box_hi[j] is not None and box_hi[j] < VACUOUS_HI))
        if not in_rows[j] and not bounded:
            rep.empty_columns.append(nm_at(names, j))
        elif not in_real[j]:
            rep.bound_only_columns.append(nm_at(names, j))
        elif len(in_real[j]) == 1:
            rep.singleton_columns.append(nm_at(names, j))
        if not upper[j]:
            rep.unbounded_above.append(nm_at(names, j))
        if not lower[j]:
            rep.unbounded_below.append(nm_at(names, j))

    hist = collections.Counter(len(s) for s in in_real)
    rep.row_counts = dict(sorted(hist.items()))

    # Output-only detection needs terms per constraint and separated bounds
    if structures.get("bounds") is not None:
        obj_vars = set()
        for t in st.terms(0):
            obj_vars |= t.variables
        outs = output_only_columns(st, con_idx, structures["bounds"], obj_vars, n)
        rep.output_columns = [nm_at(names, j) for j, _i in outs]
    return rep


def note_bound(box_lo, box_hi, j, lo, hi):
    """Tighten the recorded box of x_j with another source's bound."""
    if lo is not None:
        if box_lo[j] is None:
            box_lo[j] = lo
        else:
            box_lo[j] = max(box_lo[j], lo)
    if hi is not None:
        if box_hi[j] is None:
            box_hi[j] = hi
        else:
            box_hi[j] = min(box_hi[j], hi)


def intersect_bounds(bounds, j, lo, hi, names):
    """Intersect x_j's (lo, hi) box with another bound, in place. Crossed
    bounds prove infeasibility: say so rather than silently picking a side
    and answering a different question."""
    cur_lo, cur_hi = bounds[j]
    if lo is not None:
        if cur_lo is None:
            cur_lo = lo
        else:
            cur_lo = max(cur_lo, lo)
    if hi is not None:
        if cur_hi is None:
            cur_hi = hi
        else:
            cur_hi = min(cur_hi, hi)
    if (cur_lo is not None and cur_hi is not None
            and cur_hi < cur_lo * (1.0 - 1e-9)):
        raise InfeasibleProblem(
            f"{nm_at(names, j)} is required to be both >= {cur_lo:g} and "
            f"<= {cur_hi:g}; the model has no feasible point")
    bounds[j] = (cur_lo, cur_hi)


def fold_singleton_rows(structures, only=None):
    """Move single-variable rows into the bounds, and drop them.

    Classical singleton-row reduction; SPaircraft writes ~2500 constraints
    this way. ``only`` restricts the fold to those constraint indices: a
    singleton MODEL row must stay a row so the elastic relaxation can put
    slack on it -- folding those turned the b737 case from 38 iterations into
    a 200-iteration stall. Requires structures['bounds'] (bounds_as_rows=False);
    returns a new structures dict. Bounds are intersected, never loosened or
    clipped -- the reference solution needs the full 1e-30..1e30 box.
    """
    if structures.get("bounds") is None:
        raise ValueError(
            "fold_singleton_rows needs structures['bounds'] to fold into; "
            "run structure_detector with bounds_as_rows=False")

    rows, operators, key = log_rows_of(structures)
    names = [str(v) for v in structures.get("variables", [])]
    numer, denom = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        idx = int(r[0])
        (numer if idx >= 0 else denom)[
            idx if idx >= 0 else -idx - 1].append(r)

    bounds = []
    for b in structures["bounds"]:
        if b is None:
            bounds.append((None, None))
        else:
            bounds.append(tuple(b))

    folded = set()
    for i in sorted(k for k in set(numer) | set(denom) if k != 0):
        if only is not None and i not in only:
            continue
        if denom.get(i) or len(numer.get(i, [])) != 1:
            continue
        row = numer[i][0]
        expo = [float(e) for e in row[2:]]
        nz = [j for j, e in enumerate(expo) if abs(e) > 1e-12]
        if len(nz) != 1 or nz[0] >= len(bounds):
            continue
        j = nz[0]
        op = operator_at(operators, i)
        lo, hi = bound_from_row(float(row[1]), expo, j, op)
        if lo is None and hi is None:
            continue
        intersect_bounds(bounds, j, lo, hi, names)
        folded.add(i)

    # Renumber survivors; indices stay contiguous from 1 because the operator
    # list is positional
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

    Unpacks as (index, name, value, reason); `recover` carries the data an
    output-only variable needs after the core solve.
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


def evaluate_terms(terms, x, j=None, tj=None):
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
        # Saturate BOTH ways: solve_row_for probes brackets near log(1e300),
        # where bare math.exp overflows; only the sign matters out there
        if acc > 700:
            total += math.inf
        elif acc > -700:
            total += math.exp(acc)
    return total


def solve_row_for(num, den, j, x, lo=1e-300, hi=1e300):
    """Solve ``num/den == 1`` for ``x_j`` by bisection on ``log x_j``.

    The caller already established monotonicity in x_j, so a sign change is
    bracketed and bisection is enough. Runs once per variable after the solve.
    """
    import math

    a, b = math.log(lo), math.log(hi)
    fa = log_ratio_at(num, den, j, x, a)
    fb = log_ratio_at(num, den, j, x, b)
    # On a steep row (P**20 == posynomial) BOTH ends can come back infinite;
    # that is a valid sign change -- refusing it left the ISA fitted pressure
    # at the 1.0 placeholder (129 Pa for 70 kPa). Only NaN means the row
    # cannot be evaluated.
    if math.isnan(fa) or math.isnan(fb):
        return None
    if fa == 0.0:
        return math.exp(a)
    if fb == 0.0:
        return math.exp(b)
    if (fa > 0) == (fb > 0):
        return None                    # no sign change: not recoverable here
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = log_ratio_at(num, den, j, x, m)
        if fm == 0.0:
            return math.exp(m)
        if (fm > 0) == (fa > 0):
            a, fa = m, fm
        else:
            b, fb = m, fm
    return math.exp(0.5 * (a + b))


def log_ratio_at(num, den, j, x, t):
    """log(num/den) with log x_j overridden by t; +-inf where a side is
    non-positive, so a bisection can still read the sign."""
    import math

    n = evaluate_terms(num, x, j, t)
    if den:
        d = evaluate_terms(den, x, j, t)
    else:
        d = 1.0
    if n <= 0:
        return -math.inf
    if d <= 0:
        return math.inf
    return math.log(n) - math.log(d)


def output_only_columns(st, con_idx, bounds, in_objective, n, protect=frozenset()):
    """Variables that are computed but never fed back, peeled in rounds.

    Output-only: exactly one constraint, absent from the objective, monotone
    in the variable, and vacuously bounded in the relaxing direction --
    ``A_tri <= 100`` on ``A_tri >= f(...)`` would force f <= 100, a real
    restriction. Iterative because peeling one can expose another. Returns
    [(j, constraint_index), ...] in peel order. ``protect`` columns are never
    peeled: a grey-box-fed variable can look output-only to the algebraic
    rows while its black box pins it.
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
            if (j in taken or j in protect or j in in_objective
                    or len(rows.get(j, ())) != 1):
                continue
            i = next(iter(rows[j]))
            op = st.operator(i)

            # Monotone in x_j? Numerator exponents one sign, denominator the
            # other; mixed signs mean the constraint really does pin it
            signs = set()
            for t in st.terms(i):
                e = t.exponents.get(j)
                if e is None or abs(e) <= 1e-12:
                    continue
                signs.add((e > 0) != t.denominator)
            if len(signs) != 1:
                continue
            grows_tighter = signs.pop()

            if j < len(bounds) and bounds[j] is not None:
                lo, hi = bounds[j]
            else:
                lo, hi = None, None
            if op == "==":
                # An equality pins x_j exactly; it restricts others only via
                # x_j's own bounds, so both must be vacuous
                free = has_vacuous_bounds(bounds, j)
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
            # Callers recover in REVERSE of this order: a round-1 variable may
            # sit in the constraint defining a round-2 one
            return order


def tighten_linear_bounds(linear, L, U, names, max_passes, min_gain):
    """Interval propagation on ``coeffs . v <= rhs`` (or ``==``).

    L/U bound v in whatever space the caller works in (natural for LP, log
    for GP). Isolate ``a_k v_k <= b - S`` with S at its minimum; an equality
    also gives >= with S at its MAXIMUM -- a different sum, and reusing the
    minimum for both once "proved" SPaircraft infeasible from two unrelated
    monomial equalities. Mutates L/U in place; returns the tighten count.
    """
    import math

    NEG, POS = -math.inf, math.inf
    tightened = 0

    for _pass in range(max_passes):
        changed = False
        for rhs, a, nz, eq in linear:
            # Sum at min (and max for an equality), each carrying its own
            # count of infinite contributions
            if eq:
                ends = (True, False)
            else:
                ends = (True,)
            sums = {}
            for want_min in ends:
                tot, infs, at = 0.0, 0, -1
                for j in nz:
                    m = endpoint_contribution(a[j], L[j], U[j], want_min)
                    if m == NEG or m == POS:
                        infs += 1
                        at = j
                        if infs > 1:
                            break
                    else:
                        tot += m
                sums[want_min] = (tot, infs, at)

            for k in nz:
                for want_min in ends:
                    tot, infs, at = sums[want_min]
                    if infs > 1 or (infs == 1 and k != at):
                        continue
                    if infs == 1 and k == at:
                        rest = tot
                    else:
                        rest = tot - endpoint_contribution(a[k], L[k], U[k],
                                                           want_min)
                    if rest == NEG or rest == POS:
                        continue
                    limit = (rhs - rest) / a[k]
                    # want_min bounds a_k v_k from ABOVE, want_max from BELOW
                    upper = (a[k] > 0) == want_min
                    if upper:
                        if limit < U[k] - min_gain:
                            U[k] = limit
                            tightened += 1
                            changed = True
                    else:
                        if limit > L[k] + min_gain:
                            L[k] = limit
                            tightened += 1
                            changed = True
                    if L[k] > U[k] + 1e-6:
                        raise InfeasibleProblem(
                            f"bound propagation drove {nm_at(names, k)} to an "
                            "empty range; the model has no feasible point")
        if not changed:
            break
    return tightened


def endpoint_contribution(a_j, lo, hi, want_min):
    """a_j * v_j at whichever endpoint of [lo, hi] minimises (or maximises)
    it."""
    if want_min:
        if a_j > 0:
            return a_j * lo
        return a_j * hi
    if a_j > 0:
        return a_j * hi
    return a_j * lo


def propagate_bounds(structures, max_passes=8, min_gain=1e-6):
    """Tighten variable bounds by interval propagation.

    Works on LP/QP and GP/SP: a monomial is linear in y = log x, so the LP
    propagation applies unchanged (see :func:`_tighten_linear`). Each term of
    a posynomial separately satisfies ``c_k m_k(x) <= 1``, handing over one
    linear implication per term -- that is what makes this worth running on a
    GP. Ratios are left alone. Returns ``(structures, n_tightened)`` with a
    new bounds list; raises InfeasibleProblem if a range comes out empty.
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
        # Natural variables: rows are AG . x <= b, and x may be negative
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
        k = tighten_linear_bounds(linear, L, U, names, max_passes, min_gain)
        new_bounds = [(None if L[j] == NEG else L[j],
                       None if U[j] == POS else U[j]) for j in range(n)]
    else:
        rows, operators, key = log_rows_of(structures)
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
            op = operator_at(operators, i)
            terms = numer.get(i, [])
            # Only a single-term equality is an equality term-wise; a
            # multi-term one implies just the <= half per term
            eq = (op == "==" and len(terms) == 1)
            for r in terms:
                c = float(r[1])
                if c <= 0:
                    continue
                a = [float(e) for e in r[2:]] + [0.0] * (n - (len(r) - 2))
                nz = [j for j in range(n) if abs(a[j]) > 1e-12]
                if nz:
                    linear.append((-math.log(c), a, nz, eq))
        k = tighten_linear_bounds(linear, L, U, names, max_passes, min_gain)
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

    A monomial equality is linear in y = log x: solving for a pivot p gives
    ``x_p = c**(-1/a_p) * prod x_j**(-a_j/a_p)``, itself a monomial, so
    substitution keeps posynomial structure exactly. SPaircraft carries 752
    such equalities. Only variables with vacuous declared bounds are
    eliminated -- a real bound would come back as two monomial rows. Fill-in
    is the cost: pivots are chosen by Markowitz estimate
    ``(row_nnz - 1) * (col_nnz - 1)``, capped at ``max_fill``. Returns
    ``(structures, removed)`` in :func:`reduce_columns` form.
    """
    import math

    if structures.get("bounds") is None:
        raise ValueError(
            "eliminate_monomial_equalities needs structures['bounds']; run "
            "structure_detector with bounds_as_rows=False first")

    rows, operators, key = log_rows_of(structures)
    names = [str(v) for v in structures.get("variables", [])]
    bounds = list(structures["bounds"])
    n = max([len(r) - 2 for r in rows] + [len(bounds)])
    while len(bounds) < n:
        bounds.append((None, None))

    # Sparse form: constraint -> list of (coeff, {j: exponent}, is_denominator)
    check_width(structures, n, "eliminate_monomial_equalities")

    terms = collections.defaultdict(list)
    for r in rows:
        idx = int(r[0])
        if idx >= 0:
            i = idx
        else:
            i = -idx - 1
        e = {}
        for j, v in enumerate(r[2:]):
            if abs(float(v)) > 1e-12:
                e[j] = float(v)
        terms[i].append([float(r[1]), e, idx < 0])

    con_idx = sorted(k for k in terms if k != 0)
    alive = set(con_idx)

    # Column occupancy for the Markowitz estimate. The OBJECTIVE (index 0)
    # must be in here: without it a pivot in the objective is substituted
    # everywhere except there, silently changing it -- measured on turbofan,
    # 0.269 became 0.060 with both runs reporting convergence.
    col = collections.defaultdict(set)
    for i in [0] + list(con_idx):
        for _c, e, _d in terms[i]:
            for j in e:
                col[j].add(i)

    removed, done = [], 0
    changed = True
    while changed:
        changed = False
        # Candidate monomial equalities, cheapest pivot first
        cands = []
        for i in sorted(alive):
            if (operator_at(operators, i) != "==" or len(terms[i]) != 1
                    or terms[i][0][2]):
                continue
            c_eq, a, _d = terms[i][0]
            if c_eq <= 0:
                continue
            for pj, ap in a.items():
                if abs(ap) < min_pivot or not has_vacuous_bounds(bounds, pj):
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
                # k == 0 is the objective: never in `alive`, still needs
                # substituting
                if k == i or (k != 0 and k not in alive):
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

    st = as_detected(structures)
    info = dict(structures.get("info") or {})
    info["N_vars_substituted"] = len(removed)
    info["N_cons_total"] = len(surviving)
    out = st.rebuild(
        renumbered_terms(terms[0], pos),
        [renumbered_terms(terms[i], pos) for i in surviving],
        [operator_at(operators, i) for i in surviving],
        n=len(keep),
        bounds=[bounds[j] for j in keep],
        info=info)
    if structures.get("variables"):
        out["variables"] = [structures["variables"][j] for j in keep]

    # REVERSE elimination order: a pivot's formula may reference variables
    # eliminated later, so those must be known first. Forward recovery was off
    # by 4.3e+03 on SPaircraft while the reduced problem stayed exact.
    return out, removed[::-1]


def renumbered_terms(sparse_terms, pos):
    """The surviving [coeff, {j: e}, is_denominator] terms as Terms over
    the renumbered columns."""
    out_terms = []
    for coeff, expo, den in sparse_terms:
        exponents = {}
        for j, v in expo.items():
            if j in pos and abs(v) > 1e-12:
                exponents[pos[j]] = v
        out_terms.append(Term(coeff=float(coeff), exponents=exponents,
                              denominator=bool(den)))
    return out_terms


def fold_and_renumber(terms, drop, pos):
    """Fold removed constants into each term's coefficient and renumber the
    surviving columns."""
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


def reduce_columns(structures, guess=None, eliminate_outputs=True,
                   protect=None):
    """Remove variables the model does not connect to anything. Exact.

    Disconnected: in no real constraint or objective term -- fixed at its
    tightest finite bound and dropped. Fixed: equal bounds -- folded into the
    coefficients as a constant. Requires ``structures['bounds']``; run
    :func:`fold_singleton_rows` first or ``y >= 4`` still counts as a
    constraint. Returns ``(reduced_structures, removed)`` for
    :func:`restore_columns`. ``protect`` columns always survive: a variable
    only a grey box touches looks disconnected here while being entirely
    live. Crossed bounds on a protected column still raise.
    """
    if structures.get("bounds") is None:
        raise ValueError(
            "reduce_columns needs structures['bounds']; run structure_detector "
            "with bounds_as_rows=False (and fold_singleton_rows) first")

    if guess is None:
        # The variables' current values ARE the author's guesses -- for a
        # disconnected variable that is the only information anyone has, and
        # reporting 1.0 instead would discard it
        try:
            import pyomo.environ as pyo
            guess = [float(pyo.value(v)) for v in structures.get("variables")
                     or []]
        except Exception:
            guess = None

    rows, operators, key = log_rows_of(structures)
    names = [str(v) for v in structures.get("variables", [])]
    bounds = list(structures["bounds"])
    n = max([len(r) - 2 for r in rows] + [len(bounds)])

    check_width(structures, n, "reduce_columns")

    in_objective, in_constraint = set(), set()
    for r in rows:
        target = in_objective if int(r[0]) == 0 else in_constraint
        for j, e in enumerate(r[2:]):
            if abs(float(e)) > 1e-12:
                target.add(j)

    # Group once; the output-only scan needs terms per constraint
    numer, denom = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        idx = int(r[0])
        expo = [float(e) for e in r[2:]] + [0.0] * (n - (len(r) - 2))
        (numer if idx >= 0 else denom)[
            idx if idx >= 0 else -idx - 1].append((float(r[1]), expo))
    con_idx = sorted(k for k in set(numer) | set(denom) if k != 0)

    protect = frozenset(protect or ())
    outputs = []
    if eliminate_outputs:
        outputs = output_only_columns(as_detected(structures), con_idx, bounds,
                               in_objective, n, protect=protect)
    out_vars = {j for j, _i in outputs}
    out_cons = {i for _j, i in outputs}

    removed = []
    for j in range(n):
        if j < len(bounds) and bounds[j] is not None:
            lo, hi = bounds[j]
        else:
            lo, hi = None, None

        if lo is not None and hi is not None and lo > 0:
            if hi < lo * (1.0 - 1e-9):
                raise InfeasibleProblem(
                    f"{nm_at(names, j)} is required to be both >= {lo:g} and "
                    f"<= {hi:g}; the model has no feasible point")
            if hi <= lo * (1.0 + 1e-9) and j not in protect:
                removed.append(Removed(j, nm_at(names, j), float(lo), "fixed"))
                continue

        if j in protect or j in out_vars:
            continue                    # out_vars handled below, in peel order

        if j in in_constraint or j in in_objective:
            continue
        # Nothing refers to it: take the tightest meaningful bound, else the
        # user's guess
        if lo is not None and lo > VACUOUS_LO:
            val = float(lo)
        elif hi is not None and hi < VACUOUS_HI:
            val = float(hi)
        elif guess is not None and j < len(guess) and guess[j] > 0:
            val = float(guess[j])
        else:
            val = 1.0
        removed.append(Removed(j, nm_at(names, j), val, "disconnected"))

    # Output-only variables carry their defining constraint, not a value;
    # recorded in REVERSE peel order, the order they can be evaluated
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

    st = as_detected(structures)
    info = dict(structures.get("info") or {})
    info["N_vars_removed"] = len(removed)
    info["N_vars_output"] = len(outputs)
    info["N_cons_total"] = len(surviving)
    kept_bounds = []
    for j in keep:
        if j < len(bounds):
            kept_bounds.append(bounds[j])
        else:
            kept_bounds.append((None, None))
    out = st.rebuild(
        fold_and_renumber(st.terms(0), drop, pos),
        [fold_and_renumber(st.terms(i), drop, pos) for i in surviving],
        [st.operator(i) for i in surviving],
        n=len(keep),
        bounds=kept_bounds,
        info=info)
    if structures.get("variables"):
        out["variables"] = [structures["variables"][j] for j in keep]
    return out, removed


def restore_columns(removed, x_reduced, n_original=None):
    """Put removed variables back into a reduced solution vector.

    Constants go straight back; output-only variables are post-computed from
    their defining constraint at the solved values. ``removed`` already holds
    the output entries in evaluation order.
    """
    import math

    import numpy as np

    x_reduced = np.asarray(x_reduced, dtype=float)
    if n_original is None:
        n_original = len(x_reduced) + len(removed)

    constants = {}
    outputs = []
    for r in removed:
        if r.recover is not None:
            outputs.append(r)
        elif r.reason != 'output':
            constants[r.index] = r.value
    placed = set(constants) | {r.index for r in outputs}

    if len(x_reduced) == n_original:
        # The backend solved in the FULL frame (GP-IPOPT builds from the
        # model): positions are already right, only removed entries need
        # overwriting. Slotting a full vector through the reduced branch
        # shifted every value after the first peeled index by one.
        out = x_reduced.copy()
        for j, v in constants.items():
            out[j] = v
    else:
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
            tag, C, m = spec
            acc = math.log(C) if C > 0 else -math.inf
            for j, e in m.items():
                if j < len(out) and out[j] > 0:
                    acc += e * math.log(out[j])
            val = math.exp(acc) if -700 < acc < 700 else (
                0.0 if acc <= -700 else math.inf)
        else:
            num, den, j0 = spec
            val = solve_row_for(num, den, j0, out)
        if val is not None:
            out[r.index] = val
            r.value = float(val)
        else:
            # A silent 1.0 here would look like an answer. Say so.
            import warnings
            warnings.warn(
                f"[LC-W310] presolve could not recover the output-only "
                f"variable {r.name!r} from its defining constraint after the "
                f"solve; it is reported as 1.0, which is NOT its value",
                RuntimeWarning, stacklevel=2)
    return out


def cancellation_report(structures, x, tol=1e-6, names=None):
    """Terms in a signomial constraint that contribute nothing at ``x``.

    LCsolver writes subtractions as ratios, so denominator terms compete;
    when one supplies essentially all of it the other is inert and the
    quantity it carries is disconnected -- how a variable ends up parked at
    1e-30 with nothing complaining. Solution-dependent, so it runs after a
    solve. Returns ``[(constraint_index, side, share, variables), ...]`` for
    terms whose share falls below ``tol``, worst first.
    """
    import math

    import numpy as np

    x = np.asarray(x, dtype=float)
    st = as_detected(structures)
    if names is None:
        names = [str(v) for v in structures.get("variables", [])]

    out = []
    # Only signomial constraints: a small term in a plain posynomial is
    # ordinary, not a defect
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

    Violation is log-space for GP/SP, natural for LP/QP. Equalities
    contribute |residual|, inequalities the positive part; bounds count too.
    Solver-independent, so it can compare two structures no solver has seen.
    """
    import math

    x = list(x)
    # Dispatch as propagate_bounds does, linear flag first -- it must match,
    # and a model with negative variables can carry a signomial flag whose
    # log encoding would take logs of negatives
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
            # LCsolver stores the quadratic COEFFICIENT matrix, not the
            # Hessian: objective is x'Px + q'x + shift with no half (solve_QP
            # passes 2P to cvxopt for the same reason). Omitting the quadratic
            # term entirely reported the leftover LP's objective.
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

    Maps ``x`` through the transform and compares objective and worst
    violation; returns (objective_relative_error, violation_absolute_error).
    ``log`` is a PresolveLog used to drop removed columns; pass ``x_after``
    for other mappings. This caught the elimination bug (recovered values off
    by 4.3e+03) and would have caught the propagation bug on sight.
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

    Each pass renumbers columns, so a ``Removed.index`` is relative to the
    structure when that pass ran; keeping the lists separate and unwinding
    in reverse needs no remapping. ``print(log)`` for the account; off by
    default since the reductions are exact.
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


def floor_report(x, names=None, x_min=1e-9, rtol=1e-3):
    """Variables resting on the solver's positivity floor.

    The floor (``x_min``) is a property of the ALGORITHM, not the model -- a
    quantity at 1e-9 reads as "essentially zero" when it really means
    "nothing in this model has an opinion". Returns ``[(name, value), ...]``.
    """
    out = []
    for j, v in enumerate(x):
        if v is not None and 0 < v <= x_min * (1.0 + rtol):
            out.append((nm_at(list(names or []), j), float(v)))
    return out



def equality_graph(structures, names=None):
    """``(edges, rows, n, names)`` for the bipartite equality/variable graph.

    Single-variable equalities are kept: ``x == 3`` consumes a degree of
    freedom.
    """
    st = as_detected(as_structures(structures))
    if names is None:
        # Read names off the DETECTED object -- the same route as the rows
        # keeps name index and column index the same index
        names = [str(v) for v in (st.variables or [])]
    rows, edges = [], []
    n = len(names)
    for i in st.constraint_indices:
        if st.operator(i) != "==":
            continue
        touched = set()
        for t in st.terms(i):
            touched |= t.variables
        if touched:
            rows.append(i)
            edges.append(touched)
            n = max(n, max(touched) + 1)
    return edges, rows, n, list(names)


def unopposed_report(structures, names=None, top=25):
    """Variables no constraint resists -- quantities the optimiser moves free.

    The bounded-ness check misses these: it treats ANY equality as bounding
    both ways, but an equality constrains a COMBINATION, not an individual.
    A wing model carried p and q ("1 + 2 taper", "1 + taper") pinned only by
    whichever structural model read them; swapping models inflated q
    1.15 -> 1.56, shrinking the mean chord 20%. Reported when
    Dulmage-Mendelsohn leaves a variable under-determined AND some direction
    has no inequality resisting it, as ``[(name, direction, n_rows)]``.
    A report only; the solve's bounds are untouched.
    """
    rows, operators, key = log_rows_of(structures)
    st = as_detected(as_structures(structures))
    if names is None:
        names = [str(v) for v in (st.variables or [])]
    n = max([len(r) - 2 for r in rows] + [len(names)])
    loose = underdetermined_variables(structures, n)

    up_held = [False] * n
    down_held = [False] * n
    touch = collections.Counter()
    for i in st.constraint_indices:
        op = st.operator(i)
        terms = st.terms(i)
        seen = set()
        for t in terms:
            seen |= t.variables
        for j in seen:
            touch[j] += 1
        if op == "==":
            continue                      # handled by the DM block above
        for t in terms:
            sgn = -1.0 if t.denominator else 1.0
            for j, e in t.exponents.items():
                if sgn * e > 0:
                    up_held[j] = True     # raising j tightens p/q <= 1
                elif sgn * e < 0:
                    down_held[j] = True

    out = []
    for j in sorted(loose):
        if j >= n:
            continue
        free = [d for d, held in (("up", up_held[j]), ("down", down_held[j]))
                if not held]
        if free and touch[j]:
            out.append((nm_at(names, j), "/".join(free), touch[j]))
    out.sort(key=lambda t: -t[2])
    return out[:top]


def unopposed_text(found) -> str:
    """:func:`unopposed_report` as the paragraph a reader wants."""
    if not found:
        return ""
    L = [f"  {len(found)} variables are under-determined by the equalities AND "
         "unresisted by any inequality -- the optimiser can move these for "
         "free, and a converged answer may be resting on them:"]
    for nm, direction, k in found:
        L.append(f"    {nm}  (free to move {direction}; {k} rows mention it)")
    L.append("    A variable named by MANY rows and resisted by none is the "
             "dangerous case: it looks wired and is not.")
    return "\n".join(L)


def rigidity_report(structures, names=None, cluster_max=12):
    """Degrees of freedom, per variable, from the equality system alone.

    Every equality spends one. Reports variables an equality determines
    (outputs in design-variable clothes -- diff across a change to see the
    freedoms you removed), over-determined groups (more equations than
    unknowns), and rigid clusters (dof <= 1: a bound on any member binds all,
    invisible in the source). Inequalities are not in this graph, so rigidity
    warns a conflict is possible, not that it happened. ``cluster_max`` caps
    printable cluster size -- a 400-variable rigid block is the model, not a
    finding.
    """
    edges, rows, n, names = equality_graph(structures, names)
    match_var, match_row = max_matching(edges)

    involved = set()
    for e in edges:
        involved |= e
    rank = len(match_var)

    rep = {
        'n_equalities': len(edges),
        'n_variables': n,
        'n_involved': len(involved),
        'structural_rank': rank,
        'dof': n - rank,
        'determined': sorted(nm_at(names, j) for j in match_var),
        'overdetermined': [],
        'rigid_clusters': [],
        'incidence': {},
    }

    # Equalities touching each variable. A high count marks a hub, and a
    # bound on a hub propagates everywhere.
    inc = collections.Counter()
    for e in edges:
        for j in e:
            inc[j] += 1
    rep['incidence'] = {nm_at(names, j): c for j, c in inc.most_common()}

    # -- over-determined block (Dulmage-Mendelsohn) ------------------------
    # Unmatched rows have no variable of their own; alternating paths back
    # from them collect everything implicated.
    var_rows = collections.defaultdict(list)
    for i, e in enumerate(edges):
        for j in e:
            var_rows[j].append(i)

    unmatched = [i for i in range(len(edges)) if match_row[i] < 0]
    if unmatched:
        seen_r, seen_v, stack = set(unmatched), set(), list(unmatched)
        while stack:
            i = stack.pop()
            for j in edges[i]:
                if j in seen_v:
                    continue
                seen_v.add(j)
                owner = match_var.get(j)
                if owner is not None and owner not in seen_r:
                    seen_r.add(owner)
                    stack.append(owner)
        rep['overdetermined'] = [
            {'rows': sorted(rows[i] for i in seen_r),
             'variables': sorted(nm_at(names, j) for j in seen_v),
             'excess': len(seen_r) - len(seen_v)}]

    # -- rigid clusters ----------------------------------------------------
    # Connected components of the equality graph. A component with as many
    # equalities as variables has no freedom left; one short of that has a
    # single freedom, so every member moves in lockstep with every other.
    seen_r, comps = set(), []
    for start in range(len(edges)):
        if start in seen_r:
            continue
        seen_r.add(start)
        cr, cv, stack = [start], set(), [start]
        while stack:
            i = stack.pop()
            for j in edges[i]:
                if j in cv:
                    continue
                cv.add(j)
                for k in var_rows[j]:
                    if k not in seen_r:
                        seen_r.add(k)
                        cr.append(k)
                        stack.append(k)
        comps.append((cr, cv))

    for cr, cv in comps:
        dof = len(cv) - len(cr)
        if dof <= 1 and len(cv) <= cluster_max:
            rep['rigid_clusters'].append({
                'rows': sorted(rows[i] for i in cr),
                'variables': sorted(nm_at(names, j) for j in cv),
                'dof': dof})
    rep['rigid_clusters'].sort(key=lambda c: (c['dof'], -len(c['variables'])))
    return rep


def rigidity_text(rep, top=8):
    """:func:`rigidity_report` as the paragraph a reader wants."""
    L = [f"degrees of freedom: {rep['n_variables']} variables, "
         f"{rep['n_equalities']} equalities of structural rank "
         f"{rep['structural_rank']} -> {rep['dof']} free"]

    for blk in rep['overdetermined']:
        L.append(f"  OVER-DETERMINED: {len(blk['rows'])} equalities on "
                 f"{len(blk['variables'])} variables ({blk['excess']} more "
                 "equations than unknowns). No assignment satisfies all of "
                 "them; the solve will report infeasible or silently satisfy "
                 "them only to tolerance:")
        L.append(f"    rows {blk['rows'][:12]}"
                 + (" ..." if len(blk['rows']) > 12 else ""))
        L.append(f"    variables: {', '.join(blk['variables'][:10])}"
                 + (" ..." if len(blk['variables']) > 10 else ""))

    rigid = rep['rigid_clusters']
    if rigid:
        L.append(f"  {len(rigid)} rigid clusters -- every variable in one "
                 "moves in lockstep with the others, so a bound on any member "
                 "acts as a bound on all of them:")
        for c in rigid[:top]:
            kind = "fully determined" if c['dof'] == 0 else "1 degree of freedom"
            L.append(f"    [{kind}] {', '.join(c['variables'][:8])}"
                     + (" ..." if len(c['variables']) > 8 else "")
                     + f"  (rows {c['rows'][:6]}"
                     + (" ...)" if len(c['rows']) > 6 else ")"))
        if len(rigid) > top:
            L.append(f"    ... and {len(rigid) - top} more")

    det = rep['determined']
    if det:
        L.append(f"  {len(det)} variables are determined by equalities rather "
                 "than chosen by the optimiser -- they are outputs, and any "
                 "one of them that you added recently is a degree of freedom "
                 "you removed")
    return "\n".join(L)


# What each class means for the solve -- the report exists to answer "so what"
CLASS_INFO = {
    'Linear_Program': (
        'Linear Program (LP)',
        'one convex solve; global optimum, exact duals'),
    'Quadratic_Program': (
        'Quadratic Program (QP)',
        'one convex solve; global optimum'),
    'Geometric_Program': (
        'Geometric Program (GP)',
        'convex in log space: one convex solve, global optimum'),
    'Signomial_Program': (
        'Signomial Program (SP)',
        'convex in no variables; solved as a sequence of convex subproblems '
        '(SIA/PCCP), so the optimum is LOCAL'),
}

# Simplest first; a model is reported as the first class it satisfies
CLASS_ORDER = ['Linear_Program', 'Quadratic_Program', 'Geometric_Program',
                'Signomial_Program']


def clean_expression(text, width=88):
    """A constraint body as a reader wants it, not as Pyomo prints it."""
    for junk in ('dimensionless*', '*dimensionless', ' dimensionless'):
        text = text.replace(junk, '')
    text = ' '.join(text.split())
    return text if len(text) <= width else text[:width - 3] + '...'


def constraint_bodies(structures):
    """``{name: body}`` for every constraint on the detected model."""
    model = structures.get('model') if hasattr(structures, 'get') else None
    if model is None:
        return {}
    try:
        import pyomo.environ as pyo
        bodies = {c.name: clean_expression(str(c.expr))
                  for c in model.component_data_objects(ctype=pyo.Constraint)}
        objs = [clean_expression(str(o.expr))
                for o in model.component_data_objects(ctype=pyo.Objective)]
        if objs:
            bodies['the objective'] = objs[0]
        return bodies
    except Exception:
        return {}


def as_structures(obj):
    """Accept either the detector's output or the formulation itself.

    optimization_check(f) is the call people try first; making it work costs
    one isinstance (Detected is a dict subclass, a Formulation is not).
    """
    if isinstance(obj, dict):
        return obj
    from lcsolver.presolve.structureDetector import structure_detector
    from lcsolver.presolve.unitCorrector import unit_corrector
    return structure_detector(unit_corrector(obj), bounds_as_rows=False)


def units_diagnosis(exc):
    """A unit failure, formatted as a finding rather than raised as an error.

    Asking what is wrong with a model is exactly when it is most likely to
    be wrong; nothing downstream can run, so this is the whole report.
    """
    return ('units\n-----\n' + str(exc).rstrip() + '\n\n'
            '  Nothing further can be checked until the units balance: the\n'
            '  structure detector reads the unit-corrected model, and this one\n'
            '  has no correction. Fix the above and run this again.')


def gp_after_presolve(structures):
    """Is the problem the solver actually receives a geometric program?

    Answered on the presolved rows, not by tracking which original row went
    away -- every pass RENUMBERS, so matching content across them is
    guesswork. The non-GP markers are visible directly: a fraction (a group
    with a denominator, negative row index), a posynomial equality (an ==
    group with more than one term), or a non-positive coefficient. Returns
    None if the presolve cannot run, so the caller can decline to claim
    anything.
    """
    try:
        st = with_empty_bounds(structures)
        st = fold_singleton_rows(st)
        st, eliminated = eliminate_monomial_equalities(st)
        st, reduced_out = reduce_columns(st)
        rows, operators, key = log_rows_of(st)
    except Exception:
        return None

    groups = {}
    for r in rows:
        raw = int(r[0])
        idx = raw if raw >= 0 else -raw - 1
        if idx == 0:
            continue
        g = groups.setdefault(idx, {'numer': 0, 'denom': 0, 'neg': False})
        if raw >= 0:
            g['numer'] += 1
        else:
            g['denom'] += 1
        if float(r[1]) <= 0.0:
            g['neg'] = True

    ops = list(operators or [])
    for idx, g in groups.items():
        if g['denom'] or g['neg']:
            return False
        op = ops[idx - 1] if 0 <= idx - 1 < len(ops) else None
        if op == '==' and g['numer'] > 1:
            return False
    return True


def structure_report(structures, top=5, simplify=True) -> str:
    """What kind of problem this is, and what stops it being a simpler one.

    The detector clears a flag the moment a row rules a class out but never
    said which row; naming it is the difference between a label and an
    action. ``simplify`` also asks whether the blockers survive the presolve
    -- often the class written is not the class solved. ``top`` caps blockers
    listed per class (None for all).
    """
    from lcsolver.presolve.unitCorrector import UnitMismatch
    try:
        st = as_detected(as_structures(structures))
    except UnitMismatch as exc:
        return units_diagnosis(exc)
    blockers = (st.get('blockers') or {}) if hasattr(st, 'get') else {}
    bodies = constraint_bodies(st)

    L = ['structure', '---------']

    detected = next((k for k in CLASS_ORDER
                     if st.get(k) and st[k][0] and st[k][1] is not None), None)
    if detected is None:
        L.append('  unstructured -- no LP, QP, GP or SP form was detected')
        msg = st.get('message') if hasattr(st, 'get') else None
        if msg:
            L.append(f'  {msg}')
        return '\n'.join(L)

    # Only ask whether the solved problem simplifies to a GP; LP/QP would
    # need the linearity test rerun on the reduced rows
    simplified = detected
    if simplify and detected == 'Signomial_Program':
        if gp_after_presolve(st) is True:
            simplified = 'Geometric_Program'

    label, consequence = CLASS_INFO[detected]
    if simplified != detected:
        s_label, s_consequence = CLASS_INFO[simplified]
        L.append(f'  {label} as written')
        L.append(f'  {s_label} as solved -- every constraint that blocked it '
                 f'is removed by the presolve')
        L.append(f'    {s_consequence}')
    else:
        L.append(f'  {label}')
        L.append(f'    {consequence}')

    # Only classes SIMPLER than the one detected: a GP is not "failing to be
    # an SP"
    rank = CLASS_ORDER.index(detected)
    if rank > CLASS_ORDER.index('Geometric_Program'):
        if simplified == 'Geometric_Program':
            headline = 'Not a Geometric Program as written'
            advice = ('The solver dispatches on the as-written class, so this '
                      'still routes through the SP loop; the presolve then '
                      'hands that loop a GP, which is why it converges in a '
                      'couple of iterations.')
        else:
            headline = 'Not a Geometric Program'
            advice = ('Reformulate those and the model becomes a GP: one '
                      'convex solve, global optimum, no iteration.')
        blocker_section(L, blockers, bodies, ['Geometric_Program'], headline,
                        advice, top)
    if rank > CLASS_ORDER.index('Quadratic_Program'):
        lp = {r[0] for r in blockers.get('Linear_Program', ())}
        qp = {r[0] for r in blockers.get('Quadratic_Program', ())}
        if lp == qp:
            blocker_section(L, blockers, bodies, ['Linear_Program'],
                            'Not a Linear or Quadratic Program', None, top)
        else:
            blocker_section(L, blockers, bodies, ['Quadratic_Program'],
                            'Not a Quadratic Program', None, top)
            blocker_section(L, blockers, bodies, ['Linear_Program'],
                            'Not a Linear Program', None, top)
    elif rank > CLASS_ORDER.index('Linear_Program'):
        blocker_section(L, blockers, bodies, ['Linear_Program'],
                        'Not a Linear Program', None, top)

    return '\n'.join(L)


def blocker_section(L, blockers, bodies, classes, headline, advice, top):
    """Append one 'Not a ...' section: the rows blocking those classes,
    grouped by constraint with EVERY distinct reason (one row can fail a
    class more than one way, and the more specific reason is the
    actionable one)."""
    rows = []
    for cls in classes:
        rows += blockers.get(cls, [])
    if not rows:
        return
    order, reasons = [], {}
    for name, why, row in rows:
        if name not in reasons:
            reasons[name] = (row, [])
            order.append(name)
        if why not in reasons[name][1]:
            reasons[name][1].append(why)
    uniq = [(name, reasons[name][1], reasons[name][0]) for name in order]
    L.append('')
    n = len(uniq)
    n_obj = sum(1 for nm, _, _ in uniq if nm == 'the objective')
    n_con = n - n_obj
    parts = []
    if n_con:
        parts.append(f'{n_con} constraint' + ('s' if n_con != 1 else ''))
    if n_obj:
        parts.append('the objective')
    L.append(f'  {headline} -- {" and ".join(parts)} '
             f'block{"s" if n == 1 else ""} it:')
    if top is None:
        shown = uniq
    else:
        shown = uniq[:top]
    for name, whys, row in shown:
        body = bodies.get(name)
        if body:
            L.append(f'      {name}   {body}')
        else:
            L.append(f'      {name}')
        for why in whys:
            L.append(f'          {why}')
    if len(uniq) > len(shown):
        L.append(f'      ... and {len(uniq) - len(shown)} more')
    if advice:
        L.append(f'    {advice}')


def annihilated_report(model, names=None):
    """Constraint sides that are identically zero at the current constants.

    A posynomial cannot be zero, so a constant that annihilates a side (say
    ``(nu**2 - 1)`` with ``nu`` exactly 1) silently reclassifies the model,
    leaves a variable unbounded below in log space, or aborts the backend
    somewhere unrecognisable. Runs on the Pyomo model directly, since
    failing to detect a structure is one of the symptoms. Culprit constants
    are found by perturbing each and seeing whether the side revives.
    """
    import pyomo.environ as pyo
    from pyomo.core.expr.visitor import identify_mutable_parameters

    findings = []
    try:
        constraints = model.get_explicitConstraints()
    except Exception:
        return findings

    for con in constraints:
        for c in (con.values() if hasattr(con, 'values') else [con]):
            expr = getattr(c, 'expr', None)
            args = getattr(expr, 'args', None)
            if not args or len(args) < 2:
                continue
            for pos, side in ((0, args[0]), (len(args) - 1, args[-1])):
                try:
                    value = pyo.value(side)
                except Exception:
                    continue
                if value != 0.0:
                    continue
                # Which constants annihilated it: nudge each and re-evaluate
                culprits = []
                try:
                    params = list(identify_mutable_parameters(side))
                except Exception:
                    params = []
                for prm in params:
                    try:
                        was = pyo.value(prm)
                        prm.set_value(was * 1.5 + 1.0)
                        revived = pyo.value(side) != 0.0
                        prm.set_value(was)
                    except Exception:
                        continue
                    if revived:
                        culprits.append((str(prm), was))
                text = str(side)
                findings.append({
                    'constraint': str(getattr(c, 'name', c)),
                    'side': 'left' if pos == 0 else 'right',
                    'expression': text if len(text) <= 90 else text[:87] + '...',
                    'constants': culprits,
                    })
    return findings


def checks_setup(structures):
    """Shared front door for the check entry points.

    Returns ``(st, model, units_report)``; on a unit failure ``st`` is None
    and ``units_report`` carries the diagnosis -- bad units are a finding,
    not a crash.
    """
    from lcsolver.presolve.detected import as_detected
    from lcsolver.presolve.unitCorrector import UnitMismatch

    try:
        st = as_detected(as_structures(structures))
    except UnitMismatch as exc:
        model = None if isinstance(structures, dict) else structures
        return None, model, units_diagnosis(exc)
    # The checks need bounds separated from rows; fold a copy rather than
    # making the caller know that (unfolded rows once hid all 52 output-only
    # variables on SPaircraft)
    try:
        st = fold_singleton_rows(st if st.bounds is not None
                                 else with_empty_bounds(st))
    except Exception:
        pass
    return st, st.get("model"), None


def greybox_covered(st):
    """Names of variables referenced by a grey-box (black-box) row.

    The structural checks read only algebraic rows, so a black-box-computed
    variable looks unbounded or empty; these names subtract that false
    positive.
    """
    try:
        from lcsolver.solvers.sequential.bridge import (unwrap_variables,
                                                        greybox_blocks)
        blocks = greybox_blocks(st)
    except Exception:
        return set()
    covered = set()
    for block in blocks:
        bb = getattr(block, '_ex_model', None)
        if bb is None:
            continue
        try:
            for v in unwrap_variables(list(bb.inputVariables_optimization)
                                  + list(bb.outputVariables_optimization)):
                covered.add(str(v))
        except Exception:
            pass
    return covered


def presolve_check(structures, names=None, structure_top=5):
    """The pre-solve half of the checks: needs nothing but the model.

    Structure classification, the structural findings, rigidity, and
    unopposed variables. Black-box-computed variables are excluded from the
    empty/unbounded findings. Returns a :class:`PresolveReport`; prints
    nothing.
    """
    st, model, units_report = checks_setup(structures)

    # Run first and on the model: an annihilated side is one reason detection
    # fails, so this must survive the gate below
    probe = model if model is not None else (
        None if isinstance(structures, dict) else structures)
    annihilated = annihilated_report(probe) if probe is not None else []

    if st is None:
        rep = PresolveReport()
        rep.structure = units_report
        rep.annihilated = annihilated
        return rep

    try:
        rep = presolve_report(st, names=names)
    except ValueError:
        # No detected GP/SP rows to walk. An annihilated side is likely why;
        # return the finding that explains it, else the original raise stands.
        if not annihilated:
            raise
        rep = PresolveReport()
        rep.structure = units_report or ''
        rep.annihilated = annihilated
        return rep
    rep.annihilated = annihilated
    guesses = getattr(model, 'defaulted_guesses', None)
    if guesses:
        rep.defaulted_guesses = list(guesses)

    covered = greybox_covered(st)
    if covered:
        rep.empty_columns = [n for n in rep.empty_columns
                             if n not in covered]
        rep.unbounded_above = [n for n in rep.unbounded_above
                               if n not in covered]
        rep.unbounded_below = [n for n in rep.unbounded_below
                               if n not in covered]
        # A grey-box-fed variable can look output-only to the algebraic scan
        # while its black box pins it. Not peelable.
        rep.output_columns = [n for n in rep.output_columns
                              if n not in covered]

    if names is None:
        try:
            names = [str(v) for v in st.variables]
        except Exception:
            names = None
    try:
        rep.structure = structure_report(st, top=structure_top)
    except Exception:
        rep.structure = ''
    try:
        rep.rigidity = rigidity_report(
            st, names or [str(v) for v in st.variables])
    except Exception:
        rep.rigidity = {}
    try:
        rep.unopposed = unopposed_report(
            st, names or [str(v) for v in st.variables])
    except Exception:
        rep.unopposed = []
    return rep


def postsolve_check(structures, x=None, problem=None, names=None, x_min=1e-9,
                    skip_degeneracy_check=False):
    """The post-solve half: the checks that only mean something at a solution.

    Degenerate variables, cancelling signomial terms, and floor-resting
    variables. ``skip_degeneracy_check`` drops the expensive one (perturb and
    re-check every variable); the other two are cheap and always run. Hand
    this a SOLVED Formulation, or pass ``x`` and ``problem`` explicitly
    (takes precedence). An unsolved model raises: running these against a
    guess would describe it in the language of a result.
    """
    st, model, units_report = checks_setup(structures)
    rep = PresolveReport()
    if st is None:
        rep.structure = units_report
        return rep

    if x is None and problem is None:
        if model is None or not getattr(model, '_edi_solved', False):
            raise ValueError(
                'postsolve_check needs a solved model (or explicit x= and '
                'problem=); this one has not been solved')
        import numpy as np
        import pyomo.environ as pyo

        from lcsolver.solvers.sequential.bridge import build_problem

        solved_x = np.asarray([float(pyo.value(v)) for v in st.variables],
                              dtype=float)
        built = build_problem(st, sp_form=True)
        if solved_x.size >= built.n:
            x, problem = solved_x[:built.n], built

    if names is None:
        try:
            names = [str(v) for v in st.variables]
        except Exception:
            names = None

    if x is not None and problem is not None:
        if skip_degeneracy_check:
            rep.degenerate = []
        else:
            try:
                # Read the sparsity off the model once (the scan is quadratic
                # without it); None means unmatched, dense scan instead
                dep = constraint_dependencies(model, st.variables, problem.n,
                                              problem.constraints, x=x)
                rep.degenerate = degeneracy_report(problem, x, names=names,
                                                   depends_on=dep)
            except Exception:
                rep.degenerate = []
        try:
            rep.cancelling = cancellation_report(st, x, names=names)
        except Exception:
            rep.cancelling = []
        rep.at_floor = floor_report(
            x, names or [str(v) for v in st.variables], x_min=x_min)
    return rep


def optimization_check(structures, x=None, problem=None, names=None,
                       x_min=1e-9, structure_top=5):
    """Every check on a model, in one call, as one report.

    Before a solve: structure class, blockers, and the structural findings.
    After: ``degenerate``, ``cancelling``, ``at_floor`` -- turned on
    automatically when handed a SOLVED Formulation (explicit ``x`` and
    ``problem`` also work and take precedence). Auto-wiring never happens
    for a structure dict: the detected clone is never solved, and reading
    values off it would check initial guesses in the language of a result.
    ``structure_top`` caps blockers listed per class. Returns a
    :class:`PresolveReport` and prints nothing::

        report = optimization_check(f)
        print(report.summary())
    """
    rep = presolve_check(structures, names=names, structure_top=structure_top)

    model = None if isinstance(structures, dict) else structures
    solved = (x is not None and problem is not None) or (
        x is None and problem is None and model is not None
        and getattr(model, '_edi_solved', False))
    if solved:
        try:
            post = postsolve_check(structures, x=x, problem=problem,
                                   names=names, x_min=x_min)
            rep.degenerate = post.degenerate
            rep.cancelling = post.cancelling
            rep.at_floor = post.at_floor
        except Exception:
            pass                    # a check must never block the report
    return rep


def with_empty_bounds(structures):
    """A copy carrying an empty bounds array, so rows can be folded into it."""
    st = dict(structures)
    if st.get("bounds") is None:
        rows, ops, key = log_rows_of(st)
        width = max((len(r) - 2 for r in rows), default=0)
        n = max(width, len(st.get("variables") or []))
        st["bounds"] = [(None, None)] * n
    from lcsolver.presolve.detected import as_detected
    return as_detected(st)


def presolve(structures, fold=True, eliminate=True, propagate=False,
             reduce=True, verbose=False, fold_only=None, protect=None):
    """Run the presolve passes in an order that is safe to compose.

    Reduce and eliminate BEFORE propagate: both need vacuous bounds, and
    propagation fills exactly those in (propagating first cost 40 reductions
    on SPaircraft) -- so ``propagate`` is off by default. ``fold_only`` is
    forwarded to :func:`fold_singleton_rows` (the sequential solvers pass
    the declared-bound row block; see the note there). ``protect`` is
    forwarded to :func:`reduce_columns` (grey-box columns). Returns
    ``(structures, log)``; ``log.restore(x)`` rebuilds the full solution.
    """
    log = PresolveLog([str(v) for v in structures.get("variables", [])])
    try:
        if fold:
            before = (structures.get("info") or {}).get("N_cons_total")
            structures = fold_singleton_rows(structures, only=fold_only)
            after = structures["info"]["N_cons_total"]
            log.record("bounds", rows_folded=(before - after) if before else 0)
        if reduce:
            structures, removed = reduce_columns(structures, protect=protect)
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


def peel_output_columns(structures, protect=None):
    """The exact column reductions alone, keeping the bounds-as-rows form.

    ``solve()`` runs this ONCE, centrally, on every structured route. Bound
    rows stay rows (the convex backends read bounds out of them); the bounds
    array is synthesized empty and stripped after, so only the output-only
    peel and untouched-column removal can trigger. Returns
    ``(structures, removed)`` for :func:`restore_columns`; unchanged input
    when nothing peels.
    """
    if structures.get('bounds') is not None:
        raise ValueError(
            'peel_output_columns expects the bounds-as-rows form the solve '
            'routes carry; split-bounds structures take the full presolve')
    key = ('Signomial_Program' if structures['Signomial_Program'][0]
           else 'Geometric_Program')
    width = max((len(r) - 2 for r in structures[key][1]), default=0)
    st = dict(structures)
    st['bounds'] = [(None, None)] * max(width, len(st.get('variables') or []))
    reduced, removed = reduce_columns(st, protect=protect)
    if not removed:
        return structures, []
    reduced['bounds'] = None
    return reduced, removed


def constraint_dependencies(model, variables, n, constraints, x=None,
                            probes=16):
    """Which constraints each variable appears in: ``[j] -> [constraint index]``.

    NOT positional over every row: ``build_problem`` folds single-variable
    rows into bounds, so ``constraints`` holds only rows with 2+ distinct
    variables, in model order. The map is CHECKED by perturbing a few
    variables; if any unpredicted constraint moves this returns None and the
    caller falls back to the dense scan -- a wrong map would silently
    under-report degeneracy.
    """
    if model is None:
        return None
    try:
        import math

        import pyomo.environ as pyo
        from pyomo.core.expr.visitor import identify_variables

        rows, row_vars = [], []
        for row in model.component_data_objects(pyo.Constraint, active=True):
            vs = {id(v): v for v in identify_variables(row.body)}
            if len(vs) >= 2:            # the rest became bounds
                rows.append(row)
                row_vars.append(vs)
        if len(rows) != len(constraints):
            return None

        index = {id(v): j for j, v in enumerate(list(variables)[:n])}
        dep = [[] for _ in range(n)]
        for i, vs in enumerate(row_vars):
            for vid in vs:
                j = index.get(vid)
                if j is not None:
                    dep[j].append(i)

        if x is not None and probes:
            import numpy as np

            x = np.asarray(x, dtype=float)
            base = [log_body(c, x) for c in constraints]
            step = max(1, n // int(probes))
            for j in range(0, n, step):
                xp = x.copy()
                xp[j] *= math.exp(0.05)
                expected = set(dep[j])
                for i, c in enumerate(constraints):
                    if abs(log_body(c, xp) - base[i]) > 1e-12 and i not in expected:
                        return None      # the map missed a row: do not trust it
        return dep
    except Exception:
        return None


def degeneracy_report(problem, x, rel_step=0.05, obj_tol=1e-9,
                      viol_tol=1e-9, names=None, depends_on=None):
    """Variables the solution does not determine, tested directly.

    Degenerate: movable in BOTH directions without changing the objective or
    worsening feasibility -- the definition itself, no heuristic (the
    structural one also flags variables pinned by single-variable equalities).
    Returns a list of ``(name, value)``. ``depends_on`` is the sparsity from
    :func:`constraint_dependencies`, a pure speed-up: perturbing ``x[j]`` can
    only move rows containing it; without the map the scan is n x m body
    evaluations, the most expensive thing in a few-thousand-row solve.
    """
    import math

    import numpy as np

    x = np.asarray(x, dtype=float)
    names = list(names or [])
    cons = problem.constraints

    v0 = -math.inf
    for c in cons:
        v0 = max(v0, log_body(c, x))
    f0 = problem.objective_value(x)
    thresh = max(v0, 0.0) + viol_tol

    out = []
    for j in range(problem.n):
        free = True
        for s in (rel_step, -rel_step):
            xp = x.copy()
            xp[j] *= math.exp(s)
            if abs(problem.objective_value(xp) - f0) > obj_tol * max(1.0, abs(f0)):
                free = False
                break
            # only the rows containing x_j can move (depends_on); else all
            if depends_on is not None:
                rows = [cons[i] for i in depends_on[j]]
            else:
                rows = cons
            if any(log_body(c, xp) > thresh for c in rows):
                free = False
                break
        if free:
            out.append((nm_at(names, j), float(x[j])))
    return out


def log_body(constraint, x):
    """log g(x) for a constraint written g <= 1, floored away from -inf."""
    import math
    return math.log(max(constraint.body(x), 1e-300))


def unbuilt_blocks(model):
    """Blocks attached to `model` that never finished receiving their inputs.

    A two-phase block posts its rows only when the last input arrives, so a
    waiting block posts NOTHING -- the formulation stays solvable and answers
    an easier question. Finds SubModel, plus any object with ``_built``
    false and ``_pending`` naming what it waits for. Returns a list of
    ``(name, [missing input names])``.
    """
    from lcsolver.objects.submodel import SubModel

    out = []
    for attr in dir(model):
        if attr.startswith('_'):
            continue
        try:
            obj = getattr(model, attr)
        except Exception:
            continue
        if isinstance(obj, SubModel):
            if obj.is_built():
                continue
            out.append((obj.name or attr, obj.pending_inputs()))
        elif getattr(obj, '_built', None) is False:
            pending = getattr(obj, '_pending', None)
            out.append((getattr(obj, 'name', attr),
                        sorted(pending) if pending else []))
    return out


def unbuilt_blocks_check(model):
    """Raise if any attached block never built.  Called before every solve.

    An ERROR, not a finding: a half-assembled model is a different problem,
    and it will return a confident number for it.
    """
    from lcsolver.core import codes
    # Local import: PresolveError lives with the solver, which imports this
    # module
    from lcsolver.solvers.solver import PresolveError

    found = unbuilt_blocks(model)
    if not found:
        return
    n_inputs = sum(len(p) for _, p in found)
    lines = [f'{len(found)} model block(s) never received all their inputs, so '
             f'they posted no variables and no constraints. Every block and '
             f'every missing input is listed -- {n_inputs} in total -- so the '
             f'whole assembly can be fixed in one pass rather than one error '
             f'at a time:', '']
    for name, pending in found:
        if pending:
            lines.append(f"  {name} is still waiting for {len(pending)} input(s):")
            # wrapped, all of them: a truncated list means another solve to
            # find the rest
            row = '   '
            for item in pending:
                if len(row) + len(item) + 2 > 74:
                    lines.append(row.rstrip(','))
                    row = '   '
                row += f' {item},'
            lines.append(row.rstrip(','))
        else:
            lines.append(f"  {name} never built")
    lines += ['',
              '  Nothing about this is caught downstream. The rows those blocks',
              '  would have posted are simply absent, so the solve succeeds and',
              '  reports an optimum for a problem missing whole missions.',
              '  Assign the inputs listed above, then solve again.',
              '',
              '  f.<block>.get_status() prints every input a block needs and',
              '  what each one is currently connected to.']
    raise PresolveError(codes.tag(codes.UNBUILT_BLOCK, '\n'.join(lines)))


# older scripts in the lc* repos import these by their former names
_tighten_linear = tighten_linear_bounds
_as_structures = as_structures
_gp_after_presolve = gp_after_presolve
