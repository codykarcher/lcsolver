#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""A typed view over what the structure detector produces.

The detector's output is a plain dict whose shape has to be learned rather
than read. ``structures['Signomial_Program'][1]`` is the rows, ``[2]`` is the
operators, and a row is ``[index, coefficient, *exponents]`` where a *negative*
index means the term belongs to a denominator. None of that is written down
anywhere; it is recovered by reading whichever consumer you happen to open
first. Meanwhile ``structures['Linear_Program'][1]`` is ``[c, shift, AG, b]``,
a different shape entirely, so code that handles one cannot handle the other.

Every conversion between those shapes is a place where a transformation can
quietly change the problem, which is this repository's characteristic bug.

:class:`Detected` is a **dict subclass**, so every existing consumer keeps
working untouched while new code reads named fields. Nothing has to migrate at
once.

One representation, two spaces
------------------------------
A linear and a geometric program are not really different structures; they are
the same structure in different spaces. Both are

    sum of terms, each term a coefficient times a product of powers

compared against a right-hand side. For an **LP** the comparison is linear in
the natural variables and each term is ``a_j * x_j`` -- a single variable at
power one. For a **GP** the comparison is linear in ``y = log x`` and each term
is a monomial ``c_k * prod x_j**a_jk``. The arithmetic that operates on the
terms is identical; only the space in which the comparison is made differs.

That is not a theory. :func:`~edi.presolve._tighten_linear` already serves
bound propagation for both from one implementation, with ``space`` the only
thing distinguishing them, and the LP path fell out of the GP one for free.
:attr:`Detected.space` names that distinction so the rest of the code can rely
on it.
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field

__all__ = ["Term", "Detected", "as_detected", "DISPATCH_ORDER",
           "LinearParts"]

#: The detector sets a flag for EVERY structure the model satisfies, and they
#: are not exclusive: `z == x - y` with `x >= 4` is a linear program AND a
#: valid signomial program, and both flags come back True carrying two
#: different row encodings of the same problem.
#:
#: So "what kind is this?" and "which rows do I read?" are different questions,
#: and this codebase already answered them differently in two places --
#: `solver.py` dispatches linear-first, while `_rows_of` and the SLCP bridge
#: read signomial-first. Conflating them makes a consumer silently parse the
#: wrong encoding, which is exactly what happened on the first attempt to move
#: the bridge onto this class.
DISPATCH_ORDER = ("Linear_Program", "Quadratic_Program", "Geometric_Program",
                  "Signomial_Program")
#: Preference when reading term rows: signomial before geometric, matching
#: every existing consumer. The signomial list is the general one.
LOG_ORDER = ("Signomial_Program", "Geometric_Program")
LINEAR_ORDER = ("Linear_Program", "Quadratic_Program")

_SHORT = {"Linear_Program": "LP", "Quadratic_Program": "QP",
          "Geometric_Program": "GP", "Signomial_Program": "SP"}


@dataclass
class Term:
    """One ``coefficient * prod_j x_j ** exponent_j``.

    ``exponents`` is **sparse**: absent means zero. That removes the dense
    padding that the positional row format forces on every consumer, and with
    it the off-by-one class of bug that comes from two rows of different
    length.

    ``denominator`` replaces the negative-row-index convention. A signomial
    ``p/q <= 1`` carries its ``q`` terms with this set, and the reader no
    longer has to know that ``-i - 1`` means anything.
    """

    coeff: float
    exponents: dict = field(default_factory=dict)
    denominator: bool = False

    @property
    def is_constant(self):
        return not self.exponents

    @property
    def variables(self):
        return set(self.exponents)

    def value(self, x):
        """Evaluate at ``x``, indexed by column."""
        import math

        acc = math.log(self.coeff) if self.coeff > 0 else -math.inf
        for j, e in self.exponents.items():
            if j < len(x) and x[j] > 0:
                acc += e * math.log(x[j])
            elif e:
                return 0.0 if self.coeff > 0 else 0.0
        return math.exp(acc) if -700 < acc < 700 else (
            0.0 if acc <= -700 else float("inf"))


LinearParts = namedtuple(
    "LinearParts", "hessian linear shift A b operators")


class Detected(dict):
    """The detector's output, with names.

    A dict subclass on purpose: ``structures['Signomial_Program'][1]`` keeps
    working, so consumers migrate one at a time rather than all at once.
    """

    # -- what kind of problem is this ------------------------------------
    def _first(self, order):
        for k in order:
            entry = self.get(k)
            if entry and entry[0] and entry[1] is not None:
                return k
        return None

    @property
    def log_key(self):
        """The row list to read for GP/SP work: signomial before geometric."""
        return self._first(LOG_ORDER)

    @property
    def linear_key(self):
        """The payload to read for LP/QP work."""
        return self._first(LINEAR_ORDER)

    @property
    def key(self):
        """The row list this structure's terms live in.

        Prefers the log encoding, because that is what every term-reading
        consumer here wants. Use :attr:`dispatch_kind` to choose a solver.
        """
        return self.log_key or self.linear_key

    @property
    def dispatch_kind(self):
        """The most specific kind, for choosing a backend.

        An LP is also a QP is also a GP; this reports the narrowest, which is
        the cheapest to solve. Deliberately NOT the same question as
        :attr:`key`.
        """
        k = self._first(DISPATCH_ORDER)
        return _SHORT[k] if k else None

    @property
    def kind(self):
        """``'LP'``, ``'QP'``, ``'GP'`` or ``'SP'`` for the encoding in use."""
        k = self.key
        return _SHORT[k] if k else None

    @property
    def space(self):
        """``'log'`` if the terms in use are monomials, else ``'natural'``.

        Follows :attr:`key`, so a model that is both linear and signomial
        reports ``'log'`` -- the encoding its terms are actually stored in.
        """
        if self.log_key:
            return "log"
        return "natural" if self.linear_key else None

    # -- the pieces -------------------------------------------------------
    @property
    def variables(self):
        return self.get("variables") or []

    @property
    def bounds(self):
        """Per-variable ``(lo, hi)``, or None when bounds are still rows."""
        return self.get("bounds")

    @property
    def model(self):
        return self.get("model")

    @property
    def info(self):
        return self.get("info") or {}

    @property
    def n_variables(self):
        return len(self.variables)

    @property
    def operators(self):
        k = self.key
        return list(self[k][2]) if k else []

    # -- the linear payload ------------------------------------------------
    def linear_parts(self):
        """``(hessian, linear, shift, A, b, operators)`` for an LP or QP.

        The two store their payload in **different layouts**, which is not
        written down anywhere and is easy to miss because both are "the linear
        one":

            LP   [[c], shift, A, b]        the objective nested one deeper
            QP   [P, q, shift, A, b]

        Reading a QP with the LP layout silently yields the Hessian where the
        objective belongs and the constraint matrix where the right-hand side
        does. Both `evaluate` and `propagate_bounds` did exactly that until this
        existed.

        The semantics ARE shared, and worth stating since neither backend does:
        constraint ``i`` is ``A[i] . x + b[i] <= 0`` (or ``== 0``). Both
        backends negate ``b`` on the way into cvxopt, from opposite starting
        points, which is what makes the convention hard to read off the code.

        ``hessian`` is None for an LP.
        """
        k = self.linear_key
        if k is None:
            raise ValueError("not a linear or quadratic program")
        p = self[k][1]
        ops = list(self[k][2])
        if k == "Linear_Program":
            return LinearParts(None, p[0][0], p[1], p[2], p[3], ops)
        return LinearParts(p[0], p[1], p[2], p[3], p[4], ops)

    # -- terms ------------------------------------------------------------
    def _grouped(self):
        """Every row parsed into :class:`Term` objects, grouped by constraint.

        Memoised, because this walks the whole problem and ``terms(i)`` asks
        for one constraint out of it. Callers naturally write
        ``[st.terms(i) for i in keep]``, which without a cache re-parses every
        row of the model once per constraint -- quadratic in the constraint
        count and, on a model with a thousand columns, quadratic against a
        dense row width too. On SPaircraft that one comprehension in
        ``fold_singleton_rows`` ran for over four minutes; cached it is
        instant.

        A :class:`Detected` is treated as immutable: presolve passes build a
        new one through :meth:`rebuild` rather than editing rows in place. The
        stamp guards the case anyway, so a swapped or resized row list is
        re-parsed rather than silently answered from the cache.
        """
        import collections
        import itertools

        k = self.key
        if k is None or self.space != "log":
            return {}
        rows = self[k][1]
        stamp = (k, id(rows), len(rows))
        cached = self.__dict__.get("_grouped_cache")
        if cached is not None and cached[0] == stamp:
            return cached[1]

        groups = collections.defaultdict(list)
        for r in rows:
            idx = int(r[0])
            i = idx if idx >= 0 else -idx - 1
            # islice rather than `r[2:]`: slicing copies the row, and the row
            # is as wide as the model has variables.
            exponents = {}
            for j, v in enumerate(itertools.islice(r, 2, None)):
                v = float(v)
                if v > 1e-12 or v < -1e-12:
                    exponents[j] = v
            groups[i].append(Term(coeff=float(r[1]), exponents=exponents,
                                  denominator=idx < 0))

        self.__dict__["_grouped_cache"] = (stamp, groups)
        return groups

    @property
    def constraint_indices(self):
        """Constraint numbers, excluding the objective (which is 0)."""
        if self.space == "log":
            return sorted(i for i in self._grouped() if i != 0)
        k = self.key
        AG = self[k][1][2] if k else None
        # `AG or []` raises on a numpy array -- truth value is ambiguous.
        return list(range(0 if AG is None else len(AG)))

    def terms(self, i):
        """The :class:`Term` list for constraint ``i``; ``0`` is the objective.

        Parses the row format in **one** place instead of the five it is
        currently open-coded in.
        """
        if self.space == "log":
            return self._grouped().get(i, [])
        k = self.key
        c, _shift, AG, _b = self[k][1][:4]
        row = c if i == 0 else AG[i - 1]
        return [Term(coeff=float(v), exponents={j: 1.0})
                for j, v in enumerate(row) if abs(float(v)) > 1e-12]

    def rhs(self, i):
        """Right-hand side of constraint ``i`` in its own space.

        A GP constraint is written ``body <= 1``, so the right-hand side is
        ``1`` (``0`` in log space). An LP row carries its own.
        """
        if self.space == "log":
            return 0.0
        k = self.key
        return -float(self[k][1][3][i - 1])

    def operator(self, i):
        ops = self.operators
        idx = i - 1 if self.space == "log" else i
        return ops[idx] if 0 <= idx < len(ops) else "<="

    # -- resolving onto a model -------------------------------------------
    def values_from(self, model, as_array=False):
        """The current values of these variables, read off ``model``.

        ``unit_corrector`` clones the model before detection, because unit
        conversion rewrites constants and the user's own formulation should not
        be mutated. The consequence is a trap: ``structures['variables']`` look
        like the model's variables and are not -- they belong to the clone. Read
        values off them after a solve and you silently get the initial guess,
        with no error and no clue, because the clone was never solved.

        That is a wrong number rather than a failure, so it does not announce
        itself. This resolves each variable onto whichever model you hand it,
        by ``ComponentUID`` rather than by name -- the same route
        ``write_solution`` uses, and for the same reason: a name round-trip
        mangles indexed variables and fails outright on one whose parent
        component has been collected.

        Returns ``{name: value}``, or a list in column order with
        ``as_array=True``, which is the form the solvers want.
        """
        from edi.solvers.writeback import _name_of, _resolve_on

        import pyomo.environ as pyo

        out, ordered = {}, []
        for v in self.variables:
            target = _resolve_on(model, v)
            if target is None:
                raise ValueError(
                    f"{_name_of(v)} has no counterpart on the model given. "
                    "These variables belong to the unit-corrected clone the "
                    "structure was detected from; pass the model that clone "
                    "was made of.")
            val = float(pyo.value(target))
            out[_name_of(v)] = val
            ordered.append(val)
        return ordered if as_array else out

    # -- writing ----------------------------------------------------------
    def rebuild(self, objective, constraints, operators, n=None, **overrides):
        """A new :class:`Detected` carrying these terms.

        The counterpart to :meth:`terms`. Three presolve passes each grew their
        own row-emitting loop -- reconstructing the ``[index, coefficient,
        *exponents]`` layout and the ``-i - 1`` denominator convention by hand
        -- which is three places to get the convention wrong and three places
        that must change together if it ever moves.

        ``objective`` is a list of :class:`Term`; ``constraints`` a list of
        term-lists, numbered from 1 in order. ``n`` is the dense column count,
        defaulting to the widest column any term touches. Pass it explicitly
        when trailing columns are all-zero but must be kept: narrowing silently
        renumbers every variable after the first empty column.

        Any other keyword replaces a top-level entry -- ``variables``,
        ``bounds``, ``info``.
        """
        if n is None:
            n = max([j + 1 for terms in [objective] + list(constraints)
                     for t in terms for j in t.exponents] + [0])

        def emit(idx, terms, out):
            for t in terms:
                row = [(-idx - 1) if t.denominator else idx, float(t.coeff)]
                row += [t.exponents.get(j, 0.0) for j in range(n)]
                out.append(row)

        rows = []
        emit(0, objective, rows)
        for i, terms in enumerate(constraints, start=1):
            emit(i, terms, rows)

        out = Detected(self)
        key = self.key
        out[key] = [self[key][0], rows, list(operators)]
        for k, v in overrides.items():
            out[k] = v
        return out

    # -- presentation -----------------------------------------------------
    def __repr__(self):
        if self.kind is None:
            return "<Detected: unstructured>"
        nb = "bounds split out" if self.bounds is not None else "bounds in rows"
        return (f"<Detected {self.kind} in {self.space} space: "
                f"{self.n_variables} variables, "
                f"{len(self.constraint_indices)} constraints, {nb}>")


def as_detected(structures):
    """Wrap a detector output dict, idempotently."""
    if isinstance(structures, Detected):
        return structures
    return Detected(structures)
