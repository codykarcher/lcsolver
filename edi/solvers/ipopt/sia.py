#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Sequential Inner Approximation (SIA) for signomial programs.

A third way of solving an SP, sitting between PCCP and SLCP and taking the
useful half of each.

The idea
--------
Every constraint is put in one of three classes, and the class decides how the
sub-problem represents it:

``exact``
    A posynomial ``p <= 1`` is log-convex as written. Imposed exactly, as a
    log-sum-exp. No approximation at all.
``conservative``
    A signomial ``p/q <= 1`` is condensed by the arithmetic-geometric-mean
    inequality: ``q_hat(x) <= q(x)`` everywhere, with equality **and matching
    gradient** at the current iterate. Imposing ``p <= q_hat`` is therefore
    *harder* than the true constraint.
``linearized``
    A black-box constraint, available only as a value and a gradient. There is
    no way to be conservative with local information alone, so this one is
    linearized and needs globalizing.

Why the split matters
---------------------
If every constraint is exact or conservative, the sub-problem's feasible set is
a **subset** of the true one and contains the current iterate. Two things
follow with no extra machinery:

* whatever the sub-problem returns is feasible for the *true* problem, so
  every iterate is a valid design;
* the current iterate is feasible for the sub-problem, so the objective cannot
  increase -- descent is structural.

No line search, no merit function, no penalty parameter, no trust region.
Limit points are KKT points by the standard inner-approximation argument
(Marks & Wright 1978; the convex-concave procedure of Lipp & Boyd 2016 is the
same idea).

All of that machinery exists in a general SQP/SLCP to compensate for a model
that is *not* conservative. Here it is only needed for the black-box block, so
it is applied only there: the trust region is sized by the linearized
constraints' model accuracy, and the ratio test looks only at them. A problem
with no black-box constraints takes full steps and never rejects one.

What this buys over PCCP
------------------------
PCCP does the same condensation, and also keeps posynomial constraints exact,
so the *iterates* are similar. The differences are:

1. **Termination.** PCCP stops when the objective stops changing::

       |prev_obj - obj| / obj <= reltol

   which says "I stopped moving", not "I am optimal", and says nothing about
   feasibility or stationarity. SIA terminates on a genuine KKT residual for
   the **original** problem -- stationarity, primal feasibility and
   complementarity, all evaluated with the *true* constraint functions and
   gradients, using the sub-problem's duals. Tangency is what licenses that:
   because the condensed constraint matches the true one to first order at the
   iterate, its multiplier is the true multiplier.

2. **Black-box constraints.** PCCP needs the monomial/posynomial row
   structure. SIA takes a value/gradient callback alongside the structured
   constraints and globalizes only that part.

What this buys over SLCP
------------------------
SLCP linearizes the objective and carries a BFGS quadratic to make up for it.
On a problem that is mostly exact-or-conservative that quadratic is not a
curvature model -- the curvature condition ``s.z > 0`` fails almost every
iteration and it degenerates into a fixed proximal penalty that throttles
steps which were already safe. SIA imposes the objective exactly when it is a
posynomial and carries no quadratic at all.

Infeasible starts
-----------------
The conservative argument needs a feasible starting point. When one is not
available, each constraint gets a slack ``s_i >= 0`` and the objective a
penalty ``tau * sum(s_i)``, with ``tau`` escalating until the slacks vanish --
Lipp & Boyd's penalty CCP. Once all slacks are zero the iterate is feasible and
conservatism keeps it there, so the guarantees switch back on. ``Result``
reports which regime the run finished in.
"""
from __future__ import annotations

import math

import numpy as np
import pyomo.environ as pyo

from edi.solvers.ipopt.slcp import (CondensedEquality, Posynomial,
                                    PosynomialRatio, Problem, Signomial)

__all__ = ["SIAOptions", "SIAResult", "solve_sia", "classify",
           "SubproblemCache"]


class SIAOptions:
    """Algorithm parameters."""

    def __init__(self, **kw):
        self.max_iterations = 400
        # 400 rather than 100. SPaircraft converges in 149, so the old default
        # stopped a converging run three fifths of the way through and reported
        # it as not converged. The cost of a high cap is nothing when the KKT
        # test fires -- the run stops on its own -- and the cost of a low one is
        # a correct answer thrown away.
        # --- KKT termination, all on the TRUE problem ---------------------
        self.feasibility_tolerance = 1e-6     # max_i log g_i(x)
        self.stationarity_tolerance = 1e-6    # ||grad log L||_inf in log space
        self.complementarity_tolerance = 1e-6  # max_i |lambda_i log g_i(x)|
        # --- Phase I: find a feasible point before optimizing --------------
        self.phase1 = True             # False falls back to penalty CCP
        self.phase1_max_iterations = 50
        # When Phase I falls short, continue with the penalty path instead of
        # abandoning the solve. Penalty CCP tolerates an infeasible start by
        # construction, so a Phase I that got close has still done useful work.
        self.phase1_penalty_fallback = True
        self.phase1_margin = 1e-8      # target interiority for the Phase I
                                       # sub-problem. NOT an acceptance test:
                                       # an ACTIVE EQUALITY is feasible at
                                       # exactly log g = 0 and can never be
                                       # strictly interior, so demanding it
                                       # leaves Phase I spinning forever on
                                       # any problem with equalities.
        # --- penalty CCP, used only if phase1 is off or fails --------------
        self.tau0 = 1.0
        self.tau_factor = 5.0
        self.tau_max = 1e12
        self.tau_binding = 0.9         # raise tau once a multiplier reaches
                                       # this fraction of it -- see solve_sia
        # --- trust region, only applied to LINEARIZED constraints ---------
        self.trust_radius = 1.0        # initial |d_j| bound, log space
        self.trust_min = 1e-8
        self.trust_max = 10.0
        self.trust_expand = 2.0
        self.trust_shrink = 0.25
        self.ratio_accept = 1e-4       # accept the step if ratio exceeds this
        self.ratio_expand = 0.75
        # --- misc ----------------------------------------------------------
        self.x_min = 1e-9
        self.expand_past_blackbox = False
        #: Carry a BFGS curvature model for each linearized constraint, so the
        #: subproblem holds a convex quadratic rather than a plane. See
        #: :class:`Curvature`. The model stays convex either way; this makes it
        #: conservative wherever the curvature estimate is good enough.
        self.curvature = False
        self.step_expansion = 1.0      # >1 enables the feasibility-verified
        self.step_expansion_max = 1e4  # step extension described in solve_sia.
                                       # Set to 1.0 to take the sub-problem's
                                       # step exactly as returned.
        self.condense_numerator = False
        # Condense the NUMERATOR of p/q <= 1 as well, making the constraint a
        # monomial -- linear in log space. This is what PCCP does for an
        # equality, and it is the whole reason PCCP takes larger steps: since
        # p_hat <= p, the condensed constraint is EASIER than the true one, so
        # the sub-problem's feasible set is no longer a SUBSET of the true one
        # and an iterate may leave it. The feasible-iterate and monotone-descent
        # guarantees go with it.
        #
        # What survives is tangency: p_hat matches p in value and gradient at
        # x_k, so the sub-problem's duals still certify the ORIGINAL problem and
        # the KKT termination test remains honest. That is the trade this flag
        # offers -- PCCP's step length with SIA's stopping rule.
        self.cache_subproblem = True   # build each phase's Pyomo model once and
                                       # re-point it; see SubproblemCache. Only
                                       # applies when every body is a Posynomial
                                       # or PosynomialRatio.
        self.verbose = False
        self.tee = False
        # tol tightened from IPOPT's 1e-8 default. The sub-problem's dual
        # accuracy propagates straight into the KKT residual, and at 1e-8 it
        # leaves a floor around 2e-3 -- above the stationarity tolerance, so a
        # converged run cannot report itself converged. Measured on SPaircraft:
        # 1.96e-03 at 1e-8 against 9.51e-07 at 1e-12, for no change in the
        # objective and no extra iterations.
        self.ipopt_options = {"print_level": 0, "sb": "yes", "tol": 1e-12}
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError(f"unknown SIA option {k!r}")
            setattr(self, k, v)


class SIAResult:
    def __init__(self):
        self.x = None
        self.objective = None
        self.iterations = 0
        self.converged = False
        self.status = "not run"
        #: KKT residuals on the ORIGINAL problem at the returned point
        self.max_violation = None
        self.stationarity = None
        self.complementarity = None
        self.multipliers = None
        self.history = []
        self.objectives = []
        self.conservative = False   # True if no constraint had to be linearized
        self.slacks_active = False  # True if the run ended with slack > 0
        self.phase1_iterations = 0  # cost of finding a feasible start
        self.phase1_feasible = None # None if phase 1 was not needed
        self.blocking = None        # rows that stopped Phase I, worst first
        self.slacks = None          # per-constraint infeasibility, elastic form
        self.phase1_mode = None     # 'minmax', or 'l1-rescue' if L1 saved it
        self.infeasibility_report = None  # the same, formatted for a human


    def __repr__(self):
        """One line saying what happened, and whether to believe it.

        SLCP's Result prints itself; this printed an object address, which is
        the least useful thing a result can say when the whole point of the
        method is what it can certify about the point it returns.
        """
        state = ('converged' if self.converged else
                 'NOT converged' if self.iterations else 'no iterations')
        obj = ('?' if self.objective is None else f'{self.objective:.6g}')
        return (f"<SIAResult {state}: objective={obj}, "
                f"iterations={self.iterations}, "
                f"stationarity={self.stationarity:.2e}, "
                f"max_violation={self.max_violation:.2e}, "
                f"complementarity={self.complementarity:.2e}>")

def classify(problem):
    """``(n_exact, n_conservative, n_linearized)`` for a problem."""
    n_exact = n_cons = n_lin = 0
    for c in problem.constraints:
        if c.exact_in_logspace:
            n_exact += 1
        elif isinstance(c.body, (PosynomialRatio, CondensedEquality)):
            n_cons += 1
        else:
            n_lin += 1
    return n_exact, n_cons, n_lin


def _log_g(con, x):
    """``log g(x)`` for a constraint written ``g <= 1`` (so feasible is <= 0)."""
    return math.log(max(con.body(x), 1e-300))


def _log_viol(con, x):
    """How far ``con`` is from being satisfied, in the log, signed so that
    positive means violated.

    An inequality ``g <= 1`` is violated only from above, so its violation is
    ``log g``. An equality ``g == 1`` is violated in **either** direction, so
    its violation is ``|log g|``.

    Using ``log g`` for both -- which this module did until a black-boxed
    equality turned up in a helicopter blade model -- has two consequences,
    and both are quiet. The feasibility half of the KKT test cannot see an
    equality that has drifted negative, so a point can be certified while
    violating it. And the trust-region ratio reads that drift as *improvement*,
    so it keeps rejecting the steps that would fix it: measured, the same
    constraint produced the same ratio of -3.3 every third iteration for 300
    iterations, knocking the radius back to 1.9e-6 each time and leaving 93%
    of the run stepping 1e-5 at a time.
    """
    lg = math.log(max(con.body(x), 1e-300))
    return abs(lg) if con.operator == '==' else lg


def _violation(problem, x):
    return max((_log_g(c, x) for c in problem.constraints), default=0.0)


def _agm(posy, x_k, weight_params, n):
    """Set the AGM weight Params for ``posy`` at ``x_k``; return (coeff, expo)."""
    v = posy(x_k)
    coeff, expo = 1.0, np.zeros(n)
    for k, (c, a) in enumerate(posy.terms):
        w = c * np.prod(x_k ** a) / v
        weight_params[k].value = float(w)
        if w > 0:
            coeff *= (c / w) ** w
            expo = expo + w * a
    return coeff, expo


def _violation_structured(problem, x):
    """Worst ``log g_i(x)`` over the constraints that cost nothing to evaluate.

    Skips black-box bodies. Used only to steer the step extension, where
    spending a black-box call per trial would defeat the purpose.
    """
    worst = -math.inf
    for con in problem.constraints:
        if isinstance(con.body, Signomial) and not isinstance(
                con.body, (Posynomial, PosynomialRatio)):
            continue
        worst = max(worst, _log_g(con, x))
    return worst


def _kkt(problem, x, mults, x_min=None, bound_tol=1e-6):
    """``(stationarity, violation, complementarity)`` on the TRUE problem.

    All gradients are the true ones -- for a condensed constraint that means
    ``log_grad(p) - log_grad(q)``, not the gradient of the monomial that stood
    in for ``q`` in the sub-problem. Tangency makes the two equal at the
    iterate, which is exactly why the sub-problem's duals certify the original
    problem.

    Stationarity is the **projected** gradient, which matters as soon as a
    variable reaches a bound. There the Lagrangian gradient is balanced by the
    bound's own multiplier and need not vanish: at a lower bound only a
    negative gradient is a violation, at an upper bound only a positive one.
    Taking the raw norm instead reports a large residual at a point that is
    perfectly optimal.

    This is not a corner case here. On SPaircraft the variables that reach a
    bound are exactly the design limits the model exists to express -- wing
    thickness at ``tau_max``, engine pressure ratio at 35, taper at its floor.
    Measured, every one of them carried the sign its bound admits, so the whole
    apparent residual was this.
    """
    g = np.asarray(problem.objective.log_grad(x), dtype=float)
    comp = 0.0
    viol = 0.0
    for i, con in enumerate(problem.constraints):
        lg = _log_g(con, x)
        viol = max(viol, _log_viol(con, x))
        lam = float(mults[i])
        if lam != 0.0:
            g = g + lam * np.asarray(con.body.log_grad(x), dtype=float)
        comp = max(comp, abs(lam * lg))

    # Bound violations count as violations. Since presolve folds a row like
    # `x >= 6` into a bound, a point below it is infeasible with no constraint
    # left to say so -- and the projection below would then zero the gradient
    # and call it optimal.
    lo_all = [None] * problem.n
    hi_all = [None] * problem.n
    if problem.bounds is not None:
        for j, pair in enumerate(problem.bounds[:problem.n]):
            if pair:
                lo_all[j], hi_all[j] = pair
    for j in range(problem.n):
        lo, hi = lo_all[j], hi_all[j]
        if x_min is not None and (lo is None or lo < x_min):
            lo = x_min           # the positivity floor bounds it too
        if lo is not None and lo > 0 and x[j] > 0:
            viol = max(viol, math.log(lo) - math.log(x[j]))
            if x[j] <= lo * (1.0 + bound_tol):
                # At a lower bound only a negative gradient is a violation;
                # a positive one is held by the bound's own multiplier.
                g[j] = min(g[j], 0.0)
        if hi is not None and hi > 0 and x[j] > 0:
            viol = max(viol, math.log(x[j]) - math.log(hi))
            if x[j] >= hi * (1.0 - bound_tol):
                g[j] = max(g[j], 0.0)
    return float(np.max(np.abs(g))), viol, comp


class Curvature:
    """A convex quadratic model of a black-box constraint's log-curvature.

    The linearized class is the one place SIA gives up its guarantees, and the
    reason is narrow: in log space a linearization IS a monomial, so the
    subproblem can hold it exactly -- it is just not *conservative*, because a
    value and a gradient say nothing about curvature.

    This supplies the missing piece. Carrying a positive semi-definite ``B``
    approximating the Hessian of ``log g`` in log space, the model

        log g(x_k) + grad^T d + 1/2 d^T B d

    is still convex in ``d`` (so the subproblem stays convex and the constraint
    is still held exactly), is exact and tangent at the iterate as before, and
    is an upper bound on the truth wherever ``B`` dominates the true
    log-Hessian. That is the conservative class's three properties, now
    conditional on ``B`` rather than free.

    ``B`` is built by damped BFGS from successive gradients, which is what
    keeps it positive semi-definite -- Powell's damping, as in Nocedal and
    Wright Procedure 18.2. Curvature that would make it indefinite is damped
    away rather than accepted, which is the conservative direction: a smaller
    ``B`` under-penalises the step, and the ratio test still catches that.

    Restricted to the variables the constraint actually depends on. A black box
    over four of sixty-one variables carries a 4x4 matrix, not 61x61, which is
    the difference between sixteen extra terms in the row and nearly four
    thousand.
    """

    def __init__(self, n, damping=0.2, cap=1e3):
        self.n = n
        self.damping = damping
        self.cap = cap
        self.support = []          # variable indices this constraint touches
        self._pos = {}             # variable index -> row in B
        self.B = np.zeros((0, 0))
        self.updates = 0
        self.inflations = 0
        self.relaxations = 0

    def observe(self, grad):
        """Extend the support to cover whatever the gradient touches."""
        new = [j for j in np.nonzero(np.abs(np.asarray(grad)) > 0)[0]
               if j not in self._pos]
        if not new:
            return
        for j in new:
            self._pos[j] = len(self.support)
            self.support.append(int(j))
        m = len(self.support)
        B = np.zeros((m, m))
        if self.B.size:
            B[:self.B.shape[0], :self.B.shape[1]] = self.B
        self.B = B

    def quad(self, d):
        """``1/2 d^T B d`` as a Pyomo expression over the support."""
        if not self.support or not self.B.size:
            return 0.0
        terms = []
        for a, ja in enumerate(self.support):
            for b, jb in enumerate(self.support):
                v = self.B[a, b]
                if v != 0.0:
                    terms.append(0.5 * float(v) * d[ja] * d[jb])
        return sum(terms) if terms else 0.0

    def update(self, s_full, y_full):
        """Damped BFGS from a step and the change in the log-gradient."""
        if not self.support:
            return
        idx = self.support
        s = np.asarray(s_full, dtype=float)[idx]
        y = np.asarray(y_full, dtype=float)[idx]
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(y))):
            return
        sBs = float(s @ self.B @ s)
        sy = float(s @ y)
        if sBs <= 0 and sy <= 0:
            return
        # Powell damping keeps B positive semi-definite even where the true
        # log-Hessian is not -- which is exactly the case a signomial
        # constraint presents.
        if sy < self.damping * sBs:
            denom = sBs - sy
            theta = (1.0 - self.damping) * sBs / denom if denom > 1e-300 else 1.0
            y = theta * y + (1.0 - theta) * (self.B @ s)
            sy = float(s @ y)
        if sy <= 1e-300 or sBs <= 1e-300:
            return
        Bs = self.B @ s
        self.B = self.B - np.outer(Bs, Bs) / sBs + np.outer(y, y) / sy
        self.B = 0.5 * (self.B + self.B.T)
        w, V = np.linalg.eigh(self.B)
        w = np.clip(w, 0.0, self.cap)          # PSD, and bounded
        self.B = V @ np.diag(w) @ V.T
        self.updates += 1

    def inflate(self, shortfall, s_full):
        """Raise B along the last step when the model under-predicted.

        Validation, using data the next iteration produces anyway: if the true
        constraint came in above what the model said, the model was optimistic
        there, and the curvature along that direction was underestimated by at
        least ``2*shortfall/||s||^2``.
        """
        if not self.support or shortfall <= 0:
            return
        s = np.asarray(s_full, dtype=float)[self.support]
        ss = float(s @ s)
        if ss <= 1e-300:
            return
        extra = 2.0 * shortfall / ss
        self.B = self.B + extra * np.outer(s, s) / ss
        self._clip()
        self.inflations += 1

    def relax(self, factor=0.9):
        """Ease B back when the model turned out to be pessimistic.

        Without this the model only ever gets more conservative: ``inflate``
        raises B whenever validation catches an optimistic prediction and
        nothing ever lowers it again, so B ratchets up for the whole run. On
        the helicopter that progressively wrecked the sub-problem's
        conditioning until IPOPT itself stopped converging at iteration 721 --
        the outer iteration was still descending, and the *inner* solve gave
        up.

        Conservatism that is no longer earned is not free: it shrinks steps and
        it curves a constraint that the evidence says is behaving. Easing it
        back is the same bet the trust region makes when it expands.
        """
        if self.B.size:
            self.B = float(factor) * self.B
            self.relaxations += 1

    def _clip(self):
        if not self.B.size:
            return
        self.B = 0.5 * (self.B + self.B.T)
        w, V = np.linalg.eigh(self.B)
        self.B = V @ np.diag(np.clip(w, 0.0, self.cap)) @ V.T


class SubproblemCache:
    """Build each phase's Pyomo model once and re-point it every iteration.

    The uncached path rebuilds every constraint symbolically on every
    iteration. On SPaircraft that is thousands of log-sum-exp expressions over
    a thousand variables, reconstructed from scratch once per iteration, and it
    dominates the run -- IPOPT itself is a small fraction of the wall clock.

    Almost none of that structure moves. Each exact term is

        exp( log c_k + a_k . (d + log x_k) )
            = exp( [log c_k + a_k . log x_k]  +  [a_k . d] )

    where ``a_k . d`` is FIXED and only the bracketed constant follows the
    iterate. So the projections are built once as Pyomo expressions and the
    constants become mutable Params.

    The AGM-condensed denominator of a ``PosynomialRatio`` looks like it breaks
    this, since its exponent vector ``aq`` is recomputed every iteration -- but
    ``aq = sum_i w_i a_i``, so

        aq . d = sum_i w_i (a_i . d)

    reuses the same fixed projections and needs one mutable weight per term
    instead of a full-length coefficient vector. That is the difference between
    a handful of scalars per constraint and an n-term expression per constraint.

    This is the same device as :class:`~edi.solvers.ipopt.slcp.SubproblemCache`,
    but the SIA sub-problem is easier to cache than SLCP's for two reasons.
    There is no BFGS quadratic, which is the one part SLCP has to rebuild every
    iteration; and a problem that is cacheable at all has no black-box
    constraint, hence no trust region -- so nothing but variable bounds and a
    few scalars changes between iterations.

    A model is built per ``(minimize_violation, use_slacks)`` phase, because the
    right-hand side differs structurally between them (``t``, ``s_i``, or
    nothing at all). Each phase builds once and is then reused for all of its
    iterations.
    """

    def __init__(self, problem, options):
        self.problem = problem
        self.options = options
        self.n = problem.n
        self.usable = self._is_cacheable()
        self._phases = {}
        self.builds = 0          # for tests and instrumentation

    def _is_cacheable(self):
        """Only the shapes the bridge produces; a black box needs a rebuild."""
        for con in self.problem.constraints:
            if not isinstance(con.body, (Posynomial, PosynomialRatio,
                                        CondensedEquality)):
                return False
        return isinstance(self.problem.objective, Posynomial)

    def get(self, minimize_violation, use_slacks):
        # NOT bool(minimize_violation): it takes three values now, and
        # bool('l1') is True, so the elastic mode collided with min-max in
        # this cache and was handed a model carrying m.t and no m.s.
        mv = minimize_violation if minimize_violation == 'l1' else bool(
            minimize_violation)
        key = (mv, bool(use_slacks))
        if key not in self._phases:
            self._phases[key] = self._build(*key)
            self.builds += 1
        return self._phases[key]

    # -- construction -------------------------------------------------------
    def _build(self, minimize_violation, use_slacks):
        n = self.n
        cons = self.problem.constraints

        m = pyo.ConcreteModel()
        m.J = pyo.RangeSet(0, n - 1)
        m.I = pyo.RangeSet(0, len(cons) - 1)
        m.d = pyo.Var(m.J, initialize=0.0)
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        m.params = pyo.Block()
        # Three modes, not two:
        #   False   Phase II -- true objective, no slacks
        #   True    Phase I MIN-MAX -- one shared t, min t s.t. log g_i <= t
        #   'l1'    Phase I ELASTIC -- a slack PER constraint, min sum(s_i)
        #
        # The min-max form drives every constraint to the SAME violation
        # level, which is why a stalled Phase I reports a dozen rows sitting
        # at an identical residual and none of them stands out. The elastic
        # form is what makes an infeasibility diagnosable: at its optimum
        # nearly every s_i is zero and the few that are not ARE the answer.
        elastic = (minimize_violation == 'l1')
        if minimize_violation and not elastic:
            m.t = pyo.Var(initialize=0.0)
        slacked = (use_slacks and not minimize_violation) or elastic
        if slacked:
            m.s = pyo.Var(m.I, domain=pyo.NonNegativeReals, initialize=0.0)
            m.tau = pyo.Param(mutable=True, initialize=0.0, within=pyo.Reals)

        counter = [0]

        def scalar():
            """A fresh mutable Param, since Pyomo needs each one named."""
            name = f'p{counter[0]}'
            counter[0] += 1
            setattr(m.params, name,
                    pyo.Param(mutable=True, initialize=0.0, within=pyo.Reals))
            return getattr(m.params, name)

        def projection(a):
            """a . d -- fixed for the life of the model."""
            nz = [j for j in range(min(len(a), n)) if a[j] != 0.0]
            return sum(float(a[j]) * m.d[j] for j in nz) if nz else 0.0

        def logsumexp(terms, params):
            if len(terms) == 1:
                return params[0] + projection(terms[0][1])
            return pyo.log(sum(pyo.exp(params[k] + projection(a))
                               for k, (_c, a) in enumerate(terms)))

        # --- objective -----------------------------------------------------
        obj_expr = None
        obj_b = []
        if not minimize_violation:
            terms = self.problem.objective.terms
            obj_b = [scalar() for _ in terms]
            obj_expr = logsumexp(terms, obj_b)
            penalty = (m.tau * sum(m.s[i] for i in range(len(cons)))
                       if slacked else 0.0)
            m.obj = pyo.Objective(expr=obj_expr + penalty, sense=pyo.minimize)
        elif elastic:
            # Pure feasibility: no true objective at all, so there is no
            # scaling contest between cost and feasibility to lose.
            m.obj = pyo.Objective(expr=sum(m.s[i] for i in m.I),
                                  sense=pyo.minimize)
        else:
            m.obj = pyo.Objective(expr=m.t, sense=pyo.minimize)

        def rhs(i):
            if elastic:
                return m.s[i]
            if minimize_violation:
                return m.t
            return m.s[i] if slacked else 0.0

        # --- constraints ---------------------------------------------------
        m.cons = pyo.ConstraintList()
        b_params, q_weights, q_consts, p_consts = [], [], [], []
        condense_num = getattr(self.options, 'condense_numerator', False)
        for i, con in enumerate(cons):
            body, op = con.body, con.operator
            if isinstance(body, Posynomial):
                ps = [scalar() for _ in body.terms]
                b_params.append(ps)
                q_weights.append(None)
                q_consts.append(None)
                p_consts.append(None)
                e = logsumexp(body.terms, ps)
                if body.is_monomial and op == '==':
                    m.cons.add(e == rhs(i))
                else:
                    m.cons.add(e <= rhs(i))
            elif isinstance(body, CondensedEquality):
                # (a_p - a_q).d = sum_k w^p_k (a^p_k.d) - sum_k w^q_k (a^q_k.d),
                # so the same fixed projections serve, with one mutable weight
                # per term on each side and a single constant.
                ps = [scalar() for _ in body.p.terms]
                ws = [scalar() for _ in body.q.terms]
                qc = scalar()
                b_params.append(ps)
                q_weights.append(ws)
                q_consts.append(qc)
                p_consts.append('eq')
                expr = qc + sum(ps[k] * projection(a)
                                for k, (_c, a) in enumerate(body.p.terms)) \
                          - sum(ws[k] * projection(a)
                                for k, (_c, a) in enumerate(body.q.terms))
                if elastic:
                    # |residual| <= s_i, this row's own slack.
                    m.cons.add(expr <= m.s[i])
                    m.cons.add(-expr <= m.s[i])
                elif minimize_violation:
                    # |residual| <= t, NOT residual == t. The Phase I
                    # violation variable is a single scalar shared by every
                    # constraint: `expr == t` therefore forces EVERY
                    # signomial equality to the SAME residual, and two
                    # equalities that cannot be driven to a common value
                    # inside the trust region make the sub-problem
                    # infeasible -- which surfaces as "phase 1 could not
                    # find a feasible point after 1 iterations" on a problem
                    # that is perfectly feasible.
                    m.cons.add(expr <= m.t)
                    m.cons.add(-expr <= m.t)
                else:
                    m.cons.add(expr == rhs(i))
            else:                                     # PosynomialRatio
                ps = [scalar() for _ in body.p.terms]
                ws = [scalar() for _ in body.q.terms]
                qc = scalar()
                b_params.append(ps)
                q_weights.append(ws)
                q_consts.append(qc)
                # log q_hat = [log cq + aq . log x_k] + sum_k w_k (a_k . d)
                log_qhat = qc + sum(
                    ws[k] * projection(a)
                    for k, (_c, a) in enumerate(body.q.terms))
                if condense_num:
                    # The numerator condenses by the identical device, so it
                    # reuses the same fixed projections with its own weights.
                    pc = scalar()
                    p_consts.append(pc)
                    log_phat = pc + sum(
                        ps[k] * projection(a)
                        for k, (_c, a) in enumerate(body.p.terms))
                    m.cons.add(log_phat - log_qhat <= rhs(i))
                else:
                    p_consts.append(None)
                    m.cons.add(logsumexp(body.p.terms, ps) - log_qhat <= rhs(i))

        return _CachedPhase(model=m, obj_expr=obj_expr, obj_b=obj_b,
                            b_params=b_params, q_weights=q_weights,
                            q_consts=q_consts, p_consts=p_consts,
                            slacked=slacked,
                            minimize_violation=minimize_violation)

    # -- per-iteration update ----------------------------------------------
    def update(self, phase, x_k, tau):
        """Re-point a built phase at a new iterate. No symbolic work."""
        m, n = phase.model, self.n
        log_xk = np.log(x_k)

        if not phase.minimize_violation:
            for k, (c, a) in enumerate(self.problem.objective.terms):
                phase.obj_b[k].value = float(math.log(c) + a @ log_xk)
        if phase.slacked:
            m.tau.value = float(tau)

        for i, con in enumerate(self.problem.constraints):
            body = con.body
            if phase.p_consts[i] == 'eq':
                cp, ap = _agm(body.p, x_k, phase.b_params[i], n)
                cq, aq = _agm(body.q, x_k, phase.q_weights[i], n)
                phase.q_consts[i].value = float(
                    math.log(cp) - math.log(cq) + (ap - aq) @ log_xk)
                continue
            terms = (body.terms if isinstance(body, Posynomial)
                     else body.p.terms)
            if phase.p_consts[i] is not None:
                # Condensed numerator: b_params hold AGM WEIGHTS, not the
                # log-constants they hold in the exact case.
                pv = body.p(x_k)
                const, ap = 0.0, np.zeros(n)
                for k, (c, a) in enumerate(terms):
                    w = c * np.prod(x_k ** a) / pv
                    phase.b_params[i][k].value = float(w)
                    if w > 0:
                        const += w * math.log(c / w)
                        ap = ap + w * a
                phase.p_consts[i].value = float(const + ap @ log_xk)
            else:
                for k, (c, a) in enumerate(terms):
                    phase.b_params[i][k].value = float(
                        math.log(c) + a @ log_xk)
            if phase.q_weights[i] is None:
                continue
            # AGM weights, and the constant part of the condensed monomial.
            qv = body.q(x_k)
            const, aq = 0.0, np.zeros(n)
            for k, (c, a) in enumerate(body.q.terms):
                w = c * np.prod(x_k ** a) / qv
                phase.q_weights[i][k].value = float(w)
                if w > 0:
                    const += w * math.log(c / w)
                    aq = aq + w * a
            phase.q_consts[i].value = float(const + aq @ log_xk)

        # The positivity floor and any model bounds move with the iterate, but
        # they are bounds, so re-pointing them is free.
        floor = math.log(self.options.x_min)
        lo = floor - log_xk
        hi = np.full(n, np.inf)
        if self.problem.bounds is not None:
            for j, (blo, bhi) in enumerate(self.problem.bounds[:n]):
                if blo is not None and blo > 0:
                    lo[j] = max(lo[j], math.log(blo) - log_xk[j])
                if bhi is not None and bhi > 0:
                    hi[j] = min(hi[j], math.log(bhi) - log_xk[j])
        for j in range(n):
            ub = None if not np.isfinite(hi[j]) else float(hi[j])
            m.d[j].setlb(float(lo[j]))
            m.d[j].setub(ub)
            # Start from d = 0 (the current iterate), but inside the box: a
            # variable already at its bound has 0 outside, and Pyomo warns.
            start = min(max(0.0, float(lo[j])), ub if ub is not None else 0.0)
            m.d[j].set_value(start)
        if phase.minimize_violation == 'l1':
            # Elastic: warm-start every slack at this row's own violation,
            # which is its smallest feasible value for the linearised model.
            v0 = float(_violation(self.problem, x_k))
            for i in range(len(self.problem.constraints)):
                m.s[i].set_value(max(0.0, min(v0, _log_g(
                    self.problem.constraints[i], x_k))))
        elif phase.minimize_violation:
            m.t.set_value(float(_violation(self.problem, x_k)))
        return phase


class _CachedPhase:
    """Handles into one built phase model."""

    __slots__ = ('model', 'obj_expr', 'obj_b', 'b_params', 'q_weights',
                 'q_consts', 'p_consts', 'slacked', 'minimize_violation')

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _subproblem(problem, x_k, tau, radius, options, has_blackbox,
                curvature=None,
                minimize_violation=False, use_slacks=True, cache=None):
    """Assemble and solve the inner-approximation sub-problem in log space.

    ``minimize_violation`` selects PHASE I: the objective becomes the worst
    constraint violation ``t``, every constraint is written ``... <= t``, and
    the true objective is ignored. That problem is feasible by construction
    (raise ``t``), its objective is a single variable so there is no scaling
    contest between cost and feasibility, and the same conservative
    representations apply -- so ``t`` decreases monotonically. When it reaches
    zero the iterate is feasible for the true problem and Phase II can run with
    no slacks and no penalty at all.

    Otherwise it is PHASE II: no slacks, no penalty. The iterate is feasible,
    every constraint is exact or conservative, so the sub-problem's optimum is
    feasible for the true problem and cannot be worse than the current point.
    """
    n = problem.n
    cons = problem.constraints
    log_xk = np.log(x_k)

    if cache is not None and cache.usable:
        # Everything below is symbolic construction that does not change
        # between iterations; the cache does it once and only moves the
        # numbers. Bounds, including the positivity floor, are re-pointed in
        # update(). A cacheable problem has no black box, so no trust region.
        phase = cache.update(cache.get(minimize_violation, use_slacks),
                             x_k, tau)
        try:
            return _solve_and_extract(phase.model, problem, options,
                                      minimize_violation, use_slacks,
                                      phase.obj_expr)
        except RuntimeError:
            # A cache must never change WHETHER something solves, only how
            # fast. The Param-formulated model is numerically identical on
            # paper but not to IPOPT at tol = 1e-12, and on the hydrogen
            # aircraft the cached Phase I burned its whole iteration budget
            # where the inlined-float build below solves in a handful --
            # which then read as "phase 1 could not find a feasible point
            # after 1 iterations" with no hint that a cache was involved.
            # Fall through and build this one iteration fresh.
            pass

    m = pyo.ConcreteModel()
    m.J = pyo.RangeSet(0, n - 1)
    m.I = pyo.RangeSet(0, len(cons) - 1)
    m.d = pyo.Var(m.J, initialize=0.0)
    # Variable bounds ride on the variable. The sub-problem works in log space
    # about x_k -- x = x_k * exp(d) -- so `lo <= x <= hi` is just
    # `log(lo/x_k) <= d <= log(hi/x_k)`, which is exact and costs nothing.
    if problem.bounds is not None:
        for j, (lo, hi) in enumerate(problem.bounds[:n]):
            if lo is not None and lo > 0:
                m.d[j].setlb(math.log(lo) - log_xk[j])
            if hi is not None and hi > 0:
                m.d[j].setub(math.log(hi) - log_xk[j])
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    if minimize_violation:
        m.t = pyo.Var(initialize=float(_violation(problem, x_k)))
    if use_slacks and not minimize_violation:
        m.s = pyo.Var(m.I, domain=pyo.NonNegativeReals, initialize=0.0)

    def lse(terms):
        """log sum_k c_k exp(a_k . (d + log x_k)) -- exact, convex."""
        return pyo.log(sum(
            pyo.exp(math.log(c) + sum(a[j] * (m.d[j] + log_xk[j])
                                      for j in range(n)))
            for c, a in terms))

    # --- objective ---------------------------------------------------------
    if minimize_violation:
        # Phase I: drive the worst violation down and nothing else.
        m.obj = pyo.Objective(expr=m.t, sense=pyo.minimize)
        rhs = lambda i: m.t
    elif use_slacks:
        rhs = lambda i: m.s[i]
    else:
        # Phase II proper: the iterate is feasible and every constraint is
        # exact or conservative, so the constraint is imposed AS IT STANDS.
        # No slack variable, no penalty, nothing to distort the objective --
        # this is the pure inner approximation the guarantees are stated for.
        rhs = lambda i: 0.0
    if minimize_violation:
        pass
    elif isinstance(problem.objective, Posynomial):
        obj = lse(problem.objective.terms)
    else:
        f_k = problem.objective_value(x_k)
        gf = problem.objective.log_grad(x_k)
        obj = math.log(f_k) + sum(gf[j] * m.d[j] for j in range(n))

    if not minimize_violation:
        penalty = (tau * sum(m.s[i] for i in range(len(cons)))
                   if use_slacks else 0.0)
        m.obj = pyo.Objective(expr=obj + penalty, sense=pyo.minimize)

    m.cons = pyo.ConstraintList()
    for i, con in enumerate(cons):
        body, op = con.body, con.operator
        if con.exact_in_logspace:
            if body.is_monomial:
                c, a = body.terms[0]
                e = math.log(c) + sum(a[j] * (m.d[j] + log_xk[j])
                                      for j in range(n))
                m.cons.add(e == rhs(i) if op == "==" else e <= rhs(i))
            else:
                m.cons.add(lse(body.terms) <= rhs(i))
        elif isinstance(body, CondensedEquality):
            # Both sides condensed -> a monomial equality, affine in log space.
            # One signed multiplier, and a hyperplane rather than a null space.
            ce, ae = body.condensed(x_k)
            e = math.log(ce) + sum(ae[j] * (m.d[j] + log_xk[j])
                                   for j in range(n))
            m.cons.add(e == rhs(i))
        elif isinstance(body, PosynomialRatio):
            # log p  <=  log q_hat, with q_hat the AGM monomial under-estimator.
            # q_hat <= q everywhere, so this is HARDER than the true constraint.
            cq, aq = body.condensed_q(x_k)
            log_qhat = math.log(cq) + sum(aq[j] * (m.d[j] + log_xk[j])
                                          for j in range(n))
            if options.condense_numerator:
                cp, ap = body.condensed_p(x_k)
                log_phat = math.log(cp) + sum(ap[j] * (m.d[j] + log_xk[j])
                                              for j in range(n))
                m.cons.add(log_phat - log_qhat <= rhs(i))
            else:
                m.cons.add(lse(body.p.terms) - log_qhat <= rhs(i))
        else:
            # Black box: value and gradient only, so no conservative model
            # exists. Linearize, and let the trust region below carry it.
            v = body(x_k)
            gl = body.log_grad(x_k)
            e = math.log(max(v, 1e-300)) + sum(gl[j] * m.d[j] for j in range(n))
            # With a curvature model the row becomes a convex quadratic rather
            # than a plane: still convex, so the subproblem is unchanged in
            # kind, but now an upper bound wherever B dominates the true
            # log-curvature -- which is the property the trust region exists to
            # substitute for.
            if curvature is not None and i in curvature:
                curvature[i].observe(gl)
                e = e + curvature[i].quad(m.d)
            m.cons.add(e == rhs(i) if op == "==" else e <= rhs(i))

    # Stay in the positive orthant.
    # The positivity floor and the trust region are both simple bounds on d, so
    # they go on the variable rather than into m.cons -- 3n fewer rows on every
    # sub-problem, and IPOPT handles a bound more cheaply than a row besides.
    # Intersect with whatever the model's own bounds already put there.
    floor = math.log(options.x_min)

    def tighten(j, lo=None, hi=None):
        if lo is not None:
            cur = m.d[j].lb
            m.d[j].setlb(lo if cur is None else max(cur, lo))
        if hi is not None:
            cur = m.d[j].ub
            m.d[j].setub(hi if cur is None else min(cur, hi))

    for j in range(n):
        tighten(j, lo=floor - log_xk[j])

    # Trust region -- ONLY when something had to be linearized. With every
    # constraint exact or conservative the step is safe by construction and a
    # region would only slow it down.
    if has_blackbox:
        for j in range(n):
            tighten(j, lo=-radius, hi=radius)

    return _solve_and_extract(m, problem, options, minimize_violation,
                              use_slacks, obj if not minimize_violation else None)


def _solve_and_extract(m, problem, options, minimize_violation, use_slacks,
                       obj):
    """Solve an assembled sub-problem and read off the step and multipliers.

    Shared by the rebuild path and the cached one, so both report failures
    identically and both use the same multiplier sign convention.
    """
    n = problem.n
    cons = problem.constraints

    opt = pyo.SolverFactory("ipopt")
    if not opt.available(exception_flag=False):
        raise RuntimeError(
            "SIA needs IPOPT to solve its sub-problems; no usable installation "
            "was found. Install the ipopt executable or `pip install cyipopt`.")
    for k, v in (options.ipopt_options or {}).items():
        opt.options[k] = v
    results = opt.solve(m, tee=options.tee, load_solutions=False)
    tc = str(results.solver.termination_condition)
    if tc not in ("optimal", "locallyOptimal", "feasible"):
        raise RuntimeError(f"the SIA sub-problem failed: {tc}")
    m.solutions.load_from(results)

    d = np.array([pyo.value(m.d[j]) for j in range(n)])
    _elastic = (minimize_violation == 'l1')
    s = (np.array([pyo.value(m.s[i]) for i in range(len(cons))])
         if ((use_slacks and not minimize_violation) or _elastic)
         else np.zeros(len(cons)))

    # Multipliers. An INEQUALITY multiplier is non-negative by definition, so
    # its magnitude is the quantity wanted. An EQUALITY multiplier is not --
    # it carries a sign, and taking its magnitude makes the stationarity sum
    # unable to cancel. On Hoburg, 25 of 58 constraints are equalities, and
    # dropping their signs pins the residual at 2.5 no matter how converged
    # the iterate is. The rest of SLCP takes abs() throughout because it only
    # ever uses these for a merit function and a BFGS update, where magnitude
    # is all that matters; a KKT certificate needs the sign.
    mults = np.zeros(len(cons))
    for i in range(len(cons)):
        try:
            lam = m.dual.get(m.cons[i + 1], 0.0) or 0.0
        except Exception:
            lam = 0.0
        # Pyomo/IPOPT report the equality dual with the opposite sign to the
        # one the Lagrangian gradient f + sum(lam * g) wants, so it is negated.
        # Verified against a least-squares fit of the multipliers that zero
        # stationarity at a converged Hoburg point: magnitudes agree exactly,
        # and only the equality signs were inverted.
        mults[i] = (-float(lam) if cons[i].operator == '=='
                    else abs(float(lam)))
    if _elastic:
        # The scalar returned is the total infeasibility, sum(s_i). Zero (to
        # tolerance) means a feasible point; anything else is how much
        # constraint violation the model cannot get rid of, and `s` says
        # exactly WHERE it is.
        return d, s, mults, float(sum(s))
    if minimize_violation:
        return d, s, mults, float(pyo.value(m.t))
    return d, s, mults, float(pyo.value(obj))



def _blocking_constraints(problem, x, mults=None, k=8):
    """Which constraints are stopping Phase I, and which variables they touch.

    Phase I minimises the WORST violation, so when it stalls the useful
    question is not "what is the violation" but "which rows are holding it
    up, and are they holding each other up". This reports both signals that
    answer that:

    * **residual** -- ``log g_i`` at the stalled point. Anything above the
      feasibility tolerance is unsatisfied there.
    * **multiplier** -- from the last sub-problem. A row with a large
      multiplier on the min-max objective is one the search is actively
      trading against; a row with a big residual and a *zero* multiplier is
      along for the ride and will move once the blockers do.

    Constraints carry no names -- ``Constraint.__slots__`` is
    ``('body', 'operator')`` -- so each is identified by the variables its
    exponents actually touch, which is more use than an index anyway.
    """
    rows = []
    for i, c in enumerate(problem.constraints):
        r = _log_g(c, x)
        m = float(mults[i]) if mults is not None and i < len(mults) else 0.0
        rows.append((r, m, i, c))
    rows.sort(key=lambda t: (-t[0], -t[1]))
    out = []
    for r, m, i, c in rows[:k]:
        body = getattr(c, 'body', None)
        terms = list(getattr(body, 'terms', None) or [])
        # CondensedEquality and PosynomialRatio hold two posynomials rather
        # than terms of their own, and BOTH sides matter: reading only one
        # was reporting "vars: -" for exactly the equality rows this is
        # supposed to explain.
        for side in ('p', 'q'):
            sub = getattr(body, side, None)
            if sub is not None:
                terms.extend(getattr(sub, 'terms', None) or [])
        involved = set()
        for _coef, a in (terms or []):
            for j, e in enumerate(a):
                if e != 0.0:
                    involved.add(j)
        names = [problem.names[j] for j in sorted(involved)][:6]
        out.append({'index': i, 'operator': c.operator, 'log_g': r,
                    'multiplier': m, 'variables': names})
    return out


def format_infeasibility(problem, x, mults=None, k=8):
    """A human-readable account of why Phase I could not find a point."""
    rows = _blocking_constraints(problem, x, mults, k)
    lines = ['Phase I stopped with max log g = %.4g. Blocking rows:' %
             _violation(problem, x)]
    for r in rows:
        lines.append('  [%5d] %-2s log g = %+.4e  mult = %.3e  vars: %s'
                     % (r['index'], r['operator'], r['log_g'],
                        r['multiplier'], ', '.join(r['variables']) or '-'))
    lines.append('A row with a large residual AND a large multiplier is an '
                 'active blocker; large residual with zero multiplier will '
                 'follow once the blockers move.')
    return '\n'.join(lines)



def _phase1_l1(problem, x, options, has_blackbox, cache=None):
    """Elastic Phase I: a slack per constraint, minimising their SUM.

    ``min sum(s_i)  s.t.  log g_i(x) <= s_i,  s_i >= 0``

    This is the formulation that makes an infeasibility *diagnosable*, and the
    difference from the min-max form is not cosmetic. Min-max drives every
    constraint to a common violation level, so a stalled run shows a dozen
    rows at an identical residual with nothing to choose between them. The L1
    optimum is sparse instead: constraints that CAN be satisfied go to
    ``s_i = 0`` and drop out, and the few that cannot are the answer -- an
    approximate irreducible inconsistent subsystem, read straight off the
    solution.

    Returns ``(x, iterations, feasible, slacks, multipliers)``. ``slacks`` is
    the per-constraint infeasibility at the final point; anything above
    tolerance names a row that has to be relaxed for the model to close.
    """
    it = 0
    radius = options.trust_radius
    slacks = np.zeros(len(problem.constraints))
    mults = None
    for it in range(1, options.phase1_max_iterations + 1):
        viol = _violation(problem, x)
        if viol <= options.feasibility_tolerance:
            return x, it - 1, True, slacks, mults
        try:
            d, slacks, mults, total = _subproblem(
                problem, x, 0.0, radius, options, has_blackbox,
                minimize_violation='l1', cache=cache)
        except RuntimeError:
            if radius > options.trust_min:
                radius = max(options.trust_min, radius * options.trust_shrink)
                continue
            return x, it, False, slacks, mults
        x_new = x * np.exp(d)
        new_viol = _violation(problem, x_new)
        if options.verbose:
            nz = int(np.sum(slacks > options.feasibility_tolerance))
            print(f"  phase1-L1 {it:3d}  max log g {viol:+.3e} -> "
                  f"{new_viol:+.3e}   sum(s) = {total:.4e}  ({nz} rows slack)")
        if new_viol > viol:
            radius *= options.trust_shrink
            if radius < options.trust_min:
                return x, it, viol <= options.feasibility_tolerance, slacks, mults
            continue
        if abs(new_viol - viol) <= 1e-14 * max(1.0, abs(viol)):
            x = x_new
            if radius > options.trust_min:
                radius = max(options.trust_min, radius * options.trust_shrink)
                continue
            return (x, it,
                    _violation(problem, x) <= options.feasibility_tolerance,
                    slacks, mults)
        x = x_new
        radius = min(options.trust_max, radius * options.trust_expand)
    return (x, it, _violation(problem, x) <= options.feasibility_tolerance,
            slacks, mults)


def explain_infeasibility(problem, x, options=None, has_blackbox=False,
                          cache=None, k=12):
    """Run the elastic Phase I and report which rows cannot be satisfied.

    This is the answer to "why is my model infeasible". It re-solves the
    feasibility problem in its L1 form and lists the constraints whose slack
    stays positive -- the ones that must be relaxed -- together with the
    variables each touches.
    """
    options = options or SIAOptions()
    x1, it, feasible, slacks, mults = _phase1_l1(
        problem, np.asarray(x, dtype=float).copy(), options, has_blackbox,
        cache=cache)
    tol = options.feasibility_tolerance
    order = np.argsort(-slacks)
    lines = []
    if feasible:
        lines.append(f"FEASIBLE: elastic Phase I found a point in {it} "
                     f"iterations (all slacks <= {tol:g}).")
        return '\n'.join(lines), x1, slacks
    nz = [i for i in order if slacks[i] > tol]
    lines.append(f"INFEASIBLE at this point: {len(nz)} of "
                 f"{len(problem.constraints)} rows keep a positive slack "
                 f"after {it} elastic iterations (sum = {slacks.sum():.4e}).")
    lines.append("These are the rows that must be relaxed for the model to "
                 "close, largest first:")
    for i in nz[:k]:
        c = problem.constraints[i]
        body = getattr(c, 'body', None)
        terms = list(getattr(body, 'terms', None) or [])
        for side in ('p', 'q'):
            sub = getattr(body, side, None)
            if sub is not None:
                terms.extend(getattr(sub, 'terms', None) or [])
        involved = sorted({j for _c, a in terms
                           for j, e in enumerate(a) if e != 0.0})
        names = [problem.names[j] for j in involved][:6]
        mu = float(mults[i]) if mults is not None and i < len(mults) else 0.0
        lines.append(f"  [{i:5d}] {c.operator:2s} slack = {slacks[i]:.4e}  "
                     f"dual = {mu:.3e}   vars: {', '.join(names) or '-'}")
    return '\n'.join(lines), x1, slacks


def _phase1(problem, x, options, has_blackbox, cache=None):
    """Find a feasible point by minimizing the worst constraint violation.

    Solves ``min t  s.t.  log g_i(x) <= t`` by the same inner approximation.
    Three things make this a much better-behaved problem than the penalty
    formulation it replaces:

    * it is **always feasible** -- raise ``t`` -- so the sub-problem can never
      be unreachable and there is no penalty parameter to tune;
    * its objective is a single variable, so there is no scaling contest
      between cost and feasibility. That contest is what wrecked the penalty
      version: with an objective of order log(20000) and tau = 1, the first
      sub-problem effectively ignored 6077 constraints and took a step of
      e^59, landing in a basin it never left;
    * the same conservative representations apply, so ``t`` decreases
      monotonically.

    Returns ``(x, iterations, feasible)``. The margin asks for *strictly*
    feasible, so Phase II starts inside the set rather than on its boundary
    where round-off can push it out.
    """
    it = 0
    radius = options.trust_radius
    last_mults = None
    for it in range(1, options.phase1_max_iterations + 1):
        viol = _violation(problem, x)
        if viol <= options.feasibility_tolerance:
            return x, it - 1, True, last_mults
        try:
            d, _, last_mults, t = _subproblem(problem, x, 0.0, radius, options,
                                              has_blackbox,
                                              minimize_violation=True,
                                              cache=cache)
        except RuntimeError:
            # The min-max sub-problem is ALWAYS feasible on paper -- raise t
            # -- so a solver failure here means the trust region, not the
            # model. Shrinking is the right response and the one that was
            # missing: this used to give up immediately on anything without a
            # black box, which is how a solvable problem came back as "phase 1
            # could not find a feasible point after 1 iterations".
            if radius > options.trust_min:
                radius = max(options.trust_min, radius * options.trust_shrink)
                if options.verbose:
                    print(f"  phase1 {it:3d}  sub-problem failed, "
                          f"radius -> {radius:.3g}")
                continue
            # Out of room. Widen once instead, in case the step was too
            # SHORT to escape a badly scaled region, before giving up.
            if radius < options.trust_max:
                radius = min(options.trust_max,
                             options.trust_radius * options.trust_expand)
                continue
            return x, it, viol <= 0.0, last_mults
        x_new = x * np.exp(d)
        new_viol = _violation(problem, x_new)

        if new_viol > viol:
            # A step that makes things worse is a trust-region signal, not a
            # reason to accept it. The conservative representation says the
            # violation cannot increase; in finite precision, with IPOPT
            # solving the sub-problem to a tolerance and the approximation
            # exact only AT x_k, it sometimes does. This used to be checked
            # only for black boxes, so on a pure SP a bad step was taken
            # anyway and Phase I could wander uphill. The monotone decrease
            # the docstring promises is now actually enforced.
            if radius > options.trust_min:
                radius = max(options.trust_min,
                             radius * options.trust_shrink)
                if options.verbose:
                    print(f"  phase1 {it:3d}  step worsened "
                          f"{viol:+.3e} -> {new_viol:+.3e}, "
                          f"radius -> {radius:.3g}")
                continue
            # At trust_min the approximation is as accurate as it is going to
            # get, and the step still looks worse. ACCEPT it and carry on --
            # do NOT give up here.
            #
            # Giving up was a regression: the original code had no monotone
            # check at all on a pure SP, so it took these steps and SPaircraft
            # converged. Rejecting them outright turned a converging run into
            # "phase 1 could not find a feasible point after 15 iterations".
            # A worsening step at the smallest radius is numerical noise on a
            # conservative approximation, not evidence of a bad direction.
            if options.verbose:
                print(f"  phase1 {it:3d}  step worsened at trust_min, "
                      f"accepting {viol:+.3e} -> {new_viol:+.3e}")

        if options.verbose:
            print(f"  phase1 {it:3d}  max log g: {viol:+.3e} -> "
                  f"{new_viol:+.3e}   (model t = {t:+.3e})")
        if abs(new_viol - viol) <= 1e-14 * max(1.0, abs(viol)):
            # No further reduction AT THIS RADIUS. That is not the same as no
            # further reduction: a smaller trust region gives a tighter, more
            # accurate approximation and often moves again. Only a stall that
            # survives shrinking to trust_min is a real stall.
            if radius > options.trust_min:
                radius = max(options.trust_min, radius * options.trust_shrink)
                if options.verbose:
                    print(f"  phase1 {it:3d}  stalled at {viol:+.3e}, "
                          f"radius -> {radius:.3g}")
                x = x_new
                continue
            x = x_new
            return (x, it,
                    _violation(problem, x) <= options.feasibility_tolerance,
                    last_mults)
        x = x_new
        # Expand on a good step regardless of black-box status: the step was
        # accepted because it reduced the violation, and that is exactly when
        # a longer one is worth trying.
        radius = min(options.trust_max, radius * options.trust_expand)
    return x, it, _violation(problem, x) <= 0.0, last_mults


def solve_sia(problem: Problem, x0, options: SIAOptions = None) -> SIAResult:
    """Solve a signomial program by sequential inner approximation."""
    options = options or SIAOptions()
    x = np.asarray(x0, dtype=float).copy()
    if np.any(x <= 0):
        raise ValueError("SIA works in log space, so x0 must be strictly positive")

    # Project the start into its box. The bounds are imposed on the sub-problem
    # variable, so a start outside them is not merely a poor guess -- Phase I
    # measures feasibility with `_violation`, which reads the constraints and
    # not the bounds, so it can call a bound-violating point feasible while the
    # sub-problem cannot move to it, and the run stops at iteration 0.
    #
    # This costs nothing when the guess is already inside, and it is what makes
    # a model usable after `propagate_bounds`: propagation derives tight bounds
    # from the constraints, and a hand-written initial guess has no reason to
    # respect bounds nobody had computed yet.
    if problem.bounds is not None:
        for j, pair in enumerate(problem.bounds[:problem.n]):
            lo, hi = pair or (None, None)
            if lo is not None and lo > 0 and x[j] < lo:
                x[j] = lo
            if hi is not None and hi > 0 and x[j] > hi:
                x[j] = hi

    n_exact, n_cons, n_lin = classify(problem)
    has_blackbox = n_lin > 0

    res = SIAResult()
    res.conservative = not has_blackbox
    res.history.append(x.copy())

    tau = options.tau0
    radius = options.trust_radius
    mults = np.zeros(len(problem.constraints))

    # Build the sub-problem once per phase and re-point it thereafter. The
    # symbolic structure does not change between iterations; only a few scalars
    # per constraint do. Falls back to rebuilding for anything with a black-box
    # body, whose gradient has to be re-linearized every time anyway.
    cache = SubproblemCache(problem, options) if options.cache_subproblem \
        else None
    if cache is not None and not cache.usable:
        cache = None

    if options.verbose:
        print(f"  SIA: {n_exact} exact, {n_cons} conservative, "
              f"{n_lin} linearized"
              + ("" if has_blackbox else "  -> fully conservative, "
                                         "no globalization needed"))

    # --- Phase I ----------------------------------------------------------
    # The conservative guarantees -- feasible iterates, monotone descent --
    # hold FROM A FEASIBLE POINT. Rather than blend cost and feasibility into
    # one penalized objective and hope, get feasible first on its own terms,
    # then optimize with the guarantees switched on and no penalty at all.
    if options.phase1 and _violation(problem, x) > options.feasibility_tolerance:
        x, res.phase1_iterations, feasible, p1_mults = _phase1(
            problem, x, options, has_blackbox, cache=cache)
        res.history.append(x.copy())
        res.phase1_feasible = feasible
        if options.verbose:
            print(f"  phase 1: {res.phase1_iterations} iterations, "
                  f"{'FEASIBLE' if feasible else 'still infeasible'}, "
                  f"max log g = {_violation(problem, x):+.3e}")
        if not feasible:
            # Min-max could not close. Try the ELASTIC form before reporting
            # that, for two reasons.
            #
            # First, it often succeeds where min-max does not. Min-max has to
            # drag every constraint down together, so one stubborn row holds
            # the whole vector up. L1 lets satisfied rows fall to zero slack
            # and get out of the way.
            #
            # Second, when it does fail it says WHY in the only form that is
            # actionable: the specific rows whose slack cannot reach zero.
            # "Could not find a feasible point" names nothing, and reads like
            # a verdict on the model when Phase I is a local method that may
            # simply have started too far away.
            report, x_l1, slacks = explain_infeasibility(
                problem, x, options, has_blackbox, cache=cache)
            res.infeasibility_report = report
            res.slacks = slacks
            if _violation(problem, x_l1) <= options.feasibility_tolerance:
                x, feasible = x_l1, True
                res.phase1_feasible = True
                res.phase1_mode = 'l1-rescue'
                if options.verbose:
                    print("  phase 1: min-max failed, elastic form succeeded")
            else:
                res.blocking = _blocking_constraints(problem, x, p1_mults)
        if not feasible:
            res.status = (f"phase 1 could not find a feasible point after "
                          f"{res.phase1_iterations} iterations "
                          f"(max log g = {_violation(problem, x):.3e}); "
                          f"blocking rows: " + ", ".join(
                              f"{b['index']}({'/'.join(b['variables'][:2])})"
                              for b in (res.blocking or [])[:3]))
            res.x, res.objective = x, problem.objective_value(x)
            stat, viol, comp = _kkt(problem, x, mults, options.x_min)
            res.stationarity, res.max_violation, res.complementarity = (
                stat, viol, comp)
            # Hand off to the PENALTY path rather than giving up.
            #
            # This module's own docstring says penalty CCP is "used only if
            # phase1 is off or FAILS", but the failure branch returned here and
            # the fallback did not exist. That mattered: Phase I would stop at
            # max log g = 2.4e-6 against a 1e-6 tolerance -- a factor of two
            # short, on a problem that solves in 149 iterations if the
            # tolerance is loosened by one decade -- and the whole solve was
            # abandoned over it.
            #
            # Penalty CCP does not need a feasible start; that is its entire
            # reason for existing. Slacks absorb what remains and tau drives
            # them out. If the problem really is infeasible it fails too, and
            # the elastic report above still says which rows are responsible.
            if options.phase1_penalty_fallback:
                if options.verbose:
                    print("  phase 1 fell short; continuing with the penalty "
                          "path from the best point it reached")
                res.phase1_mode = 'penalty-fallback'
                use_slacks = True
            else:
                return res
        else:
            # Feasible now, so no slack is needed and the penalty is switched
            # off. tau only ever existed to buy feasibility.
            use_slacks = False
    else:
        use_slacks = not options.phase1

    # One curvature model per linearized constraint. Nothing is created when
    # there is nothing to linearize, so a structured problem is untouched.
    curvature = None
    if options.curvature and has_blackbox:
        curvature = {i: Curvature(problem.n)
                     for i, con in enumerate(problem.constraints)
                     if isinstance(con.body, Signomial)
                     and not isinstance(con.body, (Posynomial, PosynomialRatio))}

    curv_grad, curv_pred, lin_here = {}, {}, {}
    #: Which constraints are linearized -- the block the trust region and the
    #: ratio test exist for.
    lin_idx = [i for i, con in enumerate(problem.constraints)
               if isinstance(con.body, Signomial)
               and not isinstance(con.body, (Posynomial, PosynomialRatio))]
    x_prev_for_curv = x.copy()

    for k in range(options.max_iterations):
        try:
            d, s, mults, model_obj = _subproblem(
                problem, x, tau, radius, options, has_blackbox, curvature=curvature,
                use_slacks=use_slacks, cache=cache)
        except RuntimeError as exc:
            # An INFEASIBLE sub-problem is not a bad step -- it means the
            # trust region is too tight for the relaxed feasible set to be
            # reachable from here, which happens on the first pass from a
            # badly infeasible start. The trust-region response to a bad step
            # is to shrink; the response to an unreachable one is the
            # opposite. Widen and retry before giving up.
            if (has_blackbox and "infeasible" in str(exc).lower()
                    and radius < options.trust_max):
                radius = min(options.trust_max,
                             radius * options.trust_expand)
                if options.verbose:
                    print(f"  itr {k + 1:3d}  sub-problem unreachable, "
                          f"widening radius -> {radius:.3g}")
                continue
            res.status = f"sub-problem failure at iteration {k}: {exc}"
            res.x, res.objective = x, problem.objective_value(x)
            res.iterations = k
            break

        # What the model says each linearized constraint will be after this
        # step. Needed twice: by the ratio test below, to measure how well the
        # linearization actually predicted, and by the curvature update on the
        # next pass, to detect that the model was optimistic.
        if lin_idx:
            for i in lin_idx:
                try:
                    body = problem.constraints[i].body
                    gl = np.asarray(body.log_grad(x), dtype=float)
                    quad = 0.0
                    cv = curvature.get(i) if curvature else None
                    if cv is not None and cv.support:
                        sd = d[cv.support]
                        quad = 0.5 * float(sd @ cv.B @ sd)
                    lin_here[i] = math.log(max(body(x), 1e-300))
                    curv_pred[i] = lin_here[i] + float(gl @ d) + quad
                except Exception:
                    curv_pred.pop(i, None)
                    lin_here.pop(i, None)

        # --- KKT test on the ORIGINAL problem, AT THE POINT THE MULTIPLIERS
        # --- BELONG TO -------------------------------------------------------
        # The multipliers just returned are the sub-problem's at x, so the
        # residual has to be evaluated at x too. Testing at x_new instead
        # pairs gradients from one point with multipliers from another. That
        # is asymptotically harmless when the step is small, and badly wrong
        # when it is not -- SPaircraft carries degenerate variables (a
        # structural path that sizes nothing in the converged design, pinned
        # only by the 1e-30..1e30 box) which move tens of log-units per
        # iteration while contributing nothing. Pairing across that gap left
        # stationarity stuck near 0.57 no matter how converged the meaningful
        # variables were.
        stat, viol, comp = _kkt(problem, x, mults, options.x_min)
        if options.verbose:
            print(f"  itr {k + 1:3d}  f={problem.objective_value(x):.8f}  "
                  f"|d|={np.linalg.norm(d):.3e}  stat={stat:.3e}  "
                  f"viol={viol:.3e}  comp={comp:.3e}  tau={tau:.1e}")
        if (viol <= options.feasibility_tolerance
                and stat <= options.stationarity_tolerance
                and comp <= options.complementarity_tolerance):
            res.converged = True
            res.status = ("converged: KKT residual on the original problem "
                          "within tolerance")
            res.x, res.objective = x, problem.objective_value(x)
            res.iterations = k + 1
            break

        f_old = problem.objective_value(x)
        x_new = x * np.exp(d)

        # Extend the step while it stays feasible for the TRUE problem.
        #
        # The sub-problem is a conservative inner approximation, so its optimum
        # is feasible but PESSIMISTIC -- it stops at the edge of the condensed
        # set, which is strictly inside the real one. Walking further along the
        # same direction usually stays feasible and keeps reducing the
        # objective, and it costs one constraint evaluation to find out, against
        # a whole sub-problem solve to take another step.
        #
        # This is what lets the conservative form keep its guarantee and still
        # move: every candidate is CHECKED against the true constraints, so an
        # accepted iterate is feasible by verification rather than by
        # construction. Nothing is assumed.
        if options.step_expansion > 1.0 and (not has_blackbox
                                             or options.expand_past_blackbox):
            budget = max(viol, options.feasibility_tolerance)
            # With a black box present, judge the extension on the STRUCTURED
            # constraints alone. Evaluating the black box at each trial alpha
            # would spend the one resource this solver family exists to save --
            # a five-step expansion would cost five calls per iteration. The
            # structured constraints are posynomials and cost nothing.
            #
            # The black-box block is then policed where it already was, by the
            # trust-region ratio test, whose evaluation at the accepted point is
            # needed for the next linearization anyway. So the extension is free
            # in calls; what it risks is a step the ratio test then rejects,
            # throwing away the sub-problem solve that produced it. Off by
            # default for that reason.
            checker = _violation_structured if has_blackbox else _violation
            f_best = problem.objective_value(x_new)
            alpha = options.step_expansion
            while alpha <= options.step_expansion_max:
                trial = x * np.exp(alpha * d)
                if not np.all(np.isfinite(trial)) or np.any(trial <= 0):
                    break
                f_trial = problem.objective_value(trial)
                if f_trial >= f_best or checker(problem, trial) > budget:
                    break
                x_new, f_best = trial, f_trial
                alpha *= options.step_expansion

        if curvature is not None:
            _curv_step = np.log(x_new) - np.log(x)

        if has_blackbox:
            # Globalize the linearized block only: compare the true objective
            # reduction against the model's prediction, and size the region by
            # it. Structured constraints cannot be violated by the step, so
            # they play no part in the test.
            # The linearized constraints are what the region exists for, so
            # they are what it is sized by. The objective ratio measures the
            # wrong thing whenever the black box sits in a constraint rather
            # than the objective: a posynomial objective is modelled exactly,
            # so its ratio stays near 1 however badly the linearization is
            # behaving, and the region never shrinks when it should.
            #
            # Aggregated over the block rather than taken per constraint. A
            # per-constraint minimum lets an inactive constraint whose
            # prediction barely moves veto a perfectly good step -- measured,
            # that alone turned the three-black-box Hoburg case from 41
            # iterations and converged into 400 and not.
            v0 = max((abs(lin_here[i])
                      if problem.constraints[i].operator == '==' else lin_here[i]
                      for i in lin_idx if i in lin_here), default=None)
            vp = max((abs(curv_pred[i])
                      if problem.constraints[i].operator == '==' else curv_pred[i]
                      for i in lin_idx if i in curv_pred), default=None)
            ratio = None
            # Only measure a ratio when the predicted improvement is big
            # enough to mean something. On the helicopter the linearized
            # constraints sit at about 1e-8 and successive predictions differ
            # by 1e-11, so an absolute floor of 1e-10 let a ratio be formed
            # from two numbers that were both noise: it came out at -3.3 every
            # third iteration, knocked the radius back to 1.9e-6, and the run
            # spent 93% of its iterations stepping 1e-5 at a time. Scaled to
            # the tolerance the run is actually trying to meet, those
            # differences are correctly treated as "no predicted change" and
            # the model is judged on its accuracy instead.
            gate = 0.01 * options.feasibility_tolerance
            if v0 is not None and vp is not None and v0 - vp > gate:
                try:
                    va = max(_log_viol(problem.constraints[i], x_new)
                             for i in lin_idx)
                    ratio = (v0 - va) / (v0 - vp)
                except Exception:
                    ratio = None
            if ratio is None:
                # The linearized block predicts no improvement to measure,
                # which is the usual state once it is satisfied with margin.
                # Then the question is not "how much of the predicted gain was
                # realised" but "was the model right", and it was right if the
                # constraints landed where it said they would.
                #
                # Falling back to the objective ratio here is what stalled the
                # helicopter: at a small radius the predicted and actual
                # objective changes are both ~1e-5 and their ratio is noise,
                # so it lands between accept and expand. The step is taken,
                # the region never grows, and the solve crawls at a radius
                # some early rejection set -- 250 iterations at |d| = 1.8e-5,
                # with stationarity slowly getting worse.
                err = None
                if vp is not None:
                    try:
                        va = max(_log_viol(problem.constraints[i], x_new)
                                 for i in lin_idx)
                        err = abs(va - vp)
                    except Exception:
                        err = None
                if err is not None:
                    scale = max(options.feasibility_tolerance,
                                0.1 * float(np.linalg.norm(d)))
                    ratio = 1.0 if err <= scale else scale / err
                else:
                    pred = math.log(max(f_old, 1e-300)) - model_obj
                    actual = math.log(max(f_old, 1e-300)) \
                        - math.log(max(problem.objective_value(x_new), 1e-300))
                    ratio = actual / pred if abs(pred) > 1e-300 else 1.0

            if ratio < options.ratio_accept:
                radius = max(options.trust_min, radius * options.trust_shrink)
                if options.verbose:
                    print(f"  itr {k + 1:3d}  REJECT ratio={ratio:.3e} "
                          f"radius -> {radius:.3e}")
                if radius <= options.trust_min:
                    res.status = ("trust region collapsed at iteration "
                                  f"{k + 1}; the linearized constraints are "
                                  "not modelling the problem")
                    res.x, res.objective = x, f_old
                    res.iterations = k + 1
                    break
                continue
            if ratio > options.ratio_expand:
                radius = min(options.trust_max, radius * options.trust_expand)

        x = x_new
        res.history.append(x.copy())
        res.objectives.append(problem.objective_value(x))

        # Update the curvature models from the step just taken. Both sources
        # are free: the change in the log-gradient is a secant condition on the
        # log-Hessian, and the true constraint value here -- which the next
        # linearization needs anyway -- says whether the model was optimistic.
        # If it was, the curvature along that direction was underestimated, and
        # inflating fixes it. That is validation at no extra black-box cost.
        if curvature is not None:
            for i, cv in curvature.items():
                try:
                    body = problem.constraints[i].body
                    g_new = np.asarray(body.log_grad(x), dtype=float)
                    cv.observe(g_new)
                    prev = curv_grad.get(i)
                    if prev is not None and cv.support:
                        cv.update(_curv_step, g_new - prev)
                    curv_grad[i] = g_new
                    pred = curv_pred.get(i)
                    if pred is not None and cv.support:
                        actual = math.log(max(body(x), 1e-300))
                        margin = options.feasibility_tolerance
                        if actual > pred + margin:
                            cv.inflate(actual - pred, _curv_step)
                        elif actual < pred - margin:
                            # The model was pessimistic here, so the curvature
                            # it is carrying is more than the evidence
                            # supports. Ease it back rather than let it ratchet.
                            cv.relax()
                except Exception:
                    pass

        # --- escalate the slack penalty -----------------------------------
        # Two reasons to raise tau, and the second is easy to miss.
        #
        # 1. Slack is still open, so the iterate is infeasible.
        # 2. A MULTIPLIER has run into tau. Each slack costs tau per unit, so
        #    tau is an upper bound on every multiplier: a constraint whose
        #    true multiplier exceeds tau is cheaper to violate than to satisfy,
        #    and it goes soft. The iterate then looks feasible (the slack is
        #    tiny) and complementarity looks satisfied, while stationarity
        #    stalls at whatever the capped multipliers leave behind. This is
        #    the classic exact-penalty condition -- the penalty parameter has
        #    to dominate the multipliers, not merely close the slacks.
        slack = float(np.max(s)) if len(s) else 0.0
        lam_max = float(np.max(np.abs(mults))) if len(mults) else 0.0
        need = (slack > options.feasibility_tolerance
                or lam_max >= options.tau_binding * tau)
        if need and tau < options.tau_max:
            tau = min(options.tau_max,
                      tau * options.tau_factor,
                      max(tau * options.tau_factor,
                          options.tau_factor * lam_max))

    else:
        res.status = (f"did not converge within {options.max_iterations} "
                      "iterations")
        res.x, res.objective = x, problem.objective_value(x)
        res.iterations = options.max_iterations

    if res.x is None:
        res.x, res.objective = x, problem.objective_value(x)
    stat, viol, comp = _kkt(problem, res.x, mults, options.x_min)
    res.stationarity, res.max_violation, res.complementarity = stat, viol, comp
    res.multipliers = mults
    res.slacks_active = viol > options.feasibility_tolerance
    return res
