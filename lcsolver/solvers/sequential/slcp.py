#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Sequential Log-Convex Programming (SLCP) with an IPOPT sub-problem solver.

SLCP solves a general nonlinear program by exploiting the fact that engineering
design models are usually *mostly* GP-compatible. Constraints are split into two
groups:

* posynomials :math:`p(x) \\le 1` and monomials :math:`m(x) = 1`, which become
  convex exactly under the log transform and are imposed **directly**; and
* everything else, :math:`g(x) \\le 1` and :math:`h(x) = 1`, which is linearized
  in log space as SQP would.

The sub-problem is therefore log-convex rather than quadratic, which is what
distinguishes SLCP from LSQP. Retaining the posynomials exactly prevents the
sub-problem from stepping outside a constraint that a linear model would have
badly under-estimated -- the failure mode that costs LSQP iterations.

Reference
---------
Karcher, C. and Haimes, R., "A Method of Sequential Log-Convex Programming for
Engineering Design", Optimization and Engineering (2022).
doi:10.1007/s11081-022-09750-3

Algorithm 1 of that paper is implemented here, with its relaxed sub-problem
(Equation 15):

.. math::

    \\begin{aligned}
    \\underset{d}{\\text{minimize}} \\quad
      & \\log f(x_k)
      + \\tfrac{1}{f(x_k)}\\left(x_k \\odot \\nabla f(x_k)\\right)^T d
      + \\tfrac12 d^T \\nabla^2 \\mathcal{L}_R(y_k) d
      + K \\textstyle\\sum_i \\sigma_i^2 \\\\
    \\text{subject to} \\quad
      & \\log\\left(\\textstyle\\sum_j \\exp(P_j(d + \\log x_k) + q_j)\\right)
        \\le \\sigma_i \\\\
      & A_m (d + \\log x_k) + b_m \\le \\sigma_i \\\\
      & \\log g(x_k)
        + \\tfrac{1}{g(x_k)}\\left(x_k \\odot \\nabla g(x_k)\\right)^T d
        \\le \\sigma_i \\\\
      & \\log h(x_k)
        + \\tfrac{1}{h(x_k)}\\left(x_k \\odot \\nabla h(x_k)\\right)^T d
        = \\sigma_i
    \\end{aligned}

Why IPOPT
---------
The published implementation solves this sub-problem with cvxopt, which has no
native way to express "log-sum-exp constraints plus a quadratic penalised
objective". It gets there by calling cvxopt's geometric-programming routine and
then overwriting ``f[0]``, ``Df[0]`` and the Hessian block in the callback to
substitute the quadratic objective. That works, but the objective is smuggled
past the solver's own model.

Written for IPOPT the sub-problem is just declared: the log-sum-exp constraints,
the quadratic objective and the penalty term are all ordinary Pyomo expressions.
Convexity is preserved, so the sub-problem still has a unique global solution;
only the machinery is simpler.

The log-space gradient used throughout is (paper Equation 11)

.. math::  \\frac{\\partial \\log f(e^y)}{\\partial y_i}
           = \\frac{x_i}{f(x)}\\frac{\\partial f}{\\partial x_i}
"""

import math

import numpy as np
import pyomo.environ as pyo

from lcsolver.core.errors import SolverUnavailable


# ---------------------------------------------------------------------------
# Problem description
# ---------------------------------------------------------------------------
class Posynomial:
    """A posynomial :math:`\\sum_k c_k \\prod_j x_j^{a_{kj}}` with :math:`c_k > 0`.

    Stored as a list of ``(coefficient, exponent_vector)`` pairs. A single term is
    a monomial.
    """

    __slots__ = ('terms', 'n')

    def __init__(self, terms, n):
        self.terms = [(float(c), np.asarray(a, dtype=float)) for c, a in terms]
        self.n = int(n)
        for c, a in self.terms:
            if c <= 0:
                raise ValueError(f'posynomial coefficients must be positive, got {c}')
            if len(a) != self.n:
                raise ValueError(f'exponent vector has length {len(a)}, expected {self.n}')

    @property
    def is_monomial(self):
        return len(self.terms) == 1

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return sum(c * np.prod(x ** a) for c, a in self.terms)

    def grad(self, x):
        """Gradient in the natural variables."""
        x = np.asarray(x, dtype=float)
        g = np.zeros(self.n)
        for c, a in self.terms:
            t = c * np.prod(x ** a)
            g += t * a / x
        return g

    def log_grad(self, x):
        """d log f(e^y) / dy, via Equation 11. Exact and cheap for a posynomial."""
        x = np.asarray(x, dtype=float)
        f = self(x)
        return x * self.grad(x) / f


class Signomial:
    """A general positive function supplied as a value/gradient callback.

    ``fn(x)`` must return ``(value, gradient)`` in the natural variables, with the
    value strictly positive. This is the hook for a black-box analysis code: the
    algorithm only ever needs :math:`f` and :math:`\\nabla f` at the current
    iterate.
    """

    __slots__ = ('fn', 'n')

    def __init__(self, fn, n):
        self.fn = fn
        self.n = int(n)

    def __call__(self, x):
        return float(self.fn(np.asarray(x, dtype=float))[0])

    def grad(self, x):
        return np.asarray(self.fn(np.asarray(x, dtype=float))[1], dtype=float)

    def log_grad(self, x):
        x = np.asarray(x, dtype=float)
        v, g = self.fn(x)
        if v <= 0:
            raise ValueError(
                'SLCP requires every constraint function to stay strictly '
                f'positive; got {v}. See the Limitations section of the paper.')
        return x * np.asarray(g, dtype=float) / v



class CachedSignomial(Signomial):
    """A :class:`Signomial` that calls its function once per distinct point.

    For a black box costing seconds this is irrelevant. For one costing hours
    -- a CFD run, an airfoil solve, an FEA -- it is the difference between a
    tractable optimization and an untenable one, because the algorithms above
    reach the value and the gradient through separate entry points and revisit
    the same iterate several times per iteration:

    * the sub-problem needs ``f(x_k)`` and ``grad f(x_k)`` to linearize;
    * the KKT test needs both again at the accepted point;
    * the feasibility check needs the value a third time.

    Every one of those is the same ``x``, and ``fn`` returns value and gradient
    together, so all of it is one evaluation. Measured on a small mixed
    problem this takes the count from 4.1 evaluations per iteration to 1.0.

    The cache is keyed on the exact float pattern of ``x``, so it only ever
    returns a value the function itself produced -- no interpolation, no
    tolerance. ``maxsize`` bounds it; the default keeps every point, which is
    what you want when each one cost hours.

    ``evaluations`` counts genuine calls -- the number to report, and the one
    to budget against.
    """

    __slots__ = ('_cache', '_order', '_maxsize', 'evaluations')

    def __init__(self, fn, n, maxsize=None):
        super().__init__(fn, n)
        self._cache = {}
        self._order = []
        self._maxsize = maxsize
        self.evaluations = 0

    def _eval(self, x):
        x = np.asarray(x, dtype=float)
        key = x.tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        v, g = self.fn(x)
        out = (float(v), np.asarray(g, dtype=float))
        self._cache[key] = out
        self._order.append(key)
        self.evaluations += 1
        if self._maxsize is not None and len(self._order) > self._maxsize:
            del self._cache[self._order.pop(0)]
        return out

    def __call__(self, x):
        return self._eval(x)[0]

    def grad(self, x):
        return self._eval(x)[1]

    def log_grad(self, x):
        x = np.asarray(x, dtype=float)
        v, g = self._eval(x)
        if v <= 0:
            raise ValueError(
                'SLCP requires every constraint function to stay strictly '
                f'positive; got {v}. See the Limitations section of the paper.')
        return x * g / v


class GreyboxSignomial(CachedSignomial):
    """A grey-box equality body, ``bb(inputs) / x[out_index] == 1``.

    The tag ``out_index`` records the DEDICATED OUTPUT COLUMN the row is
    solved for, which gives the equality-restore machinery a closed form:
    the row is restored EXACTLY by ``x[out_index] *= body(x)`` -- one box
    evaluation, no Newton -- because the output variable appears nowhere
    inside the box.  sia's composite restore uses this to keep grey-box
    equalities on the manifold at the same points it restores the
    structured pins (they used to be skipped entirely on black-box
    problems, which let tangential drift accumulate in exactly the rows
    the trust machinery was told to trust; see restore_composite_bb).
    """

    __slots__ = ('out_index',)

    def __init__(self, fn, n, out_index, maxsize=None):
        super().__init__(fn, n, maxsize=maxsize)
        self.out_index = int(out_index)


class PosynomialRatio:
    """``p(x) / q(x)`` with p and q both POSYNOMIALS — the signomial-program form.

    A signomial constraint that can be written  p(x) <= q(x)  (equivalently
    ``p/q <= 1``) carries structure that SLCP's default treatment throws away:
    it linearizes the whole body into a single monomial, when in fact ``p`` is
    log-convex and can be imposed EXACTLY.

    The classical SP treatment keeps p exact and condenses only q, using the
    arithmetic-geometric-mean inequality at the current iterate:

        q_hat(x) = prod_i ( u_i(x) / w_i )^{w_i},   w_i = u_i(x_k) / q(x_k)

    ``q_hat`` is a MONOMIAL, satisfies ``q_hat(x) <= q(x)`` everywhere, and is
    tight at x_k. So imposing ``p(x) <= q_hat(x)`` is CONSERVATIVE: any point
    it admits satisfies the true constraint. The sub-problem then contains a
    log-sum-exp (p, exact) bounded by an affine function (log q_hat) — still
    convex, but with only the concave-in-log part approximated instead of all
    of it.

    This is the same condensation a signomial-program solver uses, made
    available inside SLCP so the two treatments can be compared on identical
    problems.
    """

    __slots__ = ('p', 'q', 'n')

    def __init__(self, p, q, n):
        if not isinstance(p, Posynomial) or not isinstance(q, Posynomial):
            raise TypeError('PosynomialRatio needs two Posynomial parts')
        if p.n != int(n) or q.n != int(n):
            raise ValueError('p and q must have the same dimension as the problem')
        self.p, self.q, self.n = p, q, int(n)

    def __call__(self, x):
        return self.p(x) / self.q(x)

    def grad(self, x):
        p, q = self.p(x), self.q(x)
        return self.p.grad(x)/q - p*self.q.grad(x)/q**2

    def log_grad(self, x):
        """d log(p/q) / d log x = log_grad(p) - log_grad(q)."""
        return self.p.log_grad(x) - self.q.log_grad(x)

    def condensed_q(self, x_k):
        """AGM monomial under-estimator of q at x_k, as ``(coeff, exponents)``."""
        return condense(self.q, x_k, self.n)

    def condensed_p(self, x_k):
        """AGM monomial under-estimator of the NUMERATOR.

        Condensing ``p`` as well turns ``p/q <= 1`` into a monomial inequality,
        linear in log space. It is what PCCP does for an equality constraint,
        and it is **not** conservative: since ``p_hat <= p``, the condensed
        constraint is EASIER than the true one, so the sub-problem's feasible
        set is no longer a subset of the true one and an iterate can leave it.

        What survives is tangency -- ``p_hat`` matches ``p`` in value and
        gradient at ``x_k`` -- which is what licenses a KKT certificate built
        from the sub-problem's duals. So this trades the feasible-iterate
        guarantee for a larger step while keeping the termination test honest.
        """
        return condense(self.p, x_k, self.n)


class CondensedEquality:
    """A signomial equality ``p/q == 1``, condensed on BOTH sides.

    The obvious representation is a pair of one-sided ratios, ``p/q <= 1`` and
    ``q/p <= 1``, each with its denominator condensed. That pair is correct but
    behaves badly in two ways at once, and both are severe:

    * **The step collapses.** At the iterate both halves are active and tangent
      with opposite gradients, so a step ``d`` must satisfy
      ``½dᵀH₂d <= grad f · d <= -½dᵀH₁d``. Both Hessians are positive
      semidefinite (a posynomial is log-convex), so this has a solution only
      where ``dᵀ(H₁+H₂)d <= 0`` -- the null space of the sum. The sub-problem is
      restricted to a lower-dimensional subspace wherever such an equality is
      active.
    * **The multipliers become meaningless.** The Lagrangian sees only
      ``(lam_A - lam_B) grad g_A``, so the pair is dual-degenerate: the same
      constant added to both changes nothing. A solver may return any large
      pair with the right difference, and does -- magnitudes of several
      thousand were measured on SPaircraft, whose difference is then noise at
      the solver's dual tolerance. The KKT residual inherits that noise and
      never falls below it.

    Condensing both sides instead gives ``p_hat/q_hat == 1``, a MONOMIAL
    equality and so affine in log space. One signed, well-conditioned
    multiplier, and a full ``(n-1)``-dimensional hyperplane tangent to the true
    feasible manifold rather than a null space.

    Nothing conservative is given up. An inner approximation needs an interior,
    and an equality has none; the inner-approximation argument only ever applied
    to the inequalities, which keep it untouched. What is given up is that an
    iterate can now leave the true feasible set, exactly as it can under PCCP --
    but tangency survives, so the multipliers still certify the original
    problem.
    """

    __slots__ = ('p', 'q', 'n')

    def __init__(self, p, q, n):
        self.p, self.q, self.n = p, q, n

    def __call__(self, x):
        return self.p(x) / self.q(x)

    def log_grad(self, x):
        """The TRUE gradient, for the KKT test -- not the condensed one."""
        return self.p.log_grad(x) - self.q.log_grad(x)

    def grad(self, x):
        p, q = self.p(x), self.q(x)
        return self.p.grad(x)/q - p*self.q.grad(x)/q**2

    @property
    def is_monomial(self):
        return self.p.is_monomial and self.q.is_monomial

    def condensed(self, x_k):
        """``(coeff, exponents)`` of the monomial ``p_hat/q_hat``."""
        cp, ap = condense(self.p, x_k, self.n)
        cq, aq = condense(self.q, x_k, self.n)
        return cp / cq, ap - aq


def condense(posy, x_k, n=None):
    """AGM monomial under-estimator of a posynomial at ``x_k``.

    ``q_hat(x) = prod_i (u_i(x)/w_i)**w_i`` with ``w_i = u_i(x_k)/q(x_k)``.
    By the arithmetic-geometric-mean inequality ``q_hat <= q`` everywhere, with
    equality **and matching gradient** at ``x_k``. Returned as
    ``(coeff, exponents)``.
    """
    x_k = np.asarray(x_k, dtype=float)
    n = n if n is not None else posy.n
    qv = posy(x_k)
    coeff, expo = 1.0, np.zeros(n)
    for c, a in posy.terms:
        w = c * np.prod(x_k ** a) / qv              # AGM weight, sums to 1
        if w <= 0:
            continue
        coeff *= (c / w) ** w
        expo = expo + w * a
    return coeff, expo


class Constraint:
    """One constraint in the standard form ``body <= 1`` or ``body == 1``.

    ``body`` is a :class:`Posynomial` or a :class:`Signomial`. Which of the two it
    is determines whether SLCP imposes it exactly or linearizes it.
    """

    __slots__ = ('body', 'operator')

    def __init__(self, body, operator='<='):
        if operator not in ('<=', '=='):
            raise ValueError("operator must be '<=' or '=='")
        if operator == '==' and isinstance(body, PosynomialRatio):
            raise ValueError(
                'a PosynomialRatio equality is not supported: the AGM '
                'condensation is one-sided (conservative for <=)')
        if operator == '==' and isinstance(body, Posynomial) and not body.is_monomial:
            raise ValueError(
                'a multi-term posynomial equality is not GP-compatible; supply it '
                'as a Signomial so it is linearized instead')
        self.body = body
        self.operator = operator

    @property
    def exact_in_logspace(self):
        """True when the log transform makes this constraint convex as written.

        Posynomial ``<= 1`` becomes log-sum-exp ``<= 0`` (convex); monomial ``== 1``
        becomes an affine equality. Both can be imposed directly. Everything else
        must be linearized.
        """
        return isinstance(self.body, Posynomial)

    @property
    def is_sp_form(self):
        """True for a PosynomialRatio: p exact, q condensed (see that class)."""
        return isinstance(self.body, PosynomialRatio)


class Problem:
    """A signomial program in the standard form of paper Equation 12."""

    def __init__(self, n, objective, constraints, names=None, bounds=None):
        self.n = int(n)
        self.objective = objective
        self.constraints = list(constraints)
        self.names = list(names) if names else [f'x{i + 1}' for i in range(n)]
        # Optional per-variable ``(lower, upper)``, either bound possibly None.
        # These belong on the sub-problem variable, not in ``constraints``: a
        # bound costs a solver nothing, while the same statement written as a
        # row is one more log-sum-exp to build and differentiate every
        # iteration. On SPaircraft that is 2346 rows that need not exist.
        self.bounds = list(bounds) if bounds is not None else None

    def objective_value(self, x):
        return self.objective(x)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
class Options:
    """Algorithm parameters. Defaults match the published cvxopt implementation."""

    def __init__(self, **kw):
        self.max_iterations = 500
        self.penalty_constant = 1e15      # K on the sigma relaxation
        self.lagrangian_gradient_tolerance = 1e-6
        self.step_magnitude_tolerance = 1e-4
        self.feasibility_tolerance = 1e-6    # max |constraint violation| allowed
                                             # before a run may be called converged
        self.max_step_ratio = None           # per-iteration TRUST REGION, stated as a
                                             # ratio to the CURRENT outer iterate:
                                             #   1-r <= x_sub[i]/x_outer[i] <= 1+r
                                             # Because the sub-problem works in LOG
                                             # space (x_sub = x_outer * exp(d)), this
                                             # is simply d in [log(1-r), log(1+r)] -- a
                                             # CONSTANT bound that is automatically
                                             # relative to the current iterate every
                                             # iteration. Scalar, or length-n array
                                             # (np.inf to leave a variable unbounded).
                                             # May also be a CALLABLE r(x_k) returning
                                             # such an array, for bounds that depend on
                                             # the current iterate (e.g. a fixed ratio
                                             # on a SHIFTED variable q = x - c).
        self.max_log_step = None             # per-iteration TRUST REGION on the step:
                                             # |d_j| <= max_log_step[j] in log space,
                                             # i.e. |dx_j / x_j| <~ max_log_step[j].
                                             # None (default) = unbounded, as before.
                                             # Scalar or length-n array. This is step
                                             # control for the LINEARISATION -- needed
                                             # even with exact gradients, and distinct
                                             # from surrogate-model management (TRMM),
                                             # which exact gradients do make unnecessary.
        self.penalty_escalation = 10.0       # factor by which the merit penalty on
                                             # a VIOLATED constraint is raised when
                                             # the step collapses while infeasible
        self.max_penalty_escalations = 6     # give up after this many escalations
        self.eta = 1e-4                   # Armijo parameter, in (0, 0.5)
        self.rho = 0.8                    # backtracking factor, in (0, 1)
        self.mu_margin = 1.2              # merit-multiplier margin, > 1
        self.max_step_size_tries = 30
        self.watchdog_iterations = 5      # consecutive non-monotone steps allowed
        self.exact_objective = False      # Impose a POSYNOMIAL objective exactly
                                          # (log-sum-exp) instead of linearizing
                                          # it. SLCP already keeps posynomial
                                          # CONSTRAINTS exact; the objective is
                                          # linearized with a BFGS quadratic, so
                                          # on a problem that is convex end to
                                          # end the method still marches like a
                                          # quasi-Newton scheme -- 72 iterations
                                          # on the wind turbine, which the GP
                                          # path solves in one. With this on,
                                          # and with every constraint also exact,
                                          # the sub-problem IS the original
                                          # problem and the BFGS term is dropped.
        self.cache_subproblem = False     # Build the sub-problem's Pyomo model
                                          # ONCE and re-point it each iteration
                                          # through mutable Params, instead of
                                          # rebuilding every constraint
                                          # symbolically. Only the shapes the
                                          # structure bridge produces are
                                          # cacheable (exact posynomials and
                                          # posynomial ratios, method='slcp');
                                          # anything else silently keeps the
                                          # rebuild path. Off by default.
        self.hessian_gamma = 1.0          # Weight of the background curvature
                                          # gamma*I in the limited-memory B.
                                          # On problems where the curvature
                                          # condition s.z > 0 keeps failing --
                                          # which it does whenever almost every
                                          # constraint is already exact, so the
                                          # Reduced Lagrangian has little left
                                          # in it -- the damped update
                                          # degenerates and B stays at gamma*I
                                          # forever. The quadratic is then
                                          # purely a proximal term, and gamma
                                          # is its weight: it sets the step
                                          # length directly. Lower it to take
                                          # longer steps.
        self.hessian_scaling = False      # Scale the initial limited-memory
                                          # Hessian by the Shanno-Phua ratio
                                          # y.y / s.y measured on the first
                                          # update, instead of leaving it at
                                          # the identity. The quadratic term is
                                          # the sub-problem's proximal term, so
                                          # its scale sets the step length; an
                                          # identity background on a problem
                                          # whose curvature is orders of
                                          # magnitude away throttles every
                                          # step. Only affects the
                                          # limited-memory path.
        self.hessian_memory = None        # None (default) keeps the DENSE n-by-n
                                          # BFGS matrix and the n^2-term quadratic
                                          # expression. An integer m switches to a
                                          # limited-memory representation keeping
                                          # the last m rank-two updates, which makes
                                          # the sub-problem quadratic O(n*m) terms
                                          # instead of O(n^2). Same algorithm, same
                                          # damped update; only the curvature that
                                          # has scrolled out of the window is lost.
        self.x_min = 1e-9                 # the epsilon floor of Equation 16
        self.tee = False
        self.verbose = False
        self.ipopt_options = {'print_level': 0, 'sb': 'yes'}
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError(f'unknown SLCP option {k!r}')
            setattr(self, k, v)


class Result:
    """Outcome of a solve, including the full iterate history for plotting."""

    def __init__(self):
        self.x = None
        self.objective = None
        self.iterations = 0
        self.converged = False
        self.status = 'not run'
        self.history = []          # iterates x_k, including x_0
        self.step_norms = []
        self.grad_lagrangian = []
        self.subproblem_solves = 0
        self.function_evaluations = 0
        self.max_violation = None   # max |constraint violation| at the returned
                                    # point, in `body <= 1` form. Set on EVERY
                                    # exit path, converged or not, so a caller
                                    # can always tell whether x is usable.

    def __repr__(self):
        viol = ('' if self.max_violation is None
                else f' max_violation={self.max_violation:.3e}')
        return (f'<Result {self.status!r} iterations={self.iterations} '
                f'objective={self.objective!r}{viol}>')


def _fully_log_convex(problem):
    """True when objective and every constraint are exact in log space.

    Then the sub-problem, built with the objective imposed exactly and no
    BFGS term, is the original problem: one solve is enough.
    """
    return (isinstance(problem.objective, Posynomial)
            and all(c.exact_in_logspace for c in problem.constraints))


def _apply_variable_bounds(m, problem, log_xk):
    """Put the model's variable bounds on the sub-problem step.

    The sub-problem works in log space about x_k -- x = x_k * exp(d) -- so
    `lo <= x <= hi` is `log(lo/x_k) <= d <= log(hi/x_k)`, exact and free.

    This is not optional once presolve is in play. `fold_singleton_rows` turns
    a row like `x >= 1` into a BOUND, so a sub-problem that reads only rows no
    longer sees that constraint at all -- it solves an unbounded relaxation and
    drives the variable to the positivity floor. Measured before this existed:
    `min x*y s.t. x >= 1, y >= 2` returned 1e-18 instead of 2, and reported
    itself converged on the step magnitude.
    """
    import math

    if problem.bounds is None:
        return
    for j, pair in enumerate(problem.bounds[:problem.n]):
        lo, hi = pair or (None, None)
        if lo is not None and lo > 0:
            cur = m.d[j].lb
            v = math.log(lo) - log_xk[j]
            m.d[j].setlb(v if cur is None else max(cur, v))
        if hi is not None and hi > 0:
            cur = m.d[j].ub
            v = math.log(hi) - log_xk[j]
            m.d[j].setub(v if cur is None else min(cur, v))
    seat_step_in_bounds(m)


def seat_step_in_bounds(m):
    """Start the step inside its own box.

    ``d = 0`` -- take no step -- is the right initial guess and is very nearly
    always feasible. It is not feasible when the iterate already sits ON a
    bound: the bound in log space is then ``log(lo) - log(x_k)``, which lands
    a rounding error *above* zero, so the initial value is outside its own
    bounds and Pyomo says so (W1002) once per such variable per sub-problem.
    On a converged solve of a model with active bounds that is a wall of
    warnings with nothing wrong behind it, and it trains the reader to ignore
    a class of warning worth reading.

    Seat the value in the box rather than silencing the logger, which would
    hide the genuine ones too. Call after every bound application: bounds are
    tightened in several passes and only the last one knows the final box.
    """
    for j in m.d:
        v, lo, hi = 0.0, m.d[j].lb, m.d[j].ub
        if lo is not None and v < lo:
            v = float(lo)
        if hi is not None and v > hi:
            v = float(hi)
        if v != 0.0:
            m.d[j].set_value(v)


class SubproblemCache:
    """Build the sub-problem's Pyomo model once and re-point it each iteration.

    The uncached path rebuilds every constraint symbolically on every
    iteration. For SPaircraft that is 6077 log-sum-exp expressions over 1173
    variables, constructed from scratch ~50 times, and it dominates the run --
    far more than the Hessian ever did.

    Almost none of that structure actually changes. Each exact term is

        exp( log c_k + a_k . (d + log x_k) )
            = exp( [log c_k + a_k . log x_k]  +  [a_k . d] )

    where ``a_k . d`` is FIXED and only the bracketed constant moves with the
    iterate. So the projections are built once as Pyomo expressions and the
    constants become mutable Params.

    The condensed denominator of a PosynomialRatio looks like it breaks this,
    since its AGM exponent vector ``aq`` is rebuilt every iteration -- but
    ``aq = sum_i w_i a_i``, so

        aq . d = sum_i w_i (a_i . d)

    reuses the same fixed projections and needs only one mutable weight per
    term. The objective's log-space gradient has the identical form. What is
    left varying is a handful of scalars per constraint rather than a
    full-length coefficient vector.

    Only the shapes this module's own bridge produces are cached: exact
    monomials, exact posynomials and posynomial ratios, under ``method='slcp'``.
    A Signomial body, or the ``lsqp``/``sqp`` methods, need a fresh gradient
    everywhere and fall back to rebuilding.
    """

    def __init__(self, problem, options):
        self.problem = problem
        self.options = options
        self.n = problem.n
        self.model = None
        self.usable = self._is_cacheable()

    def _is_cacheable(self):
        for con in self.problem.constraints:
            if not (isinstance(con.body, Posynomial)
                    or isinstance(con.body, PosynomialRatio)):
                return False
        return isinstance(self.problem.objective, Posynomial)

    def build(self):
        n = self.n
        cons = self.problem.constraints
        m = pyo.ConcreteModel()
        m.J = pyo.RangeSet(0, n - 1)
        m.d = pyo.Var(m.J, initialize=0.0)
        m.S = pyo.RangeSet(0, len(cons) - 1) if cons else pyo.RangeSet(0, -1)
        m.sigma = pyo.Var(m.S, domain=pyo.NonNegativeReals, initialize=0.0)
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        def projection(a):
            """The fixed part a . d, built once."""
            nz = np.nonzero(a)[0]
            return sum(float(a[j]) * m.d[j] for j in nz) if nz.size else 0.0

        # --- objective ------------------------------------------------------
        obj = self.problem.objective
        self.obj_proj = [projection(a) for _, a in obj.terms]
        self.exact_obj = getattr(self.options, 'exact_objective', False)
        self.drop_quad = self.exact_obj and _fully_log_convex(self.problem)
        if self.exact_obj:
            # Exact: log sum exp(b_k + a_k.d), with b_k the same mutable
            # constant the constraints use.
            m.obj_b = pyo.Param(range(len(obj.terms)), mutable=True,
                                initialize=0.0, within=pyo.Reals)
            self.obj_lin = pyo.log(sum(
                pyo.exp(m.obj_b[k] + self.obj_proj[k])
                for k in range(len(obj.terms))))
        else:
            m.obj_w = pyo.Param(range(len(obj.terms)), mutable=True,
                                initialize=0.0, within=pyo.Reals)
            self.obj_lin = sum(m.obj_w[k] * self.obj_proj[k]
                               for k in range(len(obj.terms)))

        # --- constraints ----------------------------------------------------
        m.cons = pyo.ConstraintList()
        self.const_params = []      # per constraint: list of mutable Params
        self.weight_params = []     # per SP constraint: AGM weights
        self.rhs_params = []        # per SP constraint: scalar rhs constant
        m.pblocks = pyo.Block()

        pcount = 0
        for i, con in enumerate(cons):
            body, op = con.body, con.operator
            if isinstance(body, Posynomial):
                terms = body.terms
                names = []
                for k, (_, a) in enumerate(terms):
                    pname = f'b_{pcount}'
                    setattr(m.pblocks, pname,
                            pyo.Param(mutable=True, initialize=0.0,
                                      within=pyo.Reals))
                    names.append(getattr(m.pblocks, pname))
                    pcount += 1
                self.const_params.append(names)
                self.weight_params.append(None)
                self.rhs_params.append(None)
                if body.is_monomial:
                    expr = names[0] + projection(terms[0][1])
                    m.cons.add(expr == m.sigma[i] if op == '=='
                               else expr <= m.sigma[i])
                else:
                    expr = sum(pyo.exp(names[k] + projection(a))
                               for k, (_, a) in enumerate(terms))
                    m.cons.add(pyo.log(expr) <= m.sigma[i])
            else:
                # PosynomialRatio: p exact, q condensed by AGM.
                pterms, qterms = body.p.terms, body.q.terms
                names = []
                for k, (_, a) in enumerate(pterms):
                    pname = f'b_{pcount}'
                    setattr(m.pblocks, pname,
                            pyo.Param(mutable=True, initialize=0.0,
                                      within=pyo.Reals))
                    names.append(getattr(m.pblocks, pname))
                    pcount += 1
                wname = f'w_{i}'
                setattr(m.pblocks, wname,
                        pyo.Param(range(len(qterms)), mutable=True,
                                  initialize=0.0, within=pyo.Reals))
                rname = f'r_{i}'
                setattr(m.pblocks, rname,
                        pyo.Param(mutable=True, initialize=0.0,
                                  within=pyo.Reals))
                wpar = getattr(m.pblocks, wname)
                rpar = getattr(m.pblocks, rname)
                self.const_params.append(names)
                self.weight_params.append(wpar)
                self.rhs_params.append(rpar)
                qproj = [projection(a) for _, a in qterms]
                lhs = sum(pyo.exp(names[k] + projection(a))
                          for k, (_, a) in enumerate(pterms))
                rhs = rpar + sum(wpar[k] * qproj[k] for k in range(len(qterms)))
                m.cons.add(pyo.log(lhs) - rhs <= m.sigma[i])

        self.model = m
        return self

    def update(self, x_k, B):
        """Re-point the cached model at a new iterate."""
        m = self.model
        n = self.n
        log_xk = np.log(x_k)
        cons = self.problem.constraints

        obj = self.problem.objective
        f_k = obj(x_k)
        if self.exact_obj:
            for k, (c, a) in enumerate(obj.terms):
                m.obj_b[k] = float(math.log(c) + a @ log_xk)
        else:
            for k, (c, a) in enumerate(obj.terms):
                # log-space gradient weight of this term: c*prod(x^a)/f
                m.obj_w[k] = float(c * np.prod(x_k ** a) / f_k)

        for i, con in enumerate(cons):
            body = con.body
            terms = body.terms if isinstance(body, Posynomial) else body.p.terms
            for k, (c, a) in enumerate(terms):
                self.const_params[i][k].value = float(math.log(c) + a @ log_xk)
            if self.weight_params[i] is not None:
                qterms = body.q.terms
                qv = body.q(x_k)
                aq = np.zeros(n)
                const = 0.0
                for k, (c, a) in enumerate(qterms):
                    w = c * np.prod(x_k ** a) / qv
                    self.weight_params[i][k] = float(w)
                    if w > 0:
                        const += w * math.log(c / w)
                        aq = aq + w * a
                self.rhs_params[i].value = float(const + aq @ log_xk)

        # Positivity floor and trust region become variable BOUNDS, which cost
        # nothing to change; in the uncached path they are extra constraints.
        floor = math.log(self.options.x_min)
        lo = floor - log_xk
        hi = np.full(n, np.inf)
        # The model's own bounds, which presolve may have folded rows into.
        # Without these the cached sub-problem solves an unbounded relaxation
        # exactly as the rebuild path did -- see _apply_variable_bounds.
        if self.problem.bounds is not None:
            for j, pair in enumerate(self.problem.bounds[:n]):
                blo, bhi = pair or (None, None)
                if blo is not None and blo > 0:
                    lo[j] = max(lo[j], math.log(blo) - log_xk[j])
                if bhi is not None and bhi > 0:
                    hi[j] = min(hi[j], math.log(bhi) - log_xk[j])
        mls = self.options.max_log_step
        if mls is not None:
            mls = np.asarray(mls, dtype=float)
            if mls.ndim == 0:
                mls = np.full(n, float(mls))
            lo = np.maximum(lo, -mls)
            hi = np.minimum(hi, mls)
        _r = getattr(self.options, 'max_step_ratio', None)
        if _r is not None:
            r = np.asarray(_r(x_k) if callable(_r) else _r, dtype=float)
            if r.ndim == 0:
                r = np.full(n, float(r))
            ok = np.isfinite(r) & (r > 0)
            if np.any(r[ok] >= 1.0):
                raise ValueError('max_step_ratio must be < 1')
            lo = np.where(ok, np.maximum(lo, np.log(1.0 - r)), lo)
            hi = np.where(ok, np.minimum(hi, np.log(1.0 + r)), hi)
        for j in range(n):
            m.d[j].setlb(float(lo[j]))
            m.d[j].setub(None if not np.isfinite(hi[j]) else float(hi[j]))
            m.d[j].set_value(0.0)

        # The objective is the one part worth rebuilding: it carries the
        # Hessian term, which is O(n*memory) under a limited-memory B and
        # O(n^2) under a dense one.
        if m.component('obj') is not None:
            m.del_component(m.obj)
        quad = 0.0 if self.drop_quad else _quadratic_expression(B, m.d, n)
        penalty = self.options.penalty_constant * sum(
            m.sigma[i] ** 2 for i in range(len(cons)))
        # Under the exact form obj_lin already carries log f; under the
        # linearized one it is only the gradient term and needs the constant.
        base = self.obj_lin if self.exact_obj else math.log(f_k) + self.obj_lin
        m.obj = pyo.Objective(expr=base + quad + penalty, sense=pyo.minimize)
        return self


def _quadratic_expression(B, d, n):
    """0.5 d^T B d, in whichever representation B is carried."""
    if isinstance(B, LimitedMemoryB):
        gamma, pairs = B.quad_terms()
        quad = 0.5 * gamma * sum(d[j] ** 2 for j in range(n))
        for sign, v in pairs:
            nz = np.nonzero(v)[0]
            if nz.size:
                proj = sum(float(v[j]) * d[j] for j in nz)
                quad = quad + 0.5 * sign * proj ** 2
        return quad
    return 0.5 * sum(B[i][j] * d[i] * d[j] for i in range(n) for j in range(n))


# ---------------------------------------------------------------------------
# Sub-problem construction
# ---------------------------------------------------------------------------
def _solve_subproblem(problem, x_k, B, options, method, cache=None):
    """Build and solve one sub-problem; return the step ``d`` and the multipliers.

    ``method`` selects how constraints enter:

    ``'slcp'``
        Posynomials and monomials imposed exactly (log-sum-exp / affine);
        everything else linearized in log space. Paper Equation 15.
    ``'lsqp'``
        Every constraint linearized in log space. The sub-problem is then a QP,
        which is exactly what the paper says SLCP degenerates to when no
        posynomial constraints are present.
    ``'sqp'``
        Every constraint linearized in the natural variables, no log transform.
    """
    n = problem.n
    cons = problem.constraints
    exact_obj = (getattr(options, 'exact_objective', False)
                 and method == 'slcp'
                 and isinstance(problem.objective, Posynomial))
    drop_quad = exact_obj and _fully_log_convex(problem)

    if cache is not None and cache.usable and method == 'slcp':
        cache.update(x_k, B)
        return _solve_pyomo_subproblem(cache.model, n, len(cons), options)

    m = pyo.ConcreteModel()
    m.J = pyo.RangeSet(0, n - 1)
    m.d = pyo.Var(m.J, initialize=0.0)

    log_xk = np.log(x_k)
    _apply_variable_bounds(m, problem, log_xk)
    f_k = problem.objective_value(x_k)

    # --- objective ---------------------------------------------------------
    if method == 'sqp':
        # Natural space: linear model of f plus the BFGS quadratic.
        gf = problem.objective.grad(x_k)
        lin = f_k + sum(gf[j] * m.d[j] for j in range(n))
    elif exact_obj:
        # The objective is a posynomial, so log f is convex in log space and
        # can be imposed exactly -- the same argument that keeps posynomial
        # CONSTRAINTS exact. Linearizing it is what makes a fully log-convex
        # problem take a quasi-Newton march instead of one solve.
        lin = pyo.log(sum(
            pyo.exp(math.log(c) + sum(a[j] * (m.d[j] + log_xk[j])
                                      for j in range(n)))
            for c, a in problem.objective.terms))
    else:
        # Log space, Equation 11: (x . grad f) / f is the log-space gradient.
        gf = problem.objective.log_grad(x_k)
        lin = math.log(f_k) + sum(gf[j] * m.d[j] for j in range(n))

    # With objective and constraints both exact the sub-problem already IS the
    # original problem; a curvature term would only bias the step.
    quad = 0.0 if drop_quad else _quadratic_expression(B, m.d, n)

    # --- constraints -------------------------------------------------------
    # Every constraint gets its own relaxation variable sigma >= 0, penalised in
    # the objective. Without this the sub-problem can be infeasible even when the
    # true problem is not -- the standard SQP inconsistent-linearization problem.
    m.S = pyo.RangeSet(0, len(cons) - 1) if cons else pyo.RangeSet(0, -1)
    m.sigma = pyo.Var(m.S, domain=pyo.NonNegativeReals, initialize=0.0)
    penalty = options.penalty_constant * sum(m.sigma[i] ** 2 for i in range(len(cons)))

    m.obj = pyo.Objective(expr=lin + quad + penalty, sense=pyo.minimize)
    m.cons = pyo.ConstraintList()
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    for i, con in enumerate(cons):
        body, op = con.body, con.operator
        keep_exact = (method == 'slcp' and con.exact_in_logspace)

        if keep_exact:
            terms = body.terms
            if body.is_monomial:
                # Monomial: log c + a.(d + log x_k) is affine in d.
                c, a = terms[0]
                expr = math.log(c) + sum(a[j] * (m.d[j] + log_xk[j]) for j in range(n))
                m.cons.add(expr == m.sigma[i] if op == '==' else expr <= m.sigma[i])
            else:
                # Posynomial: log-sum-exp, convex, imposed exactly. This is the
                # whole point of SLCP -- no linearization error here at all.
                expr = sum(pyo.exp(math.log(c)
                                   + sum(a[j] * (m.d[j] + log_xk[j]) for j in range(n)))
                           for c, a in terms)
                m.cons.add(pyo.log(expr) <= m.sigma[i])
        elif method == 'slcp' and con.is_sp_form:
            # SP form: impose  log p(x)  <=  log q_hat(x)  with p EXACT
            # (log-sum-exp) and only q condensed to a monomial (affine in log
            # space). Convex, and strictly less approximation than linearizing
            # the whole ratio.
            cq, aq = body.condensed_q(x_k)
            lhs = sum(pyo.exp(math.log(c)
                              + sum(a[j] * (m.d[j] + log_xk[j]) for j in range(n)))
                      for c, a in body.p.terms)
            rhs = math.log(cq) + sum(aq[j] * (m.d[j] + log_xk[j]) for j in range(n))
            m.cons.add(pyo.log(lhs) - rhs <= m.sigma[i])
        elif method == 'sqp':
            v = body(x_k)
            g = body.grad(x_k)
            expr = v + sum(g[j] * m.d[j] for j in range(n))
            m.cons.add(expr == 1.0 + m.sigma[i] if op == '=='
                       else expr <= 1.0 + m.sigma[i])
        else:
            # Log-space linearization: log v + (x . grad v)/v . d
            v = body(x_k)
            g = body.log_grad(x_k)
            expr = math.log(v) + sum(g[j] * m.d[j] for j in range(n))
            m.cons.add(expr == m.sigma[i] if op == '==' else expr <= m.sigma[i])

    # Keep the iterate inside the positive orthant. In log space this is a bound
    # on d; the paper's epsilon floor of Equation 16.
    if method != 'sqp':
        floor = math.log(options.x_min)
        for j in range(n):
            m.cons.add(m.d[j] >= floor - log_xk[j])

    # per-iteration trust region stated as a ratio to the current iterate
    # (see Options.max_step_ratio)
    if getattr(options, 'max_step_ratio', None) is not None:
        _r = options.max_step_ratio
        # A CALLABLE r(x_k) is evaluated at the current outer iterate. That is
        # needed when the modelled quantity is a SHIFTED variable: if x = c + q
        # for an offset c, a fixed ratio on x is NOT a fixed ratio on q, and the
        # region silently loosens or tightens as q moves.
        r = np.asarray(_r(x_k) if callable(_r) else _r, dtype=float)
        if r.ndim == 0:
            r = np.full(n, float(r))
        if r.shape != (n,):
            raise ValueError(f'max_step_ratio must be scalar or length {n}, '
                             f'got shape {r.shape}')
        for j in range(n):
            if np.isfinite(r[j]) and r[j] > 0:
                if r[j] >= 1.0:
                    raise ValueError('max_step_ratio must be < 1 (x_sub/x_outer '
                                     'has to stay positive)')
                m.cons.add(m.d[j] <= math.log(1.0 + float(r[j])))
                m.cons.add(m.d[j] >= math.log(1.0 - float(r[j])))

    # per-iteration trust region on the step (see Options.max_log_step)
    if options.max_log_step is not None:
        mls = np.asarray(options.max_log_step, dtype=float)
        if mls.ndim == 0:
            mls = np.full(n, float(mls))
        if mls.shape != (n,):
            raise ValueError(f'max_log_step must be scalar or length {n}, '
                             f'got shape {mls.shape}')
        for j in range(n):
            if np.isfinite(mls[j]):
                m.cons.add(m.d[j] <= float(mls[j]))
                m.cons.add(m.d[j] >= -float(mls[j]))

    return _solve_pyomo_subproblem(m, n, len(cons), options, method)


def _solve_pyomo_subproblem(m, n, n_cons, options, method='slcp'):
    """Hand an assembled sub-problem to IPOPT and read back (d, multipliers).

    Shared by the rebuild path and the cached one, so both report failures the
    same way.
    """
    from lcsolver.environment import ipopt_solver_factory
    opt = ipopt_solver_factory()
    if not opt.available(exception_flag=False):
        raise SolverUnavailable(
            'SLCP needs IPOPT to solve its sub-problems; no usable installation '
            'was found. Run `lcsolver-install-solvers`; see docs/ipopt.rst.')
    for k, v in (options.ipopt_options or {}).items():
        opt.options[k] = v

    # load_solutions=False: pyomo's default tries to load a solution BEFORE
    # anyone inspects the status, so a failed sub-problem dies inside
    # `load_from` with "Cannot load a SolverResults object with bad status"
    # instead of raising the clean RuntimeError below -- which solve() already
    # knows how to catch and report. Same defect, and same fix, as the main
    # ipopt path (deferring load_solutions until after the status check).
    results = opt.solve(m, tee=options.tee, load_solutions=False)
    tc = str(results.solver.termination_condition)
    if tc not in ('optimal', 'locallyOptimal', 'feasible'):
        raise RuntimeError(f'the {method.upper()} sub-problem failed: {tc}')
    m.solutions.load_from(results)

    d = np.array([pyo.value(m.d[j]) for j in range(n)])

    # Multipliers on the original constraints, for the merit function and BFGS.
    mults = np.zeros(n_cons)
    for i in range(n_cons):
        try:
            mults[i] = abs(m.dual.get(m.cons[i + 1], 0.0) or 0.0)
        except Exception:
            mults[i] = 0.0

    return d, mults


# ---------------------------------------------------------------------------
# Lagrangian gradients
# ---------------------------------------------------------------------------
def _lagrangian_gradient(problem, x, mults, method, reduced,
                         exact_objective=False):
    """Gradient of the (optionally Reduced) Lagrangian in the working space.

    The *Reduced* Lagrangian, paper Equation 14, omits the constraints that SLCP
    represents exactly:

    .. math::  \\mathcal{L}_R(y,\\lambda) = \\log f(x)
               + \\lambda \\log g(x) + \\lambda \\log h(x)

    Those constraints' curvature is already captured exactly in the sub-problem,
    so approximating it again in the BFGS Hessian sets the approximation fighting
    the true constraint. The paper is emphatic that this matters: "Imposing exact
    constraints without this modification performs worse than strict LSQP."

    ``exact_objective`` extends that same principle to the objective. With
    ``Options.exact_objective`` set, the objective is imposed exactly in the
    sub-problem as a log-sum-exp, so its curvature is already there in full;
    leaving it in the Reduced Lagrangian makes B model it a *second* time. The
    effect is the one the paper describes for constraints -- the approximation
    fights the exact term -- and it shows up as short steps and slow linear
    descent rather than as failure. On SPaircraft, whose signomial constraints
    keep B alive, this alone is the difference between crawling and converging.
    """
    if method == 'sqp':
        g = problem.objective.grad(x)
        for i, con in enumerate(problem.constraints):
            g = g + mults[i] * con.body.grad(x)
        return g

    # Objective omitted when it too is imposed exactly, for the same reason
    # the exact constraints are.
    g = (np.zeros(len(x)) if (reduced and method == 'slcp' and exact_objective)
         else problem.objective.log_grad(x))
    for i, con in enumerate(problem.constraints):
        if reduced and method == 'slcp' and con.exact_in_logspace:
            continue                     # excluded from the Reduced Lagrangian
        g = g + mults[i] * con.body.log_grad(x)
    return g


class LimitedMemoryB:
    """Limited-memory stand-in for the dense BFGS matrix ``B``.

    Two things scale as ``n^2`` in the dense path, and the second dominates:
    storing and updating ``B`` itself, and -- much worse -- building the
    sub-problem's quadratic term ``0.5 * sum_ij B[i][j] d_i d_j``, which is an
    ``n^2``-term Pyomo expression constructed from scratch every iteration. At
    n = 1173 that is 1.4 million terms per sub-problem.

    The damped BFGS update is a rank-two correction,

        B+ = B - (Bs)(Bs)^T / (s^T B s) + (r r^T) / (s^T r),

    so ``B`` is exactly ``gamma*I`` plus a sum of signed rank-one terms. Keeping
    only the most recent ``memory`` updates gives

        d^T B d = gamma * sum_j d_j^2  +  sum_k sigma_k (v_k . d)^2,

    which is ``O(n * memory)`` terms instead of ``O(n^2)``: ~12k rather than
    1.4M at n = 1173 with memory = 5.

    This is a *option*, not a replacement -- ``Options.hessian_memory = None``
    keeps the dense matrix and the original expression, unchanged.
    """

    __slots__ = ('n', 'memory', 'gamma', 'pairs', 'autoscale')

    def __init__(self, n, memory=5, gamma=1.0, autoscale=False):
        self.n = int(n)
        self.memory = int(memory)
        self.gamma = float(gamma)
        self.autoscale = bool(autoscale)
        self.pairs = []          # list of (sign, vector), newest last

    def matvec(self, s):
        """``B @ s``, in O(n * memory)."""
        s = np.asarray(s, dtype=float).ravel()
        out = self.gamma * s
        for sign, v in self.pairs:
            out = out + sign * v * float(v @ s)
        return out

    def quad_terms(self):
        """``(gamma, [(sign, vector), ...])`` for building the quadratic form."""
        return self.gamma, list(self.pairs)

    def update(self, s, z):
        """Damped BFGS, stored as two more rank-one terms."""
        s = np.asarray(s, dtype=float).ravel()
        z = np.asarray(z, dtype=float).ravel()
        if self.autoscale and not self.pairs:
            # Shanno-Phua initial scaling, applied ONCE while B is still
            # gamma*I so the stored pairs stay consistent with it. Without it
            # the background curvature is the identity in log space forever,
            # whatever the problem's actual scale -- and since the quadratic
            # acts as the sub-problem's proximal term, that sets the step
            # length directly.
            zz = float(z @ z)
            sz0 = float(s @ z)
            if zz > 0.0 and sz0 > 0.0:
                self.gamma = max(1e-8, min(1e8, zz / sz0))
        Bs = self.matvec(s)
        sBs = float(s @ Bs)
        sz = float(s @ z)
        if sBs <= 0:
            return self
        theta = 1.0 if sz >= 0.2 * sBs else (0.8 * sBs) / (sBs - sz)
        r = theta * z + (1.0 - theta) * Bs
        sr = float(s @ r)
        if abs(sr) < 1e-14:
            return self
        self.pairs.append((-1.0, Bs / math.sqrt(sBs)))
        self.pairs.append((+1.0, r / math.sqrt(abs(sr))))
        # Keep the newest `memory` updates, i.e. 2*memory vectors. Discarding
        # the oldest pair is what makes this limited-memory rather than exact;
        # the identity scaling underneath keeps the result positive definite.
        if len(self.pairs) > 2 * self.memory:
            self.pairs = self.pairs[-2 * self.memory:]
        return self

    def reset(self):
        self.pairs = []
        return self


def _damped_bfgs(B, s, z):
    """Damped BFGS update, paper Equation 13 (Nocedal & Wright Procedure 18.2).

    Damping keeps ``B`` positive definite even when the curvature condition
    fails, which it can here because ``z`` is built from the Reduced Lagrangian
    rather than the full one.
    """
    s = np.asarray(s, dtype=float).reshape(-1, 1)
    z = np.asarray(z, dtype=float).reshape(-1, 1)
    Bs = B @ s
    # These inner products are (1, 1) arrays; extract with .item() rather than
    # float(), which NumPy has deprecated for ndim > 0 and will eventually error.
    sBs = (s.T @ Bs).item()
    sz = (s.T @ z).item()
    if sBs <= 0:
        return B
    theta = 1.0 if sz >= 0.2 * sBs else (0.8 * sBs) / (sBs - sz)
    r = theta * z + (1.0 - theta) * Bs
    sr = (s.T @ r).item()
    if abs(sr) < 1e-14:
        return B
    return B - (Bs @ Bs.T) / sBs + (r @ r.T) / sr


# ---------------------------------------------------------------------------
# Merit function
# ---------------------------------------------------------------------------
def _constraint_violations(problem, x):
    """The l1 constraint violation of each constraint, in ``body <= 1`` form."""
    out = np.zeros(len(problem.constraints))
    for i, con in enumerate(problem.constraints):
        v = con.body(x) - 1.0
        out[i] = abs(v) if con.operator == '==' else max(0.0, v)
    return out


def _max_violation(problem, x):
    """Largest single constraint violation at ``x``, or 0.0 if unconstrained."""
    if not problem.constraints:
        return 0.0
    return float(np.max(_constraint_violations(problem, x)))


def _merit(problem, x, mu):
    """l1 merit function phi(x) = f(x) + sum_i mu_i |c_i(x)|_+."""
    return problem.objective_value(x) + float(np.dot(mu, _constraint_violations(problem, x)))


def _all_positive(problem, x):
    """True when the objective and every constraint body are strictly positive.

    The log transform is undefined otherwise. The paper's Limitations section
    notes the mitigation: "the step size can always be constrained to ensure
    these functions remain positive." That is what this predicate is for -- a
    trial point failing it is rejected by the line search and the step shortened,
    rather than being allowed to poison the next sub-problem.
    """
    try:
        if not (problem.objective_value(x) > 0):
            return False
        return all(con.body(x) > 0 for con in problem.constraints)
    except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
        return False


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def solve(problem, x0, method='slcp', options=None):
    """Run SLCP (or an LSQP / SQP baseline) from ``x0``.

    Parameters
    ----------
    problem : Problem
    x0 : array-like
        Strictly positive starting point.
    method : {'slcp', 'lsqp', 'sqp'}
    options : Options, optional

    Returns
    -------
    Result
    """
    if method not in ('slcp', 'lsqp', 'sqp'):
        raise ValueError("method must be one of 'slcp', 'lsqp', 'sqp'")

    # Same precondition as SIA: every sub-problem is an IPOPT solve, so
    # without one the loop makes no progress and returns the starting point
    # dressed as a result. Say what is actually wrong instead.
    from lcsolver.environment import ipopt_available
    if not ipopt_available():
        raise SolverUnavailable(
            'SLCP solves every sub-problem with IPOPT, and no usable '
            'installation was found. Run `lcsolver-install-solvers`; see '
            'docs/ipopt.rst.')

    options = options or Options()

    n = problem.n
    x = np.asarray(x0, dtype=float).copy()
    if np.any(x <= 0):
        raise ValueError('SLCP works in log space, so x0 must be strictly positive')

    B = (LimitedMemoryB(n, options.hessian_memory,
                        gamma=getattr(options, 'hessian_gamma', 1.0),
                        autoscale=getattr(options, 'hessian_scaling', False))
         if options.hessian_memory else np.eye(n))
    cache = None
    if options.cache_subproblem and method == 'slcp':
        cache = SubproblemCache(problem, options)
        if cache.usable:
            cache.build()
        else:
            cache = None
    mu = np.zeros(len(problem.constraints))
    # Persistent lower bound on the merit penalties. The mu update below is
    # rebuilt from the SUB-PROBLEM multipliers every iteration, so a constraint
    # the LINEARISED model believes is slack carries mu ~ 0 -- and then violating
    # the TRUE constraint is free in the line search. That is how a black-box
    # constraint ends up badly violated at a point the algorithm is happy to
    # stop at. This floor lets an escalation persist instead of decaying away.
    mu_floor = np.zeros(len(problem.constraints))
    escalations = 0
    res = Result()
    res.history.append(x.copy())
    watchdog = 0

    for k in range(options.max_iterations):
        try:
            d, mults = _solve_subproblem(problem, x, B, options, method,
                                         cache=cache)
        except RuntimeError as exc:
            res.status = f'sub-problem failure at iteration {k}: {exc}'
            res.x, res.objective, res.iterations = x, problem.objective_value(x), k
            res.max_violation = _max_violation(problem, x)
            return res
        res.subproblem_solves += 1

        # --- merit-function multipliers ------------------------------------
        # Nocedal & Wright Equation 18.36: mu_i must dominate |lambda_i| for the
        # step to be a descent direction on phi.
        mu = np.maximum(np.abs(mults) * options.mu_margin,
                        0.5 * (mu + np.abs(mults) * options.mu_margin))
        mu = np.maximum(mu, mu_floor)

        # --- line search ----------------------------------------------------
        phi0 = _merit(problem, x, mu)
        # Directional derivative of phi along d, in the working space.
        if method == 'sqp':
            dphi = float(np.dot(problem.objective.grad(x), d))
        else:
            dphi = float(np.dot(problem.objective.log_grad(x), d))
        dphi -= float(np.dot(mu, _constraint_violations(problem, x)))

        alpha = 1.0
        accepted = False
        for _ in range(options.max_step_size_tries):
            x_trial = (x * np.exp(alpha * d)) if method != 'sqp' else (x + alpha * d)
            if np.any(x_trial <= 0) or not np.all(np.isfinite(x_trial)):
                alpha *= options.rho
                continue
            if method != 'sqp' and not _all_positive(problem, x_trial):
                # Shorten the step rather than stepping somewhere the log
                # transform cannot be evaluated.
                alpha *= options.rho
                continue
            try:
                phi_trial = _merit(problem, x_trial, mu)
                res.function_evaluations += 1
            except (ValueError, FloatingPointError, OverflowError):
                alpha *= options.rho
                continue
            if np.isfinite(phi_trial) and phi_trial <= phi0 + options.eta * alpha * dphi:
                accepted = True
                break
            alpha *= options.rho

        if not accepted:
            # Watchdog. A step that raises the merit function is not necessarily
            # a bad step: an algorithm that has overshot a constraint must often
            # pass through worse merit values on its way back to feasibility, and
            # insisting on monotone decrease every iteration stalls it there.
            # Allow a bounded run of such steps before giving up.
            watchdog += 1
            if watchdog > options.watchdog_iterations:
                res.status = (f'line search failed at iteration {k} and the '
                              f'watchdog limit ({options.watchdog_iterations}) '
                              f'was exceeded')
                res.x, res.objective, res.iterations = x, problem.objective_value(x), k
                res.max_violation = _max_violation(problem, x)
                return res
            alpha = 1.0
            x_probe = (x * np.exp(alpha * d)) if method != 'sqp' else (x + alpha * d)
            while (np.any(x_probe <= 0) or not np.all(np.isfinite(x_probe))
                   or (method != 'sqp' and not _all_positive(problem, x_probe))):
                alpha *= options.rho
                if alpha < 1e-12:
                    res.status = f'no positive step available at iteration {k}'
                    res.x, res.objective, res.iterations = (
                        x, problem.objective_value(x), k)
                    res.max_violation = _max_violation(problem, x)
                    return res
                x_probe = (x * np.exp(alpha * d)) if method != 'sqp' else (x + alpha * d)
        else:
            watchdog = 0

        # --- take the step --------------------------------------------------
        x_new = (x * np.exp(alpha * d)) if method != 'sqp' else (x + alpha * d)

        # BFGS on the Reduced Lagrangian, evaluated at both points with the SAME
        # multipliers, so the difference isolates the curvature.
        reduced = (method == 'slcp')
        # SP-form constraints ALWAYS enter the Reduced Lagrangian: they are only
        # PARTLY exact (p is imposed exactly, but q is condensed to a monomial,
        # which discards q's curvature). Measured: excluding them fails to
        # converge from every start while including them takes ~20 iterations.
        exact_obj_opt = (getattr(options, 'exact_objective', False)
                         and isinstance(problem.objective, Posynomial))
        g_old = _lagrangian_gradient(problem, x, mults, method, reduced,
                                     exact_obj_opt)
        g_new = _lagrangian_gradient(problem, x_new, mults, method, reduced,
                                     exact_obj_opt)
        s = (np.log(x_new) - np.log(x)) if method != 'sqp' else (x_new - x)
        if isinstance(B, LimitedMemoryB):
            B.update(s, g_new - g_old)
        else:
            B = _damped_bfgs(B, s, g_new - g_old)

        x = x_new
        res.history.append(x.copy())

        step_norm = float(np.linalg.norm(alpha * d))
        res.step_norms.append(step_norm)
        grad_lag = float(np.max(np.abs(
            _lagrangian_gradient(problem, x, mults, method, reduced=False))))
        res.grad_lagrangian.append(grad_lag)

        if options.verbose:
            print(f'  {method:5s} itr {k + 1:3d}  f={problem.objective_value(x):.8f}  '
                  f'|d|={step_norm:.3e}  |gradL|={grad_lag:.3e}  alpha={alpha:.3f}')

        # --- convergence ----------------------------------------------------
        # `converged` means a KKT point to tolerance: STATIONARY *and* FEASIBLE.
        # It previously meant only "the loop stopped", and was set to True on a
        # small step alone. That is actively misleading on hard signomial /
        # black-box problems, where SLCP routinely crawls to a tiny step while
        # the Lagrangian gradient is still O(1e-1) and constraints are violated:
        # the caller sees converged=True and a plausible objective, with no way
        # to tell it from a real optimum short of recomputing the violations by
        # hand. max_violation is now always reported so that check is free.
        viol = _max_violation(problem, x)
        feasible = viol <= options.feasibility_tolerance
        if grad_lag < options.lagrangian_gradient_tolerance and feasible:
            res.converged, res.status = True, 'converged on the gradient of the Lagrangian'
            res.x, res.objective, res.iterations = x, problem.objective_value(x), k + 1
            res.max_violation = viol
            return res
        if step_norm < options.step_magnitude_tolerance:
            if feasible:
                # A genuine (if weakly certified) stop.
                res.converged, res.status = True, 'converged on the step magnitude'
                res.x, res.objective, res.iterations = x, problem.objective_value(x), k + 1
                res.max_violation = viol
                return res
            # INFEASIBLE STALL. Returning here would hand back a point that
            # violates the constraints -- which is exactly what used to be
            # reported as 'converged'. The step has collapsed because the merit
            # function is not charging enough for the violation, so raise the
            # penalty on the offending constraints and carry on. Feasibility is
            # thus part of the termination criterion, not merely reported.
            if escalations < options.max_penalty_escalations:
                escalations += 1
                bad = _constraint_violations(problem, x) > options.feasibility_tolerance
                mu_floor[bad] = np.maximum(mu_floor[bad] * options.penalty_escalation,
                                           max(1.0, abs(problem.objective_value(x))))
                mu = np.maximum(mu, mu_floor)
                # the stored curvature is for the old merit
                B = (B.reset() if isinstance(B, LimitedMemoryB)
                     else np.eye(n))
                if options.verbose:
                    print(f'  escalating merit penalty on {int(bad.sum())} violated '
                          f'constraint(s) (escalation {escalations}/'
                          f'{options.max_penalty_escalations}), max violation {viol:.3e}')
                continue
            res.converged = False
            res.status = (f'stalled at iteration {k + 1}: step {step_norm:.3e} is below '
                          f'tolerance and the point is still INFEASIBLE after '
                          f'{escalations} penalty escalations (max violation '
                          f'{viol:.3e} > {options.feasibility_tolerance:g}); '
                          f'|gradL| = {grad_lag:.3e}')
            res.x, res.objective, res.iterations = x, problem.objective_value(x), k + 1
            res.max_violation = viol
            return res

    res.status = f'did not converge within {options.max_iterations} iterations'
    res.x, res.objective, res.iterations = x, problem.objective_value(x), options.max_iterations
    res.max_violation = _max_violation(problem, x)
    return res
