#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
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
        x_k = np.asarray(x_k, dtype=float)
        qv = self.q(x_k)
        coeff, expo = 1.0, np.zeros(self.n)
        for c, a in self.q.terms:
            w = c * np.prod(x_k ** a) / qv          # AGM weight, sums to 1
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

    def __init__(self, n, objective, constraints, names=None):
        self.n = int(n)
        self.objective = objective
        self.constraints = list(constraints)
        self.names = list(names) if names else [f'x{i + 1}' for i in range(n)]

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


# ---------------------------------------------------------------------------
# Sub-problem construction
# ---------------------------------------------------------------------------
def _solve_subproblem(problem, x_k, B, options, method):
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
    m = pyo.ConcreteModel()
    m.J = pyo.RangeSet(0, n - 1)
    m.d = pyo.Var(m.J, initialize=0.0)

    log_xk = np.log(x_k)
    f_k = problem.objective_value(x_k)

    # --- objective ---------------------------------------------------------
    if method == 'sqp':
        # Natural space: linear model of f plus the BFGS quadratic.
        gf = problem.objective.grad(x_k)
        lin = f_k + sum(gf[j] * m.d[j] for j in range(n))
    else:
        # Log space, Equation 11: (x . grad f) / f is the log-space gradient.
        gf = problem.objective.log_grad(x_k)
        lin = math.log(f_k) + sum(gf[j] * m.d[j] for j in range(n))

    if isinstance(B, LimitedMemoryB):
        # d^T B d = gamma*sum(d^2) + sum_k sign_k (v_k . d)^2, which is
        # O(n*memory) terms rather than the dense O(n^2).
        gamma, pairs = B.quad_terms()
        quad = 0.5 * gamma * sum(m.d[j] ** 2 for j in range(n))
        for sign, v in pairs:
            nz = np.nonzero(v)[0]
            if nz.size:
                proj = sum(float(v[j]) * m.d[j] for j in nz)
                quad = quad + 0.5 * sign * proj ** 2
    else:
        quad = 0.5 * sum(B[i][j] * m.d[i] * m.d[j] for i in range(n) for j in range(n))

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

    opt = pyo.SolverFactory('ipopt')
    if not opt.available(exception_flag=False):
        raise RuntimeError(
            'SLCP needs IPOPT to solve its sub-problems; no usable installation '
            'was found. Install the ipopt executable or `pip install cyipopt`.')
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
    mults = np.zeros(len(cons))
    for i in range(len(cons)):
        try:
            mults[i] = abs(m.dual.get(m.cons[i + 1], 0.0) or 0.0)
        except Exception:
            mults[i] = 0.0

    return d, mults


# ---------------------------------------------------------------------------
# Lagrangian gradients
# ---------------------------------------------------------------------------
def _lagrangian_gradient(problem, x, mults, method, reduced):
    """Gradient of the (optionally Reduced) Lagrangian in the working space.

    The *Reduced* Lagrangian, paper Equation 14, omits the constraints that SLCP
    represents exactly:

    .. math::  \\mathcal{L}_R(y,\\lambda) = \\log f(x)
               + \\lambda \\log g(x) + \\lambda \\log h(x)

    Those constraints' curvature is already captured exactly in the sub-problem,
    so approximating it again in the BFGS Hessian sets the approximation fighting
    the true constraint. The paper is emphatic that this matters: "Imposing exact
    constraints without this modification performs worse than strict LSQP."
    """
    if method == 'sqp':
        g = problem.objective.grad(x)
        for i, con in enumerate(problem.constraints):
            g = g + mults[i] * con.body.grad(x)
        return g

    g = problem.objective.log_grad(x)
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

    __slots__ = ('n', 'memory', 'gamma', 'pairs')

    def __init__(self, n, memory=5, gamma=1.0):
        self.n = int(n)
        self.memory = int(memory)
        self.gamma = float(gamma)
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
    options = options or Options()

    n = problem.n
    x = np.asarray(x0, dtype=float).copy()
    if np.any(x <= 0):
        raise ValueError('SLCP works in log space, so x0 must be strictly positive')

    B = (LimitedMemoryB(n, options.hessian_memory)
         if options.hessian_memory else np.eye(n))
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
            d, mults = _solve_subproblem(problem, x, B, options, method)
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
        g_old = _lagrangian_gradient(problem, x, mults, method, reduced)
        g_new = _lagrangian_gradient(problem, x_new, mults, method, reduced)
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
