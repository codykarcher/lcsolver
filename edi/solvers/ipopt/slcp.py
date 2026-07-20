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
* everything else, :math:`g(x) \\le 1` and :math:`h(x) = 1`, which is linearised
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


class Constraint:
    """One constraint in the standard form ``body <= 1`` or ``body == 1``.

    ``body`` is a :class:`Posynomial` or a :class:`Signomial`. Which of the two it
    is determines whether SLCP imposes it exactly or linearises it.
    """

    __slots__ = ('body', 'operator')

    def __init__(self, body, operator='<='):
        if operator not in ('<=', '=='):
            raise ValueError("operator must be '<=' or '=='")
        if operator == '==' and isinstance(body, Posynomial) and not body.is_monomial:
            raise ValueError(
                'a multi-term posynomial equality is not GP-compatible; supply it '
                'as a Signomial so it is linearised instead')
        self.body = body
        self.operator = operator

    @property
    def exact_in_logspace(self):
        """True when the log transform makes this constraint convex as written.

        Posynomial ``<= 1`` becomes log-sum-exp ``<= 0`` (convex); monomial ``== 1``
        becomes an affine equality. Both can be imposed directly. Everything else
        must be linearised.
        """
        return isinstance(self.body, Posynomial)


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
        self.eta = 1e-4                   # Armijo parameter, in (0, 0.5)
        self.rho = 0.8                    # backtracking factor, in (0, 1)
        self.mu_margin = 1.2              # merit-multiplier margin, > 1
        self.max_step_size_tries = 30
        self.watchdog_iterations = 5      # consecutive non-monotone steps allowed
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

    def __repr__(self):
        return (f'<Result {self.status!r} iterations={self.iterations} '
                f'objective={self.objective!r}>')


# ---------------------------------------------------------------------------
# Sub-problem construction
# ---------------------------------------------------------------------------
def _solve_subproblem(problem, x_k, B, options, method):
    """Build and solve one sub-problem; return the step ``d`` and the multipliers.

    ``method`` selects how constraints enter:

    ``'slcp'``
        Posynomials and monomials imposed exactly (log-sum-exp / affine);
        everything else linearised in log space. Paper Equation 15.
    ``'lsqp'``
        Every constraint linearised in log space. The sub-problem is then a QP,
        which is exactly what the paper says SLCP degenerates to when no
        posynomial constraints are present.
    ``'sqp'``
        Every constraint linearised in the natural variables, no log transform.
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

    quad = 0.5 * sum(B[i][j] * m.d[i] * m.d[j] for i in range(n) for j in range(n))

    # --- constraints -------------------------------------------------------
    # Every constraint gets its own relaxation variable sigma >= 0, penalised in
    # the objective. Without this the sub-problem can be infeasible even when the
    # true problem is not -- the standard SQP inconsistent-linearisation problem.
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
                # whole point of SLCP -- no linearisation error here at all.
                expr = sum(pyo.exp(math.log(c)
                                   + sum(a[j] * (m.d[j] + log_xk[j]) for j in range(n)))
                           for c, a in terms)
                m.cons.add(pyo.log(expr) <= m.sigma[i])
        elif method == 'sqp':
            v = body(x_k)
            g = body.grad(x_k)
            expr = v + sum(g[j] * m.d[j] for j in range(n))
            m.cons.add(expr == 1.0 + m.sigma[i] if op == '=='
                       else expr <= 1.0 + m.sigma[i])
        else:
            # Log-space linearisation: log v + (x . grad v)/v . d
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

    opt = pyo.SolverFactory('ipopt')
    if not opt.available(exception_flag=False):
        raise RuntimeError(
            'SLCP needs IPOPT to solve its sub-problems; no usable installation '
            'was found. Install the ipopt executable or `pip install cyipopt`.')
    for k, v in (options.ipopt_options or {}).items():
        opt.options[k] = v

    results = opt.solve(m, tee=options.tee)
    tc = str(results.solver.termination_condition)
    if tc not in ('optimal', 'locallyOptimal', 'feasible'):
        raise RuntimeError(f'the {method.upper()} sub-problem failed: {tc}')

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

    B = np.eye(n)
    mu = np.zeros(len(problem.constraints))
    res = Result()
    res.history.append(x.copy())
    watchdog = 0

    for k in range(options.max_iterations):
        try:
            d, mults = _solve_subproblem(problem, x, B, options, method)
        except RuntimeError as exc:
            res.status = f'sub-problem failure at iteration {k}: {exc}'
            res.x, res.objective, res.iterations = x, problem.objective_value(x), k
            return res
        res.subproblem_solves += 1

        # --- merit-function multipliers ------------------------------------
        # Nocedal & Wright Equation 18.36: mu_i must dominate |lambda_i| for the
        # step to be a descent direction on phi.
        mu = np.maximum(np.abs(mults) * options.mu_margin,
                        0.5 * (mu + np.abs(mults) * options.mu_margin))

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
                    return res
                x_probe = (x * np.exp(alpha * d)) if method != 'sqp' else (x + alpha * d)
        else:
            watchdog = 0

        # --- take the step --------------------------------------------------
        x_new = (x * np.exp(alpha * d)) if method != 'sqp' else (x + alpha * d)

        # BFGS on the Reduced Lagrangian, evaluated at both points with the SAME
        # multipliers, so the difference isolates the curvature.
        reduced = (method == 'slcp')
        g_old = _lagrangian_gradient(problem, x, mults, method, reduced)
        g_new = _lagrangian_gradient(problem, x_new, mults, method, reduced)
        s = (np.log(x_new) - np.log(x)) if method != 'sqp' else (x_new - x)
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
        if grad_lag < options.lagrangian_gradient_tolerance:
            res.converged, res.status = True, 'converged on the gradient of the Lagrangian'
            res.x, res.objective, res.iterations = x, problem.objective_value(x), k + 1
            return res
        if step_norm < options.step_magnitude_tolerance:
            res.converged, res.status = True, 'converged on the step magnitude'
            res.x, res.objective, res.iterations = x, problem.objective_value(x), k + 1
            return res

    res.status = f'did not converge within {options.max_iterations} iterations'
    res.x, res.objective, res.iterations = x, problem.objective_value(x), options.max_iterations
    return res
