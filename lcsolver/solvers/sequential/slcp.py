#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Sequential Log-Convex Programming (SLCP) with an IPOPT sub-problem solver.

Split the constraints: posynomials <= 1 and monomials == 1 are convex under the
log transform and imposed exactly; everything else is linearized in log space as
SQP would. The sub-problem is log-convex rather than quadratic -- keeping the
posynomials exact stops the step leaving a constraint a linear model would have
badly under-estimated, the failure mode that costs LSQP iterations.

Implements Algorithm 1 (relaxed sub-problem, Eq. 15) of Karcher & Haimes,
"A Method of Sequential Log-Convex Programming for Engineering Design",
Optim. Eng. (2022), doi:10.1007/s11081-022-09750-3.

IPOPT instead of the paper's cvxopt: cvxopt has to smuggle the quadratic
objective past its GP routine via callback overwrites; in Pyomo/IPOPT the
sub-problem is just declared. Log-space gradient throughout is Eq. 11:
d log f(e^y)/dy_i = x_i * (df/dx_i) / f(x).
"""

import math

import numpy as np
import pyomo.environ as pyo

from lcsolver.core.errors import SolverUnavailable


# ---------------------------------------------------------------------------
# Problem description
# ---------------------------------------------------------------------------
class Posynomial:
    """A posynomial sum_k c_k prod_j x_j^a_kj, c_k > 0. Stored as (coeff, exponent_vector) pairs; one term = monomial."""

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
        """d log f(e^y) / dy, via Eq. 11. Exact and cheap for a posynomial."""
        x = np.asarray(x, dtype=float)
        f = self(x)
        return x * self.grad(x) / f


class Signomial:
    """A general positive function as a value/gradient callback.

    fn(x) returns (value, gradient) in the natural variables, value > 0.
    The black-box hook: the algorithm only needs f and grad f at the iterate.
    fn_value(x), when given, returns the value ALONE -- the cheap path for
    the value-only queries (line search, violation checks, restores) on a
    box whose derivatives cost extra (a finite-difference sweep, an adjoint
    solve); without it every value query pays the full fn.
    """

    __slots__ = ('fn', 'n', 'fn_value')

    def __init__(self, fn, n, fn_value=None):
        self.fn = fn
        self.n = int(n)
        self.fn_value = fn_value

    def __call__(self, x):
        if self.fn_value is not None:
            return float(self.fn_value(np.asarray(x, dtype=float)))
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
    """A Signomial that calls fn once per distinct point.

    The solver hits value and gradient through separate entry points at the
    same x several times per iteration; fn returns both together, so cache it.
    Measured: 4.1 evals/iteration down to 1.0 -- matters when the box is CFD/FEA.
    Keyed on the exact float pattern of x (no interpolation, no tolerance).
    maxsize bounds the cache (default keeps everything); evaluations counts
    genuine calls, the number to budget against.
    """

    __slots__ = ('_cache', '_order', '_maxsize', 'evaluations')

    def __init__(self, fn, n, maxsize=None, fn_value=None):
        super().__init__(fn, n, fn_value=fn_value)
        self._cache = {}
        self._order = []
        self._maxsize = maxsize
        self.evaluations = 0

    def _eval(self, x, need_grad=True):
        # a cached entry may be value-only (gradient None, from the cheap
        # fn_value path); it satisfies value queries and upgrades in place
        # on the first gradient request
        x = np.asarray(x, dtype=float)
        key = x.tobytes()
        hit = self._cache.get(key)
        if hit is not None and (hit[1] is not None or not need_grad):
            return hit
        if need_grad or self.fn_value is None:
            v, g = self.fn(x)
            out = (float(v), np.asarray(g, dtype=float))
        else:
            out = (float(self.fn_value(x)), None)
        if hit is None:
            self._order.append(key)
        self._cache[key] = out
        self.evaluations += 1
        if self._maxsize is not None and len(self._order) > self._maxsize:
            del self._cache[self._order.pop(0)]
        return out

    def __call__(self, x):
        return self._eval(x, need_grad=False)[0]

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
    """A grey-box equality body, bb(inputs) / x[out_index] == 1.

    out_index tags the dedicated output column the row is solved for, so the
    restore has a closed form: x[out_index] *= body(x) restores the row exactly
    in one box evaluation (the output never appears inside the box). sia's
    composite restore uses this -- skipping these rows let tangential drift
    accumulate on black-box problems; see restore_composite_bb.
    """

    __slots__ = ('out_index',)

    def __init__(self, fn, n, out_index, maxsize=None, fn_value=None):
        super().__init__(fn, n, maxsize=maxsize, fn_value=fn_value)
        self.out_index = int(out_index)


class PosynomialRatio:
    """p(x) / q(x) with p and q both posynomials -- the signomial-program form.

    Classical SP treatment: keep p exact, condense only q to the AGM monomial
    q_hat tight at x_k. q_hat <= q everywhere, so p <= q_hat is conservative,
    and only the concave-in-log part is approximated instead of the whole body.
    Same condensation an SP solver uses, here so the treatments can be compared.
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
        """AGM monomial under-estimator of the numerator.

        Condensing p too makes p/q <= 1 a monomial row, but it is NOT
        conservative (p_hat <= p, so iterates can leave the feasible set).
        Tangency at x_k survives, which keeps the KKT certificate honest.
        """
        return condense(self.p, x_k, self.n)


class CondensedEquality:
    """A signomial equality p/q == 1, condensed on both sides.

    Not the two-ratio pair (p/q <= 1 and q/p <= 1): that restricts steps to the
    null space of H1+H2 wherever active, and is dual-degenerate -- multipliers
    of several thousand measured on SPaircraft, KKT residual stuck at their
    noise. Condensing both sides gives the monomial p_hat/q_hat == 1: affine in
    log space, one well-conditioned multiplier, full tangent hyperplane.
    Nothing conservative is lost (an equality has no interior); iterates can
    leave the feasible set as under PCCP, but tangency keeps the certificate.
    """

    __slots__ = ('p', 'q', 'n')

    def __init__(self, p, q, n):
        self.p, self.q, self.n = p, q, n

    def __call__(self, x):
        return self.p(x) / self.q(x)

    def log_grad(self, x):
        """The true gradient, for the KKT test -- not the condensed one."""
        return self.p.log_grad(x) - self.q.log_grad(x)

    def grad(self, x):
        p, q = self.p(x), self.q(x)
        return self.p.grad(x)/q - p*self.q.grad(x)/q**2

    @property
    def is_monomial(self):
        return self.p.is_monomial and self.q.is_monomial

    def condensed(self, x_k):
        """(coeff, exponents) of the monomial p_hat/q_hat."""
        cp, ap = condense(self.p, x_k, self.n)
        cq, aq = condense(self.q, x_k, self.n)
        return cp / cq, ap - aq


def condense(posy, x_k, n=None):
    """AGM monomial under-estimator of a posynomial at x_k.

    q_hat(x) = prod_i (u_i(x)/w_i)**w_i, w_i = u_i(x_k)/q(x_k). q_hat <= q
    everywhere, equal and gradient-matching at x_k. Returns (coeff, exponents).
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
    """One constraint, body <= 1 or body == 1. Body type (Posynomial vs Signomial) decides exact vs linearized."""

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
        """True when the log transform makes this constraint convex as written (posynomial <= 1, monomial == 1)."""
        return isinstance(self.body, Posynomial)

    @property
    def is_sp_form(self):
        """True for a PosynomialRatio: p exact, q condensed (see that class)."""
        return isinstance(self.body, PosynomialRatio)


class Problem:
    """A signomial program in the standard form of paper Eq. 12."""

    def __init__(self, n, objective, constraints, names=None, bounds=None):
        self.n = int(n)
        self.objective = objective
        self.constraints = list(constraints)
        self.names = list(names) if names else [f'x{i + 1}' for i in range(n)]
        # optional per-variable (lower, upper), either possibly None. Bounds go
        # on the sub-problem variable, not rows: a bound is free, a row is one
        # more log-sum-exp per iteration (2346 needless rows on SPaircraft).
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
        self.feasibility_tolerance = 1e-6    # max |violation| for a run to count as converged
        self.max_step_ratio = None           # per-iteration trust region as a ratio to the
                                             # current iterate: 1-r <= x_sub/x_outer <= 1+r,
                                             # i.e. d in [log(1-r), log(1+r)] in log space.
                                             # Scalar, length-n array (np.inf = unbounded),
                                             # or callable r(x_k) for iterate-dependent
                                             # bounds (e.g. fixed ratio on a shifted var).
        self.max_log_step = None             # per-iteration trust region on the step:
                                             # |d_j| <= max_log_step[j], i.e. |dx/x| <~ it.
                                             # None = unbounded. Scalar or length-n. Step
                                             # control for the linearisation -- needed even
                                             # with exact gradients, unlike TRMM.
        self.penalty_escalation = 10.0       # factor to raise the merit penalty on a violated
                                             # constraint when the step collapses infeasible
        self.max_penalty_escalations = 6     # give up after this many escalations
        self.eta = 1e-4                   # Armijo parameter, in (0, 0.5)
        self.rho = 0.8                    # backtracking factor, in (0, 1)
        self.mu_margin = 1.2              # merit-multiplier margin, > 1
        self.max_step_size_tries = 30
        self.watchdog_iterations = 5      # consecutive non-monotone steps allowed
        self.exact_objective = False      # impose a posynomial objective exactly
                                          # (log-sum-exp) instead of linearizing.
                                          # Linearized, a fully convex problem
                                          # still marches quasi-Newton (72 iters
                                          # on the wind turbine vs 1 on the GP
                                          # path). With every constraint also
                                          # exact, the sub-problem IS the problem
                                          # and the BFGS term is dropped.
        self.cache_subproblem = False     # build the Pyomo model once and
                                          # re-point via mutable Params instead
                                          # of rebuilding symbolically. Only
                                          # bridge shapes cache (exact
                                          # posynomials/ratios, method='slcp');
                                          # anything else keeps the rebuild path.
        self.hessian_gamma = 1.0          # weight of gamma*I in the
                                          # limited-memory B. When s.z > 0 keeps
                                          # failing (most constraints exact, so
                                          # the Reduced Lagrangian is nearly
                                          # empty) B stays at gamma*I and the
                                          # quadratic is purely proximal: gamma
                                          # sets the step length. Lower it for
                                          # longer steps.
        self.hessian_scaling = False      # scale the initial limited-memory B
                                          # by the Shanno-Phua ratio y.y/s.y on
                                          # the first update. An identity
                                          # background orders of magnitude off
                                          # the true curvature throttles every
                                          # step. Limited-memory path only.
        self.hessian_memory = None        # None = dense n-by-n BFGS and the
                                          # n^2-term quadratic. Integer m = keep
                                          # the last m rank-two updates, O(n*m)
                                          # quadratic terms. Same damped update;
                                          # only scrolled-out curvature is lost.
        self.x_min = 1e-9                 # the epsilon floor of Eq. 16
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
        self.max_violation = None   # max |violation| at the returned point, in
                                    # body <= 1 form. Set on every exit path so
                                    # the caller can always tell if x is usable.

    def __repr__(self):
        viol = ('' if self.max_violation is None
                else f' max_violation={self.max_violation:.3e}')
        return (f'<Result {self.status!r} iterations={self.iterations} '
                f'objective={self.objective!r}{viol}>')


def _fully_log_convex(problem):
    """True when objective and every constraint are exact in log space -- then one sub-problem solve is enough."""
    return (isinstance(problem.objective, Posynomial)
            and all(c.exact_in_logspace for c in problem.constraints))


def _apply_variable_bounds(m, problem, log_xk):
    """Put the model's variable bounds on the sub-problem step.

    lo <= x <= hi becomes log(lo/x_k) <= d <= log(hi/x_k), exact and free.
    Not optional with presolve: fold_singleton_rows turns rows like x >= 1 into
    bounds, so without this the sub-problem solves an unbounded relaxation
    (min x*y s.t. x>=1, y>=2 returned 1e-18 and claimed convergence).
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

    d = 0 is the right initial guess, but when the iterate sits ON a bound the
    log-space bound lands a rounding error above zero and Pyomo warns (W1002)
    once per variable per sub-problem -- a wall of noise on active-bound models.
    Seat the value rather than silence the logger (that would hide genuine
    ones). Call after every bound application; only the last pass knows the box.
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

    Rebuilding symbolically dominates the run on SPaircraft (6077 log-sum-exp
    over 1173 vars, ~50 times). Each exact term is exp([log c_k + a_k.log x_k]
    + [a_k.d]): the projection a_k.d is fixed, only the constant moves, so the
    constants become mutable Params. The condensed q of a PosynomialRatio fits
    too: aq.d = sum_i w_i (a_i.d) reuses the fixed projections with one mutable
    weight per term, and the objective gradient has the same form. Only bridge
    shapes cache (exact posynomials/ratios, method='slcp'); Signomial bodies
    and lsqp/sqp need fresh gradients and fall back to rebuilding.
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

        # positivity floor and trust region become variable bounds (free to
        # change); the uncached path writes them as extra constraints
        floor = math.log(self.options.x_min)
        lo = floor - log_xk
        hi = np.full(n, np.inf)
        # the model's own bounds, which presolve may have folded rows into;
        # without them this solves an unbounded relaxation (see _apply_variable_bounds)
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

        # the objective is the one part worth rebuilding: it carries the
        # Hessian term (O(n*memory) limited-memory, O(n^2) dense)
        if m.component('obj') is not None:
            m.del_component(m.obj)
        quad = 0.0 if self.drop_quad else _quadratic_expression(B, m.d, n)
        penalty = self.options.penalty_constant * sum(
            m.sigma[i] ** 2 for i in range(len(cons)))
        # exact form: obj_lin already carries log f; linearized: gradient
        # term only, needs the constant
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
    """Build and solve one sub-problem; return the step d and the multipliers.

    method: 'slcp' = posynomials/monomials exact, rest linearized in log space
    (Eq. 15); 'lsqp' = everything linearized in log space (a QP); 'sqp' =
    everything linearized in the natural variables, no log transform.
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
        # posynomial objective: log f is convex in log space, impose it exactly
        # (linearizing it makes a fully log-convex problem march quasi-Newton)
        lin = pyo.log(sum(
            pyo.exp(math.log(c) + sum(a[j] * (m.d[j] + log_xk[j])
                                      for j in range(n)))
            for c, a in problem.objective.terms))
    else:
        # log space, Eq. 11: (x . grad f) / f is the log-space gradient
        gf = problem.objective.log_grad(x_k)
        lin = math.log(f_k) + sum(gf[j] * m.d[j] for j in range(n))

    # objective and constraints all exact: the sub-problem IS the original
    # problem, a curvature term would only bias the step
    quad = 0.0 if drop_quad else _quadratic_expression(B, m.d, n)

    # --- constraints -------------------------------------------------------
    # each constraint gets a relaxation variable sigma >= 0, penalised in the
    # objective -- otherwise the sub-problem can be infeasible when the true
    # problem is not (standard SQP inconsistent-linearization problem)
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
                # monomial: log c + a.(d + log x_k) is affine in d
                c, a = terms[0]
                expr = math.log(c) + sum(a[j] * (m.d[j] + log_xk[j]) for j in range(n))
                m.cons.add(expr == m.sigma[i] if op == '==' else expr <= m.sigma[i])
            else:
                # posynomial: log-sum-exp, convex, imposed exactly -- the whole
                # point of SLCP, no linearization error here
                expr = sum(pyo.exp(math.log(c)
                                   + sum(a[j] * (m.d[j] + log_xk[j]) for j in range(n)))
                           for c, a in terms)
                m.cons.add(pyo.log(expr) <= m.sigma[i])
        elif method == 'slcp' and con.is_sp_form:
            # SP form: log p(x) <= log q_hat(x), p exact, q condensed to a
            # monomial -- convex, less approximation than linearizing the ratio
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
            # log-space linearization: log v + (x . grad v)/v . d
            v = body(x_k)
            g = body.log_grad(x_k)
            expr = math.log(v) + sum(g[j] * m.d[j] for j in range(n))
            m.cons.add(expr == m.sigma[i] if op == '==' else expr <= m.sigma[i])

    # keep the iterate in the positive orthant: a bound on d in log space,
    # the paper's epsilon floor of Eq. 16
    if method != 'sqp':
        floor = math.log(options.x_min)
        for j in range(n):
            m.cons.add(m.d[j] >= floor - log_xk[j])

    # per-iteration trust region stated as a ratio to the current iterate
    # (see Options.max_step_ratio)
    if getattr(options, 'max_step_ratio', None) is not None:
        _r = options.max_step_ratio
        # callable r(x_k) evaluated at the outer iterate: needed for shifted
        # variables (x = c + q), where a fixed ratio on x is not one on q
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
    """Hand an assembled sub-problem to IPOPT and read back (d, multipliers). Shared by the rebuild and cached paths."""
    from lcsolver.environment import ipopt_solver_factory
    opt = ipopt_solver_factory()
    if not opt.available(exception_flag=False):
        raise SolverUnavailable(
            'SLCP needs IPOPT to solve its sub-problems; no usable installation '
            'was found. Run `lcsolver-install-solvers`; see docs/ipopt.rst.')
    for k, v in (options.ipopt_options or {}).items():
        opt.options[k] = v

    # load_solutions=False: pyomo's default loads before the status check, so a
    # failed sub-problem dies inside load_from instead of raising the clean
    # RuntimeError below. Same fix as the main ipopt path.
    results = opt.solve(m, tee=options.tee, load_solutions=False)
    tc = str(results.solver.termination_condition)
    if tc not in ('optimal', 'locallyOptimal', 'feasible'):
        from lcsolver.environment import linear_solver_failure_note
        raise RuntimeError(
            f'the {method.upper()} sub-problem failed: {tc}'
            + linear_solver_failure_note(
                (options.ipopt_options or {}).get('linear_solver')))
    m.solutions.load_from(results)

    d = np.array([pyo.value(m.d[j]) for j in range(n)])

    # multipliers on the original constraints, for the merit function and BFGS
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

    The Reduced Lagrangian (paper Eq. 14) omits constraints imposed exactly:
    their curvature is already in the sub-problem, and modeling it again in B
    sets the approximation fighting the true constraint (paper: worse than
    strict LSQP). exact_objective extends this to an exactly imposed objective
    -- leaving it in makes B model it twice, showing up as short steps and slow
    linear descent. On SPaircraft that alone is crawling vs converging.
    """
    if method == 'sqp':
        g = problem.objective.grad(x)
        for i, con in enumerate(problem.constraints):
            g = g + mults[i] * con.body.grad(x)
        return g

    # objective omitted when it too is imposed exactly, same reason as the
    # exact constraints
    g = (np.zeros(len(x)) if (reduced and method == 'slcp' and exact_objective)
         else problem.objective.log_grad(x))
    for i, con in enumerate(problem.constraints):
        if reduced and method == 'slcp' and con.exact_in_logspace:
            continue                     # excluded from the Reduced Lagrangian
        g = g + mults[i] * con.body.log_grad(x)
    return g


class LimitedMemoryB:
    """Limited-memory stand-in for the dense BFGS matrix B.

    The dense path's killer is the n^2-term Pyomo quadratic rebuilt every
    iteration (1.4M terms at n=1173). The damped update is rank-two, so B is
    gamma*I plus signed rank-one terms; keeping the last `memory` updates makes
    d^T B d O(n*memory) terms (~12k at n=1173, memory=5). An option, not a
    replacement: hessian_memory=None keeps the dense matrix unchanged.
    """

    __slots__ = ('n', 'memory', 'gamma', 'pairs', 'autoscale')

    def __init__(self, n, memory=5, gamma=1.0, autoscale=False):
        self.n = int(n)
        self.memory = int(memory)
        self.gamma = float(gamma)
        self.autoscale = bool(autoscale)
        self.pairs = []          # list of (sign, vector), newest last

    def matvec(self, s):
        """B @ s, in O(n * memory)."""
        s = np.asarray(s, dtype=float).ravel()
        out = self.gamma * s
        for sign, v in self.pairs:
            out = out + sign * v * float(v @ s)
        return out

    def quad_terms(self):
        """(gamma, [(sign, vector), ...]) for building the quadratic form."""
        return self.gamma, list(self.pairs)

    def update(self, s, z):
        """Damped BFGS, stored as two more rank-one terms."""
        s = np.asarray(s, dtype=float).ravel()
        z = np.asarray(z, dtype=float).ravel()
        if self.autoscale and not self.pairs:
            # Shanno-Phua initial scaling, applied once while B is still
            # gamma*I so the stored pairs stay consistent. Without it the
            # background curvature is identity forever, whatever the problem's
            # scale -- and the quadratic is the proximal term, so that sets
            # the step length directly.
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
        # keep the newest `memory` updates (2*memory vectors); the identity
        # scaling underneath keeps the result positive definite
        if len(self.pairs) > 2 * self.memory:
            self.pairs = self.pairs[-2 * self.memory:]
        return self

    def reset(self):
        self.pairs = []
        return self


def _damped_bfgs(B, s, z):
    """Damped BFGS update, paper Eq. 13 (Nocedal & Wright Procedure 18.2).

    Damping keeps B positive definite when the curvature condition fails,
    which it can here since z comes from the Reduced Lagrangian.
    """
    s = np.asarray(s, dtype=float).reshape(-1, 1)
    z = np.asarray(z, dtype=float).reshape(-1, 1)
    Bs = B @ s
    # inner products are (1, 1) arrays; use .item(), not float() (deprecated
    # for ndim > 0)
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
    """The l1 constraint violation of each constraint, in body <= 1 form."""
    out = np.zeros(len(problem.constraints))
    for i, con in enumerate(problem.constraints):
        v = con.body(x) - 1.0
        out[i] = abs(v) if con.operator == '==' else max(0.0, v)
    return out


def _max_violation(problem, x):
    """Largest single constraint violation at x, or 0.0 if unconstrained."""
    if not problem.constraints:
        return 0.0
    return float(np.max(_constraint_violations(problem, x)))


def _merit(problem, x, mu):
    """l1 merit function phi(x) = f(x) + sum_i mu_i |c_i(x)|_+."""
    return problem.objective_value(x) + float(np.dot(mu, _constraint_violations(problem, x)))


def _all_positive(problem, x):
    """True when the objective and every constraint body are strictly positive.

    The log transform is undefined otherwise; the line search rejects a trial
    point failing this and shortens the step (the paper's Limitations
    mitigation) rather than let it poison the next sub-problem.
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
    """Run SLCP (or an LSQP / SQP baseline) from x0.

    x0 must be strictly positive; method is 'slcp', 'lsqp' or 'sqp'.
    Returns a Result.
    """
    if method not in ('slcp', 'lsqp', 'sqp'):
        raise ValueError("method must be one of 'slcp', 'lsqp', 'sqp'")

    # same precondition as SIA: without IPOPT the loop returns the starting
    # point dressed as a result -- say what is actually wrong instead
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
    # persistent lower bound on the merit penalties. mu is rebuilt from the
    # sub-problem multipliers each iteration, so a constraint the linearised
    # model thinks is slack carries mu ~ 0 and violating the true one is free
    # in the line search -- how a black-box constraint ends up badly violated
    # at a stop point. The floor lets an escalation persist.
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
        # Nocedal & Wright Eq. 18.36: mu_i must dominate |lambda_i| for the
        # step to be a descent direction on phi
        mu = np.maximum(np.abs(mults) * options.mu_margin,
                        0.5 * (mu + np.abs(mults) * options.mu_margin))
        mu = np.maximum(mu, mu_floor)

        # --- line search ----------------------------------------------------
        phi0 = _merit(problem, x, mu)
        # directional derivative of phi along d, in the working space
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
                # shorten rather than step where the log transform can't be evaluated
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
            # watchdog: a merit-raising step isn't necessarily bad -- coming
            # back from an overshoot often passes through worse merit values,
            # so allow a bounded run of them before giving up
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

        # BFGS on the Reduced Lagrangian, both points with the SAME multipliers
        # so the difference isolates the curvature
        reduced = (method == 'slcp')
        # SP-form constraints always enter the Reduced Lagrangian: only partly
        # exact (q's curvature is discarded by condensation). Measured:
        # excluding them fails from every start; including them takes ~20 iters.
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
        # `converged` means a KKT point to tolerance: stationary AND feasible.
        # It used to mean "the loop stopped" (True on a small step alone) --
        # misleading on hard signomial/black-box problems, which crawl to a
        # tiny step while still infeasible. max_violation makes the check free.
        viol = _max_violation(problem, x)
        feasible = viol <= options.feasibility_tolerance
        if grad_lag < options.lagrangian_gradient_tolerance and feasible:
            res.converged, res.status = True, 'converged on the gradient of the Lagrangian'
            res.x, res.objective, res.iterations = x, problem.objective_value(x), k + 1
            res.max_violation = viol
            return res
        if step_norm < options.step_magnitude_tolerance:
            if feasible:
                # a genuine (if weakly certified) stop
                res.converged, res.status = True, 'converged on the step magnitude'
                res.x, res.objective, res.iterations = x, problem.objective_value(x), k + 1
                res.max_violation = viol
                return res
            # infeasible stall: the step collapsed because the merit function
            # isn't charging enough for the violation. Raise the penalty on
            # the offending constraints and carry on -- feasibility is part of
            # the termination criterion, not merely reported.
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
