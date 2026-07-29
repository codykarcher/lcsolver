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

from edi.solvers.ipopt.slcp import (Posynomial, PosynomialRatio, Problem,
                                    Signomial)

__all__ = ["SIAOptions", "SIAResult", "solve_sia", "classify"]


class SIAOptions:
    """Algorithm parameters."""

    def __init__(self, **kw):
        self.max_iterations = 100
        # --- KKT termination, all on the TRUE problem ---------------------
        self.feasibility_tolerance = 1e-6     # max_i log g_i(x)
        self.stationarity_tolerance = 1e-6    # ||grad log L||_inf in log space
        self.complementarity_tolerance = 1e-6  # max_i |lambda_i log g_i(x)|
        # --- penalty CCP, only active while infeasible --------------------
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
        self.verbose = False
        self.tee = False
        self.ipopt_options = {"print_level": 0, "sb": "yes"}
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


def classify(problem):
    """``(n_exact, n_conservative, n_linearized)`` for a problem."""
    n_exact = n_cons = n_lin = 0
    for c in problem.constraints:
        if c.exact_in_logspace:
            n_exact += 1
        elif isinstance(c.body, PosynomialRatio):
            n_cons += 1
        else:
            n_lin += 1
    return n_exact, n_cons, n_lin


def _log_g(con, x):
    """``log g(x)`` for a constraint written ``g <= 1`` (so feasible is <= 0)."""
    return math.log(max(con.body(x), 1e-300))


def _violation(problem, x):
    return max((_log_g(c, x) for c in problem.constraints), default=0.0)


def _kkt(problem, x, mults):
    """``(stationarity, violation, complementarity)`` on the TRUE problem.

    All gradients are the true ones -- for a condensed constraint that means
    ``log_grad(p) - log_grad(q)``, not the gradient of the monomial that stood
    in for ``q`` in the sub-problem. Tangency makes the two equal at the
    iterate, which is exactly why the sub-problem's duals certify the original
    problem.
    """
    g = np.asarray(problem.objective.log_grad(x), dtype=float)
    comp = 0.0
    viol = 0.0
    for i, con in enumerate(problem.constraints):
        lg = _log_g(con, x)
        viol = max(viol, lg)
        lam = float(mults[i])
        if lam != 0.0:
            g = g + lam * np.asarray(con.body.log_grad(x), dtype=float)
        comp = max(comp, abs(lam * lg))
    return float(np.max(np.abs(g))), viol, comp


def _subproblem(problem, x_k, tau, radius, options, has_blackbox):
    """Assemble and solve the inner-approximation sub-problem in log space."""
    n = problem.n
    cons = problem.constraints
    log_xk = np.log(x_k)

    m = pyo.ConcreteModel()
    m.J = pyo.RangeSet(0, n - 1)
    m.I = pyo.RangeSet(0, len(cons) - 1)
    m.d = pyo.Var(m.J, initialize=0.0)
    m.s = pyo.Var(m.I, domain=pyo.NonNegativeReals, initialize=0.0)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    def lse(terms):
        """log sum_k c_k exp(a_k . (d + log x_k)) -- exact, convex."""
        return pyo.log(sum(
            pyo.exp(math.log(c) + sum(a[j] * (m.d[j] + log_xk[j])
                                      for j in range(n)))
            for c, a in terms))

    # --- objective: exact when posynomial, linearized otherwise -----------
    if isinstance(problem.objective, Posynomial):
        obj = lse(problem.objective.terms)
    else:
        f_k = problem.objective_value(x_k)
        gf = problem.objective.log_grad(x_k)
        obj = math.log(f_k) + sum(gf[j] * m.d[j] for j in range(n))

    m.obj = pyo.Objective(
        expr=obj + tau * sum(m.s[i] for i in range(len(cons))),
        sense=pyo.minimize)

    m.cons = pyo.ConstraintList()
    for i, con in enumerate(cons):
        body, op = con.body, con.operator
        if con.exact_in_logspace:
            if body.is_monomial:
                c, a = body.terms[0]
                e = math.log(c) + sum(a[j] * (m.d[j] + log_xk[j])
                                      for j in range(n))
                m.cons.add(e == m.s[i] if op == "==" else e <= m.s[i])
            else:
                m.cons.add(lse(body.terms) <= m.s[i])
        elif isinstance(body, PosynomialRatio):
            # log p  <=  log q_hat, with q_hat the AGM monomial under-estimator.
            # q_hat <= q everywhere, so this is HARDER than the true constraint.
            cq, aq = body.condensed_q(x_k)
            log_qhat = math.log(cq) + sum(aq[j] * (m.d[j] + log_xk[j])
                                          for j in range(n))
            m.cons.add(lse(body.p.terms) - log_qhat <= m.s[i])
        else:
            # Black box: value and gradient only, so no conservative model
            # exists. Linearize, and let the trust region below carry it.
            v = body(x_k)
            gl = body.log_grad(x_k)
            e = math.log(max(v, 1e-300)) + sum(gl[j] * m.d[j] for j in range(n))
            m.cons.add(e == m.s[i] if op == "==" else e <= m.s[i])

    # Stay in the positive orthant.
    floor = math.log(options.x_min)
    for j in range(n):
        m.cons.add(m.d[j] >= floor - log_xk[j])

    # Trust region -- ONLY when something had to be linearized. With every
    # constraint exact or conservative the step is safe by construction and a
    # region would only slow it down.
    if has_blackbox:
        for j in range(n):
            m.cons.add(m.d[j] <= radius)
            m.cons.add(m.d[j] >= -radius)

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
    s = np.array([pyo.value(m.s[i]) for i in range(len(cons))])

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
    return d, s, mults, float(pyo.value(obj))


def solve_sia(problem: Problem, x0, options: SIAOptions = None) -> SIAResult:
    """Solve a signomial program by sequential inner approximation."""
    options = options or SIAOptions()
    x = np.asarray(x0, dtype=float).copy()
    if np.any(x <= 0):
        raise ValueError("SIA works in log space, so x0 must be strictly positive")

    n_exact, n_cons, n_lin = classify(problem)
    has_blackbox = n_lin > 0

    res = SIAResult()
    res.conservative = not has_blackbox
    res.history.append(x.copy())

    tau = options.tau0
    radius = options.trust_radius
    mults = np.zeros(len(problem.constraints))

    if options.verbose:
        print(f"  SIA: {n_exact} exact, {n_cons} conservative, "
              f"{n_lin} linearized"
              + ("" if has_blackbox else "  -> fully conservative, "
                                         "no globalization needed"))

    for k in range(options.max_iterations):
        try:
            d, s, mults, model_obj = _subproblem(
                problem, x, tau, radius, options, has_blackbox)
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

        f_old = problem.objective_value(x)
        x_new = x * np.exp(d)

        if has_blackbox:
            # Globalize the linearized block only: compare the true objective
            # reduction against the model's prediction, and size the region by
            # it. Structured constraints cannot be violated by the step, so
            # they play no part in the test.
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

        # --- KKT test on the ORIGINAL problem -----------------------------
        stat, viol, comp = _kkt(problem, x, mults)
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
    else:
        res.status = (f"did not converge within {options.max_iterations} "
                      "iterations")
        res.x, res.objective = x, problem.objective_value(x)
        res.iterations = options.max_iterations

    if res.x is None:
        res.x, res.objective = x, problem.objective_value(x)
    stat, viol, comp = _kkt(problem, res.x, mults)
    res.stationarity, res.max_violation, res.complementarity = stat, viol, comp
    res.multipliers = mults
    res.slacks_active = viol > options.feasibility_tolerance
    return res
