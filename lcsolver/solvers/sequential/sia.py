#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Sequential Inner Approximation (SIA) for signomial programs.

Sits between PCCP and SLCP. Each constraint gets one of three classes:
exact (posynomial, imposed as log-sum-exp), conservative (signomial
p/q <= 1 with the denominator AGM-condensed -- tangent at the iterate,
harder than the truth), or linearized (black box: value + gradient only).

With nothing linearized the sub-problem's feasible set is a subset of the
true one containing the iterate, so every iterate is feasible and descent
is structural -- no line search, merit function, or trust region (Marks &
Wright 1978; Lipp & Boyd 2016). The trust region and ratio test apply only
to the black-box block. Unlike PCCP, termination is a genuine KKT residual
on the ORIGINAL problem (tangency makes the sub-problem's duals the true
ones) and black boxes are supported; unlike SLCP, the objective is imposed
exactly and there is no BFGS quadratic. Infeasible starts get per-row
slacks and a tau penalty (penalty CCP) until the slacks vanish.
"""
from __future__ import annotations

import math

import numpy as np
import pyomo.environ as pyo

from lcsolver.core.errors import SolverUnavailable

from lcsolver.solvers.sequential.slcp import (CondensedEquality, Posynomial,
                                    seat_step_in_bounds,
                                    PosynomialRatio, Problem, Signomial)

__all__ = ["SIAOptions", "SIAResult", "solve_sia", "classify",
           "SubproblemCache"]


class SIAOptions:
    """Algorithm parameters."""

    def __init__(self, **kw):
        self.max_iterations = 400
        # 400, not 100: SPaircraft converges in 149, and a low cap throws
        # away a correct answer while a high one costs nothing.
        # --- KKT termination, all on the TRUE problem ---------------------
        self.feasibility_tolerance = 1e-6     # max_i log g_i(x)
        self.stationarity_tolerance = 1e-6    # ||grad log L||_inf in log space
        self.complementarity_tolerance = 1e-6  # max_i |lambda_i log g_i(x)|
        # --- Relative-change termination, OFF by default -------------------
        # For black boxes with noisy gradients (~1e-3, e.g. MSES), where a
        # 1e-6 stationarity is unreachable. Stop when an ACCEPTED step
        # changed nothing: objective_reltol is |f_k/f_{k-1} - 1|,
        # variable_reltol is max_j |x_k,j/x_{k-1,j} - 1|, between consecutive
        # accepted iterates. Every set tolerance must be met at a feasible
        # iterate; reported CONVERGED with no KKT certificate.
        self.objective_reltol = None
        self.variable_reltol = None
        # --- Phase I: find a feasible point before optimizing --------------
        self.phase1 = True             # False falls back to penalty CCP
        # 'composite' (default) | 'l1' | 'minmax'. See _phase1_composite.
        # min-max shares one scalar t, so coupled signomial equalities must
        # all close together -- fails on the spcomparisons aircraft at any
        # budget. Elastic L1 gives each row its own slack (SNOPT elastic
        # mode / IPOPT restoration): satisfiable rows drop to zero slack.
        self.phase1_method = 'composite'
        self.phase1_restore_iterations = 12
        # Cap on the composite trust radius, log space. Phase I is a
        # minimum-change REPAIR of the seed, not a search: left free the
        # radius reached 8 and SPaircraft came back 0.46% worse from a
        # point nowhere near its seed.
        self.phase1_trust_max = 1.0
        # Proximity weight on the elastic Phase I objective:
        #     min  sum(s_i) + w ||d||^2
        # sum(s_i) has a whole face of minimizers, and which one comes back
        # picks the local optimum Phase II finds (simpleac: 4536 vs 6485
        # depending only on presolve). The weight makes the answer the
        # SMALLEST repair of the seed -- small enough to break ties without
        # trading feasibility away.
        self.phase1_proximity = 1e-6
        # Restore the equalities after each accepted Phase II step. A
        # condensed signomial equality is tangent, not conservative, so the
        # first step leaves the feasible set even from an exact start (E175:
        # 6e-15 -> 2.257 in one step) and the run asymptotes just above
        # tolerance. The Gauss-Newton normal step pulls back onto the
        # manifold cheaply. ON by default; ~50% more wall time, but every
        # iterate is a usable design. Set False to reproduce the drift.
        self.phase2_restore = True


        self.phase1_max_iterations = 50
        # When Phase I falls short, continue with penalty CCP (which
        # tolerates an infeasible start) instead of abandoning the solve.
        self.phase1_penalty_fallback = True
        self.phase1_margin = 1e-8      # target interiority for the Phase I
                                       # sub-problem; NOT an acceptance test
                                       # (an active equality can never be
                                       # strictly interior).
        # --- penalty CCP, used only if phase1 is off or fails --------------
        self.tau0 = 1.0
        self.tau_factor = 5.0
        self.tau_max = 1e12
        # near-feasible band (multiples of feasibility_tolerance) inside
        # which tau decays back toward tau0 instead of holding: an elevated
        # penalty in the endgame drowns the objective term (MSES camber
        # run: tau parked at 5.0 from iteration 41 to the stop)
        self.tau_relief_band = 10.0
        # declared RELATIVE noise floor of the analysis (0 = off).  Sets
        # the effective feasibility band for guards/stall/recovery and
        # accepts objective steps whose predicted log-change is below it:
        # below the analysis noise, "worse" and "no change" are the same
        # measurement (MSES camber run: ratio=-1.0 rejection churn at
        # violations the box cannot resolve)
        self.analysis_noise = 0.0
        # violations within this many times feasibility_tolerance are
        # repaired IN PLACE (targeted equality restore) instead of a full
        # Phase-I re-entry, which hops to the feasible interior and gives
        # back objective it must re-earn (camber run: a 1.2e-5 violation
        # cost a 145 N hop re-earned over 12 iterations)
        self.restoration_inplace_band = 100.0
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
        # A filter cannot break a two-point cycle: neither point dominates,
        # every flip is ACCEPTED, nothing shrinks the radius (free coupled
        # 737: pi_f 1.796 <-> 1.672 for 100+ iterations). Detect two
        # sizeable accepted steps whose SUM is small and shrink.
        self.zigzag_damp = True
        # net/step below this = a cycle. 0.5 is a 151-degree reversal; a
        # 90-degree curved descent has net/step = 1.41, so valley-following
        # cannot fire it. (0.25 missed the free coupled case's orbit at 0.49.)
        self.zigzag_net_frac = 0.5
        self.zigzag_cooldown = 4       # accepted steps with expansion held
                                       # off after a detection, so the radius
                                       # cannot re-inflate into the same flip
        # --- misc ----------------------------------------------------------
        self.x_min = 1e-9
        self.expand_past_blackbox = False
        # Carry a BFGS curvature model per linearized constraint, so the
        # subproblem holds a convex quadratic rather than a plane (see
        # Curvature) -- conservative wherever the estimate is good enough.
        # ON by default: the linearized row is where SIA gives up its
        # guarantee and B restores it. Only built when has_blackbox, so a
        # pure SP is bit-identical either way.
        self.curvature = True
        self.step_expansion = 1.0      # >1 enables the feasibility-verified
        self.step_expansion_max = 1e4  # step extension described in solve_sia.
                                       # Set to 1.0 to take the sub-problem's
                                       # step exactly as returned.
        # OFF by default (2026-08-30, reversing the 2026-08 flip). Governs
        # only plain INEQUALITY ratios now -- equalities own their both-sides
        # condensation in CondensedEquality. A condensed numerator drops the
        # conservative property with NO repair path (restoration closes
        # equality manifolds, not inequality excursions): the launch-vehicle
        # coupled-losses SP limit-cycled ~3.5e-2 infeasible with True, and
        # converges in 10 iterations to the certified optimum with False.
        # Set True only to reproduce the tangent large-step mode.
        self.condense_numerator = False
        # Seed for the black-box curvature model, |d log g / d log x| units.
        # BFGS B is ZERO on the first iteration, so a zero seed leaves the
        # first step unconservative -- see Curvature.observe. 0.0 reproduces
        # the old behaviour.
        self.curvature_prior = 1.0
        # BFGS iterations per black-box row after which B counts as TRAINED
        # and the trust region is released -- see the expand path below. The
        # region only substitutes for a conservative model, and B is one
        # once it has data.
        self.curvature_warmup = 6
        # Warm-up step control for the black-box rows: cap |grad . d| <=
        # this for every row, so r <= target / max_i ||grad_i||_1 --
        # self-scaling across parameterisations. Applied for the FIRST
        # blackbox_warm_iters iterations only, then released unconditionally:
        # gating on "trained" deadlocks (capped steps too small for the BFGS
        # secant, so training never completes -- measured, 300 iterations
        # parked at the warm start). 0 disables either field.
        self.blackbox_step_target = 0.15
        self.blackbox_warm_iters = 0
        # Apply the trust box for the FIRST this-many Phase II iterations
        # only, then drop it; None keeps it for the whole solve. Growing is
        # not removing: shrink beats growth arithmetically, so a collapsed
        # box can end up carrying the dual that belongs to the black-box
        # rows and freeze stationarity (2t+2c ROM section: 1.7e-02).
        self.trust_iterations = None
        # Floor, as a multiple of feasibility_tolerance, on the PREDICTED
        # improvement in the linearized block before a violation ratio is
        # formed. 0.01 was two orders too low: the ROM section formed ratios
        # from numbers at 1e-08 (inside Ipopt noise) and rejected 90 of 300
        # iterations. Below the tolerance a violation change is not a signal.
        self.ratio_gate_rel = 1.0
        # How much worse the TRUE violation of the linearized block may get
        # on an accepted step, as a multiple of max(current violation,
        # feasibility tolerance). 0/None disables. A backstop, not a filter:
        # if it fires often the threshold is too tight, not the steps bad
        # (near a solution the bar pins to factor*tol, inside solver noise).
        self.feasibility_guard = _FEAS_GUARD          # module default, 10.0
        # Absolute floor for the guard above -- defaults OFF. It sounds like
        # the fix for noise tripping the pinned relative bar (2t+2c: 101 of
        # 192 steps, 192 -> 802 iterations for the same answer) and is not:
        # at 1e-03 it admitted an excursion Phase II could not recover from
        # and the run died at iteration 8. The plain relative guard converges
        # both sections, and 3t+3c cannot be solved cold without it.
        self.feasibility_guard_abs = 0.0
        # --- Fletcher-Leyffer filter acceptance (replaces the guard) --------
        # ON by default; supersedes feasibility_guard, kept only to reproduce
        # the old behaviour. See Filter for why a scalar guard cannot work.
        self.filter_acceptance = True
        self.filter_gamma_h = 1e-5
        self.filter_gamma_f = 1e-5
        # --- restoration ----------------------------------------------------
        # A filter accepts feasibility-worsening steps only because they are
        # RECOVERABLE; this re-enters Phase I from inside Phase II when the
        # region collapses or brackets while infeasible, instead of aborting.
        self.restoration = True
        self.restoration_max = 8          # re-entries allowed per solve
        # Iterations the run may sit infeasible without a 10% reduction
        # before restoration triggers. This, not trust-region collapse, is
        # the trigger that matters (3t+3c: held viol 2.69e-01 for 1500
        # iterations with a healthy region).
        self.restoration_patience = 5
        # --- min-norm-dual KKT termination (EXPERIMENTAL, off by default) --
        # Split equalities make the dual set unbounded, so returned
        # multipliers can read garbage at an optimal point (coupled aircraft:
        # duals to 1e13). When feasible but stationarity fails, retest with
        # MINIMUM-NORM multipliers every kkt_min_norm_every iterations and
        # always on the final one.
        self.kkt_min_norm = False
        self.kkt_min_norm_every = 25
        self.kkt_min_norm_act_tol = 1e-6
        # --- Ipopt polish of the TRUE problem (EXPERIMENTAL, off) --------
        # Conservative first-order steps crawl in shallow valleys; from a
        # settled feasible point, hand the exact log-space problem to Ipopt
        # and adopt the result only if it verifies (feasible, no worse).
        # polish_box bounds the excursion so the polish stays in the basin.
        self.polish_ipopt = False
        self.polish_box = 3.0
        self.polish_max_iter = 3000
        # --- trajectory log (DIAGNOSTIC, off) ----------------------------
        # jsonl path; one line per ACCEPTED iterate with iteration,
        # objective, violation, stationarity, |d|, and every variable
        # matching a traj_vars substring. The failure movie, replayable.
        self.traj_log = None
        self.traj_vars = ()
        # (condense_numerator trades PCCP's step length for SIA's stopping
        # rule: tangency keeps the KKT test honest, the guarantees go.)
        self.cache_subproblem = True   # build each phase's Pyomo model once and
                                       # re-point it; see SubproblemCache. Only
                                       # applies when every body is a Posynomial
                                       # or PosynomialRatio.
        self.verbose = False
        self.tee = False
        # tol tightened from IPOPT's 1e-8: sub-problem dual accuracy feeds
        # the KKT residual, and 1e-8 leaves a ~2e-3 floor (SPaircraft:
        # 9.51e-07 at 1e-12, no extra iterations). constr_viol_tol must be
        # set too -- its own 1e-4 default is a floor on Phase I: the model
        # reports feasible because 1e-04 is what it was asked to achieve.
        self.ipopt_options = {"print_level": 0, "sb": "yes", "tol": 1e-12,
                              "constr_viol_tol": 1e-12,
                              "acceptable_constr_viol_tol": 1e-10}
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
        # KKT residuals on the ORIGINAL problem at the returned point
        self.max_violation = None
        self.stationarity = None
        self.complementarity = None
        self.multipliers = None
        self.report = None          # human-readable convergence explanation
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
        """One line saying what happened and whether to believe it."""
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
    """Signed violation in the log; positive means violated.

    An inequality ``g <= 1`` uses ``log g``; an equality uses ``|log g|``.
    Using ``log g`` for both let the KKT test certify a drifted-negative
    equality and made the ratio test read the drift as improvement (same
    -3.3 ratio for 300 iterations, radius knocked to 1.9e-6).
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


# How much worse the violation may get on an objective-ratio step before it
# is rejected outright. Generous -- a backstop, not a feasibility filter.
_FEAS_GUARD = 10.0


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


def _min_norm_mults(problem, x, act_tol=1e-6):
    """Minimum-norm multipliers over the ACTIVE rows at ``x``.

    ``min ||g0 + A lam||, lam >= 0`` over rows within ``act_tol`` of active
    (split equalities span the free-sign dual). Column scaling is what
    makes this work at aircraft size -- gradient columns span ~8 decades.
    Returns a full-length vector with zeros on inactive rows, for ``_kkt``.
    """
    from scipy.optimize import lsq_linear
    g0 = np.asarray(problem.objective.log_grad(x), dtype=float)
    idx, cols, free_sign = [], [], []
    for i, con in enumerate(problem.constraints):
        if con.operator == '==':
            idx.append(i)
            cols.append(np.asarray(con.body.log_grad(x), dtype=float))
            free_sign.append(True)
        elif _log_g(con, x) >= -act_tol:
            idx.append(i)
            cols.append(np.asarray(con.body.log_grad(x), dtype=float))
            free_sign.append(False)
    mults = np.zeros(len(problem.constraints))
    if not cols:
        return mults
    A = np.column_stack(cols)
    scale = np.maximum(np.linalg.norm(A, axis=0), 1e-12)
    lo = np.array([-np.inf if fs else 0.0 for fs in free_sign])
    r = lsq_linear(A / scale, -g0, bounds=(lo, np.inf),
                   tol=1e-12, max_iter=3000, lsq_solver="lsmr",
                   lsmr_tol=1e-12)
    lam = r.x / scale
    for j, i in enumerate(idx):
        mults[i] = lam[j]
    return mults


def _polish_ipopt(problem, x, options):
    """Solve the TRUE problem locally with Ipopt from ``x``.

    Rebuilds every row symbolically in log space so Ipopt gets exact second
    derivatives. Returns the polished x, or None on black-box rows, Ipopt
    failure, or a result that does not verify.
    """
    from lcsolver.solvers.sequential.slcp import (Posynomial, PosynomialRatio,
                                        CondensedEquality)

    def _sumexp(m, terms):
        return sum(c * pyo.exp(sum(float(a[j]) * m.y[j]
                                   for j in range(problem.n)
                                   if a[j] != 0.0))
                   for c, a in terms)

    import os as _os
    _dbg = (print if _os.environ.get("SIA_POLISH_DEBUG")
            else (lambda *a, **k: None))
    x = np.asarray(x, dtype=float)
    y0 = np.log(np.maximum(x, 1e-300))
    m = pyo.ConcreteModel()
    m.y = pyo.Var(range(problem.n))
    for j in range(problem.n):
        m.y[j].set_value(float(y0[j]))
        m.y[j].setlb(float(y0[j]) - options.polish_box)
        m.y[j].setub(float(y0[j]) + options.polish_box)
    lo_floor = math.log(options.x_min) if options.x_min else None
    if problem.bounds is not None:
        for j, pair in enumerate(problem.bounds[:problem.n]):
            if pair:
                lo, hi = pair
                if lo is not None and lo > 0:
                    m.y[j].setlb(max(m.y[j].lb, math.log(lo)))
                if hi is not None:
                    m.y[j].setub(min(m.y[j].ub, math.log(hi)))
    if lo_floor is not None:
        for j in range(problem.n):
            m.y[j].setlb(max(m.y[j].lb, lo_floor))
    m.cons = pyo.ConstraintList()
    for con in problem.constraints:
        b = con.body
        if isinstance(b, Posynomial):
            e = _sumexp(m, b.terms)
            m.cons.add(e == 1.0 if con.operator == '==' else e <= 1.0)
        elif isinstance(b, (PosynomialRatio, CondensedEquality)):
            pe, qe = _sumexp(m, b.p.terms), _sumexp(m, b.q.terms)
            m.cons.add(pe == qe if con.operator == '==' else pe <= qe)
        else:
            _dbg(f"  [polish] black-box row {type(b).__name__}: cannot polish")
            return None
    if not isinstance(problem.objective, Posynomial):
        _dbg(f"  [polish] objective is {type(problem.objective).__name__}")
        return None
    m.obj = pyo.Objective(expr=_sumexp(m, problem.objective.terms),
                          sense=pyo.minimize)
    from lcsolver.environment import ipopt_solver_factory
    opt = ipopt_solver_factory()
    if not opt.available(exception_flag=False):
        return None
    for k, v in (options.ipopt_options or {}).items():
        opt.options[k] = v
    opt.options['max_iter'] = options.polish_max_iter
    opt.options['warm_start_init_point'] = 'yes'
    try:
        r = opt.solve(m, tee=False, load_solutions=False)
        tc = str(r.solver.termination_condition)
        if tc not in ('optimal', 'locallyOptimal', 'feasible'):
            _dbg(f"  [polish] ipopt: {tc}")
            return None
        m.solutions.load_from(r)
    except Exception as exc:
        _dbg(f"  [polish] ipopt raised: {exc}")
        return None
    xn = np.exp(np.array([pyo.value(m.y[j]) for j in range(problem.n)]))
    # verify on the TRUE rows before adopting
    if _violation(problem, xn) > options.feasibility_tolerance:
        _dbg(f"  [polish] rejected: violation "
             f"{_violation(problem, xn):.2e}")
        return None
    if problem.objective_value(xn) > problem.objective_value(x) \
            * (1.0 + 1e-9):
        _dbg(f"  [polish] rejected: objective "
             f"{problem.objective_value(xn):.6g} vs "
             f"{problem.objective_value(x):.6g}")
        return None
    _dbg(f"  [polish] ADOPTED: objective {problem.objective_value(x):.6g} "
         f"-> {problem.objective_value(xn):.6g}")
    return xn


def _kkt(problem, x, mults, x_min=None, bound_tol=1e-6):
    """``(stationarity, violation, complementarity)`` on the TRUE problem.

    Gradients are the true ones; tangency is why the sub-problem's duals
    certify the original problem. Stationarity is the PROJECTED gradient:
    at an active bound the Lagrangian gradient is held by the bound's own
    multiplier, and the raw norm reports a large residual at an optimal
    point (on SPaircraft the whole apparent residual was this).
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


def _restore(problem, x, options, has_blackbox, cache, done, k, why,
             targets=None):
    """Re-enter Phase I from inside Phase II. Returns ``(x, done, ok)``.

    The filter accepts feasibility-worsening steps only because they are
    recoverable; this is what makes that true. Accepted only if it actually
    restores feasibility -- coming back infeasible means the problem is
    infeasible HERE, which is worth reporting.
    """
    if not getattr(options, 'restoration', False):
        return x, done, False
    if done >= int(getattr(options, 'restoration_max', 0) or 0):
        return x, done, False
    v_before = _violation(problem, x)
    if v_before <= options.feasibility_tolerance:
        return x, done, False           # feasible already; nothing to restore
    # Proportionate response: a near-tolerance violation is repaired IN
    # PLACE by the targeted equality restore before paying for a Phase-I
    # re-entry, which hops to the feasible interior and gives back
    # objective the run must re-earn
    _band = float(getattr(options, 'restoration_inplace_band', 0.0) or 0.0)
    if _band > 0.0 and v_before <= _band * options.feasibility_tolerance:
        try:
            _fn = restore_composite_bb if has_blackbox else restore_equalities
            x_ip, _h = _fn(problem, x.copy(),
                           iters=options.phase1_restore_iterations,
                           tol=min(options.feasibility_tolerance, 1e-10))
            v_ip = _violation(problem, np.asarray(x_ip, dtype=float))
            if (np.all(np.isfinite(x_ip)) and np.all(np.asarray(x_ip) > 0)
                    and v_ip <= options.feasibility_tolerance):
                if options.verbose:
                    print(f"  itr {k + 1:3d}  RESTORATION ({why}) repaired "
                          f"in place: viol {v_before:.3e} -> {v_ip:.3e}")
                return np.asarray(x_ip, dtype=float), done, True
        except Exception:
            pass                        # fall through to Phase I
    if options.verbose:
        print(f"  itr {k + 1:3d}  RESTORATION ({why}, viol {v_before:.3e}) "
              f"-> re-entering Phase I")
    try:
        x_r = _phase1(problem, x.copy(), options, has_blackbox, cache=cache)
        if isinstance(x_r, tuple):
            x_r = x_r[0]
        x_r = np.asarray(x_r, dtype=float)
    except Exception as exc:                                  # noqa: BLE001
        if options.verbose:
            print(f"       restoration failed: {type(exc).__name__}: {exc}")
        return x, done + 1, False
    v_after = _violation(problem, x_r)
    ok = bool(np.all(np.isfinite(x_r)) and np.all(x_r > 0)
              and v_after < v_before
              and v_after <= options.feasibility_tolerance)
    # Refuse a restoration to an already-returned point: Phase I is
    # deterministic, so the descent replays byte-identically (free coupled
    # 737: a 7-state cycle burned all restoration credits).
    if ok and targets is not None:
        lx_r = np.log(x_r)
        for t in targets:
            if float(np.abs(lx_r - t).max()) < 1e-3:
                ok = False
                if options.verbose:
                    print("       restoration returned an already-visited "
                          "point; refusing (restoration loop)")
                break
        else:
            targets.append(lx_r)
    if options.verbose:
        print(f"       restoration {'OK' if ok else 'insufficient'}: "
              f"viol {v_before:.3e} -> {v_after:.3e}")
    return (x_r if ok else x), done + 1, ok


class Filter:
    """A Fletcher-Leyffer filter over ``(violation, objective)`` pairs.

    Replaces the scalar guard, whose factor K is an exchange rate with no
    correct value (ROM section: K = 10 rejected 101 of 192 steps). The
    filter asks only whether the trial is DOMINATED by a point seen:
    ``h < (1 - gamma_h) * h_j  OR  f < f_j - gamma_f * h_j``, the gammas
    being anti-cycling margins, not exchange rates. Only sound with a
    restoration phase (see solve_sia). ``h = max(log_violation, 0)``.
    """

    def __init__(self, gamma_h=1e-5, gamma_f=1e-5, h_max=None):
        self.entries = []                      # [(h, f)]
        self.gamma_h = float(gamma_h)
        self.gamma_f = float(gamma_f)
        self.h_max = h_max                     # hard cap on violation, or None
        self.rejections = 0

    @staticmethod
    def h_of(log_viol):
        """Violation measure: non-negative, zero when the block has slack."""
        return max(float(log_viol), 0.0)

    def acceptable(self, h, f):
        if self.h_max is not None and h > self.h_max:
            return False
        for hj, fj in self.entries:
            if not (h < (1.0 - self.gamma_h) * hj or f < fj - self.gamma_f * hj):
                return False
        return True

    def add(self, h, f):
        """Record a point, dropping any entry it dominates."""
        self.entries = [(hj, fj) for hj, fj in self.entries
                        if not (hj >= h and fj >= f)]
        self.entries.append((float(h), float(f)))

    def __len__(self):
        return len(self.entries)


class Curvature:
    """A convex quadratic model of a black-box constraint's log-curvature.

    A linearization is a monomial in log space -- exact but not
    conservative. With PSD ``B``, ``log g + grad.d + 1/2 d.B.d`` stays
    convex, stays tangent, and is an upper bound wherever B dominates the
    true log-Hessian: the conservative class's properties, conditional on
    B. Built by damped BFGS (Powell's damping, Nocedal & Wright Procedure
    18.2); indefinite curvature is damped away, and the ratio test catches
    an under-penalising B. Restricted to the variables the constraint
    touches (4x4, not 61x61).
    """

    def __init__(self, n, damping=0.2, cap=1e3, prior=0.0):
        self.n = n
        self.damping = damping
        self.cap = cap
        self.prior = float(prior)
        self.support = []          # variable indices this constraint touches
        self._pos = {}             # variable index -> row in B
        self.B = np.zeros((0, 0))
        self.updates = 0
        self.inflations = 0
        self.relaxations = 0
        self.last_grad = None

    def observe(self, grad):
        """Extend the support to cover whatever the gradient touches.

        New directions seed the diagonal with ``prior * |grad_j|``, not
        zero: B = 0 dominates nothing, and the first step can then leave
        the feasible set outright (helicopter: 9.975e-09 -> 5.070e-01,
        solve aborts). |grad_j| is the right scale -- measured
        |lambda|max / |grad| sits at 2.5-6.6 -- the standard B0 = gamma I
        seeding, per variable.
        """
        g = np.asarray(grad, dtype=float)
        self.last_grad = g
        new = [j for j in np.nonzero(np.abs(g) > 0)[0] if j not in self._pos]
        if not new:
            return
        for j in new:
            self._pos[j] = len(self.support)
            self.support.append(int(j))
        m = len(self.support)
        B = np.zeros((m, m))
        if self.B.size:
            B[:self.B.shape[0], :self.B.shape[1]] = self.B
        if self.prior > 0.0:
            for j in new:
                a = self._pos[j]
                B[a, a] = min(self.prior * abs(float(g[j])), self.cap)
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
        # Powell damping keeps B PSD even where the true log-Hessian is not.
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

        Free validation: the curvature along that direction was under-
        estimated by at least ``2*shortfall/||s||^2``.
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
        """Ease B back when the model turned out pessimistic.

        Otherwise B only ratchets up -- on the helicopter that wrecked the
        sub-problem's conditioning until IPOPT quit at iteration 721.
        Easing back is the same bet the trust region makes when it expands.
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

    The uncached path rebuilds thousands of log-sum-exp expressions per
    iteration and dominates the wall clock. Each exact term splits into a
    FIXED projection ``a_k . d`` plus a constant that follows the iterate,
    so projections are built once and the constants become mutable Params.
    The AGM denominator reuses the same projections with one mutable weight
    per term (``aq . d = sum_i w_i (a_i . d)``). Same device as SLCP's
    SubproblemCache, but easier: no BFGS quadratic, and a cacheable problem
    has no black box hence no trust region. One model per
    ``(minimize_violation, use_slacks)`` phase, since the RHS differs
    structurally between them.
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
        mv = (minimize_violation if minimize_violation in ('l1', 'l1_hard')
              else bool(minimize_violation))
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
        #   True    Phase I MIN-MAX -- one shared t
        #   'l1'    Phase I ELASTIC -- a slack PER constraint, min sum(s_i)
        # Elastic makes infeasibility diagnosable: the rows keeping slack
        # ARE the answer. 'l1_hard' slacks only the inequalities and imposes
        # the equalities exactly -- the tangential half of a composite step.
        # Safe only from a point ON the manifold (the AGM condensation is
        # tight there, so d = 0 is feasible); from an inconsistent start it
        # fails with "sub-problem failed: infeasible" at iteration 1.
        elastic = (minimize_violation in ('l1', 'l1_hard'))
        hard_eq = (minimize_violation == 'l1_hard')
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
            # Pure feasibility: no scaling contest with the objective. The
            # proximity tie-break is measured FROM THE PHASE-I ENTRY POINT
            # via the mutable prox_c offsets, anchoring the whole phase to
            # the seed rather than each step to its own iterate.
            w = getattr(self.options, 'phase1_proximity', 0.0)
            if w:
                m.prox_c = pyo.Param(range(n), mutable=True, initialize=0.0,
                                     within=pyo.Reals)
                prox = w * sum((m.d[j] + m.prox_c[j]) ** 2 for j in range(n))
            else:
                prox = 0.0
            m.obj = pyo.Objective(expr=sum(m.s[i] for i in m.I) + prox,
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
                if hard_eq:
                    # Imposed exactly: no slack, so the step stays on the
                    # linearised manifold.
                    m.cons.add(expr == 0.0)
                elif elastic:
                    # |residual| <= s_i, this row's own slack.
                    m.cons.add(expr <= m.s[i])
                    m.cons.add(-expr <= m.s[i])
                elif minimize_violation:
                    # |residual| <= t, NOT residual == t: a shared t forces
                    # every equality to the SAME residual, making the
                    # sub-problem infeasible on a feasible problem.
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
    def update(self, phase, x_k, tau, radius=None):
        """Re-point a built phase at a new iterate. No symbolic work.

        ``radius`` bounds |d_j| when the caller wants a trust region with
        nothing linearized: Phase II does not (every row exact or
        conservative), Phase I with hard equalities does, because a
        CondensedEquality is TANGENT, not conservative, and an unbounded
        step leaves the manifold far enough that the restoration's travel
        breaks the inequalities the step just fixed.
        """
        m, n = phase.model, self.n
        log_xk = np.log(x_k)

        if not phase.minimize_violation:
            for k, (c, a) in enumerate(self.problem.objective.terms):
                phase.obj_b[k].value = float(math.log(c) + a @ log_xk)
        if phase.slacked:
            m.tau.value = float(tau)
        # seed-referenced Phase-I proximity: re-point the offsets so the
        # tie-break measures distance from the phase's ENTRY point
        prox_c = getattr(m, 'prox_c', None)
        if prox_c is not None:
            anchor = getattr(self.options, '_phase1_anchor_logx', None)
            if anchor is not None:
                c_off = log_xk - np.asarray(anchor, dtype=float)
            else:
                c_off = np.zeros(self.n)
            for j in range(self.n):
                prox_c[j].value = float(c_off[j])

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
        if radius is not None:
            lo = np.maximum(lo, -radius)
            hi = np.minimum(hi, radius)
        for j in range(n):
            ub = None if not np.isfinite(hi[j]) else float(hi[j])
            m.d[j].setlb(float(lo[j]))
            m.d[j].setub(ub)
        # Start from d = 0 but inside the box; the old inline clamp
        # collapsed to 0.0 whenever there was no upper bound.
        seat_step_in_bounds(m)
        if phase.minimize_violation in ('l1', 'l1_hard'):
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
                minimize_violation=False, use_slacks=True, cache=None,
                force_trust=False):
    """Assemble and solve the inner-approximation sub-problem in log space.

    ``minimize_violation`` selects Phase I: minimise the worst violation
    ``t`` -- feasible by construction, no scaling contest, monotone in t.
    Otherwise Phase II: no slacks, no penalty; the optimum is feasible for
    the true problem and no worse than the current point.
    """
    n = problem.n
    cons = problem.constraints
    log_xk = np.log(x_k)

    if cache is not None and cache.usable:
        # The cache does the symbolic construction once and only moves the
        # numbers; bounds are re-pointed in update(). A cacheable problem
        # has no black box, so a trust region applies only when forced.
        phase = cache.update(cache.get(minimize_violation, use_slacks),
                             x_k, tau, radius=radius if force_trust else None)
        try:
            return _solve_and_extract(phase.model, problem, options,
                                      minimize_violation, use_slacks,
                                      phase.obj_expr)
        except RuntimeError:
            # A cache must never change WHETHER something solves. The Param
            # model is not identical to IPOPT at tol = 1e-12 (hydrogen
            # aircraft: cached Phase I burned its whole budget where the
            # inlined build solves in a handful). Rebuild this one fresh.
            pass

    m = pyo.ConcreteModel()
    m.J = pyo.RangeSet(0, n - 1)
    m.I = pyo.RangeSet(0, len(cons) - 1)
    m.d = pyo.Var(m.J, initialize=0.0)
    # Bounds ride on the variable: `lo <= x <= hi` is just
    # `log(lo/x_k) <= d <= log(hi/x_k)`, exact and free.
    if problem.bounds is not None:
        for j, (lo, hi) in enumerate(problem.bounds[:n]):
            if lo is not None and lo > 0:
                m.d[j].setlb(math.log(lo) - log_xk[j])
            if hi is not None and hi > 0:
                m.d[j].setub(math.log(hi) - log_xk[j])
        seat_step_in_bounds(m)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # Same three modes as the cached builder, kept in step deliberately:
    # the cache falls through to this path, so a mode mismatch shows up as
    # 'ConcreteModel object has no attribute s' mid-run.
    _elastic = (minimize_violation in ('l1', 'l1_hard'))
    _hard_eq = (minimize_violation == 'l1_hard')
    if minimize_violation and not _elastic:
        m.t = pyo.Var(initialize=float(_violation(problem, x_k)))
    if (use_slacks and not minimize_violation) or _elastic:
        m.s = pyo.Var(m.I, domain=pyo.NonNegativeReals, initialize=0.0)

    def lse(terms):
        """log sum_k c_k exp(a_k . (d + log x_k)) -- exact, convex."""
        return pyo.log(sum(
            pyo.exp(math.log(c) + sum(a[j] * (m.d[j] + log_xk[j])
                                      for j in range(n)))
            for c, a in terms))

    # --- objective ---------------------------------------------------------
    if _elastic:
        # Pure feasibility, L1: a sparse optimum names the unsatisfiable
        # rows. Proximity is measured FROM THE PHASE-I ENTRY POINT, not the
        # current iterate -- per-iterate proximity let Phase I drag a
        # near-optimal seed 4.8 log units on a paired-equality wind model.
        _w = getattr(options, 'phase1_proximity', 0.0)
        _anchor = getattr(options, '_phase1_anchor_logx', None)
        if _w and _anchor is not None:
            _c = np.log(x_k) - np.asarray(_anchor, dtype=float)
            _prox = _w * sum((m.d[j] + float(_c[j])) ** 2 for j in range(n))
        elif _w:
            _prox = _w * sum(m.d[j] ** 2 for j in range(n))
        else:
            _prox = 0.0
        m.obj = pyo.Objective(expr=sum(m.s[i] for i in m.I) + _prox,
                              sense=pyo.minimize)
        rhs = lambda i: m.s[i]
    elif minimize_violation:
        # Phase I: drive the worst violation down and nothing else.
        m.obj = pyo.Objective(expr=m.t, sense=pyo.minimize)
        rhs = lambda i: m.t
    elif use_slacks:
        rhs = lambda i: m.s[i]
    else:
        # Phase II proper: no slack, no penalty -- the pure inner
        # approximation the guarantees are stated for.
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
                if op == "==" and _hard_eq:
                    m.cons.add(e == 0.0)
                elif op == "==" and minimize_violation:
                    m.cons.add(e <= rhs(i))
                    m.cons.add(-e <= rhs(i))
                elif op == "==":
                    m.cons.add(e == rhs(i))
                else:
                    m.cons.add(e <= rhs(i))
            else:
                m.cons.add(lse(body.terms) <= rhs(i))
        elif isinstance(body, CondensedEquality):
            # Both sides condensed -> a monomial equality, affine in log space.
            # One signed multiplier, and a hyperplane rather than a null space.
            ce, ae = body.condensed(x_k)
            e = math.log(ce) + sum(ae[j] * (m.d[j] + log_xk[j])
                                   for j in range(n))
            if _hard_eq:
                m.cons.add(e == 0.0)
            elif minimize_violation:
                # |residual| <= slack; `e == t` against a shared t forces
                # every equality to the SAME signed residual.
                m.cons.add(e <= rhs(i))
                m.cons.add(-e <= rhs(i))
            else:
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
            # With curvature the row becomes a convex quadratic -- an upper
            # bound wherever B dominates the true log-curvature.
            if curvature is not None and i in curvature:
                curvature[i].observe(gl)
                e = e + curvature[i].quad(m.d)
            if op == "==" and _hard_eq:
                m.cons.add(e == 0.0)
            elif op == "==" and minimize_violation:
                m.cons.add(e <= rhs(i))
                m.cons.add(-e <= rhs(i))
            else:
                m.cons.add(e == rhs(i) if op == "==" else e <= rhs(i))

    # Stay in the positive orthant. The floor and trust region are simple
    # bounds on d, so they ride on the variable -- 3n fewer rows, and a
    # bound is cheaper to IPOPT than a row.
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

    # Trust region only when something was linearized; radius None means
    # the caller has RELEASED the box (see trust_iterations).
    if (has_blackbox or force_trust) and radius is not None:
        for j in range(n):
            tighten(j, lo=-radius, hi=radius)

    # Bounds are tightened in several passes above; only now is the box final.
    seat_step_in_bounds(m)

    return _solve_and_extract(m, problem, options, minimize_violation,
                              use_slacks, obj if not minimize_violation else None)


def _launch_subproblem(opt, m, options):
    """One sub-problem launch. A crashed executable (SPRAL dies on the
    regularized re-factorization after a singular KKT system -- the D8
    crash sentinel) is retried once under MA27 when the build has it; the
    [LC-W313] warning names the event either way."""
    from lcsolver.environment import (IpoptCrashed, crash_fallback_solver,
                                      ipopt_launch)
    ls = (options.ipopt_options or {}).get('linear_solver')
    try:
        with ipopt_launch(ls, recoverable=True):
            return opt.solve(m, tee=options.tee, load_solutions=False)
    except IpoptCrashed:
        fallback = crash_fallback_solver(ls)
        if fallback is None:
            raise
        opt.options['linear_solver'] = fallback
        with ipopt_launch(fallback):
            return opt.solve(m, tee=options.tee, load_solutions=False)


def _solve_and_extract(m, problem, options, minimize_violation, use_slacks,
                       obj):
    """Solve an assembled sub-problem and read off the step and multipliers.

    Shared by the rebuild path and the cached one, so both report failures
    identically and both use the same multiplier sign convention.
    """
    n = problem.n
    cons = problem.constraints

    from lcsolver.environment import ipopt_solver_factory
    opt = ipopt_solver_factory()
    if not opt.available(exception_flag=False):
        raise SolverUnavailable(
            "SIA needs IPOPT to solve its sub-problems; no usable installation "
            "was found. Run `lcsolver-install-solvers`; see docs/ipopt.rst.")
    for k, v in (options.ipopt_options or {}).items():
        opt.options[k] = v
    results = _launch_subproblem(opt, m, options)
    tc = str(results.solver.termination_condition)
    if tc not in ("optimal", "locallyOptimal", "feasible"):
        from lcsolver.environment import linear_solver_failure_note
        raise RuntimeError(
            f"the SIA sub-problem failed: {tc}"
            + linear_solver_failure_note(
                (options.ipopt_options or {}).get('linear_solver')))
    m.solutions.load_from(results)

    d = np.array([pyo.value(m.d[j]) for j in range(n)])
    _elastic = (minimize_violation in ('l1', 'l1_hard'))
    s = (np.array([pyo.value(m.s[i]) for i in range(len(cons))])
         if ((use_slacks and not minimize_violation) or _elastic)
         else np.zeros(len(cons)))

    # An inequality multiplier is non-negative, so magnitude is wanted; an
    # equality multiplier carries a sign the stationarity sum needs (Hoburg:
    # dropping equality signs pins the residual at 2.5). SLCP takes abs()
    # because it only feeds a merit function; a KKT certificate cannot.
    mults = np.zeros(len(cons))
    for i in range(len(cons)):
        try:
            lam = m.dual.get(m.cons[i + 1], 0.0) or 0.0
        except Exception:
            lam = 0.0
        # Pyomo/IPOPT report the equality dual with the opposite sign to the
        # Lagrangian convention, so it is negated -- verified against a
        # least-squares multiplier fit at a converged Hoburg point.
        mults[i] = (-float(lam) if cons[i].operator == '=='
                    else abs(float(lam)))
    if _elastic:
        # The scalar is total infeasibility sum(s_i): zero means feasible,
        # and `s` says exactly WHERE the rest is.
        return d, s, mults, float(sum(s))
    if minimize_violation:
        return d, s, mults, float(pyo.value(m.t))
    return d, s, mults, float(pyo.value(obj))



def _blocking_constraints(problem, x, mults=None, k=8):
    """Which constraints are stopping Phase I, and which variables they touch.

    Reports residual (log g_i) and the last sub-problem's multiplier: a
    large residual with a large multiplier is an active blocker; with a
    zero multiplier it follows once the blockers move. Constraints carry no
    names, so each is identified by the variables its exponents touch.
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
        # CondensedEquality/PosynomialRatio hold p and q, and BOTH sides
        # matter -- reading one reported "vars: -" for equality rows.
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




def _posy_loggrad(posy, x, n):
    """``(log p, d log p / d log x)`` analytically: the gradient is the
    value-weighted mean of the exponent vectors."""
    vals = np.array([c * np.prod(x ** a) for c, a in posy.terms])
    tot = vals.sum()
    if tot <= 0:
        return -700.0, np.zeros(n)
    w = vals / tot
    g = np.zeros(n)
    for wk, (_c, a) in zip(w, posy.terms):
        g += wk * np.asarray(a, dtype=float)
    return float(np.log(tot)), g


def _con_loggrad(con, x, n):
    """``(log g, d log g / d log x)`` for any constraint body.

    Black-box bodies are first-class: without this case restore_equalities
    crashed on the first black-box equality to reach Phase I (Hoburg UAV
    with the ROM tau coupling).
    """
    b = con.body
    if getattr(b, 'p', None) is not None:          # ratio / condensed equality
        lp, gp = _posy_loggrad(b.p, x, n)
        lq, gq = _posy_loggrad(b.q, x, n)
        return lp - lq, gp - gq
    if not hasattr(b, 'terms'):                    # black box: value + gradient
        v = float(b(x))
        if v <= 0:
            return -700.0, np.zeros(n)
        return float(np.log(v)), np.asarray(b.log_grad(x), dtype=float)
    return _posy_loggrad(b, x, n)


def restore_equalities(problem, x, iters=12, tol=1e-9, verbose=False,
                       skip_rows=None, freeze_cols=None):
    """Least-norm Gauss-Newton onto the equality manifold -- the NORMAL step.

    Works on the TRUE residuals, never the condensation, whose feasible set
    can be empty where the true equality is satisfiable (1173-variable
    aircraft, 802 equalities: 5e-12 in eleven steps while every Phase I
    formulation stalls). Least-norm keeps the repair close to the start:
    useless standalone (inequalities stay put), exactly right as the
    normal half of a composite step.
    """
    n = problem.n
    eq = [i for i, c in enumerate(problem.constraints) if c.operator == '=='
          and (skip_rows is None or i not in skip_rows)]
    if not eq:
        return np.array(x, dtype=float), 0.0
    frz = (np.asarray(sorted(freeze_cols), dtype=int)
           if freeze_cols else None)
    lo = np.array([b[0] if b and b[0] else 1e-30
                   for b in (problem.bounds or [(None, None)] * n)])
    hi = np.array([b[1] if b and b[1] else 1e30
                   for b in (problem.bounds or [(None, None)] * n)])
    x = np.array(x, dtype=float).copy()

    def _res(z):
        r = np.zeros(len(eq)); J = np.zeros((len(eq), n))
        for k, i in enumerate(eq):
            r[k], J[k] = _con_loggrad(problem.constraints[i], z, n)
        if frz is not None:
            J[:, frz] = 0.0        # held columns (e.g. grey-box outputs the
            #                        closure pass has already set exactly)
        return r, J

    for it in range(iters):
        r, J = _res(x)
        nrm = float(np.max(np.abs(r)))
        if verbose:
            print(f"    restore {it:2d}  max|h| = {nrm:.3e}")
        if nrm <= tol:
            break
        # lstsq, not pinv @ r: same least-norm solution for the under-
        # determined system without forming the pseudo-inverse.
        d = np.linalg.lstsq(J, -r, rcond=1e-10)[0]
        big = float(np.max(np.abs(d)))
        if big > 1.0:
            d *= 1.0 / big
        for alpha in (1.0, 0.5, 0.25, 0.1):
            xn = np.clip(x * np.exp(alpha * d), lo * 1.000001, hi * 0.999999)
            if float(np.max(np.abs(_res(xn)[0]))) < nrm:
                x = xn
                break
        else:
            break
    return x, float(np.max(np.abs(_res(x)[0])))


def restore_composite_bb(problem, x, iters=12, tol=1e-9, passes=3):
    """Equality restore for BLACK-BOX problems: grey-box rows by exact
    closure, structured pins by Gauss-Newton, alternated.

    A grey-box equality ``bb(inputs) / x[out] == 1`` restores exactly by
    ``x[out] *= body(x)``; the structured Newton then skips those rows with
    their output columns frozen. Alternation converges geometrically
    (airfoil model: 3 passes). Exists because the composite restore was
    gated ``not has_blackbox``, and the silent skip left box rows violated
    2-100x while reported as success.
    """
    x0 = np.array(x, dtype=float).copy()
    x = x0.copy()
    gb = [(i, c.body.out_index) for i, c in enumerate(problem.constraints)
          if c.operator == '==' and getattr(c.body, 'out_index', None)
          is not None]
    if not gb:
        return restore_equalities(problem, x, iters=iters, tol=tol)
    skip = frozenset(i for i, _ in gb)
    frz = frozenset(j for _, j in gb)
    h_gb = math.inf
    try:
        for _ in range(max(1, int(passes))):
            for i, j in gb:                   # exact closure, one eval each
                x[j] *= float(problem.constraints[i].body(x))
            x, h_st = restore_equalities(problem, x, iters=iters, tol=tol,
                                         skip_rows=skip, freeze_cols=frz)
            h_gb = max(abs(math.log(float(problem.constraints[i].body(x))))
                       for i, _ in gb)
            if h_gb <= tol:
                break
        return x, max(h_gb, h_st)
    except Exception:
        # The restore is a POLISH: it re-evaluates the analysis at shifted
        # points, and an analysis code can fail there even beside a point
        # it just solved (MSES: restart divergence at a near-identical
        # design killed a whole camber run from inside this step).  A
        # failed polish returns the un-polished point, never an exception
        return x0, math.inf


def _phase1_l1(problem, x, options, has_blackbox, cache=None):
    """Elastic Phase I: a slack per constraint, minimising their SUM.

    ``min sum(s_i)  s.t.  log g_i(x) <= s_i,  s_i >= 0``. The sparse L1
    optimum makes infeasibility diagnosable -- the rows keeping slack are
    an approximate irreducible inconsistent subsystem, where min-max shows
    a dozen rows at one identical residual. Returns ``(x, iterations,
    feasible, slacks, multipliers)``; a slack above tolerance names a row
    that must be relaxed for the model to close.
    """
    # anchor the proximity tie-break to the entry point (see the elastic
    # objective builders)
    options._phase1_anchor_logx = np.log(np.maximum(x, 1e-300))
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
        d = np.clip(d, -60.0, 60.0)   # overflow guard; see the main-loop clamp
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



def _violation_ineq(problem, x):
    """Worst ``log g_i(x)`` over the INEQUALITIES only."""
    return max((_log_g(c, x) for c in problem.constraints
                if c.operator != '=='), default=0.0)



def convergence_report(problem, res, options=None, k=8):
    """Explain what a solve did, and if it stopped short, what held it back.

    Names WHICH variables carry the residual -- nearly always a handful, in
    a sub-model that sizes nothing in the converged design (E175: 325
    iterations dragging stationarity through twelve flat fuselage-bending
    variables; the design was converged, the measure was not).
    """
    options = options or SIAOptions()
    lines = []
    x = np.asarray(res.x, dtype=float)
    stat = float(getattr(res, 'stationarity', float('nan')))
    viol = float(getattr(res, 'max_violation', float('nan')))
    comp = float(getattr(res, 'complementarity', float('nan')))
    tols = (('stationarity', stat, options.stationarity_tolerance),
            ('feasibility', viol, options.feasibility_tolerance),
            ('complementarity', comp, options.complementarity_tolerance))

    if res.converged:
        lines.append(f"CONVERGED in {res.iterations} iterations. "
                     f"objective = {res.objective:.6g}")
    else:
        binding = [(n, v, t) for n, v, t in tols
                   if v == v and v > t]          # v == v filters NaN
        lines.append(f"STOPPED after {res.iterations} iterations without "
                     f"meeting all three KKT tolerances. "
                     f"objective = {res.objective:.6g}")
        for n, v, t in binding:
            lines.append(f"  binding: {n} = {v:.3e}, tolerance {t:.1e} "
                         f"({v / t:.1f}x)")
        for n, v, t in tols:
            if (n, v, t) not in binding and v == v:
                lines.append(f"  met:     {n} = {v:.3e} <= {t:.1e}")

    # A run whose objective is stable to eight figures has converged in
    # every sense an engineer cares about, whatever the KKT residual says.
    objs = list(getattr(res, 'objectives', None) or [])
    if len(objs) >= 20:
        tail = objs[-20:]
        spread = (max(tail) - min(tail)) / max(abs(tail[-1]), 1e-300)
        lines.append(f"  objective over the last 20 iterations: relative "
                     f"spread {spread:.2e}"
                     + ("  -- STABLE; the design is settled and the residual "
                        "below is a measurement issue, not a design one"
                        if spread < 1e-6 else ""))

    # Where the stationarity residual actually lives.
    mults = getattr(res, 'multipliers', None)
    if mults is not None and not res.converged:
        g = np.asarray(problem.objective.log_grad(x), dtype=float)
        for i, con in enumerate(problem.constraints):
            lam = float(mults[i])
            if lam != 0.0:
                g = g + lam * np.asarray(con.body.log_grad(x), dtype=float)
        lo = np.full(problem.n, options.x_min)
        hi = [None] * problem.n
        if problem.bounds is not None:
            for j, pair in enumerate(problem.bounds[:problem.n]):
                if pair:
                    if pair[0] and pair[0] > options.x_min:
                        lo[j] = pair[0]
                    hi[j] = pair[1]
        for j in range(problem.n):
            if x[j] <= lo[j] * (1 + 1e-6):
                g[j] = min(g[j], 0.0)
            if hi[j] and x[j] >= hi[j] * (1 - 1e-6):
                g[j] = max(g[j], 0.0)
        names = getattr(problem, 'names', None) or [f'x[{j}]'
                                                    for j in range(problem.n)]
        order = np.argsort(-np.abs(g))[:k]
        lines.append(f"  the stationarity residual is carried by these "
                     f"variables:")
        for j in order:
            if abs(g[j]) < 1e-14:
                break
            lines.append(f"    {names[j]:<42s} dL/dlog x = {g[j]:+.3e}   "
                         f"x = {x[j]:.4e}")
        # Group by prefix: a residual confined to one sub-model is the
        # signature of a degenerate block rather than a genuine stall.
        pref = {}
        for j in order:
            key = str(names[j]).split('_')[0]
            pref[key] = pref.get(key, 0.0) + abs(g[j])
        if len(pref) == 1:
            only = next(iter(pref))
            lines.append(f"  ALL of it sits in '{only}*'. A residual confined "
                         f"to one sub-model usually means that block is flat "
                         f"-- it sizes nothing in the converged design, so its "
                         f"gradient is near zero in every direction and the "
                         f"last digits never settle. Check whether those "
                         f"variables are doing any work before treating this "
                         f"as a failure.")

    # Variables resting on the artificial positivity floor.
    floor = [j for j in range(problem.n) if x[j] <= options.x_min * 1e3]
    if floor:
        names = getattr(problem, 'names', None) or [f'x[{j}]'
                                                    for j in range(problem.n)]
        lines.append(f"  WARNING: {len(floor)} variable(s) rest on the "
                     f"positivity floor ({options.x_min:g}). The KKT "
                     f"certificate is honest -- the floor is an active bound "
                     f"and holds the gradient -- but the floor is a numerical "
                     f"device, not a modelling statement, so the answer there "
                     f"is an artifact of it:")
        for j in floor[:k]:
            lines.append(f"    {names[j]:<42s} x = {x[j]:.3e}")
    return '\n'.join(lines)


def _phase1_composite(problem, x, options, has_blackbox, cache=None):
    """Composite-step Phase I: restore the equalities, then move tangentially.

      normal      Gauss-Newton onto ``h(x) = 0``, on the TRUE residuals.
      tangential  one elastic-L1 sub-problem with the equalities imposed
                  EXACTLY, so the step stays on the linearised manifold.

    They alternate because the tangential step leaves the true manifold at
    second order. Neither half works alone: slacked equalities wander
    (stalls on the 1173-variable aircraft at every budget), hard equalities
    from an INCONSISTENT point are infeasible at iteration 1 -- but from a
    RESTORED point d = 0 is always available. Falls back to plain L1 when a
    hard sub-problem fails at the minimum radius: the manifold and trust
    region do not intersect, and slack is the honest response.
    """
    tol = options.feasibility_tolerance
    radius = options.trust_radius
    slacks = np.zeros(len(problem.constraints))
    mults = None
    it = 0
    n_eq = sum(1 for c in problem.constraints if c.operator == '==')
    if n_eq == 0:                      # nothing to restore; plain elastic L1
        return _phase1_l1(problem, x, options, has_blackbox, cache=cache)

    def _restore(z):
        fn = restore_composite_bb if has_blackbox else restore_equalities
        return fn(problem, z,
                  iters=options.phase1_restore_iterations,
                  tol=min(tol, 1e-10))

    if _violation(problem, x) <= tol:
        # Already feasible: do not touch it. Restoring first lands on a
        # DIFFERENT manifold point and picks another local optimum
        # (SPaircraft 95120 -> 95560 lbf, simpleac 4536 -> 6485).
        return x, 0, True, slacks, mults

    x, heq = _restore(x)                      # start on the manifold
    # anchor the proximity tie-break to THIS point for the whole phase
    options._phase1_anchor_logx = np.log(np.maximum(x, 1e-300))
    for it in range(1, options.phase1_max_iterations + 1):
        viol = _violation(problem, x)
        if viol <= tol:
            return x, it - 1, True, slacks, mults

        # --- tangential step -------------------------------------------------
        mode = 'l1_hard' if heq <= max(tol, 1e-7) else 'l1'
        try:
            d, slacks, mults, total = _subproblem(
                problem, x, 0.0, radius, options, has_blackbox,
                minimize_violation=mode, cache=cache, force_trust=True)
        except RuntimeError:
            if radius > options.trust_min:
                radius = max(options.trust_min, radius * options.trust_shrink)
                continue
            if mode == 'l1_hard':      # manifold and trust region miss: slack
                return _phase1_l1(problem, x, options, has_blackbox,
                                  cache=cache)
            return x, it, False, slacks, mults

        # --- normal step, closing the same composite step --------------------
        # The pair is accepted or rejected TOGETHER: a tangential step
        # judged alone reports improvement the following restoration undoes,
        # so the trust radius must own the drift.
        x_try, heq_try = _restore(x * np.exp(np.clip(d, -60.0, 60.0)))  # overflow guard
        new_viol = _violation(problem, x_try)
        if options.verbose:
            nz = int(np.sum(slacks > tol))
            print(f"  phase1-comp {it:3d}  |h| {heq_try:.1e}  max log g "
                  f"{viol:+.3e} -> {new_viol:+.3e}  sum(s) = {total:.3e}  "
                  f"({nz} slack, {mode}, r = {radius:.2g})")
        if new_viol >= viol:
            radius *= options.trust_shrink
            if radius < options.trust_min:
                # No improvement at any length: the elastic L1 answers
                # whether the rest is irreducible, and names the rows.
                return _phase1_l1(problem, x, options, has_blackbox,
                                  cache=cache)
            continue
        x, heq = x_try, heq_try
        radius = min(getattr(options, 'phase1_trust_max', options.trust_max),
                     radius * options.trust_expand)

    return x, it, _violation(problem, x) <= tol, slacks, mults


def explain_infeasibility(problem, x, options=None, has_blackbox=False,
                          cache=None, k=12):
    """Run the elastic Phase I and report which rows cannot be satisfied.

    This is the answer to "why is my model infeasible". It re-solves the
    feasibility problem in its L1 form and lists the constraints whose slack
    stays positive -- the ones that must be relaxed -- together with the
    variables each touches.
    """
    # With no IPOPT every elastic solve fails, nothing moves, and the report
    # concludes "feasible at the initial guess" -- the most damaging wrong
    # answer this function could give.
    from lcsolver.environment import ipopt_available
    if not ipopt_available():
        raise SolverUnavailable(
            'the feasibility check solves its elastic sub-problems with '
            'IPOPT, and no usable installation was found. Run '
            '`lcsolver-install-solvers`; see docs/ipopt.rst.')

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

    ``min t  s.t.  log g_i(x) <= t``: always feasible, a single-variable
    objective with no scaling contest (the penalty version once took a step
    of e^59), and t decreases monotonically under the same conservative
    representations. Returns ``(x, iterations, feasible)``. The margin asks
    for strict feasibility so Phase II starts inside the set.
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
            # The min-max sub-problem is always feasible on paper, so a
            # failure means the trust region: shrink, don't give up (giving
            # up read as "phase 1 could not find a feasible point after 1
            # iterations" on a solvable problem).
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
        d = np.clip(d, -60.0, 60.0)   # overflow guard; see the main-loop clamp
        x_new = x * np.exp(d)
        new_viol = _violation(problem, x_new)

        if new_viol > viol:
            # Conservatism says the violation cannot increase; in finite
            # precision it sometimes does. A trust-region signal, not a
            # reason to accept -- this used to be checked only for black
            # boxes, and Phase I could wander uphill on a pure SP.
            if radius > options.trust_min:
                radius = max(options.trust_min,
                             radius * options.trust_shrink)
                if options.verbose:
                    print(f"  phase1 {it:3d}  step worsened "
                          f"{viol:+.3e} -> {new_viol:+.3e}, "
                          f"radius -> {radius:.3g}")
                continue
            # At trust_min the approximation is as good as it gets: ACCEPT
            # the step. Rejecting turned a converging SPaircraft run into
            # "could not find a feasible point after 15 iterations" -- a
            # worsening step at the smallest radius is numerical noise.
            if options.verbose:
                print(f"  phase1 {it:3d}  step worsened at trust_min, "
                      f"accepting {viol:+.3e} -> {new_viol:+.3e}")

        if options.verbose:
            print(f"  phase1 {it:3d}  max log g: {viol:+.3e} -> "
                  f"{new_viol:+.3e}   (model t = {t:+.3e})")
        if abs(new_viol - viol) <= 1e-14 * max(1.0, abs(viol)):
            # Stalled AT THIS RADIUS only; a smaller region often moves
            # again. Only a stall that survives trust_min is real.
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
        # Expand on any good step; reducing the violation is exactly when a
        # longer one is worth trying.
        radius = min(options.trust_max, radius * options.trust_expand)
    return x, it, _violation(problem, x) <= 0.0, last_mults


def _relative_change_converged(problem, res, options, k):
    """The relative-change stopping rule; see ``objective_reltol``.

    Compares consecutive ACCEPTED iterates, so a rejected step never
    triggers it. The previous point is kept here because ``res.history``
    also records the seed and Phase I exit. Fills in the result and
    returns True when it fires.
    """
    o_tol, v_tol = options.objective_reltol, options.variable_reltol
    if o_tol is None and v_tol is None:
        return False
    x, f = res.history[-1], res.objectives[-1]
    prev = getattr(res, '_relative_change_prev', None)
    res._relative_change_prev = (x.copy(), f)
    if prev is None:
        return False
    x_prev, f_prev = prev
    df = abs(f / f_prev - 1.0) if f_prev != 0 else abs(f - f_prev)
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = np.max(np.abs(np.asarray(x) / np.asarray(x_prev) - 1.0))
    if options.verbose:
        # progress toward THIS stopping rule, same quantities it tests
        print("       rel: |df/f|=%.2e%s  max|dx/x|=%.2e%s" % (
            df, '' if o_tol is None else ' (tol %g)' % o_tol,
            dx, '' if v_tol is None else ' (tol %g)' % v_tol))
    fired = []
    if o_tol is not None:
        if df > o_tol:
            return False
        fired.append(f"objective {df:.2e} <= {o_tol:g}")
    if v_tol is not None:
        if not np.isfinite(dx) or dx > v_tol:
            return False
        fired.append(f"variables {dx:.2e} <= {v_tol:g}")
    # A point that moved nothing but is not a design is not an answer.
    # A declared analysis noise floors the test: sub-noise violation is
    # indistinguishable from feasible
    _eff = max(options.feasibility_tolerance,
               float(getattr(options, 'analysis_noise', 0.0) or 0.0))
    _v = _violation(problem, x)
    if _v > _eff:
        return False
    res.converged = True
    notes = ""
    if _v > options.feasibility_tolerance:
        notes += ("; feasible to the declared analysis noise "
                  f"({_v:.1e}), not to feasibility_tolerance")
    if getattr(res, '_rel_step_binding', False):
        notes += ("; the trust radius was binding at the stop, so the "
                  "step size was constrained, not chosen")
    res.status = ("converged: relative change between accepted iterates "
                  "within tolerance (" + ", ".join(fired)
                  + "); no KKT certificate" + notes)
    res.x, res.objective = x, problem.objective_value(x)
    res.iterations = k + 1
    if options.verbose:
        print(f"  itr {k + 1:3d}  STOP on relative change: "
              + ", ".join(fired))
    return True


def solve_sia(problem: Problem, x0, options: SIAOptions = None) -> SIAResult:
    """Solve a signomial program by sequential inner approximation."""
    # Checked first: without IPOPT every sub-problem fails quietly and SIA
    # returns the objective at the initial guess with converged=False -- a
    # plausible number produced by no optimization at all.
    from lcsolver.environment import ipopt_available
    if not ipopt_available():
        raise SolverUnavailable(
            'SIA solves every sub-problem with IPOPT, and no usable '
            'installation was found. Run `lcsolver-install-solvers`; see '
            'docs/ipopt.rst.')

    options = options or SIAOptions()
    x = np.asarray(x0, dtype=float).copy()
    if np.any(x <= 0):
        raise ValueError("SIA works in log space, so x0 must be strictly positive")

    # Project the start into its box: _violation reads the constraints, not
    # the bounds, so a bound-violating start can look feasible while the
    # sub-problem cannot move, stopping the run at iteration 0. Free when
    # already inside; what makes a model usable after propagate_bounds.
    if problem.bounds is not None:
        for j, pair in enumerate(problem.bounds[:problem.n]):
            lo, hi = pair or (None, None)
            if lo is not None and lo > 0 and x[j] < lo:
                x[j] = lo
            if hi is not None and hi > 0 and x[j] > hi:
                x[j] = hi

    n_exact, n_cons, n_lin = classify(problem)
    has_blackbox = n_lin > 0
    n_eq_p2 = sum(1 for c in problem.constraints if c.operator == '==')

    res = SIAResult()
    res.conservative = not has_blackbox
    res.history.append(x.copy())

    tau = options.tau0
    radius = options.trust_radius
    mults = np.zeros(len(problem.constraints))

    # Build the sub-problem once per phase and re-point it; rebuilds for
    # black-box bodies, which re-linearize every iteration anyway.
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
    # The conservative guarantees hold FROM A FEASIBLE POINT: get feasible
    # first, then optimize with no penalty at all.
    _recovering = False
    _recover_budget = 0
    _recover_best = np.inf
    # guards, stall counters and recovery measure feasibility against this:
    # a declared analysis noise floors it, since sub-noise violation is
    # indistinguishable from feasible.  The KKT certificate is unaffected
    _eff_feas = max(options.feasibility_tolerance,
                    float(getattr(options, 'analysis_noise', 0.0) or 0.0))
    _v0 = _violation(problem, x)
    if options.phase1 and _v0 > options.feasibility_tolerance:
        if options.verbose:
            print(f"  phase 1: initial point infeasible "
                  f"(max log g = {_v0:+.3e}); searching")
        _method = getattr(options, 'phase1_method', 'composite')
        if _method == 'composite':
            (x, res.phase1_iterations, feasible,
             _p1_slacks, p1_mults) = _phase1_composite(
                problem, x, options, has_blackbox, cache=cache)
            res.phase1_mode = 'composite'
        elif _method == 'l1':
            (x, res.phase1_iterations, feasible,
             _p1_slacks, p1_mults) = _phase1_l1(
                problem, x, options, has_blackbox, cache=cache)
            res.phase1_mode = 'l1'
        else:
            x, res.phase1_iterations, feasible, p1_mults = _phase1(
                problem, x, options, has_blackbox, cache=cache)
            res.phase1_mode = 'minmax'

        res.history.append(x.copy())
        res.phase1_feasible = feasible
        if options.verbose:
            print(f"  phase 1: {res.phase1_iterations} iterations, "
                  f"{'FEASIBLE' if feasible else 'still infeasible'}, "
                  f"max log g = {_violation(problem, x):+.3e}")
        if not feasible:
            # Try the ELASTIC form before reporting: it often succeeds where
            # min-max cannot (one stubborn row holds the whole vector up),
            # and when it fails it names the rows whose slack cannot close.
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
            # Hand off to the PENALTY path rather than giving up. It exists
            # for TOLERANCE SHORTFALLS (measured: Phase I stopped at
            # max log g = 2.4e-6 against 1e-6 on a problem that solves),
            # not genuine infeasibility -- with worst log g above 1.0 there
            # is nothing to iterate toward, so the run terminates with the
            # elastic report. Below that, a LOCAL Phase-I minimum is
            # routinely recoverable by the penalty path's objective
            # pressure, and feasibility is ENFORCED at exit: a run that
            # ends infeasible reports infeasible, never a design.
            _near = _violation(problem, x) <= 1.0
            if options.phase1_penalty_fallback and _near:
                if options.verbose:
                    print("  phase 1 fell short; running the penalty path "
                          "as a FEASIBILITY FINDER (bounded budget, must "
                          "deliver a feasible point or the run terminates)")
                res.phase1_mode = 'penalty-recovery'
                use_slacks = True
                _recovering = True
                _recover_budget = int(getattr(
                    options, 'feasibility_recovery_iterations', 60))
                # 60, not 30: the wind missions SP finder was terminated
                # three iterations short of tolerance at 30
            else:
                return res
        else:
            # Feasible: slacks off, penalty off -- tau only bought feasibility.
            use_slacks = False
            if options.verbose:
                print("  phase 1 complete -> entering phase 2 "
                      f"(max log g = {_violation(problem, x):+.3e})")
    else:
        use_slacks = not options.phase1
        if options.verbose:
            print(f"  phase 1 skipped: initial point feasible "
                  f"(max log g = {_v0:+.3e}) -> entering phase 2")

    # One curvature model per linearized constraint. Nothing is created when
    # there is nothing to linearize, so a structured problem is untouched.
    curvature = None
    if options.curvature and has_blackbox:
        curvature = {i: Curvature(problem.n,
                                  prior=getattr(options, 'curvature_prior', 0.0))
                     for i, con in enumerate(problem.constraints)
                     if isinstance(con.body, Signomial)
                     and not isinstance(con.body, (Posynomial, PosynomialRatio))}

    curv_grad, curv_pred, lin_here = {}, {}, {}
    _curv_trained = False
    _bb_g1 = 0.0                    # max_i ||grad log g_i||_1 over black-box rows
    if curvature:
        for _i in curvature:
            try:
                _g = np.asarray(problem.constraints[_i].body.log_grad(x), dtype=float)
                _bb_g1 = max(_bb_g1, float(np.abs(_g).sum()))
            except Exception:
                pass
    # Smallest radius REJECTED since the last accepted step: stops the
    # widen/shrink branches cycling deterministically (4-variable ROM
    # section: 300 iterations, 27 distinct black-box evaluations).
    _reject_radius = np.inf
    # The linearized block -- what the trust region and ratio test exist for.
    lin_idx = [i for i, con in enumerate(problem.constraints)
               if isinstance(con.body, Signomial)
               and not isinstance(con.body, (Posynomial, PosynomialRatio))]
    x_prev_for_curv = x.copy()

    _trust_forced = False       # a step was rejected with the box off -> re-arm
    _zz_prev = None             # last ACCEPTED iterate, for zigzag detection
    _zz_cool = 0                # expansion hold-off after a zigzag detection
    _zz_hits = 0                # detections, reported on the result
    _zz_ceiling = np.inf        # radius cap ratcheted down by REPEAT hits
    _fg_trips = 0               # times the feasibility guard condition HELD
                                # (counted whether or not it was applied)
    # TANGENT EQUALITIES get the linearized treatment: a CondensedEquality
    # is tangent -- neither inner nor outer -- so a Phase II step CAN
    # violate it, recoverably. Filter-plus-restoration, no exchange rate.
    _has_tangent_eq = any(isinstance(c.body, CondensedEquality)
                          for c in problem.constraints)
    _filter = (Filter(options.filter_gamma_h, options.filter_gamma_f)
               if (getattr(options, 'filter_acceptance', False)
                   and (has_blackbox or _has_tangent_eq))
               else None)
    _restorations = 0
    _restore_targets = []       # log-points restoration has returned; a
                                # repeat is a restoration loop, refused
    _infeas_best = np.inf       # best violation seen since going infeasible
    _infeas_stall = 0           # iterations infeasible without improving it
    _subfail = 0                # consecutive unsolvable/unreachable sub-problems
    # SQP-style safeguards: remember the best FEASIBLE iterate. Restoration
    # first; if that fails, fall back to the incumbent in strict
    # feasible-step mode (guard factor 1, not 10), and never RETURN an
    # infeasible excursion while a feasible incumbent exists.
    _x_incumbent = None
    _f_incumbent = np.inf
    # Incumbent/exit feasibility on black-box problems is judged at the
    # composite restore's ACHIEVABLE residual, not the raw tolerance --
    # judging at 1e-8 once returned the iteration-0 point byte-identically.
    _inc_tol = (max(options.feasibility_tolerance,
                    float(getattr(options, 'bb_feas_tol', 1e-5) or 0.0))
                if has_blackbox else options.feasibility_tolerance)
    # strict_feasible=True starts in feasible-step mode: small honest steps
    # instead of excursions -- wins when true-constraint nonlinearity
    # (e.g. clmax(shape)) punishes every long step.
    _strict_feas = bool(getattr(options, 'strict_feasible', False))
    # (_has_tangent_eq hoisted above the filter init; see there.)
    for k in range(options.max_iterations):
        _tgt = float(getattr(options, 'blackbox_step_target', 0.0) or 0.0)
        _radius_eff = radius
        _wi = int(getattr(options, 'blackbox_warm_iters', 0) or 0)
        _bb_growth = float(getattr(options, 'blackbox_release_growth', 0.0)
                           or 0.0)
        if has_blackbox and _tgt > 0.0 and _bb_g1 > 0.0:
            if k < _wi:
                _radius_eff = min(radius, _tgt / _bb_g1)
            elif _bb_growth > 1.0:
                # GRADUATED RELEASE: past warm-up the bb cap tapers open
                # ~30%/iteration so each bolder step is survivable and the
                # curvature models learn from steps that stand (a hard
                # release measured a radius-collapse limit cycle).
                grown = min(_tgt * _bb_growth ** (k - _wi),
                            float(getattr(options, 'blackbox_release_max',
                                          0.45) or 0.45))
                _radius_eff = min(radius, grown / _bb_g1)
        _ti = getattr(options, 'trust_iterations', None)
        _released = (_ti is not None and k >= int(_ti) and not _trust_forced)
        if _released:
            # Box off; the graduated bb cap (if configured) is the only
            # leash until it grows past relevance.
            _radius_eff = None
            if (has_blackbox and _tgt > 0.0 and _bb_g1 > 0.0
                    and _bb_growth > 1.0):
                _radius_eff = min(_tgt * _bb_growth ** (k - _wi),
                                  float(getattr(options,
                                        'blackbox_release_max', 0.45)
                                        or 0.45)) / _bb_g1
        try:
            d, s, mults, model_obj = _subproblem(
                problem, x, tau, _radius_eff, options, has_blackbox, curvature=curvature,
                use_slacks=use_slacks, cache=cache,
                force_trust=_has_tangent_eq or _trust_forced)
            # NOTE: the filter cannot size steps, only accept them --
            # without the cap tangent-equality steps thrashed +-0.5 log
            # units then froze 35 iterations in filter rejection. Filter
            # for recoverable excursions, trust region for scale.
        except RuntimeError as exc:
            # An unsolvable sub-problem `continue`s past the restoration
            # trigger, so count it here (6t+6c: "unreachable / failed"
            # cycled 1500 iterations with restorations = 0).
            _subfail += 1
            if _subfail >= max(int(getattr(options, 'restoration_patience', 0)
                                   or 0), 1):
                x_r, _restorations, _ok = _restore(
                    problem, x, options, has_blackbox, cache, _restorations, k,
                    f'sub-problem unsolvable for {_subfail} iterations', targets=_restore_targets)
                _subfail = 0
                if _ok:
                    x = x_r
                    radius = options.trust_radius
                    _reject_radius = np.inf
                    if _filter is not None:
                        _filter.entries.clear()
                        _zz_prev = None          # new basin: stale geometry
                    continue
            # An INFEASIBLE sub-problem means the trust region is too tight
            # for the relaxed set to be reachable -- widen, don't shrink.
            if (has_blackbox and "infeasible" in str(exc).lower()
                    and radius < options.trust_max):
                grown = min(options.trust_max, radius * options.trust_expand)
                if grown >= _reject_radius:
                    # Bracketed: too small to be reachable, too large to be
                    # believed. Stop rather than cycle.
                    x_r, _restorations, _ok = _restore(
                        problem, x, options, has_blackbox, cache,
                        _restorations, k, 'the trust region bracketed',
                        targets=_restore_targets)
                    if _ok:
                        x = x_r
                        radius = options.trust_radius
                        _reject_radius = np.inf
                        if _filter is not None:
                            _filter.entries.clear()
                            _zz_prev = None          # new basin: stale geometry
                        continue
                    if _x_incumbent is not None and not _strict_feas:
                        x = _x_incumbent.copy()
                        _strict_feas = True
                        radius = max(options.trust_min,
                                     0.1 * options.trust_radius)
                        _reject_radius = np.inf
                        if _filter is not None:
                            _filter.entries.clear()
                            _zz_prev = None
                        if options.verbose:
                            print(f"  itr {k + 1:3d}  bracketed + "
                                  "restoration failed; falling back to the "
                                  "feasible incumbent, strict mode ON")
                        continue
                    res.status = (
                        f"trust region bracketed at iteration {k + 1}: the "
                        f"sub-problem is infeasible at radius {radius:.3g} but the "
                        f"step at {_reject_radius:.3g} was rejected; the linearized "
                        "constraints are not modelling the problem")
                    res.x, res.objective = x, problem.objective_value(x)
                    res.iterations = k + 1
                    break
                radius = grown
                if options.verbose:
                    print(f"  itr {k + 1:3d}  sub-problem unreachable, "
                          f"widening radius -> {radius:.3g}")
                continue
            # Any OTHER failure is numerical (badly-scaled iterate): shrink
            # and re-form the model; only give up once the radius collapses.
            if has_blackbox and radius > options.trust_min:
                radius = max(options.trust_min, radius * options.trust_shrink)
                if options.verbose:
                    print(f"  itr {k + 1:3d}  sub-problem failed ({type(exc).__name__}), "
                          f"shrinking radius -> {radius:.3g}")
                continue
            res.status = f"sub-problem failure at iteration {k}: {exc}"
            res.x, res.objective = x, problem.objective_value(x)
            res.iterations = k
            break

        # Model prediction per linearized row: used by the ratio test and
        # by the next pass's curvature validation.
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
        # Testing at x_new pairs gradients from one point with multipliers
        # from another; SPaircraft's degenerate variables left stationarity
        # stuck near 0.57 however converged the meaningful ones were.
        stat, viol, comp = _kkt(problem, x, mults, options.x_min)
        if getattr(options, 'traj_log', None):
            try:
                import json as _json
                _tv = {problem.names[j]: float(x[j])
                       for j in range(problem.n)
                       if any(s in problem.names[j]
                              for s in options.traj_vars)}
                with open(options.traj_log, 'a') as _fh:
                    _fh.write(_json.dumps(dict(
                        k=k + 1, f=float(problem.objective_value(x)),
                        viol=float(viol), stat=float(stat),
                        dnorm=float(np.linalg.norm(d)), **_tv)) + "\n")
            except Exception:
                pass
        if (options.kkt_min_norm
                and viol <= options.feasibility_tolerance
                and (stat > options.stationarity_tolerance
                     or comp > options.complementarity_tolerance)
                and ((k + 1) % options.kkt_min_norm_every == 0
                     or k + 1 >= options.max_iterations)):
            m2 = _min_norm_mults(problem, x, options.kkt_min_norm_act_tol)
            s2, v2, c2 = _kkt(problem, x, m2, options.x_min)
            if max(s2 / max(options.stationarity_tolerance, 1e-300),
                   c2 / max(options.complementarity_tolerance, 1e-300)) < \
               max(stat / max(options.stationarity_tolerance, 1e-300),
                   comp / max(options.complementarity_tolerance, 1e-300)):
                stat, viol, comp = s2, v2, c2
                mults = m2
        if options.verbose:
            print(f"  itr {k + 1:3d}  f={problem.objective_value(x):.8f}  "
                  f"|d|={np.linalg.norm(d):.3e}  stat={stat:.3e}  "
                  f"viol={viol:.3e}  comp={comp:.3e}  tau={tau:.1e}")
        # --- feasibility-recovery contract ------------------------------
        # The penalty path runs ONLY to find the first feasible point; then
        # slacks come off and normal Phase II resumes. Budget expired first
        # = terminate as infeasible.
        if _recovering:
            if viol <= _eff_feas:
                _recovering = False
                use_slacks = False
                res.phase1_feasible = True
                res.phase1_mode = 'penalty-recovered'
                if options.verbose:
                    print(f"  itr {k + 1:3d}  feasibility recovered "
                          f"(viol {viol:.2e}); slacks off, Phase II resumes")
            else:
                _recover_budget -= 1
                if _recover_budget <= 0:
                    res.status = (
                        'feasibility recovery failed: Phase 1 fell short '
                        'and the bounded penalty stage did not reach a '
                        f'feasible point (viol {viol:.3e} vs tolerance '
                        f'{options.feasibility_tolerance:g})')
                    res.converged = False
                    res.x, res.objective = x, problem.objective_value(x)
                    res.iterations = k + 1
                    return res
        # --- restoration trigger: infeasible and not fixing it ---------------
        # Trust-region collapse is not the trigger that matters (3t+3c: sat
        # at viol 2.69e-01 for 1500 iterations with a healthy region). The
        # counter is CONSECUTIVE iterations infeasible, reset only by
        # feasibility -- excusing 10% progress let a creeping violation run
        # forever (ladder rungs 3/6/7/8: 1500 iterations, restorations = 0).
        if viol > _eff_feas:
            _infeas_stall += 1
            _infeas_best = min(_infeas_best, viol)
        else:
            _infeas_best, _infeas_stall = np.inf, 0
        # Incumbent memory is a GREY-BOX safeguard; on a pure SP it stays
        # dormant. Recorded unconditionally it sent black-box-free problems
        # into the strict fallback, whose unset floor rejected every step
        # (b737 anchor: 184 iterations parked vs 39 with it dormant).
        if has_blackbox and viol <= _inc_tol:
            _f_here = problem.objective_value(x)
            if _f_here < _f_incumbent:
                _x_incumbent, _f_incumbent = x.copy(), float(_f_here)
        # Suspended while feasibility recovery owns the iterate: the timer
        # once threw recovery back into failed Phase I mid-repair (viol
        # 0.77 -> 0.067); recovery polices itself with its own budget.
        if (not _recovering and
                _infeas_stall >= int(getattr(options, 'restoration_patience', 0) or 0) > 0):
            x_r, _restorations, _ok = _restore(
                problem, x, options, has_blackbox, cache, _restorations, k,
                f'infeasible for {_infeas_stall} iterations without progress', targets=_restore_targets)
            _infeas_best, _infeas_stall = np.inf, 0
            if _ok:
                x = x_r
                radius = options.trust_radius
                _reject_radius = np.inf
                if _filter is not None:
                    _filter.entries.clear()
                    _zz_prev = None          # new basin: stale geometry
                continue
            if _x_incumbent is not None and not _strict_feas:
                # Restoration failed: return to the feasible incumbent in
                # STRICT feasible-step mode (guards drop to factor 1).
                x = _x_incumbent.copy()
                _strict_feas = True
                radius = max(options.trust_min, 0.1 * options.trust_radius)
                _reject_radius = np.inf
                if _filter is not None:
                    _filter.entries.clear()
                    _zz_prev = None
                if options.verbose:
                    print(f"  itr {k + 1:3d}  restoration failed; falling "
                          "back to the feasible incumbent, strict "
                          "feasible-step mode ON")
                continue

        if (viol <= options.feasibility_tolerance
                and stat <= options.stationarity_tolerance
                and comp <= options.complementarity_tolerance):
            res.converged = True
            res.status = ("converged: KKT residual on the original problem "
                          "within tolerance")
            res.x, res.objective = x, problem.objective_value(x)
            res.iterations = k + 1
            break

        _subfail = 0                 # the sub-problem solved; the run is healthy
        f_old = problem.objective_value(x)
        # Clamp the log step before exponentiating: a near-flat penalty
        # direction can return |d_j| in the hundreds, and x*exp(d)
        # overflows to inf, after which every phase build dies on
        # "params.p0 = nan" (wind-turbine missions SP). e^60 is beyond any
        # legitimate move; the clamp turns overflow into a short step.
        d = np.clip(d, -60.0, 60.0)
        x_new = x * np.exp(d)

        # Extend the step while it stays feasible for the TRUE problem: the
        # conservative optimum is pessimistic, and walking further costs one
        # constraint evaluation against a whole sub-problem solve. Every
        # candidate is CHECKED against the true constraints, so acceptance
        # is by verification, not construction.
        if options.step_expansion > 1.0 and (not has_blackbox
                                             or options.expand_past_blackbox):
            budget = max(viol, options.feasibility_tolerance)
            # With a black box, judge the extension on the STRUCTURED rows
            # alone -- evaluating the box per trial alpha would spend the
            # calls this solver exists to save. The bb block is policed by
            # the ratio test anyway; the risk is a step it then rejects,
            # which is why this is off by default.
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

        # COMPOSITE-STEP ACCEPTANCE for tangent equalities: restore onto the
        # manifold BEFORE the acceptance test sees the trial, or the filter
        # charges the step for drift the restore removes free and rejects
        # every improving step (measured: radius 1.0 -> 1e-3, objective
        # frozen at the Phase-I point). Same accept-the-pair-together
        # principle as _phase1_composite; the post-acceptance restore
        # becomes a cheap no-op on this path.
        if n_eq_p2 and getattr(options, 'phase2_restore', False):
            import os as _os
            _dbg = _os.environ.get('SIA_DBG')
            _v_raw = _violation(problem, x_new) if _dbg else None
            _restore_fn = (restore_composite_bb if has_blackbox
                           else restore_equalities)
            x_nr, _h_nr = _restore_fn(
                problem, x_new, iters=options.phase1_restore_iterations,
                tol=min(options.feasibility_tolerance, 1e-10))
            if np.all(np.isfinite(x_nr)) and np.all(x_nr > 0):
                if has_blackbox:
                    # Grey-box rows: the restore is TRUTH, not a candidate --
                    # an un-restored iterate can carry bb-row fiction hiding
                    # real inequality violations. Keep it unconditionally;
                    # the acceptance test judges the restored point.
                    x_new = x_nr
                elif _violation(problem, x_nr) <= _violation(problem, x_new):
                    x_new = x_nr
            if _dbg:
                _net = float(np.linalg.norm(np.log(x_new) - np.log(x)))
                _obj0 = problem.objective_value(x)
                _obj1 = problem.objective_value(x_new)
                print(f'   DBG |d|={float(np.linalg.norm(d)):.3e} '
                      f'raw_viol={_v_raw:.2e} restored_viol='
                      f'{_violation(problem, x_new):.2e} net|dlogx|={_net:.3e} '
                      f'obj {_obj0:.6e}->{_obj1:.6e}', flush=True)

        if curvature is not None:
            _curv_step = np.log(x_new) - np.log(x)

        # Acceptance state shared by the black-box cascade and the tangent-
        # equality path. It used to live inside `if has_blackbox:`, leaving
        # pure-signomial problems with NO acceptance test (free coupled 737:
        # a two-point limit cycle ran 100+ iterations with rejection
        # unreachable).
        ratio = None
        va_seen = None
        v0 = None
        _rejected_by_guard = False
        if has_blackbox:
            # Globalize the linearized block only, and size the region by
            # ITS ratio: the objective ratio measures the wrong thing when
            # the box sits in a constraint. Aggregated over the block, not
            # per row -- a per-row minimum let an inactive row veto good
            # steps (Hoburg: 41 iterations converged -> 400 and not).
            v0 = max((abs(lin_here[i])
                      if problem.constraints[i].operator == '==' else lin_here[i]
                      for i in lin_idx if i in lin_here), default=None)
            vp = max((abs(curv_pred[i])
                      if problem.constraints[i].operator == '==' else curv_pred[i]
                      for i in lin_idx if i in curv_pred), default=None)
            ratio = None
            va_seen = None       # worst TRUE violation over the linearized rows
                                 # at x_new, whenever a level computed it
            _rejected_by_guard = False
            # Only measure a ratio when the predicted improvement means
            # something: an absolute 1e-10 floor formed ratios from noise
            # (helicopter: -3.3 every third iteration, 93% of the run at
            # 1e-5 steps).
            gate = (float(getattr(options, 'ratio_gate_rel', 0.01) or 0.0)
                    * options.feasibility_tolerance)
            if v0 is not None and vp is not None and v0 - vp > gate:
                try:
                    va = max(_log_viol(problem.constraints[i], x_new)
                             for i in lin_idx)
                    va_seen = va
                    ratio = (v0 - va) / (v0 - vp)
                except Exception:
                    ratio = None
            if ratio is None:
                # No predicted improvement to measure: judge model ACCURACY
                # instead (did the constraints land where it said). Falling
                # straight to the objective ratio stalled the helicopter --
                # 250 iterations at |d| = 1.8e-5 with the region never
                # growing.
                err = None
                if vp is not None:
                    try:
                        va = max(_log_viol(problem.constraints[i], x_new)
                                 for i in lin_idx)
                        va_seen = va
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
                    _noise = float(getattr(options, 'analysis_noise', 0.0)
                                   or 0.0)
                    if _noise > 0.0 and abs(pred) <= _noise:
                        # predicted change below what the analysis can
                        # resolve: a ratio here measures noise, not the
                        # model.  Accept rather than churn the radius
                        ratio = 1.0
                        if options.verbose:
                            print(f"  itr {k + 1:3d}  noise-level step "
                                  f"(|pred|={abs(pred):.2e} <= "
                                  f"analysis_noise); accepted")
                    # The objective ratio alone is not an acceptance test: a
                    # step can buy objective by leaving the feasible set
                    # (ROM section: 1619 -> 1521 while feasibility went to
                    # 2.074; never recovered). Guard it.
                    v_old = _violation(problem, x)
                    v_new = _violation(problem, x_new)
                    _g = 1.0 if _strict_feas else _FEAS_GUARD
                    if (v_new > _eff_feas
                            and v_new > _g * max(v_old, _eff_feas)):
                        ratio = -1.0

        # --- feasibility guard, applied to EVERY level of the cascade ---
        # It used to sit only in the rarely-taken objective-ratio fallback,
        # so the accuracy level accepted steps that left the feasible set
        # (3t+3c: one accepted step 9.98e-09 -> 2.69e-01, aborted at
        # iteration 7). va_seen is already computed, so this costs no extra
        # black-box calls; the linearized block IS the feasibility question.
        if (_filter is not None and va_seen is None
                and _has_tangent_eq and not has_blackbox):
            # tangent equalities can be violated by the step, so the
            # feasibility question is the TRUE violation over all rows
            va_seen = float(_violation(problem, x_new))
            if v0 is None:
                v0 = float(_violation(problem, x))
        # In FEASIBILITY-RECOVERY mode, a RATCHET on the best violation:
        # early excursions allowed (a hard monotone guard broke a
        # successful recovery), but after meaningful descent no giving back
        # more than a factor over the best (a cold start once oscillated
        # its budget away).
        if _recovering and va_seen is None:
            va_seen = float(_violation(problem, x_new))
            if v0 is None:
                v0 = float(_violation(problem, x))
        if _recovering and v0 is not None:
            _recover_best = min(_recover_best, v0)
        if (_recovering and va_seen is not None
                and _recover_best < 1.0e-2
                and va_seen > max(5.0 * _recover_best,
                                  10.0 * _eff_feas)):
            ratio = -1.0
            _rejected_by_guard = True
        if _filter is not None and va_seen is not None:
            # Dominance, not an exchange rate: refuse only if a seen point
            # was better in BOTH objective and violation.
            h_new = Filter.h_of(va_seen)
            f_new = math.log(max(problem.objective_value(x_new), 1e-300))
            if not _filter.acceptable(h_new, f_new):
                _filter.rejections += 1
                ratio = -1.0
                _rejected_by_guard = True
            elif h_new > options.feasibility_tolerance:
                # An accepted step that worsened feasibility: record the point
                # being LEFT so the iterates cannot cycle back to it.
                _filter.add(Filter.h_of(v0),
                            math.log(max(f_old, 1e-300)))
        else:
            _fg = getattr(options, 'feasibility_guard', _FEAS_GUARD)
            _fga = float(getattr(options, 'feasibility_guard_abs', 0.0) or 0.0)
            if _strict_feas:
                # feasible-step mode (post-restoration-failure): no true-
                # violation growth is tolerated beyond the tolerance itself
                _fg, _fga = 1.0, 0.0
            if (va_seen is not None and v0 is not None
                    and va_seen > options.feasibility_tolerance
                    and va_seen > _fga
                    and va_seen > max(float(_fg or 0.0), 1e-300)
                    * max(v0, options.feasibility_tolerance)):
                _fg_trips += 1
                if _fg:
                    ratio = -1.0
                    _rejected_by_guard = True

        if _strict_feas and ratio != -1.0:
            # STRICT mode measures the TRUE violation at the restored trial
            # every iteration -- the cascade's upper levels may never
            # evaluate it (|d| = 6.07 once accepted at violation 1e-8 ->
            # 9.2). The FLOOR is the restore's achievable residual, not the
            # raw tolerance (judging at 1e-8 rejected 165/165 steps).
            # Growth beyond 3x the residual or the floor is a real excursion.
            _floor_s = max(options.feasibility_tolerance,
                           float(getattr(options, 'strict_feas_floor', 0.0)
                                 or 0.0))
            _v_old_s = _violation(problem, x)
            _v_new_s = _violation(problem, x_new)
            if _v_new_s > max(3.0 * _v_old_s, _floor_s):
                ratio = -1.0
                _rejected_by_guard = True
                if options.verbose:
                    print(f"  itr {k + 1:3d}  STRICT reject: true violation "
                          f"{_v_old_s:.2e} -> {_v_new_s:.2e}")
        if ratio is None:
            ratio = 1.0     # nothing measured this step; accept it
        if ratio < options.ratio_accept:
            # A rejected iteration still ages the infeasibility clock;
            # counting only on the accepted path starved restoration
            # through a rejection streak (165 straight rejections, zero
            # restorations fired).
            if (not _recovering
                    and _violation(problem, x) > _eff_feas):
                _infeas_stall += 1
                if _infeas_stall >= int(getattr(options,
                                                'restoration_patience', 0)
                                        or 0) > 0:
                    x_r, _restorations, _ok = _restore(
                        problem, x, options, has_blackbox, cache,
                        _restorations, k,
                        f'infeasible for {_infeas_stall} iterations '
                        '(rejection streak)', targets=_restore_targets)
                    _infeas_best, _infeas_stall = np.inf, 0
                    if _ok:
                        x = x_r
                        radius = options.trust_radius
                        _reject_radius = np.inf
                        if _filter is not None:
                            _filter.entries.clear()
                            _zz_prev = None
                        continue
            if _released:
                # Rejected with the box OFF: shrinking an unapplied radius
                # would spin, so re-arm permanently, seeded from the step
                # that just failed.
                _trust_forced = True
                radius = max(options.trust_min,
                             min(radius, float(np.abs(d).max()))
                             * options.trust_shrink)
                if options.verbose:
                    print(f"  itr {k + 1:3d}  REJECT ratio={ratio:.3e} with the "
                          f"trust box released; re-arming at {radius:.3e}")
                continue
            # A FEASIBILITY rejection says the step was too long, not that
            # the model is wrong -- letting the guard write _reject_radius
            # bracketed and aborted 2t+2c at iteration 8 on a problem that
            # converges in 192 untouched. Shrink; leave the bracket alone.
            if not _rejected_by_guard:
                _reject_radius = min(_reject_radius, radius)
            radius = max(options.trust_min, radius * options.trust_shrink)
            if options.verbose:
                print(f"  itr {k + 1:3d}  REJECT ratio={ratio:.3e} "
                      f"radius -> {radius:.3e}")
            if radius <= options.trust_min:
                x_r, _restorations, _ok = _restore(
                    problem, x, options, has_blackbox, cache,
                    _restorations, k, 'the trust region collapsed',
                    targets=_restore_targets)
                if _ok:
                    x = x_r
                    radius = options.trust_radius
                    _reject_radius = np.inf
                    if _filter is not None:
                        _filter.entries.clear()   # the old trade-offs are
                        _zz_prev = None          # new basin: stale geometry
                    continue                      # about a basin we have left
                if _x_incumbent is not None and not _strict_feas:
                    x = _x_incumbent.copy()
                    _strict_feas = True
                    radius = max(options.trust_min,
                                 0.1 * options.trust_radius)
                    _reject_radius = np.inf
                    if _filter is not None:
                        _filter.entries.clear()
                        _zz_prev = None
                    if options.verbose:
                        print(f"  itr {k + 1:3d}  collapsed + restoration "
                              "failed; falling back to the feasible "
                              "incumbent, strict mode ON")
                    continue
                res.status = ("trust region collapsed at iteration "
                              f"{k + 1}; the linearized constraints are "
                              "not modelling the problem")
                res.x, res.objective = x, f_old
                res.iterations = k + 1
                break
            continue
        # Past warm-up the trust region has done its job -- B supplies the
        # conservative model -- so expand on every accepted step, or the
        # radius ratchets down and the solve crawls (16-variable ROM
        # section: stationarity pinned at 1.37e-02).
        # --- zigzag damping (see the option's comment). Arms on EVERY
        # model class -- plain signomials flip between wells too (launch-
        # vehicle SP: booster nozzle cycled forever, both steps ACCEPTED) --
        # and the hit also arms _trust_forced below, or the shrunk radius
        # would never reach the sub-problem.
        _zz_hit = False
        if (getattr(options, 'zigzag_damp', True)
                and _zz_prev is not None):
            # L2 over the WHOLE vector, not the max component: one lever
            # flipping while 1600 variables advance is progress (the max-
            # norm version replayed a byte-identical 7-state loop).
            # log space: raw units let one Reynolds-scale variable drown
            # the norm, and the comparisons below are against LOG-space
            # trust quantities
            _lx_new, _lx, _lzz = (np.log(x_new), np.log(x),
                                  np.log(_zz_prev))
            _d1 = float(np.linalg.norm(_lx_new - _lx))
            _d0 = float(np.linalg.norm(_lx - _lzz))
            _net = float(np.linalg.norm(_lx_new - _lzz))
            _s = max(_d1, _d0)
            if options.verbose:
                print(f"       zz: d1={_d1:.3e} d0={_d0:.3e} "
                      f"net={_net:.3e} ratio={_net / max(_s, 1e-300):.3f}")
            if (_s > 8 * options.trust_min
                    and _net < float(getattr(
                        options, 'zigzag_net_frac', 0.25)) * _s):
                _zz_hit = True
                _zz_hits += 1
                _zz_cool = int(getattr(options, 'zigzag_cooldown', 4))
                radius = max(options.trust_min,
                             min(radius, _s) * options.trust_shrink)
                # Make the shrink COUNT where the box is not otherwise
                # applied: from here on the sub-problem gets the radius.
                _trust_forced = True
                # A REPEAT hit means expansion re-inflated past the well
                # separation and the cycle replayed (launch-vehicle SP: 208
                # detections in 1200 iterations). Ratchet the ceiling below
                # the separation so the iterate must settle in one well.
                if _zz_hits > 1:
                    _zz_ceiling = min(_zz_ceiling,
                                      max(8.0 * options.trust_min,
                                          _s * options.trust_shrink))
                if options.verbose:
                    print(f"  itr {k + 1:3d}  ZIGZAG net={_net:.2e} vs "
                          f"step={_s:.2e}; radius -> {radius:.3e}")
        if _zz_cool > 0 and not _zz_hit:
            _zz_cool -= 1
        if _zz_hit or _zz_cool > 0:
            pass                    # hold the radius; no expansion
        elif ratio > options.ratio_expand:
            radius = min(min(options.trust_max, _zz_ceiling),
                         radius * options.trust_expand)
        elif _curv_trained:
            # Adequate step: grow by sqrt of the factor -- doubling from a
            # working radius lands on one that fails and the region
            # oscillates (ROM section: period-3 accept-accept-reject cycle).
            radius = min(min(options.trust_max, _zz_ceiling),
                         radius * math.sqrt(options.trust_expand))

        _zz_prev = x.copy()
        x = x_new
        _reject_radius = np.inf              # progress: the bracket is stale
        if getattr(options, 'phase2_restore', False) and n_eq_p2:
            _restore_fn = (restore_composite_bb if has_blackbox
                           else restore_equalities)
            x_r, _heq = _restore_fn(
                problem, x, iters=options.phase1_restore_iterations,
                tol=min(options.feasibility_tolerance, 1e-10))
            # Structured-only: keep the restore only if it helps. Grey-box:
            # the restore is TRUTH (see the pre-acceptance site) -- keep it
            # unconditionally.
            if np.all(np.isfinite(x_r)) and np.all(x_r > 0) and (
                    has_blackbox
                    or _violation(problem, x_r) <= _violation(problem, x)):
                x = x_r
        res.history.append(x.copy())
        res.objectives.append(problem.objective_value(x))

        # was the step pressed against the trust box?  The stop rule notes
        # it: a small step the region FORCED is not the same evidence as a
        # small step the design chose
        res._rel_step_binding = bool(float(np.abs(d).max()) >= 0.95 * radius)
        if _relative_change_converged(problem, res, options, k):
            break

        # Update the curvature models from the step: the gradient change is
        # a free secant condition, and the true value validates the model --
        # inflating on optimism costs no extra black-box calls.
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
                            # Pessimistic here: ease B back, don't ratchet.
                            cv.relax()
                except Exception:
                    pass
            # Trained once EVERY row has enough BFGS data -- the minimum,
            # not the mean, because the step is bounded by the worst row.
            _bb_g1 = max((float(np.abs(cv.last_grad).sum())
                          for cv in curvature.values() if cv.last_grad is not None),
                         default=_bb_g1)
            _warm = int(getattr(options, 'curvature_warmup', 0) or 0)
            if _warm and curvature:
                _curv_trained = min(cv.updates for cv in curvature.values()) >= _warm

        # --- escalate the slack penalty -----------------------------------
        # Raise tau when slack is open OR a multiplier has run into tau
        # (tau caps every multiplier, so a constraint whose true multiplier
        # exceeds it goes soft -- the exact-penalty condition). Both
        # triggers EXCLUDE equality rows: an active L1 equality's dual sits
        # at +-tau by construction, and its slack is second-order drift the
        # restore closes free. With both included tau hit 9.8e6 by
        # iteration 11 and the solve froze at the Phase-I point.
        k_ = min(len(problem.constraints),
                 len(s) if len(s) else 0,
                 len(mults) if len(mults) else 0)
        _ineq_mask = [i for i in range(k_)
                      if problem.constraints[i].operator != '==']
        slack = 0.0
        if len(s):
            s_ = np.asarray(s, dtype=float)
            vals = [s_[i] for i in _ineq_mask] + list(s_[k_:])
            slack = float(np.max(vals)) if vals else 0.0
        lam_max = 0.0
        if len(mults):
            m_ = np.abs(np.asarray(mults, dtype=float))
            vals = [m_[i] for i in _ineq_mask] + list(m_[k_:])
            lam_max = float(np.max(vals)) if vals else 0.0
        need = (slack > options.feasibility_tolerance
                or lam_max >= options.tau_binding * tau)
        # near-feasible: everything within the relief band of tolerance.
        # Escalating here drowns the objective in the endgame; decay instead
        _band = float(getattr(options, 'tau_relief_band', 10.0))
        _near = (slack <= _band * options.feasibility_tolerance
                 and lam_max < options.tau_binding * tau
                 and _violation(problem, x)
                 <= max(_band * options.feasibility_tolerance, _eff_feas))
        if need and not _near and tau < options.tau_max:
            tau = min(options.tau_max,
                      tau * options.tau_factor,
                      max(tau * options.tau_factor,
                          options.tau_factor * lam_max))
        elif _near and tau > options.tau0:
            tau = max(options.tau0, tau / options.tau_factor)

    else:
        res.status = (f"did not converge within {options.max_iterations} "
                      "iterations")
        res.x, res.objective = x, problem.objective_value(x)
        res.iterations = options.max_iterations

    if res.x is None:
        res.x, res.objective = x, problem.objective_value(x)
    # EXIT DISCIPLINE (SQP): never return an infeasible excursion while a
    # feasible incumbent exists (a budget once expired mid-excursion).
    # UNCONDITIONAL -- a converged CLAIM at an infeasible point is the
    # known false-certificate mode; the true violation is the test.
    if (_x_incumbent is not None
            and _violation(problem, res.x) > _inc_tol):
        res.x, res.objective = _x_incumbent, float(_f_incumbent)
        res.converged = False
        res.status = ((res.status or '')
                      + '  [infeasible final iterate discarded: returned '
                        'the best feasible incumbent]')
    stat, viol, comp = _kkt(problem, res.x, mults, options.x_min)
    if (options.kkt_min_norm and not res.converged
            and viol <= options.feasibility_tolerance
            and (stat > options.stationarity_tolerance
                 or comp > options.complementarity_tolerance)):
        # Stopped short with the sub-problem's duals failing: ask whether
        # MINIMUM-NORM multipliers certify the point first (split
        # equalities make the returned duals meaningless; see kkt_min_norm)
        m2 = _min_norm_mults(problem, res.x, options.kkt_min_norm_act_tol)
        s2, v2, c2 = _kkt(problem, res.x, m2, options.x_min)
        if s2 <= max(stat, options.stationarity_tolerance) \
                and c2 <= max(comp, options.complementarity_tolerance):
            stat, viol, comp, mults = s2, v2, c2, m2
            if (s2 <= options.stationarity_tolerance
                    and c2 <= options.complementarity_tolerance):
                res.converged = True
                res.status = ("converged: KKT certificate with minimum-norm "
                              "multipliers (sub-problem duals were "
                              "degenerate)")
    if (options.polish_ipopt
            and viol <= options.feasibility_tolerance
            and (stat > options.stationarity_tolerance
                 or comp > options.complementarity_tolerance)):
        xn = _polish_ipopt(problem, res.x, options)
        if xn is not None:
            m3 = _min_norm_mults(problem, xn, options.kkt_min_norm_act_tol)
            s3, v3, c3 = _kkt(problem, xn, m3, options.x_min)
            if v3 <= options.feasibility_tolerance and s3 <= stat:
                res.x = xn
                res.objective = problem.objective_value(xn)
                res.polished = True
                stat, viol, comp, mults = s3, v3, c3, m3
                if (s3 <= options.stationarity_tolerance
                        and c3 <= options.complementarity_tolerance):
                    res.converged = True
                    res.status = ("converged: Ipopt polish of the true "
                                  "problem, certified with minimum-norm "
                                  "multipliers")
                else:
                    res.status = (res.status or "") + \
                        "  [polished: objective improved, certificate " \
                        f"{s3:.2e}]"
    res.stationarity, res.max_violation, res.complementarity = stat, viol, comp
    res.multipliers = mults
    res.slacks_active = viol > options.feasibility_tolerance
    # Attached to every result: the same text explains why a converged run
    # converged, and the degenerate-block warning is worth seeing either way.
    res.feas_guard_trips = _fg_trips     # times the guard CONDITION held, whether
                                         # or not feasibility_guard applied it
    res.restorations = _restorations     # Phase I re-entries from within Phase II
    res.zigzag_hits = _zz_hits
    res.filter_rejections = (_filter.rejections if _filter is not None else 0)
    res.filter_size = len(_filter) if _filter is not None else 0
    try:
        res.report = convergence_report(problem, res, options)
    except Exception as exc:                     # never let a diagnostic
        res.report = f"(convergence report unavailable: {exc})"   # kill a solve
    if options.verbose:
        print(res.report)
    return res
