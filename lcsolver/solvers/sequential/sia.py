#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
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
        # 'l1' (default) or 'minmax'.
        #
        # min-max solves  min t  s.t.  log g_i <= t  with ONE shared scalar, so
        # every constraint is driven to the same violation level. That is a bad
        # fit for a model carrying many signomial EQUALITIES: each becomes
        # |h_i| <= t, so all of them must be satisfied simultaneously and to
        # the same tolerance, and an equality already gets a conservative AGM
        # approximation on BOTH sides (p <= q and q <= p), which can be jointly
        # infeasible even where the true equality is satisfiable.
        #
        # Elastic L1 gives each constraint its OWN slack and minimises the sum.
        # Its optimum is sparse: constraints that can be satisfied fall to zero
        # slack and stop competing with the ones that cannot. This is what
        # SNOPT calls elastic mode and IPOPT calls feasibility restoration, and
        # it is the standard answer to exactly this problem.
        #
        # Measured on the spcomparisons aircraft, which carries ~15 coupled
        # signomial equalities: min-max fails at 50, 200 AND 500 iterations
        # alike -- not a budget problem, a formulation one.
        # 'composite' (default) | 'l1' | 'minmax'. See _phase1_composite.
        self.phase1_method = 'composite'
        self.phase1_restore_iterations = 12
        # Cap on the composite trust radius, in log space. Phase I is a
        # minimum-change REPAIR of the seed, not a search: the seed carries
        # whatever the engineer knew, and on a non-convex problem the point
        # Phase I hands over decides which local optimum Phase II descends to.
        # Left to expand freely the radius reached 8 -- a factor of e^8 per
        # variable -- and SPaircraft came back 0.46% worse from a feasible
        # point nowhere near its seed.
        self.phase1_trust_max = 1.0
        # Proximity weight on the elastic Phase I objective:
        #     min  sum(s_i) + w ||d||^2
        # Feasibility alone is a badly posed thing to ask for. Any point in
        # the feasible set answers it, sum(s_i) usually has a whole face of
        # minimizers, and which one comes back is down to the interior-point
        # solver. On a non-convex problem that choice is not cosmetic: the
        # point Phase I hands over decides which local optimum Phase II
        # descends to. simpleac came back at 4536 or 6485 -- both feasible to
        # 1e-13 and stationary to 1e-8 -- depending only on whether presolve
        # had eliminated a variable first.
        #
        # The weight makes the answer the SMALLEST repair of the seed, which
        # is unique, continuous in the data, and the thing an engineer means
        # by "make my starting design feasible". Small enough not to compete
        # with the slacks: it breaks ties, it does not trade feasibility away.
        self.phase1_proximity = 1e-6
        # Restore the equalities after each accepted Phase II step.
        #
        # Phase II's guarantees -- feasible iterates, monotone descent -- are
        # stated for a problem whose constraints are exact or CONSERVATIVE. A
        # signomial equality is neither: condensing both sides gives a monomial
        # that is tangent at the iterate and on the wrong side of the true
        # constraint in both directions. So the very first step leaves the
        # feasible set even from an exactly feasible start -- measured on the
        # E175, 6e-15 to 2.257 in one step -- and the run then spends its whole
        # budget crawling back, asymptoting just above tolerance without ever
        # landing on it. That is not slow convergence, it is convergence to a
        # point that is not quite feasible, and stationarity measured there
        # stalls with it (5.93e-06 against a 1e-06 tolerance, falling 0.3% an
        # iteration).
        #
        # The normal step from Phase I fixes it: after each accepted step, pull
        # back onto the manifold. Cheap, since a step from a nearly-feasible
        # point converges in one or two Newton iterations.
        #
        # ON by default. Measured on the E175 it does what it says -- the
        # worst violation after the first step drops from 2.257 to 0.271 and
        # the converged value from 4.2e-08 to 1.7e-08. It does NOT move
        # stationarity there, and it costs about 50% more wall time, but a
        # signomial equality is the one row type that can push Phase II off the
        # feasible set from an exactly feasible start, and every iterate being a
        # usable design is worth that. Set False to reproduce the drift.
        self.phase2_restore = True


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
        # A filter cannot break a two-point cycle: if A beats B on violation
        # and B beats A on objective, neither dominates, both stay acceptable
        # forever, and every flip is an ACCEPTED step -- so nothing ever
        # shrinks the radius. Measured on the free coupled 737 case: pi_f
        # 1.796 <-> 1.672, BPR 4.49 <-> 5.39 (|d| = 0.184 log, right at the
        # radius), net drift 1e-4/iteration, 100+ iterations burned. Detect
        # the signature -- two sizeable accepted steps whose SUM is small --
        # and shrink the radius, forcing the linearization to localize.
        self.zigzag_damp = True
        # net/step below this = a cycle. 0.5 with equal step lengths is a
        # 151-degree reversal; a healthy curved descent turning 90 degrees
        # has net/step = 1.41, so this cannot fire on valley-following.
        # (0.25 was measured too strict: the wide early orbit on the free
        # coupled case ran at net/step = 0.49 and was never damped.)
        self.zigzag_net_frac = 0.5
        self.zigzag_cooldown = 4       # accepted steps with expansion held
                                       # off after a detection, so the radius
                                       # cannot re-inflate into the same flip
        # --- misc ----------------------------------------------------------
        self.x_min = 1e-9
        self.expand_past_blackbox = False
        #: Carry a BFGS curvature model for each linearized constraint, so the
        #: subproblem holds a convex quadratic rather than a plane. See
        #: :class:`Curvature`. The model stays convex either way; this makes it
        #: conservative wherever the curvature estimate is good enough.
        #:
        #: ON by default. A linearized row is the one place SIA gives up its
        #: guarantee, and B is what restores it -- leaving this off means the
        #: black-box block is bounded only by the trust region, which is a step
        #: limit, not a model. Costs nothing on a problem with no black box:
        #: the models are only built when ``has_blackbox``, so a pure SP is
        #: bit-identical either way.
        self.curvature = True
        self.step_expansion = 1.0      # >1 enables the feasibility-verified
        self.step_expansion_max = 1e4  # step extension described in solve_sia.
                                       # Set to 1.0 to take the sub-problem's
                                       # step exactly as returned.
        # ON by default (2026-08): the condensed-numerator iteration is not
        # feasibility preserving, but with Phase I restoration steps
        # (phase1_restore_iterations) repairing any excursion it is faster on
        # every anchor model and reaches the same certified points. Set False
        # for the conservative, feasibility-preserving mode.
        self.condense_numerator = True
        # Seed for the black-box curvature model, in units of |d log g / d log x|.
        # B is built by BFGS from SUCCESSIVE gradients, so it is ZERO on the first
        # iteration and the linearized rows are then not conservative at all --
        # see Curvature.observe. A prior of order one restores the property on the
        # step that has no history to learn from. 0.0 reproduces the old behaviour.
        self.curvature_prior = 1.0
        # Iterations of BFGS data per black-box row after which the curvature
        # model is treated as TRAINED and the trust region is released -- see the
        # expand path below. The trust region substitutes for a conservative
        # model; B IS the conservative model once it has data, so keeping a
        # ratcheting radius past that point only traps the solve.
        self.curvature_warmup = 6
        # Warm-up step control for the linearized (black-box) rows. Until the
        # curvature models have data their model is only as conservative as the
        # prior, so bound the step by what it can actually PREDICT rather than by
        # a fixed radius: cap |grad . d| <= this for every black-box row. Because
        # an inf-norm trust region gives |grad . d| <= ||grad||_1 * r, the cap is
        # r <= target / max_i ||grad_i||_1 -- self-scaling, so a parameterisation
        # whose gradients are 4x larger automatically gets a 4x smaller first step.
        # That is the difference between two ROM bases surviving the same radius
        # or not, and it is not something a user should have to tune per basis.
        # Applied for the FIRST blackbox_warm_iters iterations only, then released
        # unconditionally. Gating it on "curvature is trained" instead deadlocks:
        # the capped steps are too small for the BFGS secant to accept an update,
        # so training never completes and the cap never lifts -- measured, the
        # reference case then sat at its warm start for all 300 iterations.
        # 0 disables either field.
        self.blackbox_step_target = 0.15
        self.blackbox_warm_iters = 0
        # Apply the trust box for the FIRST this-many Phase II iterations only,
        # then drop it entirely. None keeps it on for the whole solve.
        #
        # Growing the radius is not the same as removing it. Measured on the 2t+2c
        # ROM section: the radius cycles 1.7e-04 .. 1.0e-03 for ~290 iterations with
        # ||d*||inf/radius = 1.000 EVERY time -- the box, not any constraint, is the
        # binding row. Six of the nine black-box drag rows then come back with
        # lambda ~ 1e-12, so the dual that should oppose the structured rows on the
        # cd variables is carried by the trust region instead. The trust region is
        # not part of the original problem, so its multiplier cannot appear in the
        # KKT test, and stationarity froze at 1.7e-02 -- entirely on cdm, P0cr and
        # the cd_i, all one sign -- while feasibility sat at 1e-08. Shrink beats
        # growth arithmetically (x0.25 per rejection against x2 / x sqrt(2)), so the
        # radius can never climb back out on its own.
        self.trust_iterations = None
        # Floor, as a multiple of feasibility_tolerance, on the PREDICTED
        # improvement in the linearized block before a violation ratio is formed
        # at all. See the gate in the ratio test.
        #
        # 0.01 was too low by two orders of magnitude on the ROM section. Measured
        # there: v0 = -1.7e-09, the model predicted vp ~ -1e-08, the step landed at
        # va ~ +2.8e-08, and the ratio came out at -3.004 -- from three numbers all
        # at 1e-08, a hundredth of the feasibility tolerance the run is trying to
        # meet and squarely inside Ipopt's own interior-point noise. That rejected
        # 90 of 300 iterations in a period-3 limit cycle, pinned the radius at
        # ~5e-04, and left cdm 8.3% and cd_4/cd_7 ~11% ABOVE the drag the black box
        # actually returns -- slack rows the solve could not close at 0.05% per
        # step. Below the feasibility tolerance a "change in violation" is not a
        # signal about the model, so no ratio should be formed from it.
        self.ratio_gate_rel = 1.0
        # How much worse the TRUE violation of the linearized block may get on an
        # accepted step, as a multiple of max(current violation, feasibility
        # tolerance). Set 0/None to disable the guard entirely.
        #
        # This is a backstop against walking out of the feasible set, not a
        # feasibility filter, so it should fire rarely. If it is firing on a large
        # fraction of iterations the threshold is too tight, not the steps too bad:
        # near a solution the current violation sits far below the tolerance, so
        # max(v0, tol) pins to the TOLERANCE and the absolute bar becomes
        # factor*tol -- 1e-05 at the defaults -- which ordinary sub-problem noise
        # can exceed without the step being remotely unsafe.
        self.feasibility_guard = _FEAS_GUARD          # module default, 10.0
        # ...AND an absolute floor, without which the relative test above is not
        # relative at all. Measured on the 2t+2c section: at a converged iterate
        # v0 sits at ~1e-09, so max(v0, tol) pins to the TOLERANCE and the bar
        # becomes factor*tol = 1e-05. Ordinary sub-problem noise clears that
        # constantly -- 101 of 192 steps tripped it, 53%, each one quartering the
        # radius, and the solve went from 192 iterations to 802 for exactly the
        # same answer (1503.021 either way).
        #
        # An absolute floor SOUNDS like the fix for that and is not -- measured,
        # it breaks the solve outright. At 1e-03 on 2t+2c the guard stops
        # rejecting the two large early steps (|d| = 2.34, 1.13) that are how the
        # run reaches its basin at all; the smaller step it takes instead lands at
        # viol 2.12e-04, above the tolerance but below the floor, and Phase II has
        # no way back from there -- every later step is rejected and the run dies
        # at iteration 8 with W 1605 against the 1503.021 it reaches untouched.
        #
        # So the floor defaults OFF. The plain relative guard converges both
        # 2t+2c (802 it) and 3t+3c (911 it); the cost is real -- 192 iterations
        # without any guard -- but 3t+3c cannot be solved cold without it at all.
        self.feasibility_guard_abs = 0.0
        # --- Fletcher-Leyffer filter acceptance (replaces the guard) --------
        # ON by default; it supersedes feasibility_guard, which is left in place
        # only so the old behaviour can be reproduced (set filter_acceptance
        # False). See :class:`Filter` for why a scalar guard cannot work.
        self.filter_acceptance = True
        self.filter_gamma_h = 1e-5
        self.filter_gamma_f = 1e-5
        # --- restoration ----------------------------------------------------
        # A filter accepts steps that worsen feasibility, which is only sound if
        # such a step is RECOVERABLE. Phase I already minimises the worst
        # violation; this re-enters it from inside Phase II when the region is
        # about to collapse or bracket while the iterate is infeasible, instead
        # of aborting. Without it Phase II's only response to an excursion is to
        # reject and shrink forever.
        self.restoration = True
        self.restoration_max = 8          # re-entries allowed per solve
        # Iterations the run may sit infeasible WITHOUT reducing the violation by
        # at least 10% before restoration is triggered. This, not trust-region
        # collapse, is the trigger that matters: measured on 3t+3c the filter
        # accepted an excursion to 2.69e-01 and the run then held that violation
        # for all 1500 iterations while the region stayed perfectly healthy.
        self.restoration_patience = 5
        # --- min-norm-dual KKT termination (EXPERIMENTAL, off by default) --
        # Splitting an equality into two always-active one-sided rows makes
        # the dual set unbounded (any common increment to the pair cancels),
        # so the sub-problem's returned multipliers can carry arbitrarily
        # large components in the null direction and the stationarity test
        # reads garbage at a genuinely optimal point (measured on the
        # coupled aircraft: duals to 1e13 with the objective stable to
        # 3e-8). With this flag, whenever the iterate is feasible but the
        # dual-based stationarity fails, the test is repeated with the
        # MINIMUM-NORM multipliers over the active rows -- the certificate
        # the point actually earns -- every `kkt_min_norm_every` iterations
        # and always on the final one.
        self.kkt_min_norm = False
        self.kkt_min_norm_every = 25
        self.kkt_min_norm_act_tol = 1e-6
        # --- Ipopt polish of the TRUE problem (EXPERIMENTAL, off) --------
        # The sequential method's conservative first-order steps crawl in
        # shallow curved valleys (measured on the coupled aircraft: the
        # design settles to <0.1% and the objective then creeps ~0.06% per
        # 400 iterations through flat trim blocks). From a settled feasible
        # point, hand the ORIGINAL problem -- rebuilt symbolically in log
        # space, every row exact, no condensation -- to Ipopt, whose exact
        # second-order steps finish shallow valleys in a handful of
        # iterations. The result is adopted only if it verifies: feasible
        # on the true rows and objective no worse. polish_box bounds the
        # excursion in log units so the polish is local to the basin the
        # sequential phase found.
        self.polish_ipopt = False
        self.polish_box = 3.0
        self.polish_max_iter = 3000
        # --- trajectory log (DIAGNOSTIC, off) ----------------------------
        # Path to a jsonl file; each ACCEPTED iterate appends one line with
        # iteration, objective, violation, stationarity, |d|, and the
        # values of every variable whose name contains one of the
        # substrings in traj_vars. The failure movie, replayable.
        self.traj_log = None
        self.traj_vars = ()
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
        # constr_viol_tol does NOT follow tol -- it keeps its own 1e-4 default,
        # which is a floor on how well the sub-problem's rows are satisfied and
        # therefore a floor on Phase I. It showed up as the composite step
        # driving the aircraft from 2.2 to 1.6e-04 in two iterations and then
        # sitting at 1e-04 forever, with the sub-problem reporting every slack
        # at zero: the model believed it was feasible because 1e-04 is what it
        # was asked to achieve.
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
        #: KKT residuals on the ORIGINAL problem at the returned point
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


#: How much worse the violation may get on a step judged only by its objective
#: ratio before that step is rejected outright. Generous -- this is a backstop
#: against walking out of the feasible set, not a feasibility filter.
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

    Solves ``min ||g0 + A lam||`` with ``lam >= 0`` over the rows whose
    log-residual is within ``act_tol`` of active (every row is a <= row
    by the time the problem is built; split equalities appear as two
    opposed active rows, whose nonnegative pair spans the free-sign
    equality dual). Column scaling is what makes this work at aircraft
    size -- gradient columns span ~8 decades and unscaled bounded
    least-squares stalls. Returns a full-length multiplier vector with
    zeros on inactive rows, suitable for ``_kkt``.
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

    Rebuilds every row symbolically in log space (posynomials as
    sum-of-exponentials; ratio and condensed-equality rows as p <= q and
    p == q), so Ipopt sees the exact nonconvex NLP with exact second
    derivatives -- the ingredient the sequential phase's first-order
    conservative steps lack in shallow valleys. Returns the polished x,
    or None if the model contains black-box rows, Ipopt fails, or the
    result does not verify.
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


def _restore(problem, x, options, has_blackbox, cache, done, k, why,
             targets=None):
    """Re-enter Phase I from inside Phase II. Returns ``(x, done, ok)``.

    The filter deliberately accepts steps that worsen feasibility, on the
    understanding that such a step is RECOVERABLE. This is what makes that true.
    Phase I already solves exactly the right sub-problem -- minimise the worst
    violation, ignore the objective -- it was simply only ever called once, at
    the start, so Phase II's only response to an excursion was to reject and
    shrink until the region died.

    Restoration is accepted only if it actually restores feasibility; a Phase I
    that comes back still infeasible tells us the problem is infeasible HERE,
    which is worth reporting rather than papering over.
    """
    if not getattr(options, 'restoration', False):
        return x, done, False
    if done >= int(getattr(options, 'restoration_max', 0) or 0):
        return x, done, False
    v_before = _violation(problem, x)
    if v_before <= options.feasibility_tolerance:
        return x, done, False           # feasible already; nothing to restore
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
    # A restoration that returns a point it has ALREADY returned to is a
    # loop, not a recovery: Phase I is deterministic, so from anywhere in
    # the same neighbourhood it lands in the same feasible well, the filter
    # is then cleared, and the descent replays byte-identically. Measured
    # on the free coupled 737 case: a 7-state cycle (f 472182.4 -> ... ->
    # 218126.2 -> restore) repeated until all restoration credits burned,
    # ~56 iterations of deterministic replay. Refusing the repeat costs one
    # Phase I call and lets Phase II continue from where it is.
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

    Replaces the scalar feasibility guard, which could not be tuned. That guard
    asked "is the violation more than K times worse than it was", and K is an
    exchange rate between feasibility and objective -- there is no correct value,
    because the right trade depends on where you are. Measured on the ROM
    section, K = 10 rejected 101 of 192 steps (near a solution the comparator
    pins to the TOLERANCE, so the relative test silently becomes an absolute bar
    at K*tol), while raising it let a step run from 1e-08 to 2.69e-01.

    A filter asks a different question, with no exchange rate in it: is this
    trial DOMINATED -- worse in objective AND worse in feasibility than a point
    already seen? If not, it is acceptable. Formally, for every entry
    ``(h_j, f_j)``::

        h < (1 - gamma_h) * h_j    OR    f < f_j - gamma_f * h_j

    ``gamma_h`` and ``gamma_f`` are anti-cycling margins, not exchange rates:
    they only stop the iterates converging onto the filter boundary, and any
    small value does that. That is the robustness the scalar guard lacked.

    A filter is only sound with a RESTORATION phase, because it deliberately
    accepts steps that worsen feasibility -- the point being that such a step is
    recoverable, not that it is harmless. See the restoration hook in
    ``solve_sia``, which re-enters Phase I.

    ``h`` must be non-negative, so it is ``max(log_violation, 0)``: a row with
    slack contributes no infeasibility.
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

        New directions are seeded with ``prior * |grad_j|`` on the diagonal, not
        with zero. B is built by BFGS from SUCCESSIVE gradients, so on the first
        iteration there is no history and a zero seed leaves the model a bare
        linearization -- the one thing the class exists to avoid. The guarantee
        is "conservative wherever B dominates the true log-curvature", and B = 0
        dominates nothing: the first step is then bounded only by the trust
        region, and on a black box that step can leave the feasible set outright
        (measured on the helicopter: 9.975e-09 -> 5.070e-01 in one step, after
        which the sub-problem is unsolvable and the solve aborts).

        |grad_j| is the right scale because in log space a smooth function's
        second derivative runs with its first: measured on that model's drag
        black box, |lambda|max / |grad| sits at 2.5-6.6. So a prior of order one
        is genuinely conservative for a unit log step, and where it is still
        optimistic ``inflate`` raises it from the next iteration's data. This is
        the standard quasi-Newton ``B0 = gamma I`` seeding, scaled per variable
        because the sensitivities here span two orders of magnitude.
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

    This is the same device as :class:`~lcsolver.solvers.sequential.slcp.SubproblemCache`,
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
        #   True    Phase I MIN-MAX -- one shared t, min t s.t. log g_i <= t
        #   'l1'    Phase I ELASTIC -- a slack PER constraint, min sum(s_i)
        #
        # The min-max form drives every constraint to the SAME violation
        # level, which is why a stalled Phase I reports a dozen rows sitting
        # at an identical residual and none of them stands out. The elastic
        # form is what makes an infeasibility diagnosable: at its optimum
        # nearly every s_i is zero and the few that are not ARE the answer.
        # 'l1' slacks everything; 'l1_hard' slacks only the INEQUALITIES and
        # imposes the equalities exactly, so the step is confined to the
        # linearised equality manifold. That is the TANGENTIAL half of a
        # composite step, expressed in the sub-problem instead of through a
        # null-space projection.
        #
        # It is only safe from a point already ON the manifold, and then it is
        # guaranteed safe: the AGM condensation is tight at its expansion
        # point (w_k = q_k(x_k)/q(x_k) gives qhat(x_k) = q(x_k) exactly), so
        # d = 0 satisfies the linearised equalities and the sub-problem cannot
        # be infeasible. From an inconsistent start the same rows are what
        # made it fail with "sub-problem failed: infeasible" at iteration 1.
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
            # Pure feasibility: no true objective at all, so there is no
            # scaling contest between cost and feasibility to lose. The
            # proximity term breaks the tie between equally feasible points.
            w = getattr(self.options, 'phase1_proximity', 0.0)
            prox = (w * sum(m.d[j] ** 2 for j in range(n))) if w else 0.0
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
    def update(self, phase, x_k, tau, radius=None):
        """Re-point a built phase at a new iterate. No symbolic work.

        ``radius`` bounds |d_j| when the caller wants a trust region even
        though nothing was linearized. Phase II does not: every row there is
        exact or conservative, so the step is safe at any length. Phase I with
        hard equalities does, because a CondensedEquality is TANGENT, not
        conservative -- condensing both sides of an equality is neither an
        inner nor an outer approximation, so an unbounded step can leave the
        true manifold far enough that the next restoration has to travel, and
        travelling is what breaks the inequalities the step just fixed.
        """
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
        if radius is not None:
            lo = np.maximum(lo, -radius)
            hi = np.minimum(hi, radius)
        for j in range(n):
            ub = None if not np.isfinite(hi[j]) else float(hi[j])
            m.d[j].setlb(float(lo[j]))
            m.d[j].setub(ub)
        # Start from d = 0 (the current iterate), but inside the box: a
        # variable already at its bound has 0 outside it. The clamp used to be
        # written inline as `min(max(0, lo), ub if ub is not None else 0.0)`,
        # which collapses to 0.0 whenever there is no upper bound -- exactly
        # the unbounded-above case it was meant to catch -- so the warning it
        # exists to prevent was emitted anyway.
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
        # update(). A cacheable problem has no black box, so a trust region
        # applies only when the caller asks for one -- see update().
        phase = cache.update(cache.get(minimize_violation, use_slacks),
                             x_k, tau, radius=radius if force_trust else None)
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
        seat_step_in_bounds(m)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # Same three modes as the cached builder: min-max (one shared t), elastic
    # L1 (a slack per row), and l1_hard (slacks on the inequalities only, with
    # the equalities imposed exactly). Kept in step deliberately -- the cache
    # falls through to this path for black-box bodies and on its own numerical
    # guard, so a mode the cache understands and this does not shows up as
    # 'ConcreteModel object has no attribute s' from whichever iteration
    # happened to fall through.
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
        # Pure feasibility, L1: the optimum is sparse, so the rows that keep a
        # slack are the ones that actually cannot be satisfied. Plus the
        # proximity term -- see SIAOptions.phase1_proximity.
        _w = getattr(options, 'phase1_proximity', 0.0)
        _prox = (_w * sum(m.d[j] ** 2 for j in range(n))) if _w else 0.0
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
                # |residual| <= slack. Writing it as `e == t` against a shared
                # t is what made Phase I fail on feasible problems: it forces
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
            # With a curvature model the row becomes a convex quadratic rather
            # than a plane: still convex, so the subproblem is unchanged in
            # kind, but now an upper bound wherever B dominates the true
            # log-curvature -- which is the property the trust region exists to
            # substitute for.
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
    # radius None means the caller has RELEASED the box (see trust_iterations).
    if (has_blackbox or force_trust) and radius is not None:
        for j in range(n):
            tighten(j, lo=-radius, hi=radius)

    # Bounds are tightened in several passes above; only now is the box final.
    seat_step_in_bounds(m)

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

    from lcsolver.environment import ipopt_solver_factory
    opt = ipopt_solver_factory()
    if not opt.available(exception_flag=False):
        raise SolverUnavailable(
            "SIA needs IPOPT to solve its sub-problems; no usable installation "
            "was found. Run `lcsolver-install-solvers`; see docs/ipopt.rst.")
    for k, v in (options.ipopt_options or {}).items():
        opt.options[k] = v
    results = opt.solve(m, tee=options.tee, load_solutions=False)
    tc = str(results.solver.termination_condition)
    if tc not in ("optimal", "locallyOptimal", "feasible"):
        raise RuntimeError(f"the SIA sub-problem failed: {tc}")
    m.solutions.load_from(results)

    d = np.array([pyo.value(m.d[j]) for j in range(n)])
    _elastic = (minimize_violation in ('l1', 'l1_hard'))
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




def _posy_loggrad(posy, x, n):
    """``(log p, d log p / d log x)`` for a posynomial, analytically.

    log p is a log-sum-exp of monomials, so the gradient is the weighted mean
    of their exponent vectors -- one pass, exact, and far cheaper than the n
    finite differences it replaces.
    """
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

    Black-box bodies are first-class here: a Signomial carries exactly the
    value and log-gradient this needs, so a black-box EQUALITY can be restored
    by the same Gauss--Newton step as a structured one. Before this case was
    added, restore_equalities crashed on the first problem that reached Phase I
    with a black-box equality present (the Hoburg UAV with the ROM section's
    tau coupling) -- the helicopter never hit the path because its start was
    seeded feasible.
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


def restore_equalities(problem, x, iters=12, tol=1e-9, verbose=False):
    """Least-norm Gauss-Newton onto the equality manifold -- the NORMAL step.

    This works on the TRUE residuals, never on the conservative condensation,
    which is the whole point. A signomial equality reaches the sub-problem as
    ``p <= q`` AND ``q <= p``, each AGM-condensed, so the approximation's
    feasible set is strictly smaller than the true one and can be empty where
    the true equality is perfectly satisfiable. Newton does not care: on a
    1173-variable aircraft with 802 signomial equalities it reaches 5e-12 in
    eleven steps, quadratically, while every Phase I formulation stalls.

    Least-norm is deliberate. The repaired point should stay as close to the
    start as the equalities allow -- it is fixing an inconsistency, not
    searching. As a standalone feasibility method this is useless (it leaves
    inequalities exactly where it found them); as the normal half of a
    composite step it is precisely right.
    """
    n = problem.n
    eq = [i for i, c in enumerate(problem.constraints) if c.operator == '==']
    if not eq:
        return np.array(x, dtype=float), 0.0
    lo = np.array([b[0] if b and b[0] else 1e-30
                   for b in (problem.bounds or [(None, None)] * n)])
    hi = np.array([b[1] if b and b[1] else 1e30
                   for b in (problem.bounds or [(None, None)] * n)])
    x = np.array(x, dtype=float).copy()

    def _res(z):
        r = np.zeros(len(eq)); J = np.zeros((len(eq), n))
        for k, i in enumerate(eq):
            r[k], J[k] = _con_loggrad(problem.constraints[i], z, n)
        return r, J

    for it in range(iters):
        r, J = _res(x)
        nrm = float(np.max(np.abs(r)))
        if verbose:
            print(f"    restore {it:2d}  max|h| = {nrm:.3e}")
        if nrm <= tol:
            break
        # lstsq, not pinv @ r: for an underdetermined system (802 equalities
        # in 1173 variables on the aircraft) it returns the same least-norm
        # solution without forming the pseudo-inverse, and this runs once per
        # Phase II iteration now, not once per solve.
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



def _violation_ineq(problem, x):
    """Worst ``log g_i(x)`` over the INEQUALITIES only."""
    return max((_log_g(c, x) for c in problem.constraints
                if c.operator != '=='), default=0.0)



def convergence_report(problem, res, options=None, k=8):
    """Explain what a solve did, and if it stopped short, what held it back.

    "did not converge within 400 iterations" is a true statement that tells an
    engineer nothing. It does not say whether the design is usable, which of
    the three KKT criteria was binding, how close it came, or -- most useful of
    all -- WHICH variables carry the residual. Nearly always the answer is a
    handful of them, and nearly always they turn out to be a sub-model that
    sizes nothing in the converged design.

    That case is worth naming, because it looks like failure and is not. On the
    E175 the objective is stable to eight figures by iteration 75 and the run
    then spends 325 more iterations dragging stationarity from 2e-04 to 6e-06,
    all of it inside twelve fuselage bending variables whose bending stations
    have run past the tail (x_hbend = 39.4 m on an aircraft whose tail is at
    23.8 m) and whose reinforcement areas are ~1e-04 m^2. The shell carries the
    loads unaided, the block is flat, and a 1e-06 stationarity ask of a flat
    block is not reasonable. The design is converged; the measure is not.
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

    # Is the objective actually still moving? A run whose objective is stable
    # to eight figures has converged in every sense an engineer cares about,
    # whatever the KKT residual says.
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

    Each round is a normal step and a tangential step, the classical split:

      normal      Gauss-Newton onto ``h(x) = 0``, on the TRUE residuals.
                  Exact, quadratic, and it ignores the inequalities entirely.
      tangential  one elastic-L1 sub-problem with the equalities imposed
                  EXACTLY, so the step reduces inequality violation without
                  leaving the (linearised) manifold.

    The tangential step leaves the true manifold at second order, which is
    what the next round's normal step is for. This is why the two halves have
    to alternate rather than run once each.

    It exists because neither half works alone on a tightly coupled model.
    Slacking the equalities lets the sub-problem trade equality residual for
    inequality residual and wander; it stalls on the 1173-variable aircraft
    with 802 signomial equalities, at every iteration budget, under both
    min-max and L1. Holding them exactly from an INCONSISTENT point is worse
    still -- the sub-problem is flatly infeasible at iteration 1. Holding them
    exactly from a RESTORED point is guaranteed feasible, because the AGM
    condensation is tight at its expansion point, so ``d = 0`` is always
    available and the sub-problem can only improve on it.

    Falls back to plain slacked L1 if a hard sub-problem fails at the minimum
    trust radius -- that means the linearised manifold and the trust region
    genuinely do not intersect, and slack is then the honest response.
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
        return restore_equalities(problem, z,
                                  iters=options.phase1_restore_iterations,
                                  tol=min(tol, 1e-10))

    if _violation(problem, x) <= tol:
        # Already feasible: do not touch it. Restoring first would still land
        # on the manifold, but at a DIFFERENT point, and on a non-convex
        # problem the starting point picks the local optimum -- it moved
        # SPaircraft from 95120 to 95560 lbf and simpleac from 4536 to 6485
        # with no constraint anywhere reporting a problem.
        return x, 0, True, slacks, mults

    x, heq = _restore(x)                      # start on the manifold
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
        # The pair is accepted or rejected TOGETHER. Judging the tangential
        # step on its own is what made this oscillate: it would report
        # 5e-02 -> 1.6e-04, and then the restoration that has to follow it
        # would put the violation back at 5e-02, because a long tangential
        # step leaves the manifold far enough that pulling it back moves the
        # inequalities too. Accepting only when the WHOLE step improves makes
        # the trust radius responsible for that drift, which is what a trust
        # radius is for.
        x_try, heq_try = _restore(x * np.exp(d))
        new_viol = _violation(problem, x_try)
        if options.verbose:
            nz = int(np.sum(slacks > tol))
            print(f"  phase1-comp {it:3d}  |h| {heq_try:.1e}  max log g "
                  f"{viol:+.3e} -> {new_viol:+.3e}  sum(s) = {total:.3e}  "
                  f"({nz} slack, {mode}, r = {radius:.2g})")
        if new_viol >= viol:
            radius *= options.trust_shrink
            if radius < options.trust_min:
                # The composite step cannot improve at any length. Either the
                # remaining violation is genuinely irreducible or it is not
                # reachable this way; the elastic L1 answers which, and names
                # the rows if it is the former.
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
    # The elastic Phase I is a sequence of IPOPT solves. With no IPOPT every
    # one of them fails, nothing moves, the slacks stay at their starting
    # zeros, and the report concludes the model is feasible at the initial
    # guess -- which is the most damaging possible wrong answer here, since
    # this function exists to be believed about feasibility.
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
    # Checked here, before anything else, because every sub-problem is an
    # IPOPT solve and the loop degrades quietly without one: each sub-problem
    # fails, the trust region never moves, and SIA returns after zero
    # iterations with `converged=False` and the objective *evaluated at the
    # initial guess*. That is a plausible-looking number produced by no
    # optimization at all, and the caller has to read `converged` to know it.
    # A missing install should say so.
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
    n_eq_p2 = sum(1 for c in problem.constraints if c.operator == '==')

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
    # Smallest radius at which a step has been REJECTED since the last accepted
    # one. The 'unreachable -> widen' and 'bad step -> shrink' branches otherwise
    # fight: shrink to r, find the sub-problem infeasible there, widen back to 2r,
    # take the same rejected step, shrink to r again -- a deterministic cycle that
    # burns the whole iteration budget re-evaluating cached points (measured on the
    # 4-variable ROM section: 300 iterations, 27 distinct black-box evaluations).
    _reject_radius = np.inf
    #: Which constraints are linearized -- the block the trust region and the
    #: ratio test exist for.
    lin_idx = [i for i, con in enumerate(problem.constraints)
               if isinstance(con.body, Signomial)
               and not isinstance(con.body, (Posynomial, PosynomialRatio))]
    x_prev_for_curv = x.copy()

    _trust_forced = False       # a step was rejected with the box off -> re-arm
    _zz_prev = None             # last ACCEPTED iterate, for zigzag detection
    _zz_cool = 0                # expansion hold-off after a zigzag detection
    _zz_hits = 0                # detections, reported on the result
    _fg_trips = 0               # times the feasibility guard condition HELD
                                # (counted whether or not it was applied)
    # TANGENT EQUALITIES get the same treatment as the linearized class:
    # a CondensedEquality is tangent -- neither inner nor outer -- so a
    # Phase II step CAN violate it, recoverably. That is exactly the
    # filter-plus-restoration situation, and it needs no exchange rate.
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
    # (_has_tangent_eq hoisted above the filter init; see there.)
    for k in range(options.max_iterations):
        _tgt = float(getattr(options, 'blackbox_step_target', 0.0) or 0.0)
        _radius_eff = radius
        _wi = int(getattr(options, 'blackbox_warm_iters', 0) or 0)
        if has_blackbox and k < _wi and _tgt > 0.0 and _bb_g1 > 0.0:
            _radius_eff = min(radius, _tgt / _bb_g1)
        _ti = getattr(options, 'trust_iterations', None)
        _released = (_ti is not None and k >= int(_ti) and not _trust_forced)
        if _released:
            _radius_eff = None
        try:
            d, s, mults, model_obj = _subproblem(
                problem, x, tau, _radius_eff, options, has_blackbox, curvature=curvature,
                use_slacks=use_slacks, cache=cache,
                force_trust=_has_tangent_eq)
            # NOTE: the filter does NOT supersede the trust region -- it
            # cannot size steps, only accept them. Measured without the
            # cap: tangent-equality subproblems have near-zero curvature
            # along the design levers, and the steps thrashed +-0.5 log
            # units in alternating directions from iteration 2 (pi_f
            # 1.69->1.82->1.57->2.16->1.43->2.20), then froze 35
            # iterations in filter rejection. Filter for recoverable
            # excursions, trust region for scale: complementary.
        except RuntimeError as exc:
            # A sub-problem that cannot be SOLVED is the strongest possible
            # signal that the linearization here is unusable -- and this path
            # `continue`s, so it used to skip the restoration trigger entirely.
            # Measured on 6t+6c: one huge first step (|d| = 3.17 in log space),
            # then "unreachable / failed" cycling 2.5 -> 5 -> 10 -> 2.5 for all
            # 1500 iterations with restorations = 0. _reject_radius also stays
            # inf here (no ratio-test rejection ever happened), so the bracket
            # hook below could never fire either. Restoration was unreachable.
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
            # An INFEASIBLE sub-problem is not a bad step -- it means the
            # trust region is too tight for the relaxed feasible set to be
            # reachable from here, which happens on the first pass from a
            # badly infeasible start. The trust-region response to a bad step
            # is to shrink; the response to an unreachable one is the
            # opposite. Widen and retry before giving up.
            if (has_blackbox and "infeasible" in str(exc).lower()
                    and radius < options.trust_max):
                grown = min(options.trust_max, radius * options.trust_expand)
                if grown >= _reject_radius:
                    # Widening would return to a radius whose step was already
                    # rejected from this iterate. The region is bracketed: too
                    # small to be reachable, too large to be believed. Stop rather
                    # than cycle -- the honest report is that the linearised model
                    # cannot represent the problem here.
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
            # Any OTHER sub-problem failure is a numerical one, not a statement
            # about the feasible set: Ipopt returning internalSolverError on a
            # model built at a badly-scaled iterate. The trust-region response to
            # that is the same as to a bad step -- shrink and re-form the model --
            # not to abandon a solve that is otherwise converging. Only give up
            # once the radius has collapsed.
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
        # --- restoration trigger: infeasible and not fixing it ---------------
        # Hooking restoration only to trust-region collapse/bracket was wrong.
        # Measured on 3t+3c: the filter (correctly) accepted an excursion to
        # viol 2.69e-01, and the run then sat there for all 1500 iterations
        # WITHOUT the region ever collapsing or bracketing, so restoration never
        # fired. The condition that matters is not "the region died", it is "we
        # are infeasible and not getting closer".
        # CONSECUTIVE iterations infeasible, reset only by reaching feasibility --
        # NOT by making progress. Excusing the counter whenever the violation
        # dropped 10% was wrong: measured on the ladder, rungs 3/6/7/8 all ran the
        # full 1500 iterations at viol 1.1e-01 .. 3.3e+00 with restorations = 0,
        # because a creeping violation kept resetting the counter while never
        # reaching feasibility. Phase II's contract IS feasible iterates, so
        # sustained infeasibility is the signal regardless of its trend.
        if viol > options.feasibility_tolerance:
            _infeas_stall += 1
            _infeas_best = min(_infeas_best, viol)
        else:
            _infeas_best, _infeas_stall = np.inf, 0
        if _infeas_stall >= int(getattr(options, 'restoration_patience', 0) or 0) > 0:
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

        # Acceptance state shared by the black-box cascade and the tangent-
        # equality path: everything from the feasibility guard down runs for
        # BOTH. It used to live inside `if has_blackbox:`, which left pure-
        # signomial problems with NO acceptance test at all -- every Phase II
        # step was taken unconditionally, the filter and the trust-region
        # ratio logic were dead code, and the radius never adapted. Measured
        # on the free coupled 737 case: a two-point limit cycle (f 201853 <->
        # 202080, |d| byte-stable at 2.315) ran for 100+ iterations with
        # zero filter rejections because rejection was unreachable.
        ratio = None
        va_seen = None
        v0 = None
        _rejected_by_guard = False
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
            va_seen = None       # worst TRUE violation over the linearized rows
                                 # at x_new, whenever a level computed it
            _rejected_by_guard = False
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
                    # The objective ratio alone is not an acceptance test: a step
                    # that buys objective by walking out of the feasible set scores
                    # well on it and is taken. Measured on the free-form ROM
                    # section, iteration 1 dropped the objective 1619 -> 1521 while
                    # feasibility went 9.975e-09 -> 2.074, and the solve never
                    # recovered. Guard it: if the step made the violation materially
                    # worse than the model implied, that is a rejection regardless
                    # of what happened to the objective.
                    v_old = _violation(problem, x)
                    v_new = _violation(problem, x_new)
                    if (v_new > options.feasibility_tolerance
                            and v_new > _FEAS_GUARD * max(v_old,
                                                          options.feasibility_tolerance)):
                        ratio = -1.0

        # --- feasibility guard, applied to EVERY level of the cascade ---
        # It used to sit inside the objective-ratio fallback only, which is
        # the branch almost never taken: measured on the 2t+2c section,
        # _violation() was called ONCE in 300 iterations. So the guard was
        # effectively dead, and levels 1 and 2 accepted steps that left the
        # feasible set. Level 2 is the hole -- it scores model ACCURACY,
        # `ratio = scale/err` with `scale = max(feas_tol, 0.1*||d||)`, and
        # with ratio_accept = 1e-4 it only rejects when err > 1e4 * scale.
        #
        # Measured on 3t+3c: iterate 0 feasible at 9.98e-09, ONE accepted
        # step to 2.69e-01 -- all of it on the black-box rows, the structured
        # rows exact to 1.1e-15 -- buying the objective 1619.14 -> 1519.20.
        # The solve never recovered and aborted at iteration 7.
        #
        # va_seen is the true violation over the linearized rows at x_new,
        # which levels 1 and 2 have already computed, so this costs no extra
        # black-box calls. The structured rows cannot be violated by the step
        # (they are exact or conservative in the sub-problem), so the
        # linearized block IS the feasibility question here.
        if (_filter is not None and va_seen is None
                and _has_tangent_eq and not has_blackbox):
            # tangent equalities can be violated by the step, so the
            # feasibility question is the TRUE violation over all rows
            va_seen = float(_violation(problem, x_new))
            if v0 is None:
                v0 = float(_violation(problem, x))
        if _filter is not None and va_seen is not None:
            # Dominance, not an exchange rate. A trial is refused only if a
            # point already seen was better in BOTH objective and violation.
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
            if (va_seen is not None and v0 is not None
                    and va_seen > options.feasibility_tolerance
                    and va_seen > _fga
                    and va_seen > max(float(_fg or 0.0), 1e-300)
                    * max(v0, options.feasibility_tolerance)):
                _fg_trips += 1
                if _fg:
                    ratio = -1.0
                    _rejected_by_guard = True

        if ratio is None:
            ratio = 1.0     # nothing measured this step; accept it
        if ratio < options.ratio_accept:
            if _released:
                # Rejected with the box OFF. Shrinking a radius that is not
                # being applied would re-solve an identical sub-problem and
                # spin, so re-arm permanently, seeded from the step that just
                # failed rather than from a stale radius.
                _trust_forced = True
                radius = max(options.trust_min,
                             min(radius, float(np.abs(d).max()))
                             * options.trust_shrink)
                if options.verbose:
                    print(f"  itr {k + 1:3d}  REJECT ratio={ratio:.3e} with the "
                          f"trust box released; re-arming at {radius:.3e}")
                continue
            # A FEASIBILITY rejection is not evidence that the linearized
            # model is wrong -- only that this step was too long to stay in
            # the feasible set. _reject_radius exists to detect the former,
            # and letting the guard write to it conflates the two: measured
            # on 2t+2c, three guard rejections were enough to bracket the
            # region and abort the whole solve at iteration 8 (W 1605, viol
            # 2e-04) on a problem that converges in 192 iterations untouched.
            # Shrink and retry, but leave the bracket alone.
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
                res.status = ("trust region collapsed at iteration "
                              f"{k + 1}; the linearized constraints are "
                              "not modelling the problem")
                res.x, res.objective = x, f_old
                res.iterations = k + 1
                break
            continue
        # Once the curvature models are TRAINED the trust region has done its
        # job and should get out of the way. It exists only to substitute for
        # a conservative model, and B supplies that as soon as it has data --
        # so past warm-up, expand on every accepted step rather than only on a
        # good ratio. Without this the radius ratchets: each early rejection
        # divides it by four, nothing multiplies it back on a merely-adequate
        # ratio, and the solve crawls at whatever radius an early rejection
        # set (measured on the 16-variable ROM section: stationarity pinned at
        # 1.37e-02 with |d| ~ 1e-03 and radius 2.4e-04, feasible to 3e-08 but
        # unable to take the step that would make it stationary).
        # --- zigzag damping (see the option's comment for the measured
        # cycle this exists for). Two sizeable accepted steps that cancel
        # mean the linearization is flipping between wells the filter
        # cannot arbitrate; only a smaller radius localizes it.
        _zz_hit = False
        if (getattr(options, 'zigzag_damp', True)
                and (_has_tangent_eq or has_blackbox)
                and _zz_prev is not None):
            # L2 over the WHOLE vector, not the max component: a single
            # lever flipping while the other 1600 variables advance is
            # progress, not a cycle. Measured with the max-norm version:
            # false detections during a healthy well-descent collapsed
            # the radius, tripped restoration to the same feasible point
            # every time, and the solve replayed a byte-identical 7-state
            # loop (f 472182.4 -> ... -> 218126.2 -> restore, twice).
            _d1 = float(np.linalg.norm(x_new - x))
            _d0 = float(np.linalg.norm(x - _zz_prev))
            _net = float(np.linalg.norm(x_new - _zz_prev))
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
                if options.verbose:
                    print(f"  itr {k + 1:3d}  ZIGZAG net={_net:.2e} vs "
                          f"step={_s:.2e}; radius -> {radius:.3e}")
        if _zz_cool > 0 and not _zz_hit:
            _zz_cool -= 1
        if _zz_hit or _zz_cool > 0:
            pass                    # hold the radius; no expansion
        elif ratio > options.ratio_expand:
            radius = min(options.trust_max, radius * options.trust_expand)
        elif _curv_trained:
            # Trained, but the step was only adequate. Grow GENTLY rather than
            # by the full factor: doubling from a radius that works lands
            # squarely on one that does not, and the region then oscillates
            # accept-accept-reject forever (measured on the ROM section: a
            # period-3 cycle, 9.8e-04 -> 2.0e-03 -> 3.9e-03 -> REJECT, ratio
            # exactly -3.004 every time, a third of the iterations wasted and
            # the objective creeping 0.04 kg per cycle). The square root of the
            # expansion factor walks up to the usable radius instead of
            # vaulting past it.
            radius = min(options.trust_max,
                         radius * math.sqrt(options.trust_expand))

        _zz_prev = x.copy()
        x = x_new
        _reject_radius = np.inf              # progress: the bracket is stale
        if (getattr(options, 'phase2_restore', False) and n_eq_p2
                and not has_blackbox):
            x_r, _heq = restore_equalities(
                problem, x, iters=options.phase1_restore_iterations,
                tol=min(options.feasibility_tolerance, 1e-10))
            # Only if it actually helps. Restoration is least-norm, so it
            # barely moves the inequalities; if it somehow makes the worst
            # violation worse, the step was not the problem and the plain
            # iterate is the safer one to keep.
            if _violation(problem, x_r) <= _violation(problem, x):
                x = x_r
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
            # Trained once EVERY black-box row has enough BFGS data for its B to
            # be a real curvature estimate rather than the seeded prior. Taking
            # the minimum, not the mean, is deliberate: one untrained row is one
            # row whose model is still only as conservative as the prior, and the
            # step is bounded by the worst of them.
            _bb_g1 = max((float(np.abs(cv.last_grad).sum())
                          for cv in curvature.values() if cv.last_grad is not None),
                         default=_bb_g1)
            _warm = int(getattr(options, 'curvature_warmup', 0) or 0)
            if _warm and curvature:
                _curv_trained = min(cv.updates for cv in curvature.values()) >= _warm

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
    if (options.kkt_min_norm and not res.converged
            and viol <= options.feasibility_tolerance
            and (stat > options.stationarity_tolerance
                 or comp > options.complementarity_tolerance)):
        # the run stopped short with the sub-problem's duals failing the
        # KKT test -- ask whether MINIMUM-NORM multipliers certify the
        # point before reporting failure (split equalities make the
        # returned duals' stationarity meaningless; see kkt_min_norm)
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
    # Attached to every result, not just the failures: the same text explains
    # why a run that DID converge converged, and the degenerate-block warning
    # is worth seeing either way.
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
