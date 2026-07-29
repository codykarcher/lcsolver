# Why SIA takes smaller steps than PCCP

Measured on SPaircraft (1121 variables, 1267 constraints after presolve: 1096
exact, 171 conservative, 0 linearized).

## It is not a bug — they condense differently

For a signomial equality, PCCP condenses the **numerator as well as the
denominator**. In `edi/solvers/cvxopt/SP.py` the branch is labelled *"signomial
equality, approximate numerator"*, and it turns

    p/q <= 1     into     p_hat/q_hat <= 1

a monomial inequality, linear in log space. SIA keeps `p` exact as a
log-sum-exp and condenses only `q`:

    log p <= log q_hat

Since the AGM condensation is an under-estimator, `p_hat <= p`, so **PCCP's
constraint is easier than the true one**. Its sub-problem feasible set is a
strict superset of SIA's, and SIA's steps are smaller by construction.

That conservatism is precisely what buys SIA its guarantees — every iterate
feasible for the true problem, descent structural, no line search or merit
function needed. SPaircraft is close to the worst case for the trade: of 171
condensed constraints, about 40 are signomial *equalities*, which are condensed
in **both** directions (`p/q <= 1` and `q/p <= 1`). Squeezing between two
approximations that are each tangent at `x_k` pins the step hard.

## The trace

Feasible from iteration 1, monotone throughout, and crawling:

```
  phase 1: 4 iterations, FEASIBLE, max log g = +2.336e-07
  itr   1  f=239940.50  |d|=1.252e+01  stat=5.049e-01  viol=2.34e-07
  itr  10  f=151865.35  |d|=2.245e-01  stat=4.972e-01  viol=1.31e-07
  itr 100  f=136000 ish |d|=2.1e-02    stat=4.8e-01    viol=1.4e-07
  itr 191  f=126737.99  |d|=2.130e-02  stat=4.645e-01  viol=1.43e-07
```

The objective falls by a nearly **constant absolute amount** (~250 per
iteration in the tail) rather than a constant fraction — the signature of a
step-limited method, not a badly conditioned one. `|d|` settles at ~2e-2 and
stops shrinking.

## Confirming it

`SIAOptions.condense_numerator` (default `False`) takes PCCP's side of the
trade. Both directions of the prediction hold:

| | wall | iterations | objective | max violation |
|---|---|---|---|---|
| conservative (`p` exact) | 57.8 s | 212 | 124,666 | 1.5e-07 (feasible) |
| PCCP-style (`p` condensed) | 70.0 s | 300 | 102,653 | **4.3e-02** (infeasible) |

Larger steps, much faster objective progress — and the feasible-iterate
guarantee is gone, exactly as the theory says it must be.

**Read that table carefully.** The better objective is partly *because* the
point is infeasible: 102,653 at 4.3% violation is not comparable to a feasible
92,789, because it is optimizing outside the feasible set. The experiment
validates the mechanism; it is not a fix.

Tangency does survive condensation — `p_hat` matches `p` in value and gradient
at `x_k` — so the sub-problem's duals still certify the *original* problem and
the KKT termination test stays honest. That is the intended use of the flag:
PCCP's step length with SIA's stopping rule, for anyone who wants it.

## The reported residual is a multiplier bug, not a distance from the optimum

The stationarity SIA reports sits at 0.4-0.5 *wherever it is* -- 0.4221 from a
cold start 30% away from the optimum, 0.5372 started at PCCP's own answer,
0.4977 at its own final point. A residual that does not shrink as the iterate
approaches the optimum is not measuring optimality.

Testing it directly, at SIA's own final point, by asking what the **best
possible** multipliers could achieve there:

```
SIA's own multipliers          : 0.497732
best any-sign multipliers      : 0.008744
best SIGN-VALID multipliers    : 0.020847   (NNLS: lambda >= 0 on inequalities,
                                             free on equalities)
```

KKT-legal multipliers exist at that point giving a stationarity of 0.021, and
SIA reports 0.498 -- **24x worse**. The residual is dominated by how the duals
are extracted from the sub-problem, not by how far the point is from optimal.

That accounts for every symptom at once: the residual pinned near 0.5
regardless of the iterate, `converged=False` on every run, and every run going
to `max_iterations`. **SIA's termination test cannot fire on this problem**, so
it never stops early, and it is far closer to optimality than it can prove.

The caveat is that 0.021 is still well above the 1e-6 tolerance, so the point
is not a converged KKT point either. There is genuine headroom, just an order
of magnitude less than the reported figure suggested.

Where to look next: the active set at that point is 1024 constraints, 648 of
them equalities. Each signomial equality enters the sub-problem as a *pair* --
`p/q <= 1` and `q/p <= 1`, both condensed -- and that pair is exactly where an
inequality multiplier would want to take a negative value, the two directions
of one equality pulling against each other. Comparing SIA's extracted duals
against the NNLS solution element-wise, looking for sign flips or a systematic
scale factor on the paired constraints, is the diagnostic that should isolate
it.

## What is still unexplained

**Neither variant converges on SPaircraft.** Conservative reaches 124,666 (34%
above the optimum near 92,789) in 212 iterations before IPOPT hits its own
iteration cap; PCCP-style reaches 102,653 but infeasible, at 300 iterations.
Both are still descending when they stop. So numerator condensation is a real
and large factor, but it is not the whole story.

Ruled out along the way, both by measurement:

* **Bound multipliers inflating the KKT residual.** `_kkt` ignores them, which
  would be wrong for a variable sitting at a bound. Measured: only 5 of 1121
  variables are at a bound, and the projected-gradient stationarity is
  identical to the plain one (0.497213 either way). The residual lives on
  interior variables — `Fuse_l_cone`, `Fuse_l_shell`, `Wing_AR`.
* **Slacks left switched on.** The trace prints `tau` escalating to 1.6e+04,
  which looks like penalty CCP running when it should not be. It is cosmetic:
  Phase I succeeds in 4 iterations, `use_slacks` correctly becomes `False`, and
  `tau` is printed but unused.

## The PCCP baseline: use the IPOPT backend

PCCP's *inner* solve is a geometric program, and which GP solver runs it
matters. Through **cvxopt** it fails on this model — `status='unknown'`, and
`solve_GP` raises. Through **IPOPT** it works, and that is the path to use:

```python
solve(m, solver='ipopt-convex')      # PCCP loop, IPOPT GP solve underneath
```

`_convex_ipopt` routes a detected signomial program into the same
`solve_SP` penalty convex-concave loop with `gp_solver=solve_gp_rows_ipopt`, so
the algorithm is identical and only the convex sub-solver changes.

The cvxopt failure is not caused by the presolve work — the only change to the
cvxopt backends is an additive `require_bounds_as_rows` guard, which fires only
when bounds are split out, and the PCCP path never splits them. But it is worth
noting where cvxopt's error points, because it is the same place the presolve
checks do:

> *a large negative value here means the geometric program is unbounded below,
> which usually means a variable has no lower bound*

The two variables the boundedness check reports as unbounded above are
`Wing_A_tri` and `M_r_out` — both also on the degenerate list, and both among
the variables gpkit itself flags on the upstream model. Upstream `wing.py`
carries `Atri <= 1e10*units('m**2')` on the line after `A_tri`'s definition,
which this port does not.

An interior-point method in log space can tolerate a variable running off to
1e30 far better than cvxopt's solver does here, which is a plausible reason the
IPOPT backend succeeds where cvxopt does not — the unboundedness is real in
both cases, and only one solver is upset by it. Restoring the upstream bound
would be the way to test that, but it is a workaround for a modelling defect
rather than a fix: `A_tri` is consumed by nothing at all (see
`docs/PRESOLVE.md`), and the honest repair is to connect it or delete it.
