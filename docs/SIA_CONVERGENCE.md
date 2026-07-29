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

## Why the step is small: the equality pair pins it to a null space

This can be made exact, and it is the sharpest statement of the gap.

The bridge writes a signomial equality as a **pair**, `p/q <= 1` and `q/p <= 1`,
and SIA condenses the denominator of each. At the iterate both are active and
tangent with *opposite* gradients. Write

    f1 = log p - log q_hat        f2 = log q - log p_hat

Both are convex, because the condensed term is a monomial and so linear in log
space. Both vanish at `x_k`, and `grad f2 = -grad f1` there by tangency. A step
`d` must satisfy both `<= 0`, so to second order

    ½ dᵀH₂d  <=  grad f1 · d  <=  -½ dᵀH₁d

which has a solution only when `dᵀ(H₁ + H₂) d <= 0`. `H₁` is the log-space
Hessian of `p` and `H₂` that of `q`; both are positive semidefinite because a
posynomial is log-convex. **So the step is confined to the null space of
`H₁ + H₂`** — the directions in which both posynomials are locally linear.

PCCP condenses the numerator too, so `p_hat` and `q_hat` are both monomials,
`H₁ = H₂ = 0`, and the same condition collapses to `grad f · d = 0`: a full
`(n-1)`-dimensional hyperplane tangent to the true feasible manifold.

That is the whole difference. Not a heuristic about step lengths — SIA's
sub-problem is genuinely restricted to a lower-dimensional subspace wherever a
signomial equality is active, and SPaircraft has about 40 of them. It also
explains why the feasibility-verified step expansion is inert: there is no
slack to expand into, because the binding set is a manifold rather than a
region.

The targeted repair is to condense the numerator **only for equality-derived
pairs**, recovering PCCP's hyperplane there while leaving genuine inequalities
exact and conservative. `condense_numerator` currently applies to every ratio,
which gives up conservatism on inequalities that never needed it.

## Result of the fix

`split_equalities=False` on SPaircraft, 250 iterations:

| | objective | violation | stationarity | time |
|---|---|---|---|---|
| PCCP (reference) | 93,141.76 | 3.3e-06 | -- | 179.7 s |
| split pair (old) | 121,355.61 | 1.5e-07 | 0.42206 | 65.9 s |
| **one equality (new)** | **95,559.98** | **1.6e-07** | 0.53966 | **23.3 s** |

The **step** problem is largely solved. The gap to PCCP closes from 30% to
2.6%, the iterate is *more* feasible than PCCP's, and the run is 7.7x faster
than PCCP and 2.8x faster than the old SIA path -- the extra speed coming from
both the smaller constraint count and the cache, which now handles the
condensed equality.

### The certificate is a separate bug

Not solved by the equality fix, and the new data says why the earlier account
was incomplete. With the pairs gone the conditioning is fully repaired -- the
largest multiplier falls from 4047 to **3.34**, with none above 10 -- and yet
stationarity is unchanged. Measured at SIA's own point:

```
SIA's multipliers          : 0.538787
best any-sign multipliers  : 1.1e-14     <- the point IS stationary
best sign-valid (NNLS)     : 0.0120
```

The objective gradient lies *exactly* in the span of the constraint gradients,
so SIA is landing on what is essentially a KKT point and mis-reporting it by a
factor of ~45. The multipliers are no longer large; they are simply wrong.

One methodological caution, recorded because it cost time here: with 1165
constraints against 1068 variables the system ``G lam = -g0`` is
**underdetermined**, so the multipliers are not unique and ``lstsq`` returns
the minimum-norm member of a family. Comparing SIA's duals element-wise
against it -- by sign, by magnitude, or by correlation -- is not diagnostic,
and an attempt to do so here produced numbers that looked damning and meant
nothing. The only sound statement is that SIA's multipliers fail stationarity.

### Resolved: it is the bound multipliers

Stationarity must be the **projected** gradient. Every variable carrying the
residual sat at a bound, each with the sign its bound admits:

```
FS_hft[2]     +0.5388  AT LOWER BOUND     (positive -- admissible)
Eng_OPR[2]    -0.1217  AT UPPER BOUND     (negative -- admissible)
Wing_tau      -0.1039  AT UPPER BOUND
Wing_b        -0.0853  AT UPPER BOUND
Wing_lambda   +0.0793  AT LOWER BOUND
```

At a lower bound only a negative gradient is a violation; at an upper bound
only a positive one. The rest is held by the bound's own multiplier. Measured
after the change:

| | objective | violation | stationarity | complementarity |
|---|---|---|---|---|
| split pair | 110,787.40 | 1.5e-07 | 0.389 | 1.2e-04 |
| **one equality** | **95,559.98** | **1.6e-07** | **0.001963** | **5.2e-07** |

A factor of 274 on stationarity, and complementarity inside tolerance. SIA now
beats the NNLS estimate of the best achievable residual (0.0120) at the same
point, because that estimate was computed against the unprojected measure.

**Why this took so long to find.** The hypothesis was tested early and
rejected, and the rejection was correct then: under the split-equality path
only 5 of 1121 variables ever reached a bound, so projecting changed nothing.
Fixing the step is what exposed it -- SIA can now move, and it moves onto the
design limits the model exists to express. The two defects were sequential, not
independent, which is why the hypotheses below all died first.

**Five hypotheses tested and refuted** before that, each by measurement:

1. *Sign convention.* All six conventions tried on the raw duals; the one in
   use is already the best (0.502 against 3.88 for the alternatives).
2. *Bound multipliers ignored by ``_kkt``.* Only 5 of 1121 variables sit at a
   bound, and the projected-gradient residual is identical to the plain one.
3. *Dual degeneracy of the equality pairs.* Real, and it was the cause of the
   step collapse -- but fixing it took the largest multiplier from 4047 to
   3.34 and left stationarity unchanged.
4. *IPOPT rescaling the problem and reporting scaled duals.* Identical to the
   last digit with ``nlp_scaling_method=none``, and with ``tol=1e-10``.
5. *Duals valid at ``d*`` but tested at ``d=0``.* Measured at both points:
   0.538848 either way. (The tempting estimate ``sum|lam| * |d| = 164 * 0.002
   = 0.35`` is a loose bound whose terms cancel in practice.)

What is known: the point is stationary (best any-sign multipliers give
1.1e-14), the multipliers are small and well-conditioned, and they nonetheless
fail stationarity by a factor of ~45.

The remaining lead is that the residual is **concentrated**, not spread. It
lives on a handful of variables -- ``Fuse_l_cone`` at 0.497, ``Fuse_l_shell``
at 0.335, ``Wing_AR`` at 0.230 -- which points at a few specific constraints
being mishandled rather than a systematic corruption of every dual. Identifying
which constraints carry those variables, and checking how each is represented
in the sub-problem against its ``log_grad``, is the next step.

The old text below is retained for the record: stationarity is unchanged at
~0.5
even though the degenerate pairs are gone. So the dual-degeneracy account
explains the step collapse, and it is confirmed on a three-variable
reproduction where multipliers fall from 3611 to 2 -- but on SPaircraft
something else holds the KKT residual up, and it is still outstanding. The
same is visible in miniature: the three-variable case reaches the exact
optimum and still reports `stat=2`, there because `y` sits at a bound and
`_kkt` ignores bound multipliers.

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

### The cause: the equality pair is dual-degenerate

It is not a sign convention. Testing all six conventions on the raw duals, the
one in use is already the best available (0.502 against 3.88 for the
alternatives). Nor is it a point mismatch between where the multipliers are
computed and where they are tested: `|d|` falls from 12.5 to 0.021 over a run
while stationarity barely moves, and a point-mismatch error would shrink with
the step.

It is the pairing. The largest multipliers come in consecutive pairs of
near-identical magnitude:

```
con 265 <= ratio mult=4047  log_g=-3.542e-08
con 266 <= ratio mult=4047  log_g=+3.542e-08
con 683 <= ratio mult=1404  log_g=-1.749e-08
con 684 <= ratio mult=1404  log_g=+1.749e-08
con 315 <= posy  mult=910.8
con 316 <= ratio mult=910.9
```

These are the two halves of one signomial equality, whose gradients are
negatives of each other. The Lagrangian sees only their **difference**,
`(lam_A - lam_B) grad g_A`, so the pair is dual-degenerate: adding the same
constant to both multipliers leaves `grad L` unchanged, and the solver is free
to return any large pair with the correct difference. It does.

That is catastrophic cancellation. With `lam ~ 4047` and a relative dual
tolerance around 1e-4, the difference carries an absolute error of about 0.4 --
precisely the stationarity floor observed. `315/316` show it directly, 910.8
against 910.9, differing in the fourth figure.

Complementarity is clean by comparison (1.4e-4; zeroing the multipliers on all
179 inactive constraints changes stationarity not at all), so the error is
specifically the paired equalities.

### Both failures have the same root

Splitting a signomial equality into two condensed inequalities produces:

1. a step confined to the null space of `H1 + H2` -- no progress;
2. a dual-degenerate multiplier pair whose difference is numerical noise -- no
   certificate.

So the fix is the same for both: **stop splitting**. Impose the equality as one
constraint with both sides condensed, which is PCCP's treatment applied only to
equalities. That gives a single signed, well-conditioned multiplier and a full
`(n-1)`-dimensional hyperplane to move in. No conservatism is lost, because an
equality never had an interior to be conservative about -- the inner
approximation argument only ever applied to the inequalities, which keep it.

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
