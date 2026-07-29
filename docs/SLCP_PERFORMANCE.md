# Why SLCP is slow on SPaircraft

Measured, not guessed. SPaircraft D8.2: 1173 variables, 6077 constraints.
PCCP solves it in 146 s to an objective of 20939.1. SLCP ran 200 sub-problems
in 1365 s and stopped at 85264, still descending.

## It is not the termination criterion

The natural suspicion — that SLCP's KKT test is stricter than PCCP's, so it
keeps working after PCCP would have stopped — is not what is happening. The
run is **feasible the whole way** (max constraint violation ~1e-7, inside the
1e-6 tolerance) and never comes close to either stopping test:

```
  it   objective        |d|      |gradL|   alpha
  33   109412.24    2.302e-01   5.106e+00  1.000
  34   108416.17    2.297e-01   4.946e+00  1.000
  ...
  40   104383.79    2.052e-01   5.090e+00  1.000
```

The step magnitude is not shrinking, the Lagrangian gradient is not shrinking,
and full steps are being accepted. It simply has not arrived. A stricter test
costs nothing when the looser test would not have fired either.

## It is not the cost per iteration either, mostly

Instrumenting the run:

```
wall 281.8 s over 40 sub-problems
  sub-problem solve  270.9 s  (96.1%)   6773 ms each
  Lagrangian grads     3.1 s   (1.1%)     26 ms each
  violations           4.7 s   (1.7%)     27 ms each
  everything else      3.1 s   (1.1%)
```

The sub-problem solve dominates, and everything SLCP adds over PCCP — the line
search, the merit function, the gradient evaluations — is under 3% of the
time. The sub-problem caching is working; assembly is not the cost. At 6.8 s
per solve for a problem this size, that is about what a GP solve costs too.

**So the cost is the iteration count: ~200 against PCCP's ~10.**

## The actual cause: the curvature model is empty

The sub-problem is *already* 98% exact:

```
constraints                6077
  exact in log space       5955   (Posynomial)
  condensed (linearized)    122   (PosynomialRatio, the signomial ones)
objective                  Posynomial with 1 term -- i.e. a monomial
```

The BFGS matrix `B` is built on the **Reduced Lagrangian**, which by design
omits every constraint imposed exactly. That is the point of Equation 14 and
it is correct: approximating curvature that is already in the sub-problem
exactly sets the approximation fighting the true term.

But here that leaves almost nothing in it. 5955 constraints are excluded, and
the objective is a single monomial — **linear in log space, so zero
curvature**. What remains is 122 condensed constraints with sign-indefinite
multipliers. Instrumenting the update:

```
 BFGS: |s|=5.128e+01 |z|=1.248e+14  s.z=-7.499e+14  -> SKIP (s.z <= 0)
 BFGS: |s|=1.332e+01 |z|=1.927e+01  s.z=-8.317e+00  -> SKIP (s.z <= 0)
 BFGS: |s|=1.422e+00 |z|=2.626e-02  s.z=+4.680e-04  -> ok
 BFGS: |s|=1.479e+00 |z|=1.676e-02  s.z=-1.512e-03  -> SKIP (s.z <= 0)
 BFGS: |s|=6.509e-01 |z|=3.365e-02  s.z=-1.007e-03  -> SKIP (s.z <= 0)
 BFGS: |s|=3.348e-01 |z|=3.632e-02  s.z=-1.336e-03  -> SKIP (s.z <= 0)
```

`|z|` is around 1e-2 against `|s|` around 1, and **its sign flips freely**: the
curvature condition `s.z > 0` fails on five of the first six updates. Powell
damping then gives `r ≈ Bs`, whose rank-two correction cancels itself, so the
update is a no-op. `B` stays at `gamma*I` for the entire run.

That is the answer. **SLCP on this problem is not a quasi-Newton method at
all.** With `B = I` the quadratic `0.5 d'Bd` is a proximal term of fixed unit
weight in log space, and the algorithm reduces to a proximal-point method with
a fixed trust radius — which converges linearly at whatever rate that radius
allows. Measured: 2–4% objective reduction per iteration, which needs about
200 iterations to cover the factor of four between the start and the optimum.

PCCP does not have this problem because it never needs a curvature model. It
condenses the signomial constraints and hands the result to a GP solver, whose
curvature comes from the exact posynomial structure.

## Confirming it

Two experiments, both consistent.

**Remove the quadratic entirely** (force `drop_quad`): the very first
sub-problem fails — IPOPT hits its iteration limit. So the term is doing real
work as regularization; it is not dead weight. Its *weight*, though, is
arbitrary.

**Weaken it** (`hessian_gamma = 0.01`): reaches an objective of 110011 at
iteration 3, which the default needs roughly 30 iterations to reach — about an
order of magnitude better early on. It then crawls again in the tail
(0.3%/iteration from iteration 18), so a single fixed weight is not the answer
either; but it demonstrates the mechanism directly.

## Ruling out the signomial approximation

`PosynomialRatio` -- keeping `p` exact and AGM-condensing only `q` -- is a
recent and still-speculative addition, so it is a natural suspect. It is not
the cause. Re-running with `sp_form=False`, which hands the same ratio over as
an opaque value/gradient callback so SLCP linearizes the whole body:

```
                      40 iterations    wall     s.z > 0 on
  sp_form=True           104384        282 s    ~1 update in 6
  sp_form=False          101259       1345 s    0 of 40 updates
```

**The curvature condition never once holds with the approximation turned off.**
So `B` is the identity in both configurations, for the same underlying reason,
and the asymptotic behaviour is identical -- the same ~2.5%/iteration linear
crawl. The 3% better objective costs 4.8x the wall time, because turning off
the SP form also loses sub-problem caching (a linearization moves every
iteration, so there is nothing to cache).

The transient does differ, interestingly. Linearizing everything gives BFGS
something real to chase early on -- it is ahead at every one of the first six
iterations and `|gradL|` genuinely falls, to 0.59 by iteration 4, which it
never does with the SP form. Then the line search collapses (`alpha` 1.000 ->
0.168 -> 0.044 -> 0.014 by iteration 14) as the linearization stops being
accurate enough for the merit function, it grinds through iterations 9-17
making almost no progress, recovers at iteration 21, and settles back into the
same linear rate.

So the two treatments fail in opposite ways -- accurate sub-problem with no
curvature, versus real curvature with an inaccurate sub-problem -- and neither
converges. That is further evidence the problem is the missing trust-region
management rather than the constraint representation.

## What would actually fix it

Not a better Hessian. On a problem where nearly everything is already exact
there is no curvature left for BFGS to learn, and no amount of memory helps.
The quadratic is functioning as a trust region, so it should be *managed* as
one: an adaptive radius, grown on successful steps and shrunk on rejected
ones, rather than a fixed unit weight. The machinery is half-present already
in `Options.max_log_step` and `Options.max_step_ratio`, which impose a trust
region but do not adapt it.

A narrower observation: `exact_objective` is worth having for a genuinely
posynomial objective and does nothing for a monomial one, since a monomial is
already exact in log space. SPaircraft minimises a single variable, so it was
never going to help here — which is why it did not.

## Options added while investigating

Both default to the previous behaviour, so nothing changes unless asked for.

* `hessian_gamma` (default 1.0) — the weight of the background `gamma*I`.
  When the curvature condition keeps failing, this is the proximal weight and
  it sets the step length directly.
* `hessian_scaling` (default False) — Shanno–Phua initial scaling, `y.y/s.y`
  taken from the first update. Note it does **not** fire on SPaircraft,
  because that first update has `s.z < 0`; it is there for problems where the
  curvature information is real but badly scaled.

`_lagrangian_gradient` also now takes `exact_objective`, and omits the
objective from the Reduced Lagrangian when it is imposed exactly — for the
same reason the exact constraints are omitted. Correct in principle, inert on
SPaircraft.
