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

## The PCCP baseline is not currently reproducible

Attempting to measure PCCP on the same model for comparison, cvxopt returns
`status='unknown'` and `solve_GP` raises. So the **146 s / 20939.1** figure
quoted in `SLCP_PERFORMANCE.md` and elsewhere cannot presently be reproduced,
and should not be cited as a baseline until it can.

This is not caused by the presolve work — the only change to the cvxopt
backends is an additive `require_bounds_as_rows` guard, which fires only when
bounds are split out, and the PCCP path never splits them.

cvxopt's own error message points at the same place the presolve checks do:

> *a large negative value here means the geometric program is unbounded below,
> which usually means a variable has no lower bound*

The two variables the boundedness check reports as unbounded above are
`Wing_A_tri` and `M_r_out` — both also on the degenerate list, and both among
the variables gpkit itself flags on the upstream model. Upstream `wing.py`
carries `Atri <= 1e10*units('m**2')` on the line after `A_tri`'s definition,
which this port does not. Whether restoring that bound is what PCCP needs is
the obvious next experiment.
