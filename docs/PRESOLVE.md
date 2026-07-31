# Presolve checks

Two questions get confused with each other, and they need different tools.

**Is this variable determined by the structure of the model?** Visible in the
sparsity pattern alone, before anything is solved. A variable in no constraint,
a variable in exactly one, a "constraint" that is really a bound, a variable
nothing can push down on. These are the classical presolve reductions and they
cost nothing to find.

**Is this variable determined at the optimum?** Not visible in the structure at
all. A variable can sit in half a dozen perfectly ordinary constraints and
still be free at the solution, because every one of them goes slack there.

Measured on SPaircraft: of the 32 variables the optimum leaves undetermined,
the structural checks find 7. Both are worth running.

## Using it

```python
from edi.preconditioner.presolve import presolve_report, degeneracy_report, fold_singleton_rows
from edi.preconditioner.structureDetector import structure_detector
from edi.preconditioner.unitCorrector import unit_corrector

structures = structure_detector(unit_corrector(model), bounds_as_rows=False)
print(presolve_report(structures))
```

```
presolve: 1173 variables, 3731 rows
  2513 rows (67%) are single-variable bounds and could be folded into variable bounds
  52 variables are fixed by equal bounds and could be substituted out
  107 variables: appears in only one constraint
  6 variables: is held only by its own bounds
  Wing_A_tri is not upper bounded
  HT_C_L_ht_fCG is not upper bounded
```

After a solve, for what structure cannot see:

```python
free = degeneracy_report(problem, x, names=names)
```

## Boundedness

The check gpkit prints as `x is not upper bounded`. In log space a posynomial
constraint `sum_k c_k prod_j x_j^a_jk <= 1` bounds `x_j` from **above** through
any term with `a_jk > 0` — pushing `x_j` up pushes the constraint toward
violation — and from **below** through any term with `a_jk < 0`. An equality
bounds both ways. For a ratio `p/q <= 1` the numerator counts as written and
the denominator counts negated, since growing `q` relaxes it. The objective
counts too: minimising a term with a positive exponent on `x` pushes `x` down.

The subtle part is which *bounds* count. Two obvious rules are both wrong:

* **Count every bound.** EDI gives every variable a default `1e-30..1e30` box,
  so everything comes back bounded both ways and the check reports nothing.
* **Ignore every single-variable row.** Now a hand-written `w >= 1` stops
  counting, and the check reports variables that are perfectly well bounded.

So the test is on the **value**, not the shape: a bound counts unless it is
vacuous (`VACUOUS_LO`/`VACUOUS_HI`, defaulting to 1e-29/1e29). `x is not upper
bounded` then means *nothing in this model holds x down except a limit chosen
to be no limit at all*. That is what makes the list actionable — on SPaircraft
it is the difference between 22 flagged variables and 2.

## Bounds as bounds

`structure_detector(..., bounds_as_rows=False)` publishes declared bounds as
`structures['bounds']` instead of materializing them as Pyomo constraints.
`fold_singleton_rows` then moves the hand-written single-variable rows into the
same place.

On SPaircraft:

| | constraints | first SIA sub-problem |
|---|---|---|
| bounds as rows (default) | 6077 | 19.9 s |
| bounds split out | 3731 | 13.7 s |
| + singleton rows folded | **1218** | **6.4 s** |

A bound costs a solver nothing; the same statement as a row is one more
log-sum-exp to build and differentiate every iteration. Four fifths of this
model's "constraints" were bounds in disguise.

Nothing is tightened, rounded or clipped. Several rows bounding the same
variable all apply and the tightest wins; an equality row fixes it. This
matters more than it sounds: SPaircraft needs its full `1e-30..1e30` box for
the reference solution to lie inside it, so a presolve that tidied those limits
would cut off the answer.

`bounds_as_rows` defaults to `True`, so existing callers are unaffected. The
cvxopt backends read bounds only from the rows and now **refuse** split
structures outright rather than silently solving an unbounded relaxation — a
relaxed problem still returns an answer, and that answer can look entirely
reasonable.

## Removing columns

`reduce_columns` is the one reduction that removes *variables*, and it removes
only what is provably inert:

* **disconnected** — appears in no real constraint and no objective term. In
  `min x**2 s.t. x >= 1, y >= 4`, nothing determines `y` and nothing depends on
  it. Fixed at its tightest finite bound and dropped.
* **fixed** — equal bounds make it a constant wearing a variable's clothing.
  Its value is folded into the coefficient of every term it appears in,
  `c * v**a`, and the column goes.
* **output-only** — computed from the design and read by nothing. See below.

Both are exact: the optimal objective is unchanged and every removed variable
gets a value feasible for the original problem.

`solve_slcp` and `solve_sia` take `presolve=True` **by default** and restore the
removed variables into `result.x`, so a caller sees the original variable
ordering and need not know anything happened. `result.removed` records what was
taken out.

On SPaircraft this removes 52 variables, all `fixed` constants
(`Wing_c_m_w=1.9`, `VT_A_vt=2.2`), and none `disconnected` — every variable is
either in a real constraint or a constant. Verified exact at the reference
solution: objective identical to 15 digits, worst violation identical, and the
restored vector round-trips bit-for-bit.

### Output-only variables

Common in engineering models and bad for the optimizer: a quantity computed
*from* the design so a human can read it, which nothing downstream consumes.
Lift-to-drag ratio is the canonical example — you want it in the report, but
no constraint depends on it, so every iteration spends effort solving for a
number that could have been worked out once at the end.

A variable is output-only when

* it appears in exactly **one** constraint,
* it is **absent from the objective**, and
* that constraint cannot restrict anything else through it — which needs both
  that the constraint is monotone in the variable (so it can always be
  satisfied by moving it) **and** that the bound in the relaxing direction is
  vacuous (so moving it is actually allowed).

That last clause is the one that is easy to get wrong. `A >= 3*x` with `A`
unbounded above says nothing whatever about `x`. Add `A <= 4` and it suddenly
forces `x <= 4/3` — a real restriction on a real variable, and eliminating `A`
would silently drop it. There is a test for exactly this.

Detection is **iterative**, because removing one output can expose another
behind it: a reporting quantity computed from a reporting quantity. Peel order
then matters for the reverse reason — a variable peeled in round 1 may sit in
the constraint defining a variable peeled in round 2, so recovery runs in
**reverse** peel order.

`eliminate_outputs=True` is the default. After the core solve,
`restore_columns` post-computes each one by solving its defining constraint at
the solved values of everything else (bisection in log space, which is safe
because monotonicity is exactly what qualified the variable). The caller gets a
full-length solution vector and cannot tell which quantities took part in the
optimization and which were worked out afterwards.

On SPaircraft this finds **52**, and the list reads like a report page:

```
LoD[0..4]            lift-to-drag ratio
C_D[0..4]            drag coefficient
Re_nacelle[1..4]     nacelle Reynolds number
C_f_nacelle[1..4]    nacelle skin friction coefficient
L_fuse[0..4]         fuselage lift
Eng_h_25[...]        engine station enthalpy
```

Combined with the fixed columns that is 1172 variables down to 1068 and 1217
constraints to 1165.

Verified against the real solution: eliminating all 52, solving, and
post-computing them back reproduces the full 1172-variable vector to a maximum
relative error of **6.0e-10**. The values a user reads are the values the
optimizer would have produced.

### Why degenerate variables are *not* removed

This is the distinction that matters, and it cuts the other way from what
intuition suggests.

A **disconnected** variable is invisible to the model at every design point, so
dropping it can never change an answer. A **degenerate** variable typically
sits in several perfectly ordinary constraints that all happen to go slack *at
this optimum*. Remove it and you remove those constraints too — and they would
bind at a different payload, range or altitude. That is the same class of error
as relaxing an equality: the weakened problem still returns a plausible number.

Nothing in `reduce_columns` touches a variable that appears in a real
constraint. So on SPaircraft `Wing_A_tri` survives, correctly, despite being
both degenerate *and* unbounded above — it appears in its own defining
constraint. The fix for it is a modelling one (delete it, or connect it to
something), not a presolve rule.

Degeneracy also cannot gate the solve that discovers it: it is a property of
the solution, so it is only knowable afterwards.

## Sub-problem caching

Related, because it attacks the same cost. `SIAOptions.cache_subproblem`
(default `True`) builds each phase's Pyomo model once and re-points it, instead
of rebuilding every constraint symbolically each iteration. Each exact term is

    exp( log c_k + a_k . (d + log x_k) ) = exp( [const] + [a_k . d] )

where `a_k . d` is fixed and only the constant follows the iterate, so the
projections are built once and the constants become mutable Params. The
AGM-condensed denominator of a ratio reuses the same projections via
`aq . d = sum_i w_i (a_i . d)`, needing one mutable weight per term rather than
a full coefficient vector.

Measured: turbofan 23.3 s → 2.0 s (11.7×), windturbine 5.7 s → 1.4 s (4.2×).
Falls back to rebuilding for a black-box body, which has to be re-linearized
every iteration regardless.

Caching is a performance change, not a numerical one — but "identical" needs
care. On simpleac the two paths land 38% apart in `V_f_fuse` and `CDA0`, with
the objective agreeing to 2.6e-07. Both are degenerate and both sit at the
positivity floor: a flat direction can land anywhere without either answer
being wrong. The equivalence test therefore compares the objective and
constraint values everywhere, and individual variables only where the problem
determines them.

## Degenerate variables

A variable is degenerate when it can be moved in **both** directions without
changing the objective and without worsening feasibility. `degeneracy_report`
tests exactly that, by perturbation, so there is no heuristic to be wrong
about.

That matters, because the obvious structural heuristic — "only bounds are
active on it" — also flags every variable pinned by a single-variable
*equality*, and those are maximally determined rather than free. An earlier
version of this check reported 77 variables on SPaircraft, including
2π constants.

Degeneracy is usually a modelling defect, and the useful question is which
kind:

* **A missing constraint.** Something that should tie the variable to the rest
  of the model was never written, so the path is disconnected rather than
  merely slack. This is the interesting case — see below.
* **Genuine slack.** A margin that simply is not binding at this design point.
  Fine, and often expected.
* **An unused output.** A quantity computed for reporting that nothing feeds
  back into.

### Worked example: the pi-tail bending model

`HT_box_M_r`, `M_r_out`, `HT_box_I_cap` and `HT_box_t_cap` all came back
degenerate, sitting at the extreme ends of their boxes — `I_cap` at 1.0e-30,
`M_r_out` at 2.7e+15, `t_web` at a physically meaningless 129170 m. That
pattern is the signature of a disconnected path, not of slack: a slack
constraint leaves a variable somewhere sensible, whereas a disconnected one
leaves it wherever the box ends.

The cause was the root-moment constraint. On a pi tail the horizontal is
supported at two points by the verticals, so treating the root as a cantilever
pinned at the fuselage centreline does not size it. The source model wrote

```python
M_r * c_root >= L_rect*(b/4) + L_tri*(b/6) - w_fuse*L_max/2
```

whose positive and negative parts nearly cancel, letting `M_r` fall to zero and
taking the whole cap-sizing chain with it.

`build(pi_tail_supports="fixed")` adds the two-fixed-support beam model
instead — hogging `wL²/12` at the supports, sagging `wL²/24` at midspan — as
posynomials with no subtraction, so no cancellation is possible. Every affected
variable moves to a physical value:

| | pinned | fixed |
|---|---|---|
| `HT_box_M_r` | 1.4e-18 | 21437 N·m |
| `M_r_out` | 2.7e+15 | 162503 |
| `HT_box_I_cap` | 1.0e-30 | 3.5e-7 m⁴ |
| `HT_box_t_cap` | 8.0e-11 | 1.1e-4 m |
| `HT_box_t_web` | 129170 | 4.1e-4 m |
| `HT_box_W_struct` | 197.6 N | 1587 N |
| **objective** | **20939.10** | **22014.53** |

The 5.1% fuel penalty is the point: sizing the HT box was previously free
because the moment path was disconnected.

`pinned` remains the default and reproduces the source model exactly, degeneracy
and all, so `verify()` against `reference.json` only matches with `pinned`.

The weakest assumption in the fixed model is the inboard load split: it takes
the inboard share of the load as its share of the span, justified by the
section being untapered over that stretch. With real taper the inboard carries
somewhat more, so this slightly under-predicts the fixed-end moment.

## Bound propagation

`propagate_bounds` tightens variable bounds by interval propagation, and works
on an **LP, a QP, a GP or an SP**. The GP case is the interesting one: a
monomial `c * prod x_j**a_j <= 1` is *linear* once written in `y = log x`, as
`a . y <= -log c`, so the ordinary LP propagation applies with nothing changed
but the space. The arithmetic is therefore shared; only the translation in and
out differs.

A posynomial gives more than it looks like it should. Every term of
`sum_k c_k m_k <= 1` is strictly positive, so each separately satisfies
`c_k m_k <= 1` — one linear implication per term, for free. Without that this
would be nearly useless on a GP, where most constraints are posynomials rather
than bare monomials.

On SPaircraft: **2478 tightenings in 0.07 s**, fully vacuous 1e-30…1e30 boxes
down from 1025 to 359, and three more variables discovered to be constants
(52 → 55 fixed columns). The one genuinely unbounded variable stays unbounded —
it must not invent bounds, and there is a test for that.

Ratios are skipped: `p <= q` bounds neither side without a point to evaluate
at, and guessing there would be silent.

**An equality needs both endpoints.** It bounds `v_k` above using the *minimum*
of the other terms and below using their *maximum*, and those are different
sums. Reusing the minimum for both manufactures contradictions between
unrelated equalities — the first version declared SPaircraft infeasible,
"proving" a variable with a wide-open box both `<= 1.6e7` and `>= 2.6e-15` from
two monomial equalities that happened to share it.

## Monomial equality elimination

`eliminate_monomial_equalities` substitutes out variables that a monomial
equality already determines. In log space such an equality is *linear*, so it
can be solved for one variable and eliminated everywhere else — Gaussian
elimination on the exponent matrix. Solving for a pivot gives

    x_p = c**(-1/a_p) * prod_{j != p} x_j**(-a_j/a_p)

which is itself a monomial with a positive coefficient, so every term it lands
in stays a monomial. **Posynomial structure survives untouched** — nothing
becomes signomial, no approximation enters, the reduction is exact.

On SPaircraft:

| | variables | constraints | nonzeros |
|---|---|---|---|
| before | 1172 | 1217 | 5392 |
| **after** | **559** | **604** | **4524** |

Both dimensions roughly halved *and* the matrix 16% sparser, in 0.1 s.

That sparsity gain is not automatic. Elimination normally costs fill-in, and it
does here if permitted: at `max_fill=64` it removes 42 more variables but
nonzeros climb to 5883, worse than the original. Pivots are chosen by a
Markowitz estimate, `(row_nnz-1)*(col_nnz-1)`, and the default cap of 16 is
where measurement put the optimum.

Verified against the reference solution: objective 93141.764922 and worst
violation +3.304e-06, identical on the full and reduced problems, with a
back-substitution round-trip of 6.0e-10.

Only variables whose **declared** bounds are vacuous are eliminated. A real
bound does not disappear when its variable does — it becomes a constraint on
the survivors, and re-adding it as two monomial rows hands back most of the
saving.

Recovery runs in **reverse** elimination order, for the same reason the
output-only peel does: a pivot's formula is captured when it is eliminated and
may name a variable eliminated in a later round. Getting this backwards leaves
the reduced problem exactly right — objective correct to 12 figures — while
returning recovered values off by 4.3e+03 relative. A single elimination cannot
expose it; it takes a chain.

## Running it: `presolve()` and the log

```python
from edi.preconditioner.presolve import presolve

reduced, log = presolve(structures)      # order is fixed and safe
print(log)                               # optional; off by default
x_full = log.restore(x_reduced)          # every variable back, exactly
```

On SPaircraft:

```
presolve:
  bounds: rows folded 2511
  columns: removed 52 fixed, 52 output
  monomial equalities: removed 562 substituted
  666 variables removed in total; each is recovered exactly and reported with the solution
```

`log.detail()` gives the per-variable account when the summary is not enough.
Nothing prints unless asked: the reductions are exact and every removed
variable comes back in the solution, so there is usually nothing for an
engineer to act on. The exception is infeasibility, which raises.

**The pass order is a constraint, not a preference.** Two interactions force
it:

* **reduce before propagate** — output-only detection needs a *vacuous* bound
  in the relaxing direction, and propagation fills exactly those in.
  Propagating first costs 40 variables on SPaircraft.
* **eliminate before propagate** — for the same reason: elimination only takes
  variables whose declared bounds are vacuous.

`propagate` is therefore off by default. It is valuable as a diagnostic and for
a solver that exploits bounds, but it buys no time once elimination has run and
blocks other reductions if run early.

Each pass renumbers the columns it leaves, so a `Removed.index` is meaningful
only in the space where it was recorded. `PresolveLog` keeps the passes
separate and unwinds them in reverse, which needs no index remapping at all.
Concatenating two passes' lists would silently mix two index spaces.

Measured end to end on SPaircraft: 1172 variables and 3728 constraints down to
**506 and 603**, solve time **14.8 s to 9.1 s**, 143 iterations to 122, at an
identical objective of 95559.91 and a full-pipeline round-trip of 6.0e-10.

## Sensitivities across a reduction

Sensitivities to `Constant`s survive every reduction here, including monomial
elimination, and the reason is architectural rather than lucky.

**Presolve transforms the detected structure, never the model.** The `structures`
dict is what gets folded, reduced and eliminated; the Pyomo `Formulation` the
user holds is untouched. `sensitivities()` obtains duals via
`constraint_duals`, which prefers a populated `dual` Suffix and otherwise
recovers them from the primal solution by KKT — and the SLCP/SIA path solves a
separate `Problem` object, so it never populates a Suffix on the user's model.
Sensitivity recovery therefore always runs against the *original* constraints,
needing only `x`.

So the requirement reduces to one thing: **the full primal vector must be
restored**, which `restore_columns` does, verified at 6.0e-10 on SPaircraft.

Tested on the hardest case — a constant appearing *only* in the equality that
elimination consumes, so that after the reduction it survives nowhere but in
the coefficients it was folded into:

```
analytic   dlog(f*)/dlog(K) = -0.5
normal     sens(K) = -0.500000   (method=kkt)
eliminated sens(K) = -0.500000   (method=kkt, 1 substituted)
```

The one thing that would break this is a backend that reported duals for the
*reduced* problem onto the user's model. Nothing does today: the cvxopt
backends refuse split-bound structures outright, and the plain IPOPT route does
not presolve. A future backend that presolved *and* wrote back duals would need
a dual postsolve to match — for an eliminated equality the multiplier is
recoverable from stationarity with respect to the eliminated variable,

    lam_eq = -(dlog f/dy_p + sum_i lam_i dlog g_i/dy_p) / a_p

which is the dual mirror of the primal back-substitution, but it is not
implemented because nothing currently needs it.

## Signomial cancellation

`cancellation_report` looks for the pi-tail failure directly, rather than
inferring it from a variable parked at 1e-30.

EDI writes a constraint containing a subtraction as a ratio, moving the
negative terms into the denominator alongside the left-hand side, so
`M_r*c >= A + B - C` becomes `(A + B) / (M_r*c + C) <= 1`. The two terms in
that denominator are in direct competition: whatever `C` supplies, `M_r` need
not. When `C` supplies essentially all of it, `M_r` is inert — the constraint
holds regardless of what it does.

```python
for i, side, share, variables in cancellation_report(structures, x):
    print(f"constraint {i}: {variables} contributes {share:.1e} of its {side}")
```

Each entry is a term whose share of its own group falls below `tol` (1e-6 by
default), worst first. Only constraints with a denominator are examined — a
small term in a plain posynomial is ordinary and not a defect.

This is solution-dependent, so it runs after a solve. LP presolve has no reason
to look for it, since LP has no signomials, but on a signomial program it is
the check most likely to find a real modelling error: a subtraction that
silently disconnects the quantity it was meant to size.

### Cross-check: the checks agree with gpkit on the same model

Run against the pinned SPaircraft, the cancellation check finds the pi-tail
constraint on its own, with no prior knowledge of it:

```
pinned: con 1051  denominator share=5.29e-24  HT_c_root_ht, HT_box_M_r
fixed:  (gone)
```

That is `M_r * c_root` contributing 5e-24 of its group — the disconnected term
itself, rather than the 1e-30 symptom.

It also flags a second constraint that survives both variants, `A_tri >=
0.5*(1-taper)*c_root*b`, and there all three checks converge: `Wing_A_tri` is
the one remaining degenerate variable *and* the one variable reported "not
upper bounded". The cause is that `A_tri` is defined by that constraint and
consumed by nothing, so it floats upward until the subtracted term is
negligible beside it.

The upstream SPaircraft agrees. Its own notebook output records

```
A_{tri} : 1e+30 [m²]
value near upper bound: M_{r_{out}}..., A_{tri}..., d_{nacelle}..., \alpha_{max}...
```

— the same four variables, from gpkit's own diagnostics. Upstream `wing.py`
also carries `Atri <= 1e10*units('m**2')` on the line after the definition,
which this port does not; that bound exists only to silence the warning, and
`A_tri` reaches 1e+30 upstream regardless.

### A limitation this exposes

`VACUOUS_HI` is unit-blind. A wing area of 1e10 m² is exactly as meaningless
as 1e30, but only the latter trips the threshold — so upstream's `Atri <= 1e10`
would count as a real bound and silence the check, even though it says nothing
about any aircraft. Catching that needs a bound compared against the
variable's own scale rather than against an absolute number, which is not
implemented.

## What is not checked

Duplicate rows are reported but not removed (38 survive on SPaircraft), and
dominated columns and forcing rows are not implemented.

Elimination runs on declared bounds and does not distinguish them from bounds
*derived* by propagation. Derived bounds are implied by the constraints and so
could be discarded on an eliminated variable, which would let propagation run
first and elimination still reach the variables it tightened. As it stands the
two are best run in the other order.
