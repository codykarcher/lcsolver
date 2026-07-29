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
from edi.presolve import presolve_report, degeneracy_report, fold_singleton_rows
from edi.structure.structureDetector import structure_detector
from edi.units.unitCorrector import unit_corrector

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

`fold_singleton_rows` is the only reduction that actually transforms the
problem. Duplicate rows are reported but not removed, and dominated columns and
forcing rows are not implemented.
