# Discrepancies found while rebuilding

Differences between the published papers, the published source code, and what
is actually needed to make each model reproduce its stated results.

Policy: where dimensional analysis, an internal cross-check, or the reference
implementation makes the intent unambiguous, the rebuild uses the corrected
form and says so both here and in a comment at the site of the change. Where
the intent is genuinely ambiguous, the rebuild follows the paper as printed
and the discrepancy is flagged as open.

---

## 1. Wind turbine GP — equation (14) is missing the tip speed ratio

**Status: corrected.** High confidence.

**Paper:** W. Hoburg and P. Abbeel, "Fast Wind Turbine Design via Geometric
Programming", AIAA SDM 2012, section IV.A.

**As printed, equation (14):**

```
s*b  >=  (dCp/dy) * 1/(8 y^2 F G)  +  eps*a*b
```

**Correct:**

```
s*b  >=  (dCp/dy) * 1/(8 lambda y^2 F G)  +  eps*a*b
```

### Evidence

Three independent confirmations.

**1. It follows from the paper's own (13).** Equation (13) states

```
dCp/dy = 8 s lambda y^2 F G (b - eps(lambda y + s))
```

Substituting the induced-velocity relation (6), `ab = s(lambda y + s)`:

```
dCp/dy = 8 lambda y^2 F G (s b - eps a b)
```

so the `lambda` sits with `8 y^2 F G`, exactly where (14) omits it.

**2. It is required by the paper's own (18).** Section IV.B states that with
`F = G = 1` and `eps -> 0`, constraints (14), (15) and (16) reduce to

```
dCp <= 16 a b^2 y dy / (1 + sqrt(1 + 4ab/(lambda^2 y^2)))
```

Reducing the corrected (14) symbolically gives precisely this expression
(verified with sympy — the difference simplifies to 0). Reducing (14) as
printed does not; it is short by a factor of `lambda`.

**3. Only the corrected form yields the Betz limit.** The paper states that
in the light-loading limit the model must give `Cp = 16/27 = 0.5926`.

| lambda | Cp, corrected (14) | Cp, (14) as printed |
|---|---|---|
| 4  | 0.5602 | 0.1401 |
| 8  | 0.5820 | 0.0727 |
| 16 | 0.5896 | 0.0368 |
| 32 | 0.5918 | 0.0185 |

The corrected form approaches 16/27 from below, reaching 99.87% of it at
`lambda = 32`. As printed, Cp *decreases* as `1/lambda` and never approaches
the Betz limit at all.

### Consequence

None for the paper's conclusions — Figure 1 reproduces correctly with the
corrected form (peak Cp within 0.03 of the plotted maxima across all four
`c_l/c_d` curves, with and without tip loss), so the figure was evidently
produced with the correct equation. The typo is confined to the printed
equation.

---

## 2. Wind turbine GP — number of blades not stated

**Status: open, guessed.** `B = 3`.

Figure 1's tip-loss curves depend on the blade count `B` through the Prandtl
correction (23)/(25), but the paper does not state it. The rebuild assumes
`B = 3` and flags it at the definition site.

This assumption is not load-bearing for the main result: with
`tip_loss=False` the model is independent of `B`, so the solid curves of
Figure 1 are verified without it. With `B = 3` the dashed curves also land
within tolerance, which is weak evidence the assumption is right.

---

## 3. SimPleAC — wing weight coefficient differs from the GP paper

**Status: no action, documented.**

`gplibrary`'s `SimPleAC.py` uses `W_W_coeff1 = 2e-5 1/m` and carries an inline
comment recording the original value as `12e-5`. The Hoburg & Abbeel GP paper
that SimPleAC descends from uses the larger value. The rebuild follows the
source (2e-5), since the reference solution it is verified against was
produced with it.

This is a deliberate model change, not a typo — SimPleAC is a teaching
descendant of the paper's GP rather than a reimplementation of it. Other
deliberate differences: `CDA0` is free rather than fixed at 0.035 m^2, and a
fuel-volume model has been added that makes the problem an SP.

---

## 4. EDI `examples/Kirschen2sp.py` — unit errors

**Status: corrected separately** (see git history on this branch's parent).

Six unit/typo defects were found and fixed in the pre-existing EDI example
before this rebuild started, including a missing `V_TO**2` in the takeoff
constraint and a `rdot_req / I_z` that should have been a product. The model
is now dimensionally consistent throughout but still does not converge; the
author confirms it was never in working order.

---

## 5. EDI: `structure_detector` crashes on unclassifiable models

**Status: fixed.**

`parseDict_GP` checked the monomial and signomial cases, then *assumed*
signomial fraction without testing its status. An expression outside the GP
algebra (a transcendental, say) satisfies none of the four categories, so the
walker leaves `leadingCoefficients` as `None` and `len()` raised

    TypeError: object of type 'NoneType' has no len()

It now returns `None` for "not representable", and the three call sites
translate that into the module's existing `unstructured_dict()` idiom with a
message naming the offending constraint and side.

---

## 6. EDI: constant-only constraints silently defeat structure detection

**Status: fixed.** This one produced a *wrong answer*, not a crash.

A constraint containing no `Var` — `Qmax >= Q` where both were substituted —
was fed to the posynomial machinery, which zeroes it to a bare negative
number (`10 - 100 = -90`, then `+1` gives `-89`). The negative leading
coefficient reads as a subtraction, and the whole model is declared
unstructured. A model that **is** a GP was then misrouted to IPOPT, which
failed with `Error in step computation` and no hint that structure detection
was the cause.

Such constraints are now dropped before the constraint numbering is
established, or reported as infeasible if false as written.

The filtering has to happen before the loop, not inside it: `parseDict_GP` is
handed the `enumerate` index and uses it to group monomials by constraint, so
skipping one mid-loop leaves a gap in the numbering and later indexing walks
off the end of the operator list. (That was the first attempt, and it turned
a silent misroute into an `IndexError`.)

---

## 7. TASOPT: notes rather than defects

Nothing in TASOPT 2.16 has turned out to be wrong so far — five modules
reproduce to machine precision. Three things are worth recording anyway
because they look like bugs and are not:

* **`fusew.f` assigns `Afweb` twice** with unrelated meanings — first the
  shell web area, later the floor web area. The port names them separately.
* **`fusew.f` uses the unlimited `wfb`** in `hfb = sqrt(Rfuse^2 - wfb^2)`
  while using the clamped `wfblim` for `thetafb`, so `wfb > Rfuse` would take
  the square root of a negative number. No sane input reaches it; preserved
  rather than "fixed".
* **`surfw.f` recomputes `Vout`** identically just before returning. Dead
  code; not reproduced.

The one real trap was in *my* verification drivers, not in TASOPT: passing
literal constants through these implicit-interface calls under `-O` delivered
garbage (a `gas_prat` called with a literal `4.0d0` pressure ratio received
`0.0`), which initially looked like a porting error. All drivers now pass
named variables only.

---

## 8. Wing rebuild lands at a different optimum — RESOLVED

**Status: fixed.** It was a real model error, not a solver difference.

**Cause: material properties guessed instead of read.** `CFRPUD` was assumed
to be `E = 190 GPa, sigma = 1500 MPa`; `gpkitmodels/GP/materials/composite.py`
says `E = 137 GPa, sigma = 1700 MPa`. E being 39% too stiff made the
deflection-angle recursion

    th[i+1] >= th[i] + 0.5 deta (b/2) (M[i+1] + M[i]) / (E I)

loose by exactly that factor, letting the optimizer buy span it had not paid
for. Every other constraint was already correct.

**Result after the fix** — AR error drops from 12.8% to 1.1%, and the
structural weights land within 0.3%:

| quantity | before | after | reference |
|---|---|---|---|
| Cd | 0.007095 | 0.0072466 | 0.0071952 |
| AR | 22.72 | 19.92 | 20.15 |
| W_spar | 21.94 | 22.168 | 22.150 |
| W_core | 10.86 | 10.963 | 10.953 |

The rebuild now sits 0.7% *above* the reference and satisfies all 103
evaluable reference constraints, i.e. it is feasible-but-slightly-suboptimal
there — some constraint is marginally tighter than its counterpart. Left as
is.

**Method, which generalizes.** Substitute your optimum into the *reference*
model and evaluate every one of its constraints. Whichever it violates names
the defect directly, with no reading or reasoning. Six of 130 came back
violated here, all the same `th` recursion, and the only term in it not read
from source was E.

Two traps, both of which silently return a clean bill of health:

* substituting **bare floats** makes gpkit read them in the wrong units, so
  even `t >= tmin` appears violated — substitute quantities *with* units;
* not substituting the model's **constants** leaves 102 of 130 constraints
  unevaluable, and a bare `except: continue` counts them as passing. Always
  report evaluated-vs-skipped.

### Original record

The following was written before the cause was found, and is kept because the
reasoning in it turned out to be right.

The EDI rebuild of `gplibrary/GP/aircraft/wing` solves cleanly but reaches a
different point than the gpkit reference:

| quantity | rebuild | reference | delta |
|---|---|---|---|
| Cd (objective) | 0.007095 | 0.007195 | -1.4% |
| AR | 22.72 | 20.15 | +12.8% |
| S (ft^2) | 44.35 | 42.85 | +3.5% |

Ruled out: the beam chain (the reference's constraints print equivalent to
the rebuilt ones), `WingCore` (initially guessed as `0.5*tau*cave^2`, actually
`Abar*cave^2` with a fixed `Abar = 0.0753449` — correcting it moved the
objective from 0.007056 to 0.007095), and the tip-relaxation constant
(`wing_test` uses 1e-1, `box_spar` 1e-2, and the two give 0.007682 vs
0.007195).

Unexplained: in the reference solution the manoeuvre load `q` sits well above
the lower bound its own constraint states — 1605.8 N/m at the root against
`N*W/b*cbar[0] = 422.9 N/m` — while the gust case's `q` matches its bound
exactly. Since `q` appears only on the loosening side of the shear chain, the
optimizer should drive it to that bound.

**Caveat on the accepted explanation** (this proved to be the correct
instinct). The discrepancy was accepted as plausibly a solver difference
(MOSEK vs cvxopt). Recorded at the time:
`wing_test` calls `Model.solve()`, not `localsolve()`, so this is a genuine
GP rather than an SP. GP solvers converge to the global optimum, so two
correct solvers on the same GP should agree to solver tolerance, not to 13%
in a design variable. That makes a residual model difference the more likely
explanation, and anything built on the wing inherits it.

Cheap way to settle it later: solve the *reference* model with two different
gpkit backends. If they agree with each other and disagree with the rebuild,
the difference is in the model, not the solver.

---

## 9. EDI: cvxopt non-convergence surfaced as ZeroDivisionError

**Status: fixed.**

`solve_GP` read cvxopt's results without checking `status`. cvxopt reports
failure there but still returns numbers, and those numbers are in log space.
On a diverging (typically unbounded) GP the transformed objective runs to a
large negative value, `exp()` of it underflows to exactly 0.0, and the
equality-dual normalization

```python
res['y'].append(1/res['primal objective']*y2)
```

then raised a bare `ZeroDivisionError` from inside the solver — a traceback
saying nothing about the actual cause, which is almost always a model missing
a lower bound.

Found while building the empennage: the tail boom was unbounded (nothing
forced a nonzero diameter, so it went weightless and the moment arm grew
without limit). cvxopt terminated with a singular KKT matrix and a
transformed objective of -1.7e5; `exp(-1.7e5)` is 0.0.

`solve_GP` now checks the status first and raises with the transformed
objective value and the likely cause. The empennage's real defect — a missing
`TailBoomBending` constraint set — was then obvious rather than buried.

---

## 10. gassolar/gas: the fuselage is charged for drag twice

**Status: reproduced, since the reference numbers depend on it.**

``AircraftPerf`` in ``gassolar/gas/gas.py`` builds the area-drag sum by
looping over components and appending a term for **each** of ``"Cf"``,
``"Cd"``, ``"C_d"`` that the component's flight model happens to define:

```python
for dc, dm in zip(areadragcomps, areadragmodel):
    if "Cf" in dm.varkeys:  dvars.append(dm["Cf"]*dc["S"]/static.wing["S"])
    if "Cd" in dm.varkeys:  dvars.append(dm["Cd"]*dc["S"]/static.wing["S"])
    if "C_d" in dm.varkeys: dvars.append(dm["C_d"]*dc["S"]/static.wing["S"])
```

``TailBoomAero`` defines only ``Cf`` and ``TailAero`` only ``Cd``, so those
contribute once each. But ``FuselageAero`` defines **both**, and since its
own constraint is ``Cd/mfac >= Cf*k``, the fuselage is charged roughly
``(1 + k) = 2.15x`` its actual drag.

The arithmetic at the reference solution:

| term | value |
|---|---|
| fuselage via Cf | 0.003872 |
| fuselage via Cd | 0.004467 |
| htail + vtail + boom | 0.001974 |
| **sum** | **0.010313** |
| reference CDA | 0.010314 |

Dropping the duplicate gives 0.006441 and MTOW lands 28% low, so this is not
cosmetic — it is load-bearing for every published number from this model.

---

## 11. gassolar/gas: the wing load factors end up swapped

**Status: reproduced, since the reference numbers depend on it.**

``gas.py`` lines 194-195:

```python
loading[0].substitutions[loading[0].Nmax] = 5
loading[1].substitutions[loading[0].Nmax] = 2
```

The second line keys *loading[1]'s* substitution dict with **loading[0]'s**
varkey. Both entries therefore target the manoeuvre case, the later one wins
and sets it to 2, and the gust case never receives a substitution at all —
it keeps ``Nmax``'s default of 5.

The solved reference confirms the outcome: ``SparLoading.N = 2`` and
``GustL.N = 5``, with gust moments dominating throughout (2420 vs 1503 N*m at
the root). The evident intent, reading the two lines, was manoeuvre 5 and
gust 2.

Consequence for anyone reusing the model: the wing is sized by a 5-g *gust*
case rather than a 5-g manoeuvre, which are not the same load distribution —
the gust case adds the incremental lift term ``2 pi agust/cl (1 + Ww/W)``
and is relieved by wing weight.
