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
