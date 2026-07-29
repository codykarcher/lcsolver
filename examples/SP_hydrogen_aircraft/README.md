# SP_hydrogen_aircraft

A hydrogen-electric transport aircraft as a signomial program, derived from
the TASOPT v3 Python port in [`../tasopt/`](../tasopt/).

**Status: solves to a certified KKT point under EDI's default SIA solver.**
31 iterations; stationarity, feasibility and complementarity all within 1e-6
on the true problem; `python -m examples.SP_hydrogen_aircraft.model` runs the
build, the solve, and the self-checks.

## Headline solution

180 pax, 3000 km, cruise M 0.78 at 11 km. Objective: minimum MTOW.

| quantity | value | sanity anchor |
|---|---|---|
| MTOW | 40.0 t | payload fraction 0.46 (no gear/systems/climb — see caveats) |
| block fuel | 1504 kg H2 (+10% reserve = 1654) | continuous Breguet: 1426 kg, 1.7% off |
| boil-off | 69 kg | the tank's thermal design, visible in the mission |
| wing | 81 m², AR 9.9, 2.4 t | loading 489 kg/m², transport range |
| L/D (incl. cooling drag) | 14.0 | cooling drag costs ~1.2 L/D points |
| tank | 265 kg for 1654 kg H2 | gravimetric 0.81, in the published 0.6–0.85 band |
| stack | 5.8 t at 8.1 MW | 1.39 kW/kg, inside the PEM system band |
| fan | D = 1.47 m per side | interior optimum — see "the bypass valley" below |
| current density | 10 000 A/m² | **at the fit's validity cap** — see caveats |

## What this demonstrates

The hydrogen physics — cryogenic tank, PEM stack, electric drivetrain,
ducted fan — carried from a machine-precision-verified TASOPT v3 port into
an SP that *actually solves*, with the provenance of every reformulation
recorded, in the manner of York, Hoburg and Drela's turbofan SP.

| module | derived from | MAIDAS said | reformulation used |
|---|---|---|---|
| `cryo_tank.py` | `tasopt_py/cryo/` | skin monomial; head area a fractional power of a posynomial; radius stack-up a difference | AR fixed → bracket collapses exactly; differences → tight posynomial inequalities |
| `fuel_cell.py` | `engine_v3/fuelcell_1d.py` | polarisation curve `not_gp` (logarithmic) | monomial fit **generated from the port**, 1.57% worst error on a stated band |
| `powertrain.py` | `propsys/`, `engine_v3/ducted_fan.py` | losses monomial; flux terms `not_gp` | efficiencies as constants; fan weight `D^2.7` straight from the port |
| `wing_h2.py` | `structures/surface.py` via SPaircraft's `wingbox.py` | exact taper ratio `not_gp`; box relations posynomial | Hoburg's posynomial box — the already-worked-out SP form of the physics MAIDAS mapped |

The airframe integration lesson stands from the earlier attempt: SPaircraft's
`wing.py` is closed by the fuselage/gear/trim block and cannot be taken
alone, but its `wingbox.py` is self-contained — nine posynomial constraints
needing only a planform and a load case. The hydrogen wing supplies that load
case with **no fuel-bending relief**, because the fuel is in the fuselage:
one constraint *removed*, and a real structural penalty of hydrogen.

## Structure audit

94 rows after presolve: **78 GP-exact, 16 conservative (signomial)**. The
sixteen are exactly the documented set: stack heat balance (×4), actuator
disc (×4), weight decrement (×3), momentum (×4), tank volume closure (×1) —
every one of York's "sum on the greater side" shape. (The four momentum rows
are mathematically reducible to GP by dividing through by the monomial side;
the detector keeps them conservative. A rewrite MAIDAS could learn.)

## The convergence war story

This model is the record of a claim worth keeping: **every SIA failure was a
model defect, and the solver's failure mode named it.**

1. **Phase 1 "failed after 1 iteration".** An EDI bug, now fixed in
   `edi/solvers/ipopt/sia.py`: the cached Phase-I subproblem hits IPOPT's
   iteration limit at `tol=1e-12` where the identical fresh-built model
   solves. A cache must never change *whether* something solves — on cached
   failure it now falls back to a fresh build for that iteration.
2. **A 1/k stationarity tail, thousands of iterations.** The iterate was
   sliding along a *null valley*: the split of total range across identical
   cruise segments is arbitrary. Pinning each segment to `R/N` removed the
   null space — and an SP row with it.
3. **Still creeping — `mdot_a` drifting up monotonically.** The model
   rewarded **infinite bypass**: an ideal actuator disc with nothing charging
   for fan size wants `mdot → ∞`, `u_j → u_0`, `eta_p → 1`. The missing
   charge is the bypass-ratio trade TASOPT itself exists to sweep: blade
   mass `∝ D^2.7` and nacelle area `∝ D²` from the port's
   `ducted_fan_weight`, plus the actuator-disc row tying flow to area. With
   those, **SIA converged in 29 iterations.** The non-convergence was the
   diagnostic; the fan diameter now sits at an interior optimum (1.47 m).
4. **The weight decrement pointed the wrong way.** `W[i] >= W[i+1] + burn`
   lets the optimiser drop later segments to their lower bound and fly a
   10 t aircraft for three quarters of the cruise. Caught by a Breguet
   cross-check reading 15% low — not by the solver, which satisfied the
   wrong constraint happily. Worth a tonne of MTOW and 40% of the fuel.
5. **The Breguet check itself was wrong twice**: first omitting cooling drag
   from L/D (16% phantom gap), then comparing against a total that includes
   boil-off, which no Breguet integral models (5.5% near-threshold residual,
   decomposed rather than tolerated). `verify()` now compares propulsion
   burn at effective L/D and closes to 1.7%.

Plus the five bugs from the earlier crude-airframe rounds (waste heat free
after segment 0; fan power charged for free-stream kinetic energy; lift
allowed to see an arbitrarily light aircraft; wing area too cheap; cruise at
C_Lmax) — **ten model defects total, every one surfaced by a diagnostic
rather than by reading the answer and squinting.**

## Verification against TASOPT.jl

`verify_against_tasopt.py` closes the derivation loop: it solves the SP, then
feeds the **solved design** back through the port's high-fidelity routines
(machine-precision-verified against TASOPT.jl; the fuel-cell chain re-run in
Julia directly, agreeing to every printed digit). This measures what each
reformulation costs at the optimum — where the model is used, not where it
was fitted. E175-class mission:

| quantity | SP | TASOPT | diff | what the difference is |
|---|---|---|---|---|
| cell voltage | 0.7509 V | 0.7532 V | −0.3% | monomial fit, at its cap |
| stack cells / area / heat | — | — | ≤0.8% | follows the fit |
| tank dry mass | 258 kg | 330 kg | **−22%** | lumped insulation, AR-fixed heads |
| tank length | 2.97 m | 3.09 m | −4% | same |
| fan shaft power | 4.22 MW | 4.74 MW | **−11%** | ideal disc vs `epf = 0.9` gas path |
| fan mass flow | 207 kg/s | 204 kg/s | +1.6% | momentum closure |
| fan diameter | 1.13 m | 1.33 m | −15% | disc area vs M 0.6 fan face |
| motor mass | 211 kg | 1137 kg | **−81%** | 10 kW/kg flat vs sized PMSM, direct drive |

Reading it: the *derived* constraints hold to ~1% at the optimum; the
*idealizations* cost 11–22% and are each understood (the fan gap is almost
exactly the polytropic efficiency the disc omits); and the **motor is the
model's weakest claim** — the port's PMSM cannot reach 10 kW/kg at 2.1 MW at
any speed (the shaft binds at 8.6 kW/kg near 60 krpm), so the flat constant
silently assumes several smaller machines per fan. A corrected powertrain
would add roughly 1–2 t to the E175-class MTOW.

## Caveats — read before quoting numbers

* **`j` sits at the polarisation fit's validity cap (10 000 A/m²).** The
  optimiser wants an even lighter, hotter stack; the cap stands in for the
  concentration-loss wall just beyond it, which the monomial cannot
  represent. The binding constraint is a *fit boundary*, not physics.
* Cruise-only mission: no climb, no diversion, reserves are a flat 10%.
* Fuselage and tail are area scalings; no trim, no tail sizing, no landing
  gear, no systems weight. The 0.46 payload fraction is optimistic for
  exactly these reasons.
* Fan tip speed fixed at 300 m/s; bus voltage fixed at 800 V.
* `model_integrated.py` is retained as the record of why SPaircraft's full
  airframe cannot be adopted module-by-module.

## Running it

```python
from examples.SP_hydrogen_aircraft.model import build, verify

verify(tee=True)     # build + SIA solve + self-checks, ~30 iterations
```
