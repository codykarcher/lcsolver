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
| tank | gravimetric ≈0.77 | in the published 0.6–0.85 band; includes support rings |
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
| tank dry mass | 353 kg | 331 kg | +6.5% | support rings now modelled — they *were* the gap |
| tank length | 2.97 m | 3.09 m | −4% | same |
| fan shaft power | 4.22 MW | 4.74 MW | **−11%** | ideal disc vs `epf = 0.9` gas path |
| fan mass flow | 207 kg/s | 204 kg/s | +1.6% | momentum closure |
| fan diameter | 1.13 m | 1.33 m | −15% | disc area vs M 0.6 fan face |
| motor mass | 211 kg | 1137 kg | **−81%** | 10 kW/kg flat vs sized PMSM, direct drive |

Reading it: the *derived* constraints hold to ~1% at the optimum. The tank
row is a story worth keeping: the verification first read **−22%**, blamed
on "lumped insulation" — a wrong diagnosis. Decomposing piece by piece
showed every modelled component within 13% (erring heavy: skin −1%, heads
+13%, insulation +13%) and the entire gap was **support rings the SP did not
model at all** — 93 kg of stiffeners carrying the vessel and its fuel, the
same class of finding as SPaircraft's missing torsion constraint. A monomial
fitted to the port's ring sizing (4.2% worst error; the load exponent of
0.07 says a ring is mostly its own perimeter mass) closed it to +6.5%
conservative — after SIA's phase 1 priced the load term out of the model:
carrying ``W^0.069`` made the subproblem ill-conditioned and the documented
cases stopped solving cold, so the load is frozen at nominal for 8% on a
component that is a quarter of the tank. The *idealizations* cost 11–15%
and are each understood (the fan gap is almost exactly the polytropic
efficiency the disc omits); and the **motor is the model's weakest claim** — the port's PMSM cannot reach 10 kW/kg at 2.1 MW at
any speed (the shaft binds at 8.6 kW/kg near 60 krpm), so the flat constant
silently assumes several smaller machines per fan. A corrected powertrain
would add roughly 1–2 t to the E175-class MTOW.

## Full-aircraft comparison: TASOPT.jl's own LH2 aircraft

TASOPT.jl ships a complete hydrogen aircraft — `example/cryo_input.toml`, an
**LH2-burning turbofan** with the fuselage tank, 180 pax x 215 lbf over
3000 nmi — and `size_aircraft!` closes it at 76.0 t MTOW. Re-running this SP
at that exact mission (payload 172,146 N, 5,556 km) gives the honest
side-by-side. Propulsion differs by design: they burn the hydrogen, we run
it through a fuel cell.

| | TASOPT.jl LH2-TF | this SP (FC-electric) | reading |
|---|---|---|---|
| MTOW | 76.0 t | 47.3 t | not comparable headline: see rows below |
| block H2 | 9,634 kg | 3,707 kg | ~1.55x fuel-cell chain efficiency x ~1.35x full mission vs cruise-only |
| tank dry / gravimetric | 3,426 kg / **73.8%** (their printout) | ≈1,000 kg / 78.9% | 5-point gap with rings modelled; remainder is their `ftankadd` support extras and heavier insulation stack |
| wing / MTOW | 16.1% (12.2 t, with flaps, slats, ribs) | 5.5% (bare box x1.2) | their buildup shows secondary structure ~doubles a bare box |
| fuselage / MTOW | 26.6% | 25.4% | closer than the crude k_fuse deserves |
| empty fraction | 55.5% | 55.1% | **coincidence**: missing gear/systems offset by the 10 t stack |

**The fuel-cell full-aircraft path in TASOPT.jl is scaffolding, not a
model.** The enum, engine model and weight hooks all exist
(`prop_sys_arch = "fuel_cell_with_ducted_fan"`), but no example exercises it
and it cannot run: `read_input`'s first propsys switch predates the option
(leaving `eng_has_BLI_cores` unset), the wing-relief switch in
`size_aircraft!` omits the arch (leaving `Weng1` unset), and with both
patched locally the sizing loop NaNs. So for a fuel-cell-electric transport,
**this SP is currently the only one of the two that closes** — and the
component-level verification above is the strongest available check until
TASOPT.jl's FC path is finished.

## Replicating TASOPT.jl's LH2 turbofan (`model_lh2tf.py`)

The full-aircraft comparison above left a 63,000 lb gap and a diagnosis:
part modeling scope, part wrong parameters. `model_lh2tf.py` settles which is
which by replicating their aircraft — same propulsion architecture
(H2-burning turbofan, their engine's own solved TSFC and weight-per-thrust),
every calibration read from *their design*, never tuned to their MTOW. The
convergence sequence is the answer to the diagnosis:

| step | what was set | MTOW [lb] | error |
|---|---|---:|---:|
| calibrations only, free planform | k_fuse (validated at 259 vs 260), hull geometry, f_ns = 1.64, Wadd = 0.065 MTOW, engines, TSFC, 20% reserves | 138,268 | −17.6% |
| + their planform | AR = 10.1, taper 0.25, sweep factor 1/cos²(26°) | 150,843 | −10.1% |
| + their policies | CL ≤ 0.57, boil-off ≤ 0.4%/hr, k_beam = 1.196 (declared) | 160,509 | −4.3% |
| + their tank parameters | vent pressure 2 atm, heat-leak factor 1.3 | **164,920** | **−1.7%** |

Final breakdown, every row within 2% except the tank:

| | SP [lb] | TASOPT [lb] | diff |
|---|---:|---:|---:|
| wing | 26,566 | 26,988 | −1.6% |
| fuselage | 44,624 | 44,678 | −0.1% |
| empennage | 2,975 | 3,020 | −1.5% |
| engines | 14,367 | 14,621 | −1.7% |
| tank | 6,055 | 7,556 | **−19.9%** |
| gear+systems | 10,720 | 10,901 | −1.7% |
| fuel | 20,913 | 21,247 | −1.6% |
| **MTOW** | **164,920** | **167,711** | **−1.7%** |

Wing area lands at 122.0 vs their 121.5 m², L/D 14.58 vs 14.71, tank
9.29 × 2.31 m vs 9.51 × 2.28, insulation 12.7 cm at exactly their 0.4%/hour
boil-off policy.

What the sequence taught:

* **Parameters were most of it.** Free-planform optimization flew off to
  AR 6 (the Hoburg box under-prices span with sweep unmodelled); their CL
  policy, boil-off policy, vent pressure and heat-leak factor were all
  simply *settings* the first comparison had not set.
* **Two genuine modeling gaps, both now priced:** the Hoburg box vs
  TASOPT's beam theory is worth 19.6% on the box at their exact point
  (carried as the declared constant `k_beam = 1.196`, measured once, not
  tuned), and the lumped tank thermal/structural model is worth ~20% on the
  tank — 0.9% of MTOW — from their layered k(T) insulation integral,
  support-angle and `ftankadd` details the SP deliberately lumps.
* **`k_fuse = 260 N/m²` was never wrong.** Their hull works out to
  259 N/m²; the earlier fuselage gap was missing nose/tailcone geometry.

On "did you port all of SPaircraft": no — only `wingbox.py`, because
`wing.py`/`fuselage.py`/tails/gear close through the trim block and cannot
be taken piecewise (documented above). The sweep block that `wing.py` would
have contributed is exactly what the +12,600 lb step 2 recovered.

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
