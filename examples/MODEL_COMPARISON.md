# SPaircraft vs TASOPT: what is actually different

Both size a transonic tube-and-wing airliner around a turbofan. They share a
lineage — SPaircraft's fuselage bending model is ported from TASOPT, its wing
drag polar is a fit to TASOPT's airfoil database, and its nacelle drag is
TASOPT's formula written out — but they are different *kinds* of program, and
most of the differences follow from that.

This lists the differences, roughly in order of how much they matter.

---

## 1. The fundamental one: a simulation versus an optimization problem

**TASOPT is an analysis code with an optimizer bolted on.** `wsize.f` is a
fixed-point loop: guess a weight, size the structure, march the mission, get a
new weight, repeat. Inside it are three separate Newton solves — `tfoper`'s
9-variable engine, `htsize`'s 2x2 tail-area/wing-position system, `balance`'s
single-step trim. Optimization happens *outside*, by `simpop`/`gradop` calling
the whole analysis repeatedly.

**SPaircraft is one optimization problem.** Every physical relation is a
monomial or posynomial inequality; the "solve" and the "optimize" are the same
step. There is no fixed-point loop and no inner Newton — the weight-fuel
coupling that `wsize` iterates on is just a constraint the solver satisfies
along with everything else.

Consequences that follow:

| | TASOPT | SPaircraft |
|---|---|---|
| result of a run | one converged aircraft | the *optimal* aircraft |
| sensitivities | finite differences over the whole code | dual variables, free |
| what "converged" means | outer loop residual + 3 inner Newtons | KKT point of one problem |
| failure mode | an inner Newton stalls | infeasible, or a signomial iteration cycles |
| design variables | chosen by the outer optimizer | chosen by the solver |
| guaranteed global optimum | no | yes for the GP part; local for the signomial part |

## 2. Engine: continuous cycle vs. simultaneous multi-point

This is the largest physics difference.

**TASOPT** separates on-design (`tfsize`) from off-design (`tfoper`). Sizing
picks the flow areas for a required thrust; every other flight point then
solves a 9-unknown Newton system (three pressure ratios, three corrected mass
flows, burner temperature, turbine exit pressure, fan-face Mach) against those
fixed areas.

**SPaircraft** has no such split. The engine *geometry* — `A_2`, `A_{2.5}`,
`A_5`, `A_7` and the map design mass flows — is shared across all flight
segments; the *cycle state* is per-segment; and all of it solves at once. The
optimizer therefore chooses the design point rather than being told it.

Within the cycle:

| | TASOPT | SPaircraft |
|---|---|---|
| gas properties | polynomial fits per species (N2/O2/CO2/H2O/Ar/fuel), integrated continuously | piecewise-constant `cp` — 1003 J/kg/K at 250 K, 1008 at 350, 1099 at 800, 1216 in the burner, 1280 in the HPT |
| air composition | mass fractions from `airfrac.inc` | implicit in the `cp` values |
| turbine cooling | `tfcool`: row-by-row from metal temperature, Stanton-number heat balance, hot-streak allowance | one constant bleed fraction `alpha_c` (0.19 for the D8.2) |
| compressor maps | `ecmap` shape function + `Ncmap` speed inversion (a Newton solve of its own) | a monomial fit to the E3 map, enforced as a ±10% *band*: `pi_f (1.7/pi_fD) ~ (1.06 m~^0.137)^10` |
| component efficiency | from the map, varying with operating point | constant per component |
| turbine maps | `etmap`, two quadratic penalties on pressure ratio and speed-flow | constant polytropic efficiency |
| burner | enthalpy balance with a full species mass balance | `f` from a `cp`-weighted energy balance |

A caveat on the map difference: in shipped TASOPT the compressor map's
off-design penalties are **switched off** (`CK = DK = 0` in the active
`tfmap.inc` set), so `ecmap` collapses to a straight line in pressure ratio
with no mass-flow dependence at all. The gap in map fidelity is smaller in
practice than the source suggests.

## 3. Drag: solved vs. fitted, component by component

| component | TASOPT | SPaircraft |
|---|---|---|
| induced | Trefftz-plane vortex sheet, wing and tail solved **together** with an image system, so tail height and downwash interference are resolved (`trefftz1`) | fitted span efficiency in the wing polar; no explicit wing–tail interference |
| wing profile | spanwise quadrature against a tri-cubic spline of a viscous airfoil database (`airfun` over `C.air`, 39 Mach x 7 c_l x 7 tau) | York's monomial fit **to the same database**, in (Re, tau, cosΛ·M, C_L) |
| fuselage | axisymmetric potential flow + boundary layer (`axisol`, `blax`, `fusebl`), returning a dissipation area | a **constant** `C_{D_{fuse}} = 0.018081`, scaled by `(M/M_{fuseD})^2` |
| nacelle | flat-plate `cfturb` with a vortex-sheet overspeed model | **the same formula**, written out as constraints |
| tails | `surfcd` with the airfoil database | fitted polars |
| compressibility | inside the airfoil database (`airfun`'s own `cdw` output is hard-wired to zero) | inside the polar fit |
| BLI | explicit `Phiinl`/`Kinl` defects handed to the engine, plus negative drag increments | one constant factor, `D_{reduct} = 0.98416` |

The fuselage is the biggest gap: TASOPT solves a boundary layer, SPaircraft
uses a constant. That also means SPaircraft's BLI benefit is an input, not a
result — the 1.6% credit is asserted, where TASOPT computes the ingested
defect from the BL solution.

## 4. Mission profile

| | TASOPT | SPaircraft |
|---|---|---|
| points | 17: static, rotate, takeoff, cutback, 5 climb, 2 cruise, 5 descent, 1 spare | 5: 3 climb, 2 cruise |
| descent | modelled | **absent** |
| takeoff | `takeoff.f`: ground roll, balanced field length, both climb-out angles, engine-out case | absent |
| climb | integrated trajectory with flight-path angle | excess-power formulation |
| cruise | marched with fuel burn | range from segment times |
| engine at each point | full off-design solve | same constraint set, different segment index |

SPaircraft imposes a maximum climb time (16 min), a minimum rate of climb
(500 ft/min) and a minimum cruise altitude instead of modelling the phases
TASOPT integrates.

## 5. Stability, trim and control

**TASOPT** does a real moment balance: `cglpay` finds the forward and aft CG
limits from worst-case payload loading; `balance` trims by tail lift, tail
area or wing position; `htsize` sizes the tail and places the wing
simultaneously against forward-CG trim power and an aft-CG stability margin;
the neutral point includes the engine inlet's normal force.

**SPaircraft** uses static-margin constraints plus a neutral-point
*approximation* from Unified's aircraft design rules (a signomial equality in
aspect ratios and tail volume). CG range enters as a specified `dx_{CG}`
rather than being derived from a loading study, and there is no engine-inlet
term.

TASOPT also sizes the vertical tail by engine-out trim or volume coefficient;
SPaircraft carries the engine-out case (`D_{wm}`, `T_e`, `y_{eng}`) but closes
it at aircraft level with a windmilling-drag coefficient of 0.5.

## 6. Where SPaircraft is *more* detailed

Not everything runs one way.

* **Landing gear.** SPaircraft has a full model — tip-over angles, strut
  sizing, wheel and tire sizing from Currey/Raymer correlations, retraction
  geometry. TASOPT uses two weight fractions of MTOW (`flgnose`, `flgmain`)
  and a fixed position.
* **Double-bubble fuselage.** SPaircraft models the two-lobe cross-section
  explicitly, with the joining web, its pressure load, and the `theta_db`
  geometry. TASOPT's fuselage is circular with a `dRfuse` vertical extension
  and a `wfb` web — it can represent the shape, but with less structural
  detail on the web itself.
* **Pi-tail.** SPaircraft carries a pi-tail bending case (`M_r`,
  `\pi_{M-fac}`) with the HT mounted on twin verticals. TASOPT has no pi-tail
  configuration.

## 7. Things SPaircraft has that are artefacts, not modelling choices

Found while rebuilding it, and reproduced deliberately because the reference
numbers depend on them. See `convexengineering/DISCREPANCIES.md`.

* The **main landing gear weight factor is computed in newtons where the
  correlation wants lbf**, making `F_{w_m}` 4.448x too large and the wheel
  assembly about 2.4x too heavy. The nose-gear line two rows above converts
  correctly, and the tire-diameter line below it also converts — which is what
  makes it look like an oversight (§17-18).
* The **horizontal tail box taper is pinned at 0.3**, never tied to the
  planform taper the optimizer picks — unlike the wing and vertical tail,
  which are tied (§ HT docstring).
* The **pi-tail structural path is degenerate** in the converged D8.2: it is
  held up only by artificial bounds, not by any physical constraint.
* Several **fit exponents are extreme** — the VT drag fit carries `M^1022.7`
  and `M^-114.577`, the wing polar `C_L^-1.44114`. These are curve fits over a
  narrow box, not physics, and they behave badly outside it.
* The atmosphere replaces **Sutherland's law with a monomial fit** carrying a
  dimensional fudge factor (`6.64 K^0.28`) that exists only to make the units
  resolve.

TASOPT's equivalents: the airfoil database has a quadratic penalty fence
outside its tabulated box rather than extreme exponents, and its atmosphere is
the standard table.

## 8. Fidelity summary

Where each is stronger:

**TASOPT is more physical** in the engine cycle (real gas, real maps,
row-by-row cooling), the fuselage (BL solve vs. a constant), induced drag
(Trefftz plane with wing–tail interference), trim and CG (real loading study),
and mission coverage (takeoff, descent, BFL).

**SPaircraft is more capable as a design tool**: it returns an optimum rather
than an analysis point, gives exact sensitivities for free, chooses its own
engine design point, and models the landing gear and double-bubble structure
in more detail. It also solves in seconds where a TASOPT optimization is many
analysis runs.

The honest characterisation is that SPaircraft trades physical fidelity —
mostly in the engine and the fuselage — for the ability to pose the whole
aircraft as a single tractable optimization. Much of the fidelity it keeps is
borrowed from TASOPT directly, as fits to TASOPT's own data.
