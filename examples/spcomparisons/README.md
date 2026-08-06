# spcomparisons — four size classes x five architectures

Twenty aircraft, sized as signomial programs from one shared component
library. The point is to change **one thing at a time**: rows 1 and 2 differ
only in the airframe, rows 1, 3, 4 and 5 only in what carries the energy.

To solve one case by hand -- for debugging, sweeps, or the geometry
visualizer -- see [RUNNING.md](RUNNING.md).

## Layout

```
components/          the reusable models
  flight_state.py      standard troposphere
  wing.py wingbox.py   Hoburg box + York transonic airfoil fit
  fuselage.py          pressure vessel, floor, cone, bending; seats abreast is a parameter
  horizontal_tail.py vertical_tail.py landing_gear.py
  turbofan/            full 1-D cycle (York/Hoburg/Drela)
  cryo_tank.py         LH2 vessel, insulation, boil-off
  fuel_cell.py         PEM stack sized by its worst operating point
  powertrain.py        inverter, motor, ducted fan, actuator disc
  battery.py           pack sized by whichever of energy or power binds
  electric.py          presents a powertrain to the airframe as an engine
classes.py           Citation X, E175, 737-800, 787-8
architectures.py     conventional, d8, h2burn, h2fc, battery
aircraft.py          assembly: build(size_class, architecture)
run.py               process-parallel solver, one JSON per case
continuation.py      warm-start rescue for cases that fail cold
report.py            reads results/, prints the tables
```

Six of the eight structural components are **byte-identical** to SPaircraft's.
Only `fuselage.py` needed a parameter (seats abreast), which is why a
component library was the right shape for this: the wing, tails, gear and
trim chain genuinely do not care what is making thrust.

## What is free, and what is not

**Free:** wing area, aspect ratio, sweep (carried as `cos Λ`, which keeps it
monomial), thickness, lift coefficient, **cruise Mach**, **cruise altitude**,
fuselage radius, nose length, tailcone taper, floor beam depth, and every
weight that follows from them.

**Specified:** passengers, seats abreast (an integer cabin decision no
continuous optimiser should make), range, and the architecture.

### Freeing the fuselage took real work

Four shape variables are substitutions in SPaircraft's `subs/optimalD8.py`,
and three are **unbounded** the moment the pin comes off, each in a different
direction:

| variable | freed behaviour | physics supplied instead |
|---|---|---|
| `l_nose` | nose wetted area grows with length, so it collapses to zero | cockpit floor (4 m, size-independent) plus 1.2-calibre fineness |
| `lambda_cone` | `l_cone = R/λ`, so λ runs to infinity and the cone vanishes | at least 2 calibres; a genuine trade, since a short cone shortens the tail arm and the tail volume coefficients charge for it |
| `h_floor` | `A_floor ∝ 1/h_floor`, so a deeper beam is always lighter | at most 0.10·R_fuse, the space under the floor |
| `w_db` | a real trade: stiffens the shell, adds skin | architecture switch — the double bubble is what makes a D8 a D8 |

Fuselage **radius** stays free throughout. SPaircraft already has the row
that keeps it honest — seats, aisles, systems width and the double-bubble web
against `2·w_fuse` — it just assumed the D8's two aisles, so aisle count is
now class-driven.

## Two limitations that shape every number here

**The atmosphere runs out at 11 km.** `flight_state.py` is the standard
troposphere and keeps lapsing above the tropopause, so it is ~10 K too cold
at 41 kft and ~23 K at 48 kft, which makes the air too dense and lift and
drag at altitude both optimistic. This is inherited: SPaircraft's own
verified reference point cruises at 38,478 ft with this same atmosphere, and
its `MinCruiseAlt` is *above* the tropopause. Capping at 11 km makes every
case infeasible against that floor.

Extending it is not a patch. An isothermal layer fits a monomial
`p = 7.64e12·h^-2.108` to **2.45%** over 11–16 km, which is GP-native — but
stretched across climb too (3–16 km) the same form is **39%** out, so one
power law cannot cover both layers and the model has no way to pick a regime.
The bias applies identically to all twenty cases, which a comparison *across*
architectures tolerates far better than an absolute number would.

**Ceilings are the model's, not a type certificate's.** SPaircraft already
requires a 1.5% climb gradient at top of climb — about 700 ft/min at cruise
speed, well above the 100 ft/min that defines a service ceiling and the
0 ft/min that defines an absolute one. Cruise-climb afterwards is correct for
a fuel burner: the ceiling *rises* as the aircraft burns down, and step climb
is the aircraft following it.

A **battery** aircraft is the exception, and it is a real architectural
difference rather than a modelling convenience: it does not get lighter, so
its ceiling does not rise, and it is the one architecture here held to level
cruise.

**Buffet margin (FAR 25.251) is not modelled.** On a real transport that is
what caps cruise altitude, before anything in this model does. A cruise
angle-of-attack limit stands in for it, badly.

## Reading the results

Two things must be said before any MTOW here is compared with a real
aircraft.

**The optimiser does not design the reference aircraft.** With Mach free and
fuel as the objective, it flies slow and high on a big wing, because nothing
in the objective values speed:

| | Mach chosen | Mach real | AR chosen | AR real |
|---|---:|---:|---:|---:|
| Citation X | 0.60 | 0.90 | 13.3 | 7.9 |
| E175 | 0.64 | 0.78 | 12.9 | 8.9 |
| 737-800 | 0.73 | 0.79 | 11.8 | 9.5 |

That is the correct answer to *"what is the minimum-fuel aircraft for this
payload and range"* and the wrong answer to *"what is a Citation X"*. The
business jet is the extreme case, because a business jet's product **is**
speed and minimising fuel discards exactly that.

**Empty weight is systematically light.** Against the real aircraft, OEW
lands at 0.76–0.77 for the E175 and 737 — a consistent ~23% shortfall, which
is about what a conceptual-design SP with idealised structures, no systems
detail and no certification margin should be expected to miss by. The
Citation is much worse at **0.44**, and that is a genuine extrapolation
failure: SPaircraft is calibrated on a 180-passenger tube and does not
survive a 20x reduction in payload.

So: **compare architectures against each other, within a size class.** The
absolute numbers are not certification-grade and the Citation column should
be read as a trend, not a design.


## What "infeasible" means here, and why it took three tries to get right

Most of the effort in this study went not into the aircraft but into being
able to tell three different failures apart:

1. the design genuinely cannot exist,
2. the solver could not find a starting point,
3. a hard-coded constant from the source model does not scale.

They are indistinguishable from the outside -- all three print "phase 1 could
not find a feasible point" -- and getting them confused produces confident
nonsense. Every early conclusion in this project about hydrogen or batteries
being infeasible turned out to be category 2 or 3.

### The solver could not answer the question as written

LCsolver's Phase I minimised the WORST violation: ``min t s.t. log g_i <= t``,
one shared scalar. That formulation deliberately drives every constraint to a
common violation level, so a stalled run reported a dozen rows sitting at an
identical residual with nothing to choose between them. It cannot say which
row is the problem, because it has equalised them by construction.

Four defects were fixed in ``lcsolver/solvers/ipopt/sia.py``:

* **Equalities were written ``expr == t``**, forcing every signomial equality
  to the SAME residual. Two that could not be driven to a common value made
  the sub-problem infeasible -- on models that were perfectly feasible. This
  is what produced "failed after 1 iterations" on solvable problems, and it
  scaled with the number of SignomialEqualities, which is why the 1,200-row
  aircraft hit it and the 109-row hydrogen models never did. Now
  ``|residual| <= t``.
* **Monotone decrease was not enforced.** The docstring promised it; the
  check was gated behind ``has_blackbox``, so on a pure SP a step that made
  things worse was accepted anyway.
* **Sub-problem failure gave up immediately** instead of shrinking the trust
  region and retrying.
* **A stall was believed at the first radius** rather than re-tested at a
  smaller one, where the approximation is more accurate.

Then the formulation itself was replaced with the **elastic L1** form:
``min sum(s_i) s.t. log g_i <= s_i``, a slack per constraint. Its optimum is
sparse -- rows that can be satisfied fall to zero and drop out, and the few
that cannot ARE the answer, an approximate irreducible inconsistent
subsystem. ``solve_sia`` now falls back to it automatically: it either finds
the point min-max could not (reported as ``phase1_mode='l1-rescue'``) or
names the rows that must be relaxed, with their duals and variables.

The difference on one case, ``e175/battery``:

| | min-max | elastic L1 |
|---|---|---|
| rows implicated | ~12, all at 0.38128 | **4 of 3373** |
| duals | all ~1e-13 | up to 3.3e-1 |
| actionable | no | yes |

### Three D8.2 constants pretending to be physics

With a diagnosis that names rows, three failures resolved into hard-coded
values inherited from ``subs/optimalD8.py`` -- none of them a physical limit:

| what the diagnosis named | what it was | fix |
|---|---|---|
| ``LG_h_hold``, highest dual in the report (0.678) | pinned at 1.0 m: a 787 with an 8-seater's cargo hold | scaled on fuselage radius |
| ``Eng_h_t_*``, ``Eng_f``, ``Eng_fp1`` on every segment (40 rows, sum 6.18) | ``D82_SPaircraft``, a 180-passenger cycle asked for 787 thrust | GE90 for the 787, CFM56 for the small classes |
| ``PT_F_net``, ``PT_mdot_a``, ``PT_u_j`` on every segment | ``n_fans=2``, ``D_fan <= 4.5 m`` -- the 8-seat fuel-cell model's propulsion | fan count and diameter cap scale by class |

The third is the cautionary one. Reported as "phase 1 could not find a
feasible point", ``b737/h2fc`` reads as *hydrogen fuel cells do not scale to
single-aisle*. What it actually said was *this 737 has two small fans*.

**So no failure in this study should be read as a statement about hydrogen or
batteries unless the elastic report names rows that are physically about
energy, power or mass -- and the tables mark which is which.**


## Comparison to published studies

A caveat on this section: the figures below are quoted from memory of the
literature, not fetched from the sources, and should be checked against the
originals before being relied on. They are stated as **directions and rough
bands**, which is the level at which this model can be judged anyway.

### LH2-burning turbofan

The recurring findings across Brewer's *Hydrogen Aircraft Technology* (1991),
the EU **Cryoplane** project (Airbus-led, ~2000-2003), Verstraete's long-range
LH2 work (*Int. J. Hydrogen Energy*, 2013) and the UK ATI **FlyZero**
concepts (2022) agree on direction and roughly on magnitude:

| | published direction | this model (737-class, LH2 vs Jet-A, same airframe) |
|---|---|---|
| fuel **mass** | falls ~2.8x, the LHV ratio | −61% (LHV ratio alone is −64%) |
| MTOW | **down**, ~5-15% | −4.6% |
| OEW | **up**, ~10-25% | +9.3% |
| fuselage length | **up**, ~10-25% | +30% |
| energy consumed | **up**, ~8-15% | +8.8% |

Every one lands in or near the published band, and the two that matter most
-- energy up ~9%, MTOW down modestly -- sit squarely inside it. The
fuselage growth is at the top of the range or slightly past it, which is
consistent with this model putting the whole tank in the fuselage at constant
radius rather than widening the hull.

Worth noting *how* that result was reached, because the first version of it
was wrong: mission energy initially came out **flat**, which is outside every
published study. Two omissions caused it -- boil-off was not charged against
the fuel budget, so the optimiser deleted the insulation entirely, and
TASOPT's `ftankadd` fittings/supports term was zero. With both corrected the
tank's gravimetric efficiency falls to 0.696, correctly *below* TASOPT's
published 0.738 for a tank holding a third the fuel, and the energy penalty
appears.

### Hydrogen fuel cell

The consensus (FlyZero's regional concept, and the broader literature) is
that FC-electric is credible for **regional** aircraft and not for
single-aisle or widebody, limited by system-level stack specific power
(roughly 1-3 kW/kg installed, including cooling) and by the cooling drag of
rejecting heat at low temperature.

This model reproduces the ordering: the E175 and Citation fuel-cell cases
converge, and the fuel-cell E175 comes out **heavier** than its Jet-A
equivalent (73,304 lb against 61,395), which is the expected penalty. The
737 and 787 fuel-cell cases have not converged, and that should NOT be read
as the model agreeing they are infeasible -- the failures are a solver bug
(see below), not a physical verdict.

### Battery-electric

The literature here is unusually firm. Epstein & O'Flarity (*J. Propulsion
and Power*, 2019) and Schäfer et al. (*Nature Energy*, 2019) both conclude
that battery-electric transport is confined to small aircraft on short
sectors, and that a single-aisle would need cell-level specific energy of
roughly 800 Wh/kg -- far beyond the ~250-300 Wh/kg cells and ~200 Wh/kg packs
available now.

An independent check with this project's own numbers, before the matrix: the
electric range equation `R = e·eta·(L/D)·f_batt/g` gives **144-260 nmi** at a
200 Wh/kg pack, and a 180-passenger 1,620 nmi mission would need a pack of
**367% of MTOW**. The battery columns here are therefore expected to be
infeasible for the larger classes on physical grounds -- but that is a
prediction from the range equation, not something the matrix has yet
demonstrated, because the battery cases are currently blocked by the same
solver bug rather than by physics.

**This distinction matters and the tables preserve it.** A case that fails
because a design cannot exist is a finding. A case that fails because the
solver's Phase I mishandles signomial equalities is a bug. They are marked
differently (`x(sub)` versus `x(loc)`) and neither is reported as evidence
about hydrogen or batteries.

## Running it

```
python run.py                # all 20, one process each
python continuation.py       # warm-start whatever failed cold
python report.py             # tables
```
