# spaircraft_h2 — SPaircraft, converted to liquid hydrogen

The D8.2 airframe of [`../spaircraft/`](../spaircraft/) with its kerosene
fuel system replaced by a liquid-hydrogen one. Same 1,172 variables, same
trim chain, tails, landing gear and turbofan cycle; **converges in 81
iterations**, exactly as the kerosene model does.

This is the port the earlier `SP_hydrogen_aircraft/` models were not: the
hydrogen physics moved *into* a full-fidelity airframe rather than beside a
scaffolded one.

## What changed, and nothing else

| # | change | where |
|---|---|---|
| 1 | **The wing is dry.** `f_wingfuel` → ~0, removing the wing-tank volume row *and* the fuel's bending relief in the root-moment constraint | `model.py` |
| 2 | **A cryogenic tank**, sized by the fuel load and fitted inside the fuselage | `add_cryo_tank` from `SP_hydrogen_aircraft` |
| 3 | **The tank lengthens the shell**: `l_shell == nrows·pitch + l_tank` | `fuselage.py` |
| 4 | **The engine burns hydrogen**: `D82_LH2`, identical to `D82_SPaircraft` but `hf = 120` MJ/kg instead of 43.003 | `../turbofan/model.py` |

Change 3 is the whole structural hook: everything downstream of `l_shell` —
skin, insulation, floor, cone station, both bending distributions — follows
through SPaircraft's own structure with no further edits. Windows were moved
to scale with seated length rather than shell length, since the tank bay has
none.

Losing the wing-fuel bending relief (change 1) is a genuine structural
penalty of hydrogen, not a modelling convenience: kerosene in the wing
offloads the root moment, and LH2 in the fuselage cannot.

## Result: the same airframe on two fuels

Both solved through identical code, same solver, same seed.

| | Jet-A D8.2 | LH2 D8.2 | Δ |
|---|---:|---:|---:|
| MTOW [lb] | 134,781 | 128,612 | −4.6% |
| dry [lb] | 74,601 | 81,541 | +9.3% |
| fuel [lb] | 21,480 | 8,371 | −61.0% |
| wing [lb] | 22,886 | 21,857 | −4.5% |
| fuselage [lb] | 31,354 | 34,230 | +9.2% |
| tails [lb] | 701 | 1,236 | +76.4% |
| tank dry [lb] | — | 3,663 | new |
| **mission energy [GJ]** | **418.8** | **455.5** | **+8.8%** |

**Fuselage geometry** — the radius is unchanged, so the tank pays entirely in
length:

| | Jet-A | LH2 | Δ |
|---|---:|---:|---:|
| overall length [m] | 32.67 | 42.49 | +30.1% |
| shell length [m] | 18.22 | 28.05 | +53.9% |
| radius [m] | 1.68 | 1.68 | 0.0% |
| nose / cone [m] | 8.84 / 5.60 | 8.84 / 5.60 | 0.0% |

**Drag** — the longer fuselage is charged for, through
`Dfuse == 0.5 rho V^2 * C_D_fuse * l_fuse * R_fuse * (M/M_fuseD)^2`:

| cruise | Jet-A | LH2 | Δ |
|---|---:|---:|---:|
| fuselage drag [lbf] | 258 | 296 | +14.5% |
| total drag [lbf] | 1,157 | 1,221 | +5.5% |
| **L/D** | **25.12** | **23.45** | **−6.6%** |
| fuselage share of drag | 22.3% | 24.2% | |

Fuel mass falls 61% against hydrogen's 64.2% LHV advantage — the shortfall is
the aircraft growing to carry the tank. Energy rises **8.8%**, inside the
+8 to +15% band published for LH2 conversions of conventional transports.

## The tank, against TASOPT's documented case

Two corrections moved this from a tank that was frankly too good to one that
verifies.

**1. Boil-off was not charged.** Nothing in the first version of this port
paid for insulation, so the optimiser deleted it: `t_insul` ran to *zero*,
and the tank collected its weight and its length for free. Every segment now
carries `g * m_boil * thr` against the fuel budget.

Charging it, rather than imposing TASOPT's fixed 0.4 %/hour policy, makes
insulation thickness a genuine trade — foam weight and the fuselage length it
costs, against the hydrogen it boils away. The trade lands at **18.4 cm and
0.42 %/hour**, which is TASOPT's policy number recovered as an *outcome*
rather than assumed.

**2. `ftankadd` was zero.** TASOPT's own parameter for mounts, fill and vent
lines, baffles and vapour management, as a fraction of structural weight.
Calibrated at 0.35 by sweeping the *port's* sizer against their published
tank (below).

| | this model | TASOPT port, same inputs | Δ |
|---|---:|---:|---:|
| tank dry [lb] | 2,345 | 2,279 | +2.9% |
| tank length [m] | 7.25 | 7.25 | 0.0% |

| | LH2 D8.2 | TASOPT documented |
|---|---:|---:|
| fuel carried [lb] | 8,371 | 21,247 |
| tank dry [lb] | 3,663 | 7,556 |
| tank length [m] | 9.82 | 9.51 |
| insulation [cm] | 18.4 | 12.7 |
| **gravimetric** | **0.696** | **0.738** |

Gravimetric now sits *below* theirs, which is the direction square-cube
demands: this tank holds a third of the fuel, and a smaller tank has the
worse surface-to-volume ratio. The earlier 0.81 was backwards — better than
TASOPT's while a third the size.

**Ruled out along the way:** their tank is not vacuum-jacketed. Running the
port's `size_outer_tank` on their case lands at gravimetric 0.48, far from
their published 0.738, so the documented tank is single-wall foam-insulated.

**What the calibration cannot separate:** `ftankadd ≈ 0.4` and "`ftankadd ≈
0.1` with roughly twice the insulation density" fit their two published
numbers about equally well, since insulation density moves weight without
moving length either. Two numbers cannot distinguish them. The term is a
calibrated lump, not a claim about where their extra mass physically sits.

## What this comparison is not

TASOPT's documented LH2 case is a 737-class tube-and-wing at R_fuse = 2.54 m;
this is a D8.2 with a double bubble, BLI and rear engines. The airframe-level
numbers are **not** like-for-like and no attempt is made here to pretend
otherwise — SPaircraft only converges in its `optimalD8` configuration, so
there is no 737 variant to run. `../../SP_hydrogen_aircraft/model_lh2tf.py`
is the TASOPT replication (−1.7% on MTOW); the comparison above is
deliberately at *component* level, where inputs can be matched exactly.

## A presolve bug this exposed

`verify()` passes `presolve=False`. Adding the cryogenic tank makes EDI's
presolve overflow:

```
edi/presolve.py restore_columns -> _solve_for -> _eval_terms
OverflowError: math range error      (math.exp of an accumulated log)
```

The kerosene SPaircraft presolves fine through the identical path, and the
trigger is *not* the near-zero `f_wingfuel` — it reproduces at 1e-6, 1e-4 and
1e-2 alike. It is one of the tank's fractional-power rows creating a column
presolve eliminates and then cannot back-solve. Worth fixing in EDI; recorded
here rather than worked around silently.

## Two traps worth knowing

* **`unit_corrector` returns a corrected copy.** Calling it for its side
  effect and then reading the original model gives a self-inconsistent answer
  that still reports "converged" — a wing heavier than the dry weight, in the
  first run of this port.
* **The model mixes lbf and N.** SPaircraft declares weights in lbf, the tank
  and wing box in N, and EDI's corrector reconciles them inside the
  constraints. Read `pyomo.environ.units.get_units(v)` before scaling
  anything; assuming one or the other silently rescales half the buildup.

## Running it

```python
from spaircraft_h2.model import verify
verify()          # ~35 s, 81 iterations
```
