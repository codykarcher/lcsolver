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
| MTOW [lb] | 134,781 | 124,125 | **−7.9%** |
| dry [lb] | 74,601 | 77,824 | +4.3% |
| fuel [lb] | 21,480 | 7,601 | **−64.6%** |
| wing [lb] | 22,886 | 22,387 | −2.2% |
| fuselage [lb] | 31,354 | 33,205 | +5.9% |
| tank dry [lb] | — | 1,786 | new |
| wing area [m²] | 152 | 169 | +11.4% |
| aspect ratio | 12.0 | 10.7 | −10.3% |
| fuselage length [m] | 32.7 | 39.1 | +19.7% |
| shell length [m] | 18.2 | 24.6 | **+35.2%** |
| **mission energy [GJ]** | **418.8** | **413.6** | **−1.2%** |

The fuel mass drop (−64.6%) is almost exactly hydrogen's LHV advantage
(1/2.79 = −64.2%), which is the arithmetic working. MTOW falls because the
fuel saved outweighs the tank added; the fuselage grows 35% in shell length
to house a 6.4 m tank; and the optimiser answers by growing wing area 11% and
dropping aspect ratio 10%.

## The number to distrust

**Mission energy comes out flat (−1.2%).** Published LH2 conversions of
conventional transports land nearer +8 to +15% energy, and there are two
named reasons this model is optimistic:

1. **The tank is ~20% light.** Verified directly against the TASOPT v3 port
   (see `../../SP_hydrogen_aircraft/verify_against_tasopt.py`): the lumped
   thermal/structural model reads about 20% under TASOPT's layered `k(T)`
   insulation integral and support details. Gravimetric efficiency here is
   0.81 against TASOPT's printed 0.738.
2. **The planform re-optimised.** This is not a fixed-geometry swap — S and
   AR both moved, so part of the energy parity is the optimiser recovering
   losses elsewhere. A fixed-geometry comparison would show a larger penalty.

The *direction* (MTOW down, energy roughly flat-to-worse, fuselage much
longer) is right and is what the hydrogen literature reports. The magnitude
of the energy term is not yet trustworthy.

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
