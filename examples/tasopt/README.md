# TASOPT in Python

A port of TASOPT 2.16 (Drela's Transport Aircraft System OPTimization) from
Fortran to Python, verified against the original.

## Why a port at all

Three goals, in the order they were asked for:

1. an independent Python implementation of TASOPT;
2. a test case for MAIDAS structure detection;
3. a reference against which to check a York-style signomial-program
   formulation of the same aircraft (see `../convexengineering/`).

The third is the reason the port has to be *numerically faithful* rather than
merely equivalent in spirit: it only serves as a baseline for the SP
formulation if it reproduces the original's numbers.

## Verification method

Every ported routine is checked against the original **compiled from source**,
not against published tables or hand-worked examples.

For each module, a small driver in `fortran_ref/` calls the original
subroutine over a sweep of inputs and writes CSV. The drivers are compiled
with TASOPT's own build flags from `src/Makefile`:

```
gfortran -O -fdefault-real-8 -fdollar-ok
```

`-fdefault-real-8` matters: TASOPT is written with bare `real` declarations
and relies on that flag for double precision. Compiling without it silently
gives single precision and ~1e-7 agreement, which would look like a porting
bug.

The resulting CSVs are committed under `tests/data/`, so `pytest` runs without
gfortran or the TASOPT sources present. Tolerance is 1e-13 relative — the
port evaluates the same expressions but not always in the same association
order, so the last bit or two can differ; anything looser would indicate a
genuine formula difference.

```bash
python -m pytest tests/ -q
```

## Status

| module | Fortran source | status | agreement |
|---|---|---|---|
| `atmosphere` | `atmos.f` | **verified** | 205 values, max rel 4.4e-16 |
| `gas.properties` | `gasfun.f` | **verified** | 2046 values, max rel 3.6e-14 |
| `gas.mixture` | `gascalc.f` | **verified** | 212 values, max rel 4.4e-16 |
| `engine` | `tfsize.f`, `tfoper.f`, `tfcalc.f` | not started | |
| `structures` | `surfw.f`, `fusew.f`, `tailpo.f` | not started | |
| `aero` | `surfcd.f`, `cdsum.f`, `trefftz.f`, `blax.f` | not started | |
| `mission` | `mission.f`, `wsize.f`, `balance.f` | not started | |

The gas tables (11 gases, ~600 numbers each) are **generated**, not
transcribed — `fortran_ref/extract_gas_tables.py` parses them out of the
Fortran `DATA` blocks. Hand-copying 6600 numbers would have been a reliable
source of exactly the kind of typo this project is trying to find.

## Notes on the original

Things worth knowing that are not obvious from the code:

**The atmosphere is deliberately smooth.** TASOPT does not use the piecewise
ICAO standard atmosphere. The tropopause is blended with a softplus over
about 2 K:

```
T(h) = Tblend*log(1 + exp((TSL + Tlapse*h - Tpause)/Tblend)) + Tpause
```

As `Tblend -> 0` this is exactly `max(TSL + Tlapse*h, Tpause)`. The blending
exists because TASOPT is gradient-optimized and a kink at the tropopause
upsets the optimizer. Pressure is likewise a fitted rational-exponent
expression rather than the exact hydrostatic integral.

**Entropy interpolates against log T, enthalpy against T.** In `gasfun.f`
each gas table carries both `t` and `tl = log t`. This is not redundancy:
`dh = cp dT` but `ds = cp d(ln T)`, so using `log T` for entropy makes `cp`
the correct Hermite endpoint slope for both quantities.

**The gas routines extrapolate rather than clamp.** Outside the tabulated
temperature range the same cubic is evaluated with the fractional coordinate
outside [0,1]. The port preserves this; clamping would silently change
results near the table edges.

**Newton loops give up quietly.** `gas_tset`, `gas_prat` and `gas_delh` run
at most 10 iterations, print a message on failure, and return the unconverged
value. The port raises `ConvergenceError` instead, so a silently wrong cycle
calculation cannot propagate; callers wanting the original behaviour can
catch it.

## A trap when writing Fortran drivers

Passing **literal constants** through these implicit-interface calls under
`-O` produced garbage — a `gas_prat` called with a literal `4.0d0` pressure
ratio received `0.0`, and the routine then failed to converge with no
indication that the input was wrong. Every driver here passes named variables
only. If you add a driver and get inexplicable non-convergence, check this
first.
