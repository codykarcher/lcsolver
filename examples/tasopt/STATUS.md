# TASOPT Python port

A module-by-module port of TASOPT 2.16, each verified against the compiled
Fortran to machine precision. TASOPT is built with `-fdefault-real-8`, so
`implicit real` is double precision; the reference drivers use the same flag.

## Verified

| module | source | check |
|---|---|---|
| `atmosphere` | `atmos.f` | 4.4e-16 |
| `gas.properties` | `gasfun.f` | 2046 values, 3.6e-14 |
| `gas.mixture` | `gascalc.f` | 212 values, 4.4e-16 |
| `gas.burn` | `gasburn.f` | 60 values, 1e-13 |
| `structures.fuselage` | `fusew.f` | 66x3 configs, 3.5e-16 |
| `structures.surface` | `surfw.f` | 111x3 planforms, 3.8e-16 |
| `aero.moment` | `surfcm.f`, `tailpo.f` | 15x3, 5.2e-16 |
| `aero.drag` | `surfcd.f` | 30 values, 1e-13 |
| `engine.cooling` | `tfcool.f` | 105 values, 1e-13 |

41 tests. Reference CSVs are committed, so the suite runs without a Fortran
compiler; the drivers in `fortran_ref/` regenerate them.

Where a routine needs a table this port does not carry, the dependency is
injected rather than stubbed out — `surfcd2` takes an `airfoil` callable in
place of `airfun`, which is what lets a fitted surrogate be substituted, and
is exactly what the signomial formulation does.

## Not yet ported

* **Engine cycle** — `tfsize.f` (841), `tfoper.f` (3419), `tfcalc.f` (766).
  The largest remaining piece. `gasburn` and `tfcool` above are its
  prerequisites and are done.
* **Induced drag** — `trefftz.f` (641), a Trefftz-plane solve with its own LU
  factorization.
* **Drag summation** — `cdsum.f` (490), which orchestrates `surfcd`,
  `trefftz` and the fuselage boundary layer.
* **Sizing loop** — `wsize.f` (1727), `mission.f` (985), `balance.f` (744).

So this is a verified physics library, not yet a runnable aircraft.

## Signomial forms

`sp/surface_drag.py` writes `surfcd` as an EDI signomial program and checks it
against the exact port over six designs, agreeing to 5.4e-8. That is the York
claim made checkable for one routine; doing it for the rest is what would
substantiate "TASOPT can be made almost entirely SP".

Three things obstruct GP/SP compatibility in that routine and are worth
expecting elsewhere: quotients of differences (the spanwise integral factors),
a `1 - x` blend (the sweep/unsweep mix), and posynomials appearing on the
greater side of a relation.
