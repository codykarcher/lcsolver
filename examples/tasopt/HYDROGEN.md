# Hydrogen support — what landed, and where

`tasopt_py` can now burn hydrogen. The code for it is in commit `bac7654`,
whose message is about presolve and monomial-equality elimination and says
nothing about any of this: a concurrent session on this branch swept the
staged files into its own commit before they could be committed under their
own message. History was not rewritten to fix that, because another session
was actively writing to the branch at the time and rebasing shared history
underneath a live writer is worse than a wrong commit message. This file is
the record instead.

## Why it was needed

TASOPT 2.16 cannot burn hydrogen — not badly, *at all*. `gasfun.f` defines
eleven species and none of them is H2, and `gaschem` ends with

```fortran
        write(*,*) 'GASCHEM: undefined gas index:', igas
        stop
```

so a hydrogen case halts the program rather than returning a wrong number.
See `DISCREPANCIES.md` §54.

## What was added

| piece | source | check |
|---|---|---|
| `H2` species, `igas = 40`, 45-point table | TASOPT.jl `gasdata.jl` | 25 × 6 values, **bit-exact** |
| `gaschem` / `gasfuel` entry for H2 | TASOPT.jl `gascalc.jl` | stoichiometry balances to 1e-5 |
| `hvap` argument to `gas_burn` | TASOPT.jl `gascalc.jl` | 32 burn cases, **bit-exact** |

The table is **generated** by `tools/gen_h2_table.py`, not transcribed — 45 × 6
hand-copied numbers is 270 chances at a typo invisible in a plausible-looking
answer, the same reasoning as `fortran_ref/extract_gas_tables.py`.

`hvap` is the fuel's heat of vaporisation, which 2.16's `gas_burn` has no
argument for: it assumes the fuel arrives as a gas, which is fine for kerosene
and wrong for anything cryogenic. Default zero reproduces the Fortran exactly,
which is what keeps `737.out` byte-identical. §55.

## How it is verified

Julia 1.12.6 is installed in `~/software/julia-1.12.6`, and TASOPT.jl is a
runnable reference exactly as the compiled Fortran is. `julia_ref/` holds the
drivers that dump reference values out of it; the CSVs are committed so the
suite runs without Julia. Same arrangement as `fortran_ref/`.

Both agree **bit for bit**, which is the expected result rather than a happy
one: the port's cubic Hermite interpolation is the same algorithm on the same
table, so anything other than an exact match would mean one of the two had
been transcribed wrong.

## What this does *not* buy

The engine can burn hydrogen. The **aircraft still cannot fly on it**, because
there is no fuel tank model anywhere in 2.16 — fuel is a weight and a volume
in the wing box. `TASOPT.jl/src/cryo_tank` (1971 lines) is the real hydrogen
enabler and is not ported. See `HANDOFF.md` for the scope of what remains.
