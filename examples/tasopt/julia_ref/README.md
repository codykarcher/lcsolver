# Reference drivers for the TASOPT.jl-derived modules

Everything under `tasopt_py/` that comes from **TASOPT 2.16** is verified
against the compiled Fortran; the drivers for that live in `../fortran_ref/`.

A few things do not exist in 2.16 at all — hydrogen is the first — and those
come from [TASOPT.jl](https://github.com/MIT-LAE/TASOPT.jl) v3 instead. The
drivers here dump reference values out of the Julia package the same way
`fortran_ref/` dumps them out of the Fortran, so the port is diffed against a
running reference rather than transcribed by eye.

## Running them

```bash
# once
cd ~/software && curl -fLO https://julialang-s3.julialang.org/bin/mac/aarch64/1.12/julia-1.12.6-macaarch64.tar.gz
tar xzf julia-1.12.6-macaarch64.tar.gz
git clone https://github.com/MIT-LAE/TASOPT.jl /tmp/tjl
cd /tmp/tjl && ~/software/julia-1.12.6/bin/julia --project=. -e 'using Pkg; Pkg.instantiate()'

# then, per driver
cd /tmp/tjl && ~/software/julia-1.12.6/bin/julia --project=. \
    /path/to/julia_ref/dump_gas_h2.jl
```

Each driver writes a CSV into `/tmp`; copy it to `../tests/data/`. The CSVs
are committed, so the test suite runs without Julia — same arrangement as the
Fortran references.

## Gotchas

* `gasfun` and friends are **not** exported from `TASOPT`. They live in the
  `TASOPT.engine` submodule: `TASOPT.engine.gasfun(40, 300.0)`.
* `gasfun` returns six values, `(s, s_t, h, h_t, cp, r)` — matching this
  port's `GasState`, not the Fortran's argument order.
* `gas_burn` takes a **ninth** argument, `hvap`, that the Fortran's does not.
  Passing zero reproduces 2.16 exactly.
