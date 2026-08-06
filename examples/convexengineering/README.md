# convexengineering models, rebuilt in LCsolver

Reimplementations of the GP/SP aircraft-design models published by the MIT
Convex Engineering Group (https://github.com/convexengineering), together with
the papers they came from.

Each model directory holds:

| file | role |
|---|---|
| `model.py` | the LCsolver reimplementation; `build()` returns a `Formulation` |
| `reference.json` | recorded solution of the original gpkit model (committed) |
| `reference.py` | script that regenerates `reference.json` (needs gpkit) |

Run a model's self-check with `python model.py`. It solves the LCsolver
formulation, diffs every variable against `reference.json`, and reports
feasibility. This needs only LCsolver — not gpkit.

## Verification approach

Ground truth is the **published source**, not the paper tables. The papers
report 3-significant-figure values and, per the authors, contain typos; the
source is more precise and more self-consistent. Where a model has no
published source (the wind turbine GP), the paper is the only reference and
that is noted in the model docstring.

Discrepancies found between paper and source, and any corrections made, are
collected in [DISCREPANCIES.md](DISCREPANCIES.md).

## Reproducing the reference solutions

The original models are ~8 years old and do not run on a current scientific
Python stack. They need an isolated environment:

```bash
conda create -y -n gpkit-ref python=3.11 "numpy<2" scipy
conda activate gpkit-ref
pip install cvxopt ad pandas matplotlib
pip install <path-to-gpkit-checkout>      # NOT PyPI gpkit 1.1.0 — see below
```

Then clone the reference repos and point `PYTHONPATH` at them:

```bash
for r in gpkit gplibrary gpfit SPaircraft gassolar solar jho turbofan; do
  git clone https://github.com/convexengineering/$r.git
done
export PYTHONPATH=$PWD/gplibrary:$PWD/gpfit:$PWD/SPaircraft:...
```

### Environment landmines

Each of these cost real time; they are recorded so the next person can skip
them.

* **PyPI `gpkit==1.1.0` is broken** — the sdist omits `gpkit/breakdowns.py`,
  so `import gpkit` fails with `ModuleNotFoundError: No module named
  'gpkit.breakdowns'`. Install from a git checkout instead.
* **`numpy>=1.24` breaks the fit machinery.** `gpfit`'s `FitCS` and gpkit's
  `NomialArray` build arrays from ragged lists of monomials, which numpy made
  a hard error in 1.24. This affects `solar` and `gassolar`, which use fitted
  wind/solar-availability constraints. `numpy<2` alone is *not* enough.
* **`gpfit` renamed its API.** `gpfit.fit_constraintset.FitCS` (dict-based)
  became `gpfit.constraint_set.FitConstraintSet` (Fit-object-based) in commit
  `07b6362`. Models older than that need the pre-rename file, recoverable with
  `git show 07b6362~1:gpfit/fit_constraintset.py`.
* **A `PYTHONPATH` that pins the base interpreter's `site-packages`** (as a
  login shell may set) leaks the wrong numpy into the isolated env. Run the
  reference interpreter with `env -u PYTHONPATH` — and note this applies to
  **pip commands too, not just imports**. Running
  `<isolated-python> -m pip install gpkit` with that PYTHONPATH set made pip
  see the *base* env's gpkit and uninstall it while installing into the
  isolated env, leaving the base install as a broken namespace package. See
  `../ENVIRONMENT_CHANGES.md`.
* **Subsystem models are unbounded standalone.** `Wing`, `Fuselage`, etc. have
  no lower bounds on their own; they must be driven by the loading/mission
  wrappers used in each directory's `*_test.py`.

## Status

See [STATUS.md](STATUS.md) for what is rebuilt, verified, and outstanding.
