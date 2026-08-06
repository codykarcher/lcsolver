<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/codykarcher/lcsolver/main/logo/LCsolver_dark.svg">
  <img src="https://raw.githubusercontent.com/codykarcher/lcsolver/main/logo/LCsolver.svg" height="100" alt="LCsolver">
</picture>

[![tests](https://github.com/codykarcher/lcsolver/actions/workflows/tests.yml/badge.svg)](https://github.com/codykarcher/lcsolver/actions/workflows/tests.yml)
[![cross-platform](https://github.com/codykarcher/lcsolver/actions/workflows/test.yml/badge.svg)](https://github.com/codykarcher/lcsolver/actions/workflows/test.yml)
[![coverage](https://github.com/codykarcher/lcsolver/actions/workflows/coverage.yml/badge.svg)](https://github.com/codykarcher/lcsolver/actions/workflows/coverage.yml)

LCsolver is a package targeted at formulating and solving optimization problems that exploit Log Convexity to achieve solutions.  LCsolver (abbreviated as LCS) consists of three major components:

1. A lightweight wrapper around the Pyomo modeling language that has been designed to mimicmany of the features found in [GPkit](https://github.com/convexengineering/gpkit) and [CVXPY](https://github.com/cvxpy/cvxpy) while also providing a simple, clean interface for black-box analysis codes that are common in engineering design applications.
2. A set of methods that detect the structure of the optimization problem and classifies it for sorting to the appropriate optimization algorithm
3. Two novel algorithms for solving particularly complex optimization problems (SLCP and SIA)

## Installation

LCsolver began as a contribution to Pyomo itself — the Engineering Design Interface, [`pyomo.contrib.edi`](https://github.com/Pyomo/pyomo/pull/2937), public since August 2023 — and is now distributed as a standalone package so that it can evolve independently of the Pyomo release cycle.

Install from GitHub for now — `pip install lcsolver` is coming once the PyPI release is published.

It takes two steps, because one of the solvers cannot come from pip at all. The package and its Python dependencies install normally; IPOPT does not. There is no IPOPT executable on PyPI under any name, and cyipopt is published there as source only, so `pip install cyipopt` compiles against an IPOPT that has to be on the machine already. Getting one is what `lcsolver-install-solvers` is for.

### 1. The package

**Everything in one command (recommended)** — conda can supply IPOPT, so this covers most of step 2 as well:

```
git clone https://github.com/codykarcher/lcsolver.git
cd lcsolver
conda env create -f environment.yml     # python, cvxopt, ipopt, cyipopt
conda activate lcsolver
pip install -e .
```

**Or with pip alone:**

```
pip install git+https://github.com/codykarcher/lcsolver.git
```

or, from a clone, `pip install -e .`.

### 2. The solvers

```
lcsolver-install-solvers
```

Run this once in every environment you install into, including the conda one above — part of what it fetches is per-environment and cannot be shared. It prints exactly what it will run and asks before touching anything; `--dry-run` shows the plan and exits.

It finds IPOPT wherever it can — conda-forge, else Homebrew or apt, else a source build — installs cyipopt, and installs Pyomo's PyNumero ASL library, which the in-process route needs and which ships with neither pyomo nor cyipopt. Whatever is already present is left alone.

Then confirm what you ended up with — which `ipopt` binary will actually run, which linear solver it carries, and whether cyipopt agrees:

```
lcsolver-check-solvers
```

**You do not need IPOPT to try LCsolver.** With the package alone, a detected LP, QP, GP or SP solves through cvxopt — `solve()` falls back on its own and says so — and the test suite passes, skipping what it cannot run. IPOPT is needed for general nonlinear programs, for black-box constraints, and for the SLCP and SIA routes.

### 3. MA27, if you want it

Either route above leaves you on a **MUMPS** build of IPOPT, because MUMPS is the only linear solver that may be redistributed. That is a working install. For geometric and signomial programs MA27 is markedly more robust; it is free for academic use, but it has to be fetched by hand and IPOPT rebuilt against it:

```
lcsolver-install-solvers --ma27 <path-to-extracted-MA27-sources>
```

That serves as a first install and as an upgrade on top of an existing MUMPS one: it rebuilds IPOPT, relinks cyipopt to match, and records where the build went. Nothing needs to go in a shell profile — LCsolver finds a build it installed even when that build is on no `PATH`, and prefers an MA27 build over a MUMPS one whatever `PATH` order says. See [docs/ipopt.rst](docs/ipopt.rst) for why this is worth doing.

For any other environment on the same machine, `lcsolver-install-solvers --relink-cyipopt` points that environment's cyipopt at the MA27 build without rebuilding IPOPT again.

## Solving

LCsolver detects the structure of a formulation and routes it to an appropriate solver.

```python
from lcsolver.solvers.solver import solve

res = solve(f)                    # auto: IPOPT for LP/QP/GP/SP, and everything else
res = solve(f, convex_backend='cvxopt')  # structured backends for a detected LP/QP/GP/SP
res = solve(f, solver='ipopt')    # force IPOPT (general NLP, and black-box models)
```

After any successful solve the solution is written back onto the model, so

```python
import pyomo.environ as pyo
pyo.value(f.x)                    # the optimum, not the initial guess
```

### Sensitivities

After a solve, LCsolver reports how strongly the optimum responds to each `Constant`,
the way GPkit does for a geometric program:

```python
solve(f)
f.print_sensitivities()
```
```
========================================================================
Sensitivities to constants    [d log(f*) / d log(c)]
objective = 254.872    duals: kkt
========================================================================
  W_0                             +0.9953   ++++++++++++++++++++
  e                               -0.4795   ----------
  k                               +0.4108   +++++++++
  ...
```

The numbers are log-log sensitivities (elasticities), so they are unitless and
can be ranked against each other: `+0.4108` means a 1% increase in `k` costs
about 0.41% of objective. They come from the constraint duals via the envelope
theorem, so the cost is one solve regardless of how many constants the model
has — not the `2N` re-solves a finite difference would need — and every partial
derivative is taken symbolically, so there is no step size to tune. Available
for LP, QP, GP and SP on both the cvxopt and IPOPT cores; for a signomial
program the result is a local approximation from the final convex subproblem and
is flagged as such. See the [documentation](docs/sensitivities.rst).

### IPOPT

Two routes are supported. `method='pyomo'` uses `SolverFactory('ipopt')`, Pyomo's
AMPL-based interface, and needs the `ipopt` executable. `method='cyipopt'`
uses `pyomo.contrib.pynumero` and calls the IPOPT library in-process. The default,
`method='auto'`, prefers the Pyomo route but switches to cyipopt when the model
contains black-box (grey-box) constraints, which the AMPL route cannot evaluate.

Which linear solver is underneath matters more than anything else about the
install, and it is fixed when IPOPT is built. Every prebuilt IPOPT — conda,
apt, Homebrew — is a MUMPS build, because MUMPS is the only one that may be
redistributed, and MUMPS is not what you want underneath a geometric or
signomial program. IPOPT's own default is `ma27`. See
[docs/ipopt.rst](docs/ipopt.rst) for why, and
[lcsolver/scripts/install_ipopt.sh](lcsolver/scripts/install_ipopt.sh) (driven
by `lcsolver-install-solvers --ma27`) to build against HSL MA27.

If you have both a conda IPOPT and a source-built MA27 one, LCsolver uses the
MA27 build — it prefers MA27 over a MUMPS build regardless of `PATH` order, and
finds a build it installed even if that build is on no `PATH` at all. This is
deliberate: `conda activate` prepends `$CONDA_PREFIX/bin` in every new shell, so
honouring `PATH` strictly would mean the build you made specifically to get MA27
is silently used by nothing. `lcsolver-check-solvers` always names the binary it
picked and why.

To override — force one binary, or restore strict `PATH` order:

```
export LCSOLVER_IPOPT_EXECUTABLE=/path/to/ipopt/build/bin/ipopt
export LCSOLVER_IPOPT_AUTOSELECT=0
```

```python
from lcsolver.solvers.ipopt import ipopt_solve
res = ipopt_solve(f, options={'tol': 1e-8, 'max_iter': 500}, tee=True)
```

## Usage

The core object in LCsolver is the `Formulation`  object, which inherits from the `pyomo.environ.ConcreteModel`.  Essentially, a `Formulation` is a Pyomo `Model` with some extra stuff, but can be treated exactly as if it were a Pyomo `Model`.  However, an LCsolver `Formulation` has some additional features that can help simplify model construction.

Below is a simple example to get started, but additional resources can be found in the [examples](https://github.com/codykarcher/lcsolver/tree/main/examples) folder or in the LCsolver [documentation](https://github.com/codykarcher/lcsolver/tree/main/docs)

<!-- BEGIN readme_example -->
<!-- generated from examples/readme_example.py -- edit that file and run `python utilities/sync_readme.py` -->
```python
# =================
# Import Statements
# =================
import lcsolver
from lcsolver import Formulation, BlackBoxFunctionModel, units

# ===================
# Declare Formulation
# ===================
f = Formulation()

# =================
# Declare Variables
# =================
x = f.Variable(name='x', guess=1.0, units='m'  , description='The x variable')
y = f.Variable(name='y', guess=1.0, units='m'  , description='The y variable')
z = f.Variable(name='z', guess=1.0, units='m^2', description='Model output')

# =================
# Declare Constants
# =================
c = f.Constant(name='c', value=[1.0, 2.0], units='', size=2, description='A constant c')

# =====================
# Declare the Objective
# =====================
f.Objective(c[0] / x + c[1] / y)


# ===================
# Declare a Black Box
# ===================
class UnitCircle(BlackBoxFunctionModel):
    def __init__(self):  # The initialization function
        # Initialize the black box model
        super().__init__()

        # A brief description of the model
        self.description = 'This model evaluates the function: z = x**2 + y**2'

        # Declare the black box model inputs
        self.inputs.append(name='x', units='ft', description='The x variable')
        self.inputs.append(name='y', units='ft', description='The y variable')

        # Declare the black box model outputs
        self.outputs.append(
            name='z', units='ft**2', description='Resultant of the unit circle'
        )

        # Declare the maximum available derivative
        self.availableDerivative = 1

    def BlackBox(self, x, y):  # The actual function that does things
        # Convert to the declared input units (ft) and strip to plain floats
        x, y = self.sanitizeInputs(x, y, strip_units=True)

        z    = x**2 + y**2  # Compute z
        dzdx = 2 * x        # Compute dz/dx
        dzdy = 2 * y        # Compute dz/dy

        # Attach the declared units: z in ft**2, the gradient in ft**2/ft
        res = self.packOutputs(z, [dzdx, dzdy])

        return res

# =======================
# Declare the Constraints
# =======================
f.ConstraintList([
    [ z, '==', [x, y], UnitCircle() ], 
    x + y <= 1.0 * units.m
    ])

# =============================================
# Black Box can be run as a function!
# =============================================
uc = UnitCircle()
bbo = uc.BlackBox(0.5 * units.m, 0.5 * units.m)

# =======================
# Solve Model
# =======================
sol = lcsolver.solve(f)
print(sol.summary())
```
<!-- END readme_example -->

## Acknowledgement

This package is spun out of Pyomo, acknowledged below.

Pyomo: Python Optimization Modeling Objects  
Copyright (c) 2008-2023  
National Technology and Engineering Solutions of Sandia, LLC  
Under the terms of Contract DE-NA0003525 with National Technology and
Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
rights in this software.

Development of this module was conducted as part of the Institute for
the Design of Advanced Energy Systems (IDAES) with support through the
Simulation-Based Engineering, Crosscutting Research Program within the
U.S. Department of Energy’s Office of Fossil Energy and Carbon Management.

This software is distributed under the 3-clause BSD License.
