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

## Dependencies

LCS has some standard dependencies that install on a typical python build:  pyomo, numpy, scipy, pint, packaging, and cvxopt. Optional packages include mpi4py, matplotlib, and pandas.  However, the highest quality LCS solvers depend on IPOPT to converge the hardest and most relevant engineering design problems.  Ipopt can be installed through the cyipopt package, but ships by default with the MUMPS linear algebra package, which is known to have performance difficulties.  We strongly encourage that users obtain MA27 and build IPOPT on this solver as opposed to the default MUMPS.  LCS is able to build and run without MA27 by default and provides easy options to upgrade later, described below.  

## Installation

Installation takes two steps due to the IPOPT dependency.

Quickstart TLDR:  First obtain MA27 which is free for individual use

```
https://www.hsl.rl.ac.uk/download/MA27/1.0.0/a/
```

Then run the following commands
```
pip install git+https://github.com/codykarcher/lcsolver.git
lcsolver-install-solvers --ma27 <path-to-extracted-MA27-sources>
```

However, more detailed build instructions are below

### 1. The package

Installing via pip is the simplest method

```
pip install git+https://github.com/codykarcher/lcsolver.git
```

or, from a clone, `pip install -e .`.

**If you wish to build in an environment** run the following

```
git clone https://github.com/codykarcher/lcsolver.git
cd lcsolver
conda env create -f environment.yml     # python, cvxopt, ipopt, cyipopt
conda activate lcsolver
pip install -e .
```

Pypi support is coming soon.

### 2. The solvers

After installing the core package, run the following to install the solvers:

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

### 3. MA27, if you want it (you do, trust me)

Either route above leaves you on a **MUMPS** build of IPOPT, because MUMPS is the only linear solver that may be redistributed. That is a working install. For geometric and signomial programs MA27 is markedly more robust; it is free for individual use, but it has to be fetched by hand and IPOPT rebuilt against it.  You can request a download of MA27 here:

```
https://www.hsl.rl.ac.uk/download/MA27/1.0.0/a/
```

and then install LCS with MA27 using the following command
```
lcsolver-install-solvers --ma27 <path-to-extracted-MA27-sources>
```

This command serves as both a first install and as an upgrade on top of an existing MUMPS one: it rebuilds IPOPT, relinks cyipopt to match, and records where the build went. Nothing needs to go in a shell profile — LCsolver finds a build it installed even when that build is on no `PATH`, and prefers an MA27 build over a MUMPS one whatever `PATH` order says. See [docs/ipopt.rst](docs/ipopt.rst) for why this is worth doing.

For any other environment on the same machine, `lcsolver-install-solvers --relink-cyipopt` points that environment's cyipopt at the MA27 build without rebuilding IPOPT again.

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

## Citing LCsolver

Citation metadata is in [CITATION.cff](CITATION.cff), which GitHub renders as a
"Cite this repository" button and most reference managers read directly.

The algorithm behind the SLCP route is published separately:

> Karcher, C. and Haimes, R., "A Method of Sequential Log-Convex Programming for
> Engineering Design," *Optimization and Engineering*, 2022.
> [doi:10.1007/s11081-022-09750-3](https://doi.org/10.1007/s11081-022-09750-3)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to report a bug, ask a question,
or open a pull request, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for the
conduct expected of participants.

## Acknowledgement

Generative AI was used to assist in the development, documentation, and testing
of this software package.  Human developers have reviewed and verified the code
to ensure its quality and accuracy.

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
