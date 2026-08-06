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

LCS has some standard dependencies that install on a typical python build:  pyomo, numpy, scipy, pint, packaging, and cvxopt. Optional packages include mpi4py, matplotlib, and pandas.  However, the highest quality LCS solvers depend on IPOPT to converge the hardest and most relevant engineering design problems.  IPOPT cannot be installed from pip at all: there is no IPOPT executable on PyPI, and the cyipopt package is source-only there, so it *compiles against* an IPOPT that must already exist rather than providing one.  `lcsolver-install-solvers` obtains IPOPT and then builds cyipopt against it.  Every IPOPT you can install prebuilt ships with the MUMPS linear algebra package, which is known to have performance difficulties.  We strongly encourage that users obtain MA27 and build IPOPT on this solver as opposed to the default MUMPS.  LCS is able to build and run without MA27 by default and provides easy options to upgrade later, described below.  

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
<!-- generated from examples/boyd.py -- edit that file and run `python utilities/sync_readme.py` -->
```python
# ===========
# Description
# ===========
# The box design problem, formulated as a Geometric Program
# From:  Boyd, Kim, Vandenberghe, and Hassibi
#        A Tutorial on Geometric Programming
#        Optimization and Engineering
#        2007
#
# Maximize the volume of a box (stated as minimizing the inverse volume)
# subject to a wall area limit, a floor area limit, and aspect ratio limits
# on both the height and the depth. Used as the introductory test problem in
# the SLCP paper; the optimum is 5.196e-3 1/m^3 (a volume of 192.45 m^3) at
# w = 5.774 m, h = 2.887 m, d = 11.547 m.

# =================
# Import Statements
# =================
import lcsolver
from lcsolver import Formulation, units

# ===================
# Declare Formulation
# ===================
f = Formulation()

# =================
# Declare Variables
# =================
w = f.Variable(name="w", guess = 5.0, units = "m", description="Box width")
h = f.Variable(name="h", guess = 5.0, units = "m", description="Box height")
d = f.Variable(name="d", guess = 5.0, units = "m", description="Box depth")

# =================
# Declare Constants
# =================
Aflr  = f.Constant( name="Aflr" , value=1000.0 , units="m^2" , description="Maximum floor area")
Awall = f.Constant( name="Awall", value=100.0  , units="m^2" , description="Maximum wall area")
alpha = f.Constant( name="alpha", value=0.5    , units="-"   , description="Minimum height aspect ratio h/w")
beta  = f.Constant( name="beta" , value=2.0    , units="-"   , description="Maximum height aspect ratio h/w")
gamma = f.Constant( name="gamma", value=0.5    , units="-"   , description="Minimum depth aspect ratio d/w")
delta = f.Constant( name="delta", value=2.0    , units="-"   , description="Maximum depth aspect ratio d/w")

# =====================
# Declare the Objective
# =====================
f.Objective(1 / (h * w * d))

# =======================
# Declare the Constraints
# =======================
f.ConstraintList([
    2 * (h * w + h * d) <= Awall,
    w * d <= Aflr,
    alpha <= h / w,
    h / w <= beta,
    gamma <= d / w,
    d / w <= delta,
    ])

# ===========
# Solve Model
# ===========
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
or open a pull request.

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
