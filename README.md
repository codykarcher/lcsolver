# Engineering Design Interface

The Pyomo Engineering Design Interface (EDI) is a lightweight wrapper on the Pyomo language that is targeted at composing engineering design optimization problems.  The language and interface have been designed to mimic many of the features found in [GPkit](https://github.com/convexengineering/gpkit) and [CVXPY](https://github.com/cvxpy/cvxpy) while also providing a simple, clean interface for black-box analysis codes that are common in engineering design applications.

## Statement of Need

Design optimization in aerospace, energy, and mechanical engineering is characterized by models that mix closed-form physics with legacy analysis codes, and by quantities that carry units whose mismatch is a common and expensive source of error.  Disciplined convex modeling packages such as CVXPY and geometric-programming packages such as GPkit give excellent ergonomics and strong guarantees, but only within their problem class, and neither accommodates an arbitrary external solver in the constraint set.  General algebraic modeling languages such as Pyomo impose no such restriction, but leave the engineer to manage units manually and to hand-roll the interface to any external analysis code.

EDI targets the gap: an engineer writes a single unit-annotated model in which some constraints are algebraic and others are evaluated by external codes; EDI checks unit consistency, detects the mathematical structure of the algebraic portion, and routes the problem to a solver appropriate to that structure.

## Installation

EDI began as a contribution to Pyomo itself (`pyomo.contrib.edi`) and is now distributed as a standalone package.  Install it directly from GitHub:

```
pip install git+https://github.com/codykarcher/edi.git
```

or from a local clone:

```
git clone https://github.com/codykarcher/edi.git
cd edi
pip install -e .
```

The structured solver backends require `cvxopt`, which is an optional extra:

```
pip install "edi[solvers] @ git+https://github.com/codykarcher/edi.git"
```

## Solving

EDI detects the structure of a formulation and routes it to an appropriate solver.

```python
from edi.solvers.solver import solve

res = solve(f)                    # auto: cvxopt for LP/QP/GP/SP, IPOPT otherwise
res = solve(f, solver='ipopt')    # force IPOPT (general NLP, and black-box models)
res = solve(f, solver='cvxopt')   # force the structured backends
```

After any successful solve the solution is written back onto the model, so

```python
import pyomo.environ as pyo
pyo.value(f.x)                    # the optimum, not the initial guess
```

### Sensitivities

After a solve, EDI reports how strongly the optimum responds to each `Constant`,
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
AMPL-based interface, and needs the `ipopt` executable on PATH. `method='cyipopt'`
uses `pyomo.contrib.pynumero` and needs `pip install cyipopt`. The default,
`method='auto'`, prefers the Pyomo route but switches to cyipopt when the model
contains black-box (grey-box) constraints, which the AMPL route cannot evaluate.

Getting IPOPT itself is worth doing carefully. `conda install -c conda-forge
ipopt` works and takes a minute, but like every prebuilt IPOPT it is built
against the MUMPS linear solver — the only one that may be redistributed — and
MUMPS is not what you want underneath a geometric or signomial program. IPOPT's
own default is `ma27`. See [docs/ipopt.rst](docs/ipopt.rst) for why, and
[tools/install_ipopt.sh](tools/install_ipopt.sh) to build against HSL MA27.

```python
from edi.solvers.ipopt import ipopt_solve
res = ipopt_solve(f, options={'tol': 1e-8, 'max_iter': 500}, tee=True)
```

## Usage

The core object in EDI is the `Formulation`  object, which inherits from the `pyomo.environ.ConcreteModel`.  Essentially, a `Formulation` is a Pyomo `Model` with some extra stuff, but can be treated exactly as if it were a Pyomo `Model`.  However, an EDI `Formulation` has some additional features that can help simplify model construction.

Below is a simple example to get started, but additional resources can be found in the [examples](https://github.com/codykarcher/edi/tree/main/examples) folder or in the EDI [documentation](https://github.com/codykarcher/edi/tree/main/docs)

```python
# =================
# Import Statements
# =================
import pyomo.environ as pyo
from pyomo.environ import units
from edi import Formulation, BlackBoxFunctionModel

# ===================
# Declare Formulation
# ===================
f = Formulation()

# =================
# Declare Variables
# =================
x = f.Variable(name = 'x', guess = 1.0, units = 'm'  , description = 'The x variable')
y = f.Variable(name = 'y', guess = 1.0, units = 'm'  , description = 'The y variable')
z = f.Variable(name = 'z', guess = 1.0, units = 'm^2', description = 'The unit circle output')

# =================
# Declare Constants
# =================
c = f.Constant(name = 'c', value = 1.0, units = '', description = 'A constant c', size = 2)

# =====================
# Declare the Objective
# =====================
f.Objective(
    c[0]*x + c[1]*y
)

# ===================
# Declare a Black Box
# ===================
class UnitCircle(BlackBoxFunctionModel):
    def __init__(self): # The initialization function
        
        # Initialize the black box model
        super().__init__()

        # A brief description of the model
        self.description = 'This model evaluates the function: z = x**2 + y**2'
        
        # Declare the black box model inputs
        self.inputs.append(name = 'x', units = 'ft' , description = 'The x variable')
        self.inputs.append(name = 'y', units = 'ft' , description = 'The y variable')

        # Declare the black box model outputs
        self.outputs.append(name = 'z', units = 'ft**2',  description = 'Resultant of the unit circle evaluation')

        # Declare the maximum available derivative
        self.availableDerivative = 1

        # Post-initialization setup
        self.post_init_setup()

    def BlackBox(self, x, y): # The actual function that does things
        x = pyo.value(units.convert(x,self.inputs['x'].units)) # Converts to correct units then casts to float
        y = pyo.value(units.convert(y,self.inputs['y'].units)) # Converts to correct units then casts to float

        z = x**2 + y**2 # Compute z
        dzdx = 2*x      # Compute dz/dx
        dzdy = 2*y      # Compute dz/dy

        z *= units.ft**2
        dzdx *= units.ft # units.ft**2 / units.ft
        dzdy *= units.ft # units.ft**2 / units.ft
        
        return z, [dzdx, dzdy] # return z, grad(z), hess(z)...

# =======================
# Declare the Constraints
# =======================
f.ConstraintList(
    [
        [ z, '==', [x,y], UnitCircle() ] ,
        z <= 1*units.m**2
    ]
)
```

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
