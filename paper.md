---
title: 'LCsolver: An Engineering Design Interface for Pyomo'
tags:
  - Python
  - optimization
  - engineering design
  - geometric programming
  - signomial programming
  - Pyomo
authors:
  - name: Cody J. Karcher
    orcid: 0000-0002-6062-6465
    corresponding: true
    affiliation: 1
  - name: Michael Bynum
    orcid: 0000-0000-0000-0000
    affiliation: 2
affiliations:
  - name: California State University, Long Beach, CA, USA
    index: 1
  - name: Sandia National Laboratories, Albuquerque, NM, USA
    index: 2
date: 6 August 2026
bibliography: paper.bib
---

<!--
NOTE TO AUTHORS (delete before submission)
  REMAINING: (1) Bynum's ORCID is still a placeholder -- JOSS checks them, and a
  contribution statement is still to be added; (2) a Zenodo DOI, which requires a
  tagged GitHub release (needed at acceptance, not submission); (3) a SAND number
  before public release.
-->

# Summary

Engineering design optimization problems are usually assembled by hand from three
awkwardly-fitting pieces: an algebraic modeling language, a set of physical units,
and one or more external analysis codes that cannot be expressed algebraically at
all. `LCsolver` is a lightweight layer over the
Pyomo modeling language [@bynum2021pyomo] that makes these three pieces fit
together. It provides a `Formulation` object that behaves exactly like a Pyomo
`ConcreteModel` while adding unit-aware variable and constant declarations, a
structured interface for wrapping black-box analysis codes as differentiable
constraints, and automatic detection of problem structure so that a model written
once can be recognized as linear, quadratic, geometric, or signomial and handed to
an appropriate solver.

# Statement of need

Design optimization in aerospace, energy, and mechanical engineering is
characterized by models that mix closed-form physics with legacy analysis codes,
and by quantities that carry units whose mismatch is a common and expensive source
of error.

`LCsolver` targets that gap. An engineer writes a single unit-annotated model in
which some constraints are algebraic and others are evaluated by external codes;
`LCsolver` checks unit consistency, detects the mathematical structure of the
algebraic portion, and routes the problem to a solver appropriate to that
structure. The same model can therefore be solved as a geometric program when it
happens to be one, and as a general nonlinear program when it is not, without
being rewritten.

The software began in 2023 as a contributed sub-package of Pyomo itself, the
Engineering Design Interface [@karcher2023edi], and is distributed here as a
standalone package so that it can evolve independently of the Pyomo release cycle.

# State of the field

Existing tools address parts of this problem. Disciplined convex modeling packages
such as `CVXPY` [@diamond2016cvxpy] and geometric-programming packages such as
`GPkit` [@burnell2020gpkit] give excellent ergonomics and strong guarantees, but
only within their problem class. A model that is *almost* a geometric program falls
outside both, and neither accommodates an arbitrary external solver in the
constraint set — the analysis codes that dominate real engineering practice cannot
be expressed at all. General algebraic modeling languages such as Pyomo impose no
such restriction, but leave the engineer to manage units by hand and to write the
interface to any external analysis code from scratch, and they discard the
structure that makes a log-convex problem tractable.

`LCsolver` borrows its ergonomics from `GPkit` and `CVXPY` while retaining the
generality and solver ecosystem of Pyomo. The distinguishing capability is that
structure is *detected* rather than *declared*: the user is not required to know,
or to commit to, which problem class their model belongs to.

# Software design

`LCsolver` is a thin layer over Pyomo rather than a fork or a new modeling
language, and that constraint drove the significant design decisions.

A `Formulation` subclasses Pyomo's `ConcreteModel`, so every existing Pyomo tool,
solver interface, and transformation continues to work on an `LCsolver` model. The
alternative — a native model object with a Pyomo exporter — would have bought
freedom in the API at the cost of the ecosystem, which is most of Pyomo's value.

Structure detection is implemented as an expression-tree walker rather than by
asking users to declare a problem class, as `CVXPY` and `GPkit` do. Declaration is
simpler to implement and yields better error messages, but it requires the modeler
to classify their own problem, which is precisely the expertise the tool exists to
supply.

Black-box constraints are routed through Pyomo's grey-box interface rather than
through callbacks or finite differences, so an external code contributes exact
derivatives to the same KKT system as the algebraic constraints. The cost is that
the author must supply a Jacobian; the benefit is that the resulting problem is
solved rather than sampled.

Because a transformation that silently changes the problem is this architecture's
characteristic failure mode, the package carries an unusual amount of internal
checking: unit consistency is verified as the model is built, an equivalence
assertion compares the problem before and after each presolve reduction, and
post-solve checks report constraints that were declared as validity limits but
turned out to bind.

# Functionality

- **Unit-aware modeling.** Variables and constants are declared with units, guesses,
  and descriptions; unit consistency is checked as the model is built rather than at
  solve time.
- **Black-box constraints.** A `BlackBoxFunctionModel` base class exposes an
  external analysis code, including its derivatives, as a Pyomo constraint.
- **Structure detection and solver routing.** Linear and quadratic programs go to
  IPOPT in their natural variables, or to `cvxopt`; geometric programs are solved
  in log space, where they are convex, so global optimality is preserved.
  Signomial programs default to sequential inner approximation (SIA), which
  terminates on a KKT residual for the original problem, with a penalty
  convex–concave loop and Sequential Log-Convex Programming available as
  alternatives. Models carrying black-box constraints are routed to SIA, which
  imposes each box through its linearization inside a trust-region loop while
  keeping every algebraic constraint exact. Solutions are written back onto the
  model in every case.
- **Sequential Log-Convex Programming.** For the common case of a model that is
  *almost* GP-compatible, `LCsolver` implements SLCP [@karcher2022slcp]. Posynomial
  and monomial constraints are imposed exactly in a log-convex subproblem while the
  remainder is linearized in log space, recovering much of the conditioning and
  reliability of a geometric program without requiring the whole model to be one.
- **Sensitivities to constants.** After a solve, `LCsolver` reports the log-log
  sensitivity of the optimum to every declared constant, ranking a model's
  assumptions by how much they actually matter. These come from the constraint
  duals via the envelope theorem, so they cost one solve rather than the
  two-per-constant a finite difference would need, and each partial derivative is
  taken symbolically rather than by differencing. For a signomial program the duals
  are recovered from the final convex subproblem; `LCsolver` tests them against the
  stationarity condition and reports them as unreliable when they fail it, rather
  than presenting an uncertified number as though it were exact.

# Research impact

The software's predecessor has been the basis of two completed master's theses in
mechanical and aerospace engineering, both of which shaped its direction.
@avila2025aircraft develops a signomial-programming framework for the conceptual
design of commercial transport aircraft, minimizing takeoff weight across
integrated aerodynamic, structural, weight, and mission-performance models.
@shoda2026liftingline embeds lifting-line aerodynamic analysis directly in
signomial-programming form, so that wing analysis and design optimization share a
single formulation. The SLCP algorithm the package implements was developed and
validated in the peer-reviewed literature [@karcher2022slcp].

# Example

The complete path from model to result, on a geometric program small enough
to check by hand — minimize $x + y$ subject to $xy \geq A$:

```python
import lcsolver
from lcsolver import Formulation

f = Formulation()
x = f.Variable(name='x', guess=2.0, units='m',   description='width')
y = f.Variable(name='y', guess=2.0, units='m',   description='height')
A = f.Constant(name='A', value=4.0, units='m^2', description='required area')

f.Objective(x + y)
f.ConstraintList([x * y >= A])

sol = lcsolver.solve(f)
print(sol.summary())
```

```
Report
------
   Problem auto-detected as a geometric program (GP)
   Solved with ipopt (pyomo, log-transformed) [gp form: sum]

Objective
---------
   4.00 m

Variables
---------
   x  :  2.00   [m]   width
   y  :  2.00   [m]   height

Constants
---------
   A  :  4      [m**2]   required area

Sensitivities
-------------
   A  :    +0.5000   +++++++++++

Post Solve Report
-----------------
   Status: optimal
```

The reported sensitivity is exact: the optimum is $2\sqrt{A}$, so
$d \log f^* / d \log A = 1/2$. A constraint evaluated by an external analysis
code enters the same constraint list as `[z, '==', [x, y], UnitCircle()]`,
where `UnitCircle` is a `BlackBoxFunctionModel` subclass wrapping the
analysis code and its derivatives; the repository `README` shows the complete
version.

# Author Contribution Statement

Author Karcher was responsible for the primary development of the LCsolver package,
including the primary interface, structure detectors, and back end solvers.  
Author Bynum was responsible for the primary interface to black box analysis models
through the Pyomo grey-box interface, oversaw development work, guided the scope
of the software, and provided critical feedback regarding software design decisions.

# AI usage disclosure

Generative AI assistance (Anthropic's Claude) was used during the development of
this software, principally for test authoring, documentation, refactoring, and code
review. The mathematical formulations, algorithm designs, and solver strategies are
the authors' own and derive from the peer-reviewed work cited above. All
AI-assisted contributions were reviewed by the authors before being committed. This
paper was drafted by the authors with AI assistance for editing.

Correctness is established by verification in addition to inspection. The
package carries a suite of roughly 600 tests, run on Linux, macOS, and Windows
across four Python versions, with a coverage floor enforced in continuous
integration.  Tests reference published optima from the literature when they are
available to ensure correctness.

# Acknowledgements

Development of portions of this software was conducted at Sandia National
Laboratories. Sandia National Laboratories is a multimission laboratory managed and
operated by National Technology and Engineering Solutions of Sandia, LLC., a wholly
owned subsidiary of Honeywell International, Inc., for the U.S. Department of
Energy's National Nuclear Security Administration under contract DE-NA0003525.

Earlier development was supported through the Institute for the Design of Advanced
Energy Systems (IDAES) by the Simulation-Based Engineering, Crosscutting Research
Program within the U.S. Department of Energy's Office of Fossil Energy and Carbon
Management.

<!-- SAND number required before public release. -->

# References
