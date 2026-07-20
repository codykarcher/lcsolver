---
title: 'EDI: An Engineering Design Interface for Pyomo'
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
date: 19 July 2026
bibliography: paper.bib
---

<!--
NOTE TO AUTHORS (delete before submission)
  * Michael Bynum's ORCID is still a placeholder and must be filled;
    JOSS checks them. Karcher's is set.
  * JOSS papers are SHORT: roughly 250-1000 words. This draft is near the top of
    that range. Resist expanding it; JOSS reviews the SOFTWARE, and the paper is
    only a pointer.
  * Repository requirements are now satisfied: BSD-3-Clause LICENSE, Sphinx docs,
    110 automated tests, CI, and CONTRIBUTING.md.
  * REMAINING BLOCKERS: (1) push the repo -- LICENSE, docs/, tests/, .github/,
    paper.md and the whole solvers/ipopt tree are still untracked locally, so the
    public repo does not yet contain the things JOSS checks; (2) Bynum's ORCID;
    (3) a Zenodo DOI, which requires a tagged GitHub release; (4) a SAND number.
-->

# Summary

Engineering design optimization problems are usually assembled by hand from three
awkwardly-fitting pieces: an algebraic modeling language, a set of physical units,
and one or more external analysis codes that cannot be expressed algebraically at
all. `EDI` — the Engineering Design Interface — is a lightweight layer over the
Pyomo modeling language [@bynum2021pyomo] that makes these three pieces fit
together. It provides a `Formulation` object that behaves exactly like a Pyomo
`ConcreteModel` while adding unit-aware variable and constant declarations, a
structured interface for wrapping black-box analysis codes as differentiable
constraints, and automatic detection of problem structure so that a model written
once can be recognised as linear, quadratic, geometric, or signomial and handed to
an appropriate solver.

The interface deliberately borrows its ergonomics from `GPkit` [@burnell2020gpkit]
and `CVXPY` [@diamond2016cvxpy], which are pleasant to write models in but
restricted to particular problem classes, while retaining the generality and solver
ecosystem of Pyomo.

# Statement of need

Design optimization in aerospace, energy, and mechanical engineering is
characterised by models that mix closed-form physics with legacy analysis codes,
and by quantities that carry units whose mismatch is a common and expensive source
of error. Existing tools address parts of this problem. Disciplined convex modeling
packages such as `CVXPY` and geometric-programming packages such as `GPkit` give
excellent ergonomics and strong guarantees, but only within their problem class,
and neither accommodates an arbitrary external solver in the constraint set.
General algebraic modeling languages such as Pyomo impose no such restriction, but
leave the engineer to manage units manually and to hand-roll the interface to any
external analysis code.

`EDI` targets the gap. An engineer writes a single unit-annotated model in which
some constraints are algebraic and others are evaluated by external codes; `EDI`
checks unit consistency, detects the mathematical structure of the algebraic
portion, and routes the problem to a solver appropriate to that structure. The same
model can therefore be solved as a geometric program when it happens to be one, and
as a general nonlinear program when it is not, without being rewritten.

`EDI` began as a contribution to Pyomo itself and is distributed here as a
standalone package so that it can evolve independently of the Pyomo release cycle.

# Functionality

- **Unit-aware modeling.** Variables and constants are declared with units, guesses,
  and descriptions; unit consistency is checked as the model is built rather than at
  solve time.
- **Black-box constraints.** A `BlackBoxFunctionModel` base class provides a
  structured way to expose an external analysis code, including its derivatives, as
  a Pyomo constraint via the grey-box interface.
- **Structure detection.** A model walker classifies the algebraic structure of the
  formulation, recognising linear, quadratic, geometric, and signomial forms.
- **Solver routing.** Detected structure is dispatched to an appropriate backend:
  linear, quadratic and geometric programs to `cvxopt` or, optionally, to IPOPT
  applied to the log-transformed problem; signomial programs to a solver built on
  successive monomial approximation with a penalty convex–concave step, or to
  Sequential Log-Convex Programming; and anything else, including models
  containing black-box constraints, to IPOPT via Pyomo. Solutions are written
  back onto the model in every case.
- **Sequential Log-Convex Programming.** For the common case of a model that is
  *almost* GP-compatible, `EDI` implements SLCP [@karcher2022slcp]. Posynomial and
  monomial constraints are imposed exactly in a log-convex subproblem while the
  remainder — including constraints evaluated by external codes — is linearised
  in log space. This recovers much of the conditioning and reliability of a
  geometric program without requiring the whole model to be one.

# Example

<!-- Keep this short; JOSS wants a taste, not a tutorial. -->

```python
from edi import Formulation, BlackBoxFunctionModel

f = Formulation()
x = f.Variable(name='x', guess=1.0, units='m',   description='x variable')
y = f.Variable(name='y', guess=1.0, units='m',   description='y variable')
z = f.Variable(name='z', guess=1.0, units='m^2', description='unit circle output')
c = f.Constant(name='c', value=1.0, units='', description='a constant', size=2)

f.Objective(c[0] * x + c[1] * y)
```

# Acknowledgements

<!-- Sandia co-author => the NTESS/DOE disclaimer and a SAND number are required. -->
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
