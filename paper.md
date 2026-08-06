---
title: 'LCsolver: A Python Package for Exploiting Log-Convexity in Engineering Design Optimization'
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
  REMAINING: (1) Bynum's ORCID is still a placeholder -- JOSS checks them; (2) a Zenodo DOI, which requires a
  tagged GitHub release (needed at acceptance, not submission); (3) a SAND number
  before public release.
-->

# Summary

Engineering design optimization often takes the form of designing a product that
maximizes system performance subject to the constraints of physics, budget, and schedule.
These problems, when cast in the language of mathematics, are
*numerical optimization* problems.  The LCsolver package seeks to facilitate the
formulation of engineering design optimization problems and support back-end
numerical algorithms that provide solutions to these problems as formulated.
Specifically, LCsolver builds a thin layer on top of the
Pyomo modeling language [@bynum2021pyomo] to provide design engineers with a
clean interface to constructing optimization problems, then programmatically
detects the mathematical structure of the optimization formulation, identifies
the proper solver for the identified problem structure, and executes a back-end
solver to obtain an optimal design.

# Statement of need

The field of Multi-Disciplinary Analysis and Optimization (MDAO) seeks to solve
engineering design problems, typically by creating large interconnected frameworks
of software analysis tools, and then connecting these tools to general-purpose
optimization algorithms that treat the analyses as opaque
[@martins2013multidisciplinary]. However, a parallel track of work has emerged
that challenges this paradigm, showing that when the optimization is given
equal priority with the analysis, optimal designs can be determined with remarkable
efficiency [@hoburg2014geometric].  Key to these developments has been the discovery
that many design problems are well represented as Geometric Programs
[@boyd2007tutorial], which are a type of optimization problem that becomes *convex*
upon a transformation to log-log coordinates.  This log-convex property is what
gives LCsolver its name and is the key to rapid design optimization.

Basic Geometric Programs (GPs) are well understood and can be rapidly solved by
existing primal-dual interior point methods.  While many academic problems
have been shown to conform to the rigid set of mathematical rules that define
GPs, real engineering design problems rarely fit into so neat a box.  The need
is for a reliable software tool that can exploit log-convex structure when
it is present, while still allowing for the messy edges where the strict GP
formulation fails.  LCsolver provides this capability.

# State of the field

Many tools exist for MDAO, the most notable being NASA's OpenMDAO code [@gray2019openmdao].
While OpenMDAO is powerful with near unbounded modeling capability, it has no tools
for exploiting log-convexity and leans heavily on an analysis-centered interface
that is not conducive to formulating optimization-centered design problems.

On the other end of the spectrum, tools like `CVXPY` [@diamond2016cvxpy] and
`GPkit` [@burnell2020gpkit] provide excellent user interfaces to mathematical
modeling and allow for log-convex exploitation, but are narrowly limited in terms
of their modeling scope and solver capability.  In these cases, CVXPY is limited
to convex optimization problems and GPkit solves only geometric programs and
their generalization, signomial programs.  CVXPY and GPkit also lock out the use
of black-box analysis models (such as computational fluid dynamics, or finite
element analysis) that have become the hallmark of modern engineering design.

Pyomo [@bynum2021pyomo] is a highly flexible modeling framework, capable of
capturing complex mathematical relationships and passing them to premier
solvers such as IPOPT [@wachter2006implementation].  However, Pyomo has three gaps.
First, Pyomo's interface and syntax favor modeling flexibility over the clean
engineering focused language of GPkit.  Second, Pyomo has no awareness of
log-convexity, nor of any other optimization structures present in the problems
it creates.  Third, even premier solvers are unable to converge on many
engineering problems of interest, and so new algorithms are required beyond what
Pyomo currently offers.

# Software design

The LCsolver (LCS) package is composed of four primary elements.  First is a thin
wrapping layer over Pyomo that mimics the front end of GPkit and similar
engineering design focused tools.  This approach does come at the cost of some
flexibility in the native LCS interface, but since all LCS models *are* Pyomo
models, power users can still express full model control as needed.

Second is a structure detector that classifies the constructed optimization
formulation.  LCS at time of writing detects the following optimization
formulation types:  Linear Programs (LPs), Quadratic Programs (QPs), Geometric
Programs (GPs), Signomial Programs (SPs), and variants of each of these that
contain black-box constraints that are not inspectable by the solver.  This
classification is done by means of a Pyomo walker that walks the objectives
and constraints of the problems to check if they are compatible with the
underlying mathematics of each form.

Third is a series of pre-solve checks that are performed on the structure
detected problem.  A check is run for unit consistency, ensuring that the
human design engineer has not accidentally set a constraint of wing area
(units of length squared) to be less than a fixed wing span (units of
length), which exposes a problematic gap between the modeler's intent and
the problem being represented that must be closed.  The model is also checked
for unbounded variables that may be driven incorrectly to positive or
negative infinity.  Further checks target the quieter failure modes that
produce plausible-looking answers rather than errors, including variables 
that appear in no constraint at all, constraint rows that duplicate one 
another, and other conditioning checks.

Finally, LCS solves the problem as formulated and returns the result.  For
convex formulations (LP, QP, GP), LCS provides two options.  CVXOPT
[@andersen2013cvxopt] is a free Python package for solving convex optimization
problems that does well on small-to-medium-sized problems that are well
conditioned.  Alternatively, IPOPT is available as part of the provided installer,
though with the default MUMPS linear solver.  Users are encouraged in the
documentation to install MA27 for solving more challenging problems.

LCS defaults to IPOPT when it is available.  However, it is on the more
complex problems that LCS is most differentiated from its predecessors.

For signomial programs and problems with black-box analysis models, LCS
provides two algorithms: Sequential Log-Convex Programming (SLCP)
[@karcher2022slcp] and the Sequential Inner Approximation (SIA) algorithm,
an evolution of SLCP that uses conservative constraint approximations to
exploit additional log-convex structure and improve problem convergence
(publication pending).  These two algorithms enable LCS to give the human
design engineer access to the majority of Pyomo's modeling flexibility,
while still exploiting log-convexity for highly efficient solutions.

Black-box integration is handled through Pyomo's grey-box framework,
bridging the previous gap between GPkit and OpenMDAO.  In the event that
the formulated problem has no detectable underlying structure, LCS passes the
problem to IPOPT for a non-linear solve, meaning that structure is exploited
where possible, but does not constrain the user unnecessarily.

Upon return, LCS also compiles the dual variables from the optimization solve
into sensitivities of the objective to critical modeling parameters and reports
these out to the user, similar to a core feature of GPkit.

# Research impact

Prior versions of LCS (branded as the Engineering Design Interface for Pyomo)
have been used to develop signomial programming compatible models for subsonic
transport aircraft design [@avila2025aircraft] and for computing aircraft
induced drag via Trefftz Plane Analysis [@shoda2026liftingline].  LCS also serves
as the definitive implementation of Sequential Log-Convex Programming
[@karcher2022slcp] and the newly developed Sequential Inner Approximation algorithm.

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

# Author contribution statement

Author Karcher was responsible for the primary development of the LCsolver package,
including the primary interface, structure detectors, and back-end solvers.

Author Bynum was responsible for the primary interface to black-box analysis models
through the Pyomo grey-box interface, oversaw development work, guided the scope
of the software, and provided critical feedback regarding software design decisions.

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
