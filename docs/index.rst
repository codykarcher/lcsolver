LCsolver
========

LCsolver is a standalone Python package for composing and solving engineering
design optimization problems. It is written on top of the `Pyomo
<https://pyomo.readthedocs.io>`_ modeling language, and its interface is
designed to mimic many of the features found in `GPkit
<https://github.com/convexengineering/gpkit>`_ and `CVXPY
<https://github.com/cvxpy/cvxpy>`_ while also providing a simple, clean
interface for the black-box analysis codes that are common in engineering
design applications.

The package detects the mathematical structure of a model -- linear,
quadratic, geometric, or signomial program -- and routes it to a solver that
exploits that structure, checks the model on the way in and the solution on
the way out, and reports sensitivities with every solve.

(LCsolver began life as a contributed package inside Pyomo, the "Engineering
Design Interface". It is now developed and distributed independently.)


Statement of Need
-----------------

Design optimization in aerospace, energy, and mechanical engineering is
characterized by models that mix closed-form physics with legacy analysis
codes, and by quantities that carry units whose mismatch is a common and
expensive source of error. Disciplined convex modeling packages such as CVXPY
and geometric-programming packages such as GPkit give excellent ergonomics and
strong guarantees, but only within their problem class, and neither
accommodates an arbitrary external solver in the constraint set. General
algebraic modeling languages such as Pyomo impose no such restriction, but
leave the engineer to manage units by hand and to write the interface to any
external analysis code from scratch.

LCsolver targets that gap: an engineer writes a single unit-annotated model in
which some constraints are algebraic and others are evaluated by external
codes, and LCsolver checks unit consistency, detects the mathematical
structure of the algebraic portion, and routes the problem to a solver
appropriate to that structure. The same model can be solved as a geometric
program when it happens to be one, and as a general nonlinear program when it
is not, without being rewritten.


Installation
------------

LCsolver is installed directly from GitHub:

::

    pip install git+https://github.com/codykarcher/lcsolver.git


(``pip install lcsolver`` follows once the PyPI release is published.)

cvxopt comes with it. Optional extras add plotting and MPI support:

::

    pip install "lcsolver[plotting] @ git+https://github.com/codykarcher/lcsolver.git"    # matplotlib, pandas
    pip install "lcsolver[parallel] @ git+https://github.com/codykarcher/lcsolver.git"    # mpi4py

IPOPT is the second step, and it cannot come from pip --- see :doc:`ipopt` for
why not:

::

    lcsolver-install-solvers        # once per environment
    lcsolver-check-solvers          # what you ended up with

IPOPT is the default convex backend and the only one that can evaluate a
black-box constraint, so it is worth installing properly rather than quickly:
every prebuilt IPOPT ships with the MUMPS linear solver, which is not the one
you want. ``lcsolver-install-solvers --ma27 <path>`` builds one that is.


User's Guide
------------

.. toctree::
   :maxdepth: 4

   quickstart.rst
   formulation.rst
   variables.rst
   constants.rst
   objectives.rst
   constraints.rst
   blackboxconstraints.rst
   advancedruntimeconstraints.rst
   holographic.rst
   submodels.rst
   solvers.rst
   checks.rst
   ipopt.rst
   results.rst
   sensitivities.rst
   examples.rst
   additionaltips.rst


Reference
---------

.. toctree::
   :maxdepth: 2

   api.rst


Technical Notes
---------------

Three longer engineering notes, for readers who want the machinery rather than
the interface: the presolve reductions and their invariants, the SIA
convergence argument, and measured SLCP behavior across the paper's test
problems.

.. toctree::
   :maxdepth: 2

   PRESOLVE.md
   SIA_CONVERGENCE.md
   SLCP_PERFORMANCE.md


Developers
----------

LCsolver is developed and maintained by `Cody Karcher
<https://github.com/codykarcher>`_.
