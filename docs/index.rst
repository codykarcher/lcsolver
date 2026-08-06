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

Three longer engineering notes live alongside the documentation source in the
repository, for readers who want the machinery rather than the interface:
``docs/PRESOLVE.md`` (the presolve reductions and their invariants),
``docs/SIA_CONVERGENCE.md`` (the SIA convergence argument), and
``docs/SLCP_PERFORMANCE.md`` (measured SLCP behavior across the paper's test
problems).


Developers
----------

LCsolver is developed and maintained by `Cody Karcher
<https://github.com/codykarcher>`_.
