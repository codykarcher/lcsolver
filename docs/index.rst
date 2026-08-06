Engineering Design Interface (LCsolver)
==================================

The Pyomo Engineering Design Interface (LCsolver) is a lightweight wrapper on the Pyomo language that is targeted at composing engineering design optimization problems.  The language and interface have been designed to mimic many of the features found in `GPkit <https://github.com/convexengineering/gpkit>`_ and `CVXPY <https://github.com/cvxpy/cvxpy>`_ while also providing a simple, clean interface for black-box analysis codes that are common in engineering design applications.


Installation
------------

LCsolver is distributed as a standalone package and is installed directly from
GitHub:

::

    pip install git+https://github.com/codykarcher/lcsolver.git


Optional extras provide the solver backends and plotting support:

::

    pip install "lcsolver[solvers] @ git+https://github.com/codykarcher/lcsolver.git"     # cvxopt
    pip install "lcsolver[plotting] @ git+https://github.com/codykarcher/lcsolver.git"    # matplotlib, pandas
    pip install "lcsolver[parallel] @ git+https://github.com/codykarcher/lcsolver.git"    # mpi4py

The IPOPT backend additionally requires either the ``ipopt`` executable on your
``PATH`` or ``pip install cyipopt``. IPOPT is the default convex backend and the
only one that can evaluate a black-box constraint, so it is worth installing
properly rather than quickly --- in particular, every prebuilt IPOPT ships with
the MUMPS linear solver, which is not the one you want. See :doc:`ipopt`.

LCsolver began as a contribution to Pyomo itself and is distributed separately so it
can evolve independently of the Pyomo release cycle.


User's Guide
------------

.. toctree::
   :maxdepth: 4

   quickstart.rst
   formulation.rst
   variables.rst
   constants.rst
   objectives.rst
   blackboxobjectives.rst
   constraints.rst
   blackboxconstraints.rst
   advancedruntimeconstraints.rst
   solvers.rst
   ipopt.rst
   sensitivities.rst
   slcp.rst
   examples.rst
   additionaltips.rst


Developers
----------

The pyomo LCsolver interface is developed and maintained by `Cody Karcher <https://github.com/codykarcher>`_ 
