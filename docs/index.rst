Engineering Design Interface (EDI)
==================================

The Pyomo Engineering Design Interface (EDI) is a lightweight wrapper on the Pyomo language that is targeted at composing engineering design optimization problems.  The language and interface have been designed to mimic many of the features found in `GPkit <https://github.com/convexengineering/gpkit>`_ and `CVXPY <https://github.com/cvxpy/cvxpy>`_ while also providing a simple, clean interface for black-box analysis codes that are common in engineering design applications.


Installation
------------

EDI is distributed as a standalone package and is installed directly from
GitHub:

::

    pip install git+https://github.com/codykarcher/edi.git


Optional extras provide the solver backends and plotting support:

::

    pip install "edi[solvers] @ git+https://github.com/codykarcher/edi.git"     # cvxopt
    pip install "edi[plotting] @ git+https://github.com/codykarcher/edi.git"    # matplotlib, pandas
    pip install "edi[parallel] @ git+https://github.com/codykarcher/edi.git"    # mpi4py

The IPOPT backend additionally requires either the ``ipopt`` executable on your
``PATH`` or ``pip install cyipopt``.

EDI began as a contribution to Pyomo itself and is distributed separately so it
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
   sensitivities.rst
   slcp.rst
   examples.rst
   additionaltips.rst


Developers
----------

The pyomo EDI interface is developed and maintained by `Cody Karcher <https://github.com/codykarcher>`_ 
