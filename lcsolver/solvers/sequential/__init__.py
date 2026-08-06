#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Sequential methods for signomial programs: PCCP, SLCP and SIA.

These are METHODS, not backends: each iterates convex subproblems, and the
backend folders (``cvxopt/``, ``ipopt/``) provide the atomic solves they and
the router build on. PCCP takes its inner GP solver as a parameter
(``gp_solver=``); SLCP and SIA currently run their subproblems on the IPOPT
machinery in :mod:`lcsolver.solvers.sequential.slcp`.

``bridge`` translates a detected structure (grey-box rows included) into the
:class:`~lcsolver.solvers.sequential.slcp.Problem` form SLCP and SIA consume.
"""

from lcsolver.solvers.sequential.bridge import (build_problem,  # noqa: F401
                                                solve_sia, solve_slcp)
