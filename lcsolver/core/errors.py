#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Exceptions that say what kind of problem this is.

"This model cannot be solved" and "this machine has nothing to solve it
with" are different events -- formulation vs install -- and keeping them
apart lets the test suite skip cleanly on a machine without IPOPT
(see tests/conftest.py).
"""


class SolverUnavailable(RuntimeError):
    """No usable installation of the required solver was found. Subclasses
    RuntimeError deliberately, so existing callers catching RuntimeError
    keep working; catch this to tell "no IPOPT" from "IPOPT failed"."""
