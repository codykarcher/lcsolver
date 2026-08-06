#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Exceptions that say what kind of problem this is.

The distinction that matters here is between *this model cannot be solved* and
*this machine has nothing to solve it with*. They are not the same event: the
first is about the user's formulation, the second is about the install, and
only the second is somebody else's problem to fix.

Keeping them apart is what lets the test suite skip cleanly on a machine
without IPOPT instead of reporting dozens of failures that say nothing about
the code --- see ``tests/conftest.py``.
"""


class SolverUnavailable(RuntimeError):
    """No usable installation of the required solver was found.

    Subclasses ``RuntimeError`` deliberately: every existing caller and test
    that catches ``RuntimeError`` keeps working, and nothing that treats this
    as a plain runtime error becomes wrong. Callers that want to tell "you
    have no IPOPT" apart from "IPOPT could not solve this" can catch it
    specifically.
    """
