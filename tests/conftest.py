#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Test-suite policy for a machine with no IPOPT.

IPOPT cannot be installed by pip (see ``lcsolver/install.py``), so a reviewer,
a contributor, or a CI runner may reasonably not have one. Without the hook
below, that machine reports around fifty failures, none of which say anything
about the code --- they all say "no usable IPOPT installation was found". A
wall of red that means "your machine is missing an optional system package" is
worse than useless: it buries whatever is genuinely broken.

So a solve that fails *because this machine has nothing to solve with* is
reported as a skip. Two things keep that from becoming a way to hide bugs:

* Only :class:`~lcsolver.core.errors.SolverUnavailable` is converted, and it is
  raised in exactly the handful of places that check for a missing install. A
  solver that is present and fails raises something else and still fails.
* The conversion happens **only when IPOPT really is absent**. If IPOPT is
  installed and the code claims otherwise, that is a bug in the detection
  logic, and it fails as loudly as any other bug.

CI installs IPOPT in a dedicated job precisely so these tests do not sit
skipped forever; the bare-install job is what proves a fresh `pip install`
still gives a green suite.
"""

import pytest

from lcsolver.core.errors import SolverUnavailable

_IPOPT_PRESENT = None


def _ipopt_present():
    """Is there any usable IPOPT here --- executable or cyipopt? Cached."""
    global _IPOPT_PRESENT
    if _IPOPT_PRESENT is None:
        try:
            from lcsolver.solvers.solver import _ipopt_available
            _IPOPT_PRESENT = bool(_ipopt_available())
        except Exception:
            _IPOPT_PRESENT = False
    return _IPOPT_PRESENT


_ASL_PRESENT = None


def _asl_present():
    """Is Pyomo's compiled PyNumero ASL library available?

    A second environment gap of the same kind as a missing IPOPT: cyipopt
    cannot build an NLP without it, so the in-process route -- and therefore
    every black-box model -- cannot run. It is not installable by pip and not
    installable by conda on a current Python; it has to be compiled.
    """
    global _ASL_PRESENT
    if _ASL_PRESENT is None:
        try:
            from lcsolver.environment import _pynumero_asl_available
            _ASL_PRESENT = bool(_pynumero_asl_available())
        except Exception:
            _ASL_PRESENT = False
    return _ASL_PRESENT


def _convert(exc):
    if not _ipopt_present():
        pytest.skip(f'no IPOPT on this machine: {exc}')
    if 'PyNumero ASL' in str(exc) and not _asl_present():
        pytest.skip(f'no PyNumero ASL library on this machine: {exc}')
    # Everything the solve needs is installed and something still reported it
    # missing. That is a real defect, not an environment gap.
    raise exc


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except SolverUnavailable as exc:
        _convert(exc)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    # Fixtures solve too, and a failure there is an error rather than a
    # failure -- same cause, same treatment.
    try:
        return (yield)
    except SolverUnavailable as exc:
        _convert(exc)
