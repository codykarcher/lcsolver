#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Test-suite policy for a machine with no IPOPT.

IPOPT isn't pip-installable, so without this hook a bare machine shows ~50
failures that all mean "missing optional system package". Convert those to
skips -- but only SolverUnavailable, and only when IPOPT really is absent, so
a present-but-broken solver still fails. CI installs IPOPT in a dedicated job
so these don't sit skipped forever.
"""

import pytest

from lcsolver.core.errors import SolverUnavailable

_IPOPT_PRESENT = None


def _ipopt_present():
    """Is there any usable IPOPT here --- executable or cyipopt? Cached."""
    global _IPOPT_PRESENT
    if _IPOPT_PRESENT is None:
        try:
            from lcsolver.solvers.solver import any_ipopt_available
            _IPOPT_PRESENT = bool(any_ipopt_available())
        except Exception:
            _IPOPT_PRESENT = False
    return _IPOPT_PRESENT


_ASL_PRESENT = None


def _asl_present():
    """Is Pyomo's compiled PyNumero ASL library available? Same environment
    gap as a missing IPOPT: without it cyipopt cannot build an NLP, so no
    black-box model can run. Not pip/conda installable; must be compiled."""
    global _ASL_PRESENT
    if _ASL_PRESENT is None:
        try:
            from lcsolver.environment import pynumero_asl_available
            _ASL_PRESENT = bool(pynumero_asl_available())
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


# ---------------------------------------------------------------------------
# --no-skip: an environment audit. Every skip in this suite is an
# environment gap (no IPOPT, no cyipopt, no ASL library, no pint), and a
# broken install skips the same way a missing one does. Under --no-skip a
# skip is a failure that names the gap, so a machine that claims to have
# everything has to prove it.
# ---------------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        '--no-skip', action='store_true', default=False,
        help='fail every skipped test, naming the skip reason; use to audit '
             'an install that is supposed to have IPOPT, cyipopt and the '
             'PyNumero ASL library')


def _fail_skipped(report, config):
    # an xfail also reports as skipped; it is a known defect, not a gap
    if (report.skipped and not hasattr(report, 'wasxfail')
            and config.getoption('--no-skip')):
        reason = report.longrepr
        if isinstance(reason, tuple):           # (path, lineno, message)
            reason = reason[2]
        report.outcome = 'failed'
        report.longrepr = f'skipped, refused under --no-skip: {reason}'


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    _fail_skipped(report, item.config)
    return report


_CONFIG = None


def pytest_configure(config):
    global _CONFIG
    _CONFIG = config


@pytest.hookimpl(wrapper=True)
def pytest_collectreport(report):
    # a module-level importorskip skips at collection, before any item exists
    result = yield
    if _CONFIG is not None:
        _fail_skipped(report, _CONFIG)
    return result
