"""A crashed ipopt executable is a named, recoverable failure -- not a raw
pyomo ERROR log and a bare ApplicationError.

Found on the SPaircraft D8 under SPRAL: two of 229 SIA sub-problems killed
the executable (SIGBUS, SIGSEGV) after SPRAL declared the KKT system
singular. pyomo printed the whole solver log at ERROR level and raised an
exception the SIA main loop did not catch. ipopt_launch captures the log,
names the signal and the linear solver, and raises IpoptCrashed -- a
RuntimeError the sequential loops already treat as a failed sub-problem,
which they now retry under MA27 first.
"""
import logging
import warnings

import pytest

try:
    from pyomo.common.errors import ApplicationError

    from lcsolver.environment import (
        IpoptCrashed,
        crash_fallback_solver,
        ipopt_launch,
    )
    available = True
except Exception:                                    # pragma: no cover
    available = False


def _pyomo_style_crash(rc):
    """What pyomo does when the executable dies: log, then raise."""
    log = logging.getLogger('pyomo.opt')
    log.error('Solver (ipopt) returned non-zero return code (%s)' % rc)
    log.error('Solver log:\nIpopt 3.14.20: linear_solver=spral')
    raise ApplicationError('Solver (ipopt) did not exit normally')


@pytest.mark.skipif(not available, reason='LCsolver import failed')
class TestIpoptLaunch:

    def test_signal_is_named_and_the_log_is_not_printed(self, caplog):
        with caplog.at_level(logging.ERROR, logger='pyomo.opt'):
            with pytest.raises(IpoptCrashed) as ctx:
                with ipopt_launch('spral'):
                    _pyomo_style_crash(-10)
        msg = str(ctx.value)
        assert 'SIGBUS' in msg and 'signal 10' in msg
        assert "linear_solver='spral'" in msg
        assert 'MA27' in msg
        assert 'Solver log' not in caplog.text     # captured, not printed

    def test_positive_exit_code_is_reported_as_such(self):
        with pytest.raises(IpoptCrashed) as ctx:
            with ipopt_launch('mumps'):
                _pyomo_style_crash(2)
        assert 'exit code 2' in str(ctx.value)

    def test_it_is_a_runtime_error(self):
        """The SIA/SLCP main loops catch RuntimeError for sub-problem
        failures; a crash must land in that handler."""
        assert issubclass(IpoptCrashed, RuntimeError)

    def test_recoverable_files_a_tagged_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(IpoptCrashed):
                with ipopt_launch('spral', recoverable=True):
                    _pyomo_style_crash(-11)
        tagged = [str(w.message) for w in caught
                  if str(w.message).startswith('[LC-W313]')]
        assert len(tagged) == 1
        assert 'SIGSEGV' in tagged[0]

    def test_not_recoverable_files_no_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(IpoptCrashed):
                with ipopt_launch('spral'):
                    _pyomo_style_crash(-10)
        assert not [w for w in caught
                    if str(w.message).startswith('[LC-W313]')]

    def test_a_clean_solve_passes_through_and_logging_is_restored(self, caplog):
        log = logging.getLogger('pyomo.opt')
        before = (list(log.handlers), log.propagate)
        with caplog.at_level(logging.WARNING, logger='pyomo.opt'):
            with ipopt_launch('ma27'):
                log.warning('harmless note from pyomo')
        assert (list(log.handlers), log.propagate) == before
        assert 'harmless note' in caplog.text      # re-emitted, not lost

    def test_other_exceptions_are_untouched(self):
        with pytest.raises(ValueError):
            with ipopt_launch('ma27'):
                raise ValueError('not a crash')


@pytest.mark.skipif(not available, reason='LCsolver import failed')
class TestFallback:

    def test_ma27_never_falls_back_to_itself(self):
        assert crash_fallback_solver('ma27') is None
        assert crash_fallback_solver(None) is None

    def test_never_raises_on_a_bad_executable(self):
        assert crash_fallback_solver('spral', '/no/such/ipopt') in (None, 'ma27')


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
