"""Which linear solver runs when solve() is told nothing, and which one a
crashed sub-problem is retried under.

The open-source build carries MUMPS + SPRAL; MA27 is optional. So the
default is MA27 when present (fastest, most robust measured), else SPRAL
(certifies everything MA27 does), else MUMPS -- chosen explicitly so the
choice is recorded and SPRAL's runtime env and mc64 default always apply.
SPRAL is never chosen in-process (its OpenMP runtime cannot share a Python
process with conda's), and a crashed sub-problem retries under the most
robust OTHER solver the build has: MA27, else MUMPS.
"""
import pytest

try:
    from lcsolver import environment as env
    available = True
except Exception:                                    # pragma: no cover
    available = False


def _have(*names):
    """Patch the probes so only ``names`` exist."""
    def exe_probe(name, executable=None, library=None):
        return name in names

    def cy_probe(name):
        return name in names
    return exe_probe, cy_probe


@pytest.mark.skipif(not available, reason='LCsolver import failed')
class TestDefault:

    def test_order_is_ma27_spral_mumps(self):
        assert env.DEFAULT_LINEAR_SOLVER_ORDER == ('ma27', 'spral', 'mumps')

    def test_ma27_wins_when_present(self, monkeypatch):
        exe_probe, _ = _have('ma27', 'mumps', 'spral')
        monkeypatch.setattr(env, 'linear_solver_available', exe_probe)
        assert env.default_linear_solver('pyomo') == 'ma27'

    def test_the_open_source_build_defaults_to_spral(self, monkeypatch):
        exe_probe, _ = _have('mumps', 'spral')
        monkeypatch.setattr(env, 'linear_solver_available', exe_probe)
        assert env.default_linear_solver('pyomo') == 'spral'

    def test_mumps_only_build_defaults_to_mumps(self, monkeypatch):
        exe_probe, _ = _have('mumps')
        monkeypatch.setattr(env, 'linear_solver_available', exe_probe)
        assert env.default_linear_solver('pyomo') == 'mumps'

    def test_nothing_probed_leaves_ipopt_to_itself(self, monkeypatch):
        exe_probe, _ = _have()
        monkeypatch.setattr(env, 'linear_solver_available', exe_probe)
        assert env.default_linear_solver('pyomo') is None

    def test_in_process_never_picks_spral(self, monkeypatch):
        _, cy_probe = _have('mumps', 'spral')
        monkeypatch.setattr(env, 'cyipopt_linear_solver_available', cy_probe)
        assert env.default_linear_solver('cyipopt') == 'mumps'

    def test_an_inconclusive_in_process_probe_defers(self, monkeypatch):
        monkeypatch.setattr(env, 'cyipopt_linear_solver_available',
                            lambda name: None)
        assert env.default_linear_solver('cyipopt') is None


@pytest.mark.skipif(not available, reason='LCsolver import failed')
class TestCrashFallback:

    def test_spral_crash_falls_back_to_ma27_first(self, monkeypatch):
        exe_probe, _ = _have('ma27', 'mumps', 'spral')
        monkeypatch.setattr(env, 'linear_solver_available', exe_probe)
        assert env.crash_fallback_solver('spral') == 'ma27'

    def test_without_ma27_it_falls_back_to_mumps(self, monkeypatch):
        exe_probe, _ = _have('mumps', 'spral')
        monkeypatch.setattr(env, 'linear_solver_available', exe_probe)
        assert env.crash_fallback_solver('spral') == 'mumps'

    def test_never_retries_the_solver_that_crashed(self, monkeypatch):
        exe_probe, _ = _have('ma27')
        monkeypatch.setattr(env, 'linear_solver_available', exe_probe)
        assert env.crash_fallback_solver('ma27') is None
        exe_probe, _ = _have('mumps')
        monkeypatch.setattr(env, 'linear_solver_available', exe_probe)
        assert env.crash_fallback_solver('mumps') is None

    def test_nothing_else_available_means_none(self, monkeypatch):
        exe_probe, _ = _have('spral')
        monkeypatch.setattr(env, 'linear_solver_available', exe_probe)
        assert env.crash_fallback_solver('spral') is None


@pytest.mark.skipif(not available, reason='LCsolver import failed')
def test_the_report_names_spral_and_the_default():
    report = env.check_solvers(probe=False)
    assert 'spral' in report['ipopt'] and 'default' in report['ipopt']
    report['ipopt'].update(executable='/x/ipopt', ma27=False, mumps=True,
                           spral=True, default='spral')
    text = env._fmt(report)
    assert 'mumps, spral' in text and '(default: spral)' in text


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
