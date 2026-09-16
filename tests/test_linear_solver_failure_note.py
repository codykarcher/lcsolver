"""A failed solve under a non-MA27 linear solver names the retry.

MA27 is the measured most-robust solver on this problem class (the D8
sentinel breaks MUMPS outright -- docs/linear_solvers.rst), so any failure
under another solver carries a note pointing back at it.  MA27's own
failures carry no note: there is nothing more robust to suggest.
"""
import pytest

try:
    import lcsolver
    from lcsolver import Formulation
    from lcsolver.environment import linear_solver_failure_note
    available = True
except Exception:                                    # pragma: no cover
    available = False


def _ipopt_here():
    try:
        from lcsolver.solvers.solver import _ipopt_available
        return bool(_ipopt_available())
    except Exception:
        return False


def _mumps_here():
    try:
        from lcsolver.environment import linear_solver_available
        return bool(linear_solver_available('mumps'))
    except Exception:
        return False


@pytest.mark.skipif(not available, reason='LCsolver import failed')
class TestNote:

    def test_ma27_carries_no_note(self):
        assert linear_solver_failure_note('ma27') == ''

    def test_unspecified_carries_no_note(self):
        assert linear_solver_failure_note(None) == ''
        assert linear_solver_failure_note('') == ''

    def test_other_solvers_name_the_retry(self):
        for name in ('mumps', 'spral', 'ma97'):
            note = linear_solver_failure_note(name)
            assert repr(name) in note
            assert 'MA27' in note

    def test_the_note_never_raises(self):
        assert isinstance(
            linear_solver_failure_note('mumps', '/no/such/ipopt'), str)


def _infeasible():
    f = Formulation()
    x = f.Variable('x', 1.0, '-', 'x', bounds=[0.1, 10.0])
    f.Objective(x)
    f.ConstraintList([x >= 5.0, x <= 0.2])
    return f


@pytest.mark.skipif(not available or not _ipopt_here() or not _mumps_here(),
                    reason='needs an IPOPT carrying MUMPS')
class TestThroughSolve:

    def test_mumps_failure_points_at_ma27(self):
        with pytest.raises(RuntimeError) as ctx:
            lcsolver.solve(_infeasible(), linear_solver='mumps')
        msg = str(ctx.value)
        assert "'mumps'" in msg
        assert 'MA27' in msg

    def test_ma27_failure_stays_clean(self):
        with pytest.raises(RuntimeError) as ctx:
            lcsolver.solve(_infeasible(), linear_solver='ma27')
        assert 'retry with' not in str(ctx.value)


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
