"""Sequential inner approximation (lcsolver.solvers.ipopt.sia).

The properties being tested are the ones that motivate the method: that a
problem whose constraints are all exact-or-conservative needs no globalization,
that every iterate stays feasible, and that termination is a genuine KKT
certificate on the original problem rather than "the objective stopped
changing".
"""
import numpy as np
import pytest

from lcsolver.solvers.ipopt.slcp import (Constraint, Posynomial, PosynomialRatio,
                                    Problem, Signomial)
from lcsolver.solvers.ipopt.sia import SIAOptions, classify, solve_sia

N = 2


def _mono(c, a):
    return Posynomial([(c, a)], N)


def _posy(terms):
    return Posynomial(terms, N)


def _objective():
    """min x + y"""
    return _posy([(1.0, [1.0, 0.0]), (1.0, [0.0, 1.0])])


def _bounds():
    return [Constraint(_mono(0.1, [1.0, 0.0])),
            Constraint(_mono(0.1, [0.0, 1.0])),
            Constraint(_mono(0.05, [-1.0, 0.0])),
            Constraint(_mono(0.05, [0.0, -1.0]))]


def _signomial_constraint():
    """3 <= x + y + x*y, written 3/(x+y+xy) <= 1."""
    return Constraint(PosynomialRatio(
        _mono(3.0, [0.0, 0.0]),
        _posy([(1.0, [1.0, 0.0]), (1.0, [0.0, 1.0]), (1.0, [1.0, 1.0])]), N))


def _problem_structured():
    """Optimum is x = y = 1, objective 2."""
    return Problem(N, _objective(),
                   _bounds() + [_signomial_constraint(),
                                Constraint(_mono(1.0, [-1.0, -1.0]))])


def _blackbox(x):
    """x*y >= 1 delivered only as a value and a gradient."""
    x = np.asarray(x, dtype=float)
    v = 1.0 / (x[0] * x[1])
    g = np.array([-1.0 / (x[0] ** 2 * x[1]), -1.0 / (x[0] * x[1] ** 2)])
    return v, g


def _problem_blackbox():
    return Problem(N, _objective(),
                   _bounds() + [_signomial_constraint(),
                                Constraint(Signomial(_blackbox, N))])


X0 = np.array([2.0, 0.5])


def test_solves_a_signomial_program():
    r = solve_sia(_problem_structured(), X0, SIAOptions(max_iterations=80))
    assert r.converged
    assert r.objective == pytest.approx(2.0, rel=1e-6)
    assert r.x[0] == pytest.approx(1.0, rel=1e-4)
    assert r.x[1] == pytest.approx(1.0, rel=1e-4)


def test_classification_drives_the_algorithm():
    p = _problem_structured()
    n_exact, n_cons, n_lin = classify(p)
    assert (n_exact, n_cons, n_lin) == (5, 1, 0)
    r = solve_sia(p, X0, SIAOptions(max_iterations=80))
    # No constraint had to be linearized, so no globalization was used.
    assert r.conservative is True


def test_a_black_box_constraint_is_detected_and_globalized():
    p = _problem_blackbox()
    assert classify(p) == (4, 1, 1)
    r = solve_sia(p, X0, SIAOptions(max_iterations=80))
    assert r.conservative is False       # the trust region was in play
    assert r.converged


def test_black_box_reproduces_the_structured_answer():
    """The same constraint, given only as value+gradient, must land in the
    same place -- that is the whole point of accepting black boxes."""
    a = solve_sia(_problem_structured(), X0, SIAOptions(max_iterations=80))
    b = solve_sia(_problem_blackbox(), X0, SIAOptions(max_iterations=80))
    assert b.objective == pytest.approx(a.objective, rel=1e-6)
    assert b.x[0] == pytest.approx(a.x[0], rel=1e-5)
    assert b.x[1] == pytest.approx(a.x[1], rel=1e-5)


def test_termination_is_a_kkt_certificate_not_a_stall_test():
    """All three residuals must be inside tolerance, on the TRUE problem."""
    opts = SIAOptions(max_iterations=80)
    r = solve_sia(_problem_structured(), X0, opts)
    assert r.converged
    assert r.max_violation <= opts.feasibility_tolerance
    assert r.stationarity <= opts.stationarity_tolerance
    assert r.complementarity <= opts.complementarity_tolerance
    assert "KKT" in r.status


def test_tightening_the_tolerance_is_respected():
    """A stall test would return the same point either way; a KKT test must
    keep working."""
    loose = solve_sia(_problem_structured(), X0,
                      SIAOptions(max_iterations=80,
                                 stationarity_tolerance=1e-3))
    tight = solve_sia(_problem_structured(), X0,
                      SIAOptions(max_iterations=200,
                                 stationarity_tolerance=1e-9))
    assert loose.converged and tight.converged
    assert tight.iterations > loose.iterations
    assert tight.stationarity < loose.stationarity


def test_every_iterate_is_feasible_once_the_slacks_close():
    """Conservatism means the sub-problem's optimum satisfies the TRUE
    constraints, so the trajectory cannot leave the feasible set."""
    p = _problem_structured()
    r = solve_sia(p, X0, SIAOptions(max_iterations=80))
    # Skip the first few while the penalty is still pulling it in.
    tail = r.history[max(1, len(r.history) // 2):]
    for x in tail:
        worst = max(c.body(x) for c in p.constraints)
        assert worst <= 1.0 + 1e-6, f"iterate {x} violates by {worst - 1:.2e}"


def test_objective_is_monotone_once_feasible():
    r = solve_sia(_problem_structured(), X0, SIAOptions(max_iterations=80))
    tail = r.objectives[len(r.objectives) // 2:]
    for a, b in zip(tail, tail[1:]):
        assert b <= a + 1e-9


def test_multipliers_come_back_and_are_nonnegative():
    r = solve_sia(_problem_structured(), X0, SIAOptions(max_iterations=80))
    assert r.multipliers is not None
    assert np.all(r.multipliers >= 0.0)
    assert np.any(r.multipliers > 0.0)      # something is active at the optimum


def test_rejects_a_nonpositive_start():
    with pytest.raises(ValueError, match="strictly positive"):
        solve_sia(_problem_structured(), np.array([1.0, 0.0]))


def test_unknown_option_is_refused():
    with pytest.raises(AttributeError, match="unknown SIA option"):
        SIAOptions(trust_radius_typo=1.0)


def test_cached_signomial_evaluates_once_per_point():
    """Value and gradient come from one call; revisits are free."""
    from lcsolver.solvers.ipopt.slcp import CachedSignomial
    calls = {"n": 0}

    def fn(x):
        calls["n"] += 1
        return _blackbox(x)

    s = CachedSignomial(fn, N)
    x = np.array([2.0, 0.5])
    s(x), s.grad(x), s.log_grad(x), s(x)
    assert calls["n"] == 1
    assert s.evaluations == 1
    s(np.array([1.0, 1.0]))
    assert s.evaluations == 2


def test_caching_cuts_black_box_calls_without_changing_the_answer():
    """The reason it exists: a black box costing hours must not be called
    four times per iteration when once will do."""
    from lcsolver.solvers.ipopt.slcp import CachedSignomial
    counts = {}

    def build(wrapper, tag):
        counts[tag] = 0

        def fn(x):
            counts[tag] += 1
            return _blackbox(x)

        body = wrapper(fn, N)
        return Problem(N, _objective(),
                       _bounds() + [_signomial_constraint(),
                                    Constraint(body)])

    plain = solve_sia(build(Signomial, "plain"), X0,
                      SIAOptions(max_iterations=80))
    cached = solve_sia(build(CachedSignomial, "cached"), X0,
                       SIAOptions(max_iterations=80))

    assert cached.objective == pytest.approx(plain.objective, rel=1e-9)
    assert cached.iterations == plain.iterations
    assert counts["cached"] < counts["plain"] / 3
    assert counts["cached"] <= cached.iterations + 2


def test_cache_is_exact_not_interpolating():
    """A nearby-but-different point must trigger a real evaluation."""
    from lcsolver.solvers.ipopt.slcp import CachedSignomial
    calls = {"n": 0}

    def fn(x):
        calls["n"] += 1
        return _blackbox(x)

    s = CachedSignomial(fn, N)
    s(np.array([1.0, 1.0]))
    s(np.array([1.0 + 1e-15, 1.0]))
    assert calls["n"] == 2
