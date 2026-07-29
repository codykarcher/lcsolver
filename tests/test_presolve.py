"""Structural checks, and the bounds-as-bounds path through the detector.

Two things are under test here and they fail in opposite ways.

:mod:`edi.presolve` is diagnostic -- it never changes the problem, so the way
it goes wrong is by reporting nothing useful. The check that matters is that
it *finds* the defect it is looking for and stays quiet on a clean model.

``bounds_as_rows=False`` does change how the problem is carried, and there the
failure mode is severe and silent: bounds that go into
``structures['bounds']`` and are then read by nobody simply vanish, and the
solver happily returns the optimum of an unbounded relaxation. So the central
test uses an **active** bound -- one holding the optimum away from where the
objective wants to go -- because a bound that is slack at the solution proves
nothing when it is dropped.
"""
import numpy as np
import pytest

# The SIA solver is held back pending publication; without it these exercise
# nothing, so skip rather than error on a checkout that does not have it.
pytest.importorskip("edi.solvers.ipopt.sia")

from edi import Formulation
from edi.presolve import (
    InfeasibleProblem,
    cancellation_report,
    degeneracy_report,
    eliminate_monomial_equalities,
    fold_singleton_rows,
    presolve,
    presolve_report,
    propagate_bounds,
    reduce_columns,
    restore_columns,
)
from edi.solvers.ipopt.slcp_bridge import (
    build_problem,
    presolve_structures,
    solve_sia,
)
from edi.structure.structureDetector import (
    require_bounds_as_rows,
    structure_detector,
)
from edi.units.unitCorrector import unit_corrector


def _detect(f, bounds_as_rows=True):
    return structure_detector(unit_corrector(f), bounds_as_rows=bounds_as_rows)


def _active_bound_model():
    """min 1/x  s.t.  x*y >= 1,  with x declared in [0.1, 3].

    Minimising 1/x pushes x up, and nothing in the constraints stops it -- the
    only thing holding x at 3 is its declared upper bound. So the optimum is
    1/3 with the bound and 0 without it, and losing the bound is unmissable.
    """
    f = Formulation()
    x = f.Variable('x', 1.0, '', 'x', bounds=[0.1, 3.0])
    y = f.Variable('y', 1.0, '', 'y', bounds=[0.1, 10.0])
    f.Objective(1.0 / x)
    f.Constraint(x * y >= 1.0)
    return f


# ---------------------------------------------------------------------------
# bounds carried as bounds
# ---------------------------------------------------------------------------
def test_bounds_are_published_and_rows_drop():
    """Splitting bounds out removes rows and publishes the same numbers."""
    rows_st = _detect(_active_bound_model(), bounds_as_rows=True)
    split_st = _detect(_active_bound_model(), bounds_as_rows=False)

    assert rows_st['bounds'] is None            # unchanged for every old caller
    assert split_st['bounds'] is not None

    # Same variables, and the declared box comes through untouched.
    assert len(split_st['bounds']) == len(split_st['variables'])
    assert (0.1, 3.0) in split_st['bounds']
    assert (0.1, 10.0) in split_st['bounds']

    # Four bound rows (two per variable) are gone.
    assert (rows_st['info']['N_cons_total']
            - split_st['info']['N_cons_total']) == 4
    assert rows_st['info']['N_cons_bounds'] == 4
    assert split_st['info']['N_cons_bounds'] == 0


def test_active_bound_survives_the_split():
    """The optimum is the same either way -- the bound is not lost.

    This is the whole point. The bound is active, so if the split dropped it
    the objective would fall to 0 and read as a better answer.
    """
    a = solve_sia(_detect(_active_bound_model(), bounds_as_rows=True),
                  x0=np.array([1.0, 1.0]))
    b = solve_sia(_detect(_active_bound_model(), bounds_as_rows=False),
                  x0=np.array([1.0, 1.0]))

    assert a.objective == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert b.objective == pytest.approx(a.objective, rel=1e-8)


def test_problem_carries_bounds_only_when_split():
    assert build_problem(_detect(_active_bound_model(), True)).bounds is None
    assert build_problem(_detect(_active_bound_model(), False)).bounds is not None


def test_backends_that_ignore_bounds_refuse_them():
    """A backend reading only rows must fail loudly, not solve a relaxation."""
    split = _detect(_active_bound_model(), bounds_as_rows=False)
    with pytest.raises(ValueError, match='bounds_as_rows'):
        require_bounds_as_rows(split, 'solve_GP')
    require_bounds_as_rows(_detect(_active_bound_model(), True), 'solve_GP')


# ---------------------------------------------------------------------------
# structural checks
# ---------------------------------------------------------------------------
def test_unbounded_above_is_reported():
    """max-like variable with nothing holding it down.

    ``w`` appears only as ``w >= 1``, which is a floor. Nothing bounds it
    above, and the objective does not mention it. That is exactly gpkit's
    "w is not upper bounded".
    """
    f = Formulation()
    x = f.Variable('x', 1.0, '', 'x')
    w = f.Variable('w', 1.0, '', 'w')
    f.Objective(x)
    f.Constraint(x >= 2.0)
    f.Constraint(w >= 1.0)
    rep = presolve_report(_detect(f))

    assert 'w' in rep.unbounded_above
    assert 'w' not in rep.unbounded_below     # the floor bounds it below
    assert not rep.clean


def test_the_default_box_does_not_rescue_an_unbounded_variable():
    """1e-30..1e30 is not a bound, but a real box is.

    Every EDI variable carries a box, so counting it would make the check
    vacuous -- that is the mistake this test exists to prevent. The same model
    with a meaningful upper bound must come back clean, which is what
    distinguishes "ignore the value" from "ignore the default".
    """
    def model(hi):
        f = Formulation()
        x = f.Variable('x', 1.0, '', 'x', bounds=[1e-30, 1e30])
        w = f.Variable('w', 1.0, '', 'w', bounds=[1e-30, hi])
        f.Objective(x)
        f.Constraint(x * w >= 1.0)      # bounds both below, neither above
        return f

    vacuous = presolve_report(_detect(model(1e30)))
    assert 'w' in vacuous.unbounded_above
    assert 'w' not in vacuous.unbounded_below    # the constraint floors it

    real = presolve_report(_detect(model(50.0)))
    assert 'w' not in real.unbounded_above


def test_clean_model_reports_clean():
    f = Formulation()
    x = f.Variable('x', 1.0, '', 'x')
    y = f.Variable('y', 1.0, '', 'y')
    f.Objective(x + y)
    f.Constraint(x * y >= 1.0)
    f.Constraint(x >= 0.1)
    f.Constraint(y >= 0.1)
    rep = presolve_report(_detect(f))

    assert not rep.empty_columns
    assert rep.clean, str(rep)


def test_singleton_rows_are_counted_not_confused_with_real_ones():
    """`x >= 2` is a bound in disguise; `x*y >= 1` is a real constraint."""
    f = Formulation()
    x = f.Variable('x', 1.0, '', 'x')
    y = f.Variable('y', 1.0, '', 'y')
    f.Objective(x + y)
    f.Constraint(x * y >= 1.0)
    f.Constraint(x >= 2.0)
    f.Constraint(y >= 0.5)
    rep = presolve_report(_detect(f))

    assert len(rep.singleton_rows) == 2
    # each variable is in exactly one *real* constraint, so both are singletons
    assert set(rep.singleton_columns) == {'x', 'y'}


def test_report_is_printable():
    rep = presolve_report(_detect(_active_bound_model()))
    assert 'presolve:' in str(rep)


# ---------------------------------------------------------------------------
# folding singleton rows into bounds
# ---------------------------------------------------------------------------
def _singleton_row_model():
    """min 1/x  s.t.  x*y >= 1,  x <= 3  -- the cap written as a constraint.

    Same problem as `_active_bound_model`, except the binding limit is a
    hand-written row rather than a declared bound. Folding it must not lose
    it: the optimum is 1/3 with the row and 0 without.
    """
    f = Formulation()
    x = f.Variable('x', 1.0, '', 'x')
    y = f.Variable('y', 1.0, '', 'y')
    f.Objective(1.0 / x)
    f.Constraint(x * y >= 1.0)
    f.Constraint(x <= 3.0)
    f.Constraint(y <= 10.0)
    f.Constraint(x >= 0.1)
    return f


def test_folding_keeps_an_active_row():
    """The folded row is binding, so losing it would show up as a 0 objective."""
    st = _detect(_singleton_row_model(), bounds_as_rows=False)
    folded = fold_singleton_rows(st)

    before = solve_sia(st, x0=np.array([1.0, 1.0]))
    after = solve_sia(folded, x0=np.array([1.0, 1.0]))

    assert before.objective == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert after.objective == pytest.approx(before.objective, rel=1e-8)


def test_folding_removes_the_rows_and_tightens_the_bounds():
    st = _detect(_singleton_row_model(), bounds_as_rows=False)
    folded = fold_singleton_rows(st)

    assert folded['info']['N_cons_folded'] == 3       # x<=3, y<=10, x>=0.1
    assert folded['info']['N_cons_total'] == 1        # only x*y >= 1 survives
    assert st['info']['N_cons_total'] == 4            # input left alone

    names = [str(v) for v in folded['variables']]
    lo, hi = folded['bounds'][names.index('x')]
    assert hi == pytest.approx(3.0, rel=1e-9)
    assert lo == pytest.approx(0.1, rel=1e-9)


def test_folding_takes_the_tightest_of_several_rows():
    f = Formulation()
    x = f.Variable('x', 1.0, '', 'x')
    y = f.Variable('y', 1.0, '', 'y')
    f.Objective(1.0 / x)
    f.Constraint(x * y >= 1.0)
    f.Constraint(x <= 8.0)
    f.Constraint(x <= 2.0)        # tighter, and the one that must win
    f.Constraint(x <= 5.0)
    f.Constraint(y <= 10.0)       # keeps y from running off; not under test
    folded = fold_singleton_rows(_detect(f, bounds_as_rows=False))

    names = [str(v) for v in folded['variables']]
    assert folded['bounds'][names.index('x')][1] == pytest.approx(2.0, rel=1e-9)
    assert solve_sia(folded, x0=np.array([1.0, 1.0])).objective == \
        pytest.approx(0.5, rel=1e-6)


def test_folding_needs_split_bounds():
    with pytest.raises(ValueError, match='bounds_as_rows'):
        fold_singleton_rows(_detect(_singleton_row_model(), bounds_as_rows=True))


# ---------------------------------------------------------------------------
# removing disconnected and fixed columns
# ---------------------------------------------------------------------------
def _disconnected_model():
    """min x + 1/x  s.t.  x >= 1,  y >= 4.

    `y` appears in no real constraint and in no objective term. Nothing
    determines it and nothing depends on it, so it can be fixed at 4 and
    dropped without touching the answer.
    """
    f = Formulation()
    x = f.Variable('x', 2.0, '', 'x', bounds=[0.1, 10.0])
    y = f.Variable('y', 5.0, '', 'y', bounds=[1e-30, 1e30])
    f.Objective(x + 1.0 / x)
    f.Constraint(x >= 1.0)
    f.Constraint(y >= 4.0)
    return f


def _reduced(f):
    st = fold_singleton_rows(_detect(f, bounds_as_rows=False))
    return st, reduce_columns(st)


def test_disconnected_variable_is_identified_and_removed():
    st, (small, removed) = _reduced(_disconnected_model())

    assert [(nm, why) for _j, nm, _v, why in removed] == [('y', 'disconnected')]
    assert removed[0].value == pytest.approx(4.0, rel=1e-9)  # tightest bound
    assert len(small['variables']) == len(st['variables']) - 1
    assert 'y' not in [str(v) for v in small['variables']]


def test_removing_it_does_not_change_the_answer():
    st, (small, removed) = _reduced(_disconnected_model())

    full = solve_sia(st, x0=np.array([2.0, 5.0]))
    cut = solve_sia(small, x0=np.array([2.0]))

    assert cut.objective == pytest.approx(full.objective, rel=1e-8)
    assert cut.objective == pytest.approx(2.0, rel=1e-6)   # min of x + 1/x


def test_removed_values_come_back_in_the_solution():
    st, (small, removed) = _reduced(_disconnected_model())
    cut = solve_sia(small, x0=np.array([2.0]))

    x = restore_columns(removed, cut.x, n_original=len(st['variables']))
    names = [str(v) for v in st['variables']]
    assert len(x) == len(names)
    assert x[names.index('y')] == pytest.approx(4.0, rel=1e-9)
    # x + 1/x is flat at its minimum, so the location is recovered far less
    # precisely than the value -- the objective is right to 8 figures while x
    # is only good to about 1e-4. That is curvature, not the reduction.
    assert x[names.index('x')] == pytest.approx(1.0, rel=1e-3)


def test_a_fixed_variable_is_folded_into_the_coefficients():
    """Equal bounds make it a constant; the answer must not move."""
    def model(as_variable):
        f = Formulation()
        x = f.Variable('x', 2.0, '', 'x', bounds=[0.1, 10.0])
        f.Objective(x)
        if as_variable:
            k = f.Variable('k', 3.0, '', 'k', bounds=[3.0, 3.0])
            f.Constraint(x >= k * 2.0)
        else:
            f.Constraint(x >= 6.0)
        return f

    st = fold_singleton_rows(_detect(model(True), bounds_as_rows=False))
    small, removed = reduce_columns(st)

    assert [(nm, why) for _j, nm, _v, why in removed] == [('k', 'fixed')]
    assert solve_sia(small, x0=np.array([2.0])).objective == \
        pytest.approx(6.0, rel=1e-6)


def test_a_variable_in_a_real_constraint_is_never_removed():
    """Even when it is slack, and even when the objective ignores it.

    This is the boundary: `y` here is degenerate at the optimum, but it sits
    in a genuine constraint that would bind at a different design point.
    Removing it would delete that constraint too.
    """
    f = Formulation()
    x = f.Variable('x', 2.0, '', 'x', bounds=[0.1, 10.0])
    y = f.Variable('y', 5.0, '', 'y', bounds=[0.1, 10.0])
    f.Objective(x)
    f.Constraint(x >= 1.0)
    f.Constraint(x * y >= 0.2)          # slack at the optimum, but real
    st = fold_singleton_rows(_detect(f, bounds_as_rows=False))
    _small, removed = reduce_columns(st)

    assert [nm for _j, nm, _v, _why in removed] == []


def test_presolve_is_on_by_default_and_is_invisible_to_the_caller():
    """The default path reduces, solves, and hands back the original layout.

    Both the reduced and unreduced solves must agree, and `result.x` must
    still be indexed by the original variable ordering -- a caller should not
    have to know whether anything was removed.
    """
    st = _detect(_disconnected_model())
    names = [str(v) for v in st['variables']]

    on = solve_sia(st, x0=np.array([2.0, 5.0]))
    off = solve_sia(_detect(_disconnected_model()), x0=np.array([2.0, 5.0]),
                    presolve=False)

    assert on.objective == pytest.approx(off.objective, rel=1e-6)
    assert len(on.x) == len(names)
    assert [nm for _j, nm, _v, _why in on.removed] == ['y']
    assert on.x[names.index('y')] == pytest.approx(4.0, rel=1e-9)
    assert off.removed == []


def test_presolve_works_when_bounds_are_still_rows():
    """The default detector output has bounds as rows; presolve must cope."""
    st = _detect(_disconnected_model(), bounds_as_rows=True)
    assert st['bounds'] is None
    reduced, log = presolve_structures(st)

    assert reduced['bounds'] is not None
    assert [r.name for r in log.removed_variables] == ['y']


def test_reduction_needs_split_bounds():
    with pytest.raises(ValueError, match='bounds_as_rows'):
        reduce_columns(_detect(_disconnected_model(), bounds_as_rows=True))


# ---------------------------------------------------------------------------
# signomial cancellation
# ---------------------------------------------------------------------------
def test_cancellation_finds_the_term_that_does_nothing():
    """A subtraction large enough to satisfy the constraint by itself.

    This is the pi-tail failure in miniature: `m >= a - c` with `c` far bigger
    than `a` holds for any `m`, so `m` is disconnected. EDI writes it as
    `a / (m + c) <= 1`, and the check is that `m`'s share of that denominator
    is negligible.
    """
    f = Formulation()
    m = f.Variable('m', 1.0, '', 'm', bounds=[1e-30, 1e30])
    a = f.Variable('a', 1.0, '', 'a', bounds=[0.5, 2.0])
    f.Objective(m)
    f.Constraint(a >= 1.0)
    f.Constraint(m >= a - 1e6)          # 1e6 swamps a; m does nothing
    st = _detect(f)
    res = solve_sia(st, x0=np.array([1.0, 1.0]))

    names = [str(v) for v in st['variables']]
    rep = cancellation_report(st, res.x, names=names)
    inert = {v for _, _, _, vs in rep for v in vs}
    assert 'm' in inert, rep


def test_cancellation_quiet_when_the_subtraction_is_small():
    """Same shape, but the subtracted term no longer dominates."""
    f = Formulation()
    m = f.Variable('m', 1.0, '', 'm', bounds=[1e-30, 1e30])
    a = f.Variable('a', 1.0, '', 'a', bounds=[0.5, 2.0])
    f.Objective(m)
    f.Constraint(a >= 1.0)
    f.Constraint(m >= a - 0.1)          # m still has to supply most of it
    st = _detect(f)
    res = solve_sia(st, x0=np.array([1.0, 1.0]))

    names = [str(v) for v in st['variables']]
    inert = {v for _, _, _, vs in cancellation_report(st, res.x, names=names)
             for v in vs}
    assert 'm' not in inert


def test_cancellation_ignores_plain_posynomials():
    """A small term in a posynomial is ordinary, not a defect."""
    f = Formulation()
    x = f.Variable('x', 1.0, '', 'x', bounds=[0.1, 10.0])
    y = f.Variable('y', 1.0, '', 'y', bounds=[0.1, 10.0])
    f.Objective(x)
    f.Constraint(x >= y + 1e-12)        # tiny term, but no subtraction
    f.Constraint(y >= 1.0)
    st = _detect(f)
    res = solve_sia(st, x0=np.array([1.0, 1.0]))

    assert cancellation_report(st, res.x,
                               names=[str(v) for v in st['variables']]) == []


# ---------------------------------------------------------------------------
# output-only variables
# ---------------------------------------------------------------------------
def _output_model():
    """min x + 1/x  s.t.  x >= 0.5,  A == 3*x,  T >= 2*A.

    `A` and `T` are reporting quantities: computed from the design, read by
    nothing. `T` sits behind `A`, so peeling `A` is what exposes it -- the
    second round of the scan.
    """
    f = Formulation()
    x = f.Variable('x', 2.0, '', 'x', bounds=[0.1, 10.0])
    A = f.Variable('A', 1.0, '', 'A', bounds=[1e-30, 1e30])
    T = f.Variable('T', 1.0, '', 'T', bounds=[1e-30, 1e30])
    f.Objective(x + 1.0 / x)
    f.Constraint(x >= 0.5)
    f.Constraint(A == 3.0 * x)
    f.Constraint(T >= 2.0 * A)
    return f


def test_output_only_variables_are_found_behind_each_other():
    st = fold_singleton_rows(_detect(_output_model(), bounds_as_rows=False))
    _small, removed = reduce_columns(st)
    out = {r.name for r in removed if r.reason == 'output'}
    assert out == {'A', 'T'}


def test_output_values_are_post_computed_correctly():
    """The reported value must be the one the optimizer would have produced."""
    st = _detect(_output_model())
    names = [str(v) for v in st['variables']]

    on = solve_sia(st)                                    # default: eliminated
    off = solve_sia(_detect(_output_model()), presolve=False)

    assert on.objective == pytest.approx(off.objective, rel=1e-6)
    assert len(on.x) == len(names)                        # full vector back

    xv = on.x[names.index('x')]
    assert on.x[names.index('A')] == pytest.approx(3.0 * xv, rel=1e-6)
    assert on.x[names.index('T')] == pytest.approx(6.0 * xv, rel=1e-6)
    # and it agrees with solving them inside the optimization
    assert on.x[names.index('A')] == pytest.approx(off.x[names.index('A')],
                                                   rel=1e-4)


def test_eliminating_outputs_shrinks_the_solved_problem():
    st = fold_singleton_rows(_detect(_output_model(), bounds_as_rows=False))
    small, removed = reduce_columns(st)
    kept, _ = reduce_columns(st, eliminate_outputs=False)

    assert len(small['variables']) == len(st['variables']) - 2
    assert small['info']['N_cons_total'] == st['info']['N_cons_total'] - 2
    assert small['info']['N_vars_output'] == 2
    # with the mode off, nothing is removed and the constraints stay
    assert len(kept['variables']) == len(st['variables'])


def test_a_bounded_output_is_not_eliminated():
    """A real bound turns the defining constraint into a genuine restriction.

    With `A` capped at 4, `A >= 3*x` forces `x <= 4/3` -- a restriction on a
    variable that IS in the objective. Eliminating `A` would silently drop it.
    """
    def model(hi):
        f = Formulation()
        x = f.Variable('x', 1.0, '', 'x', bounds=[0.1, 10.0])
        A = f.Variable('A', 1.0, '', 'A', bounds=[1e-30, hi])
        f.Objective(1.0 / x)             # pushes x UP, so the cap binds
        f.Constraint(A >= 3.0 * x)
        return f

    st = fold_singleton_rows(_detect(model(4.0), bounds_as_rows=False))
    _small, removed = reduce_columns(st)
    assert [r.name for r in removed if r.reason == 'output'] == []

    # ... whereas with a vacuous cap it is a pure output again
    st2 = fold_singleton_rows(_detect(model(1e30), bounds_as_rows=False))
    _s2, removed2 = reduce_columns(st2)
    assert [r.name for r in removed2 if r.reason == 'output'] == ['A']


def test_a_variable_the_objective_uses_is_never_an_output():
    f = Formulation()
    x = f.Variable('x', 1.0, '', 'x', bounds=[0.1, 10.0])
    A = f.Variable('A', 1.0, '', 'A', bounds=[1e-30, 1e30])
    f.Objective(A)                       # A is the thing being minimised
    f.Constraint(A >= 3.0 * x)
    f.Constraint(x >= 2.0)
    st = fold_singleton_rows(_detect(f, bounds_as_rows=False))
    _small, removed = reduce_columns(st)
    assert [r.name for r in removed if r.reason == 'output'] == []


# ---------------------------------------------------------------------------
# sub-problem caching
# ---------------------------------------------------------------------------
def test_cached_and_rebuilt_subproblems_agree():
    """Caching is a performance change and must not be a numerical one.

    The objective and every constraint value must match. Individual variables
    are checked only where the problem determines them -- a flat direction can
    land anywhere without either answer being wrong, which is exactly what a
    degenerate variable is.
    """
    from edi.solvers.ipopt.sia import SIAOptions

    def model():
        f = Formulation()
        x = f.Variable('x', 2.0, '', 'x', bounds=[0.1, 10.0])
        y = f.Variable('y', 2.0, '', 'y', bounds=[0.1, 10.0])
        z = f.Variable('z', 2.0, '', 'z', bounds=[0.1, 10.0])
        f.Objective(x + y)
        f.Constraint(x * y >= 4.0)
        f.Constraint(z * x >= 0.5)
        f.Constraint(x + z >= 1.0)
        return f

    out = {}
    for flag in (False, True):
        st = _detect(model())
        out[flag] = solve_sia(st, options=SIAOptions(cache_subproblem=flag))

    assert out[True].objective == pytest.approx(out[False].objective, rel=1e-8)
    assert out[True].max_violation == pytest.approx(
        out[False].max_violation, abs=1e-6)

    st = _detect(model())
    names = [str(v) for v in st['variables']]
    problem = build_problem(st)
    free = {nm for nm, _v in degeneracy_report(problem, out[False].x,
                                               names=names)}
    for j, nm in enumerate(names):
        if nm not in free:
            assert out[True].x[j] == pytest.approx(out[False].x[j], rel=1e-5), \
                f"{nm} moved and is not degenerate"


def test_cache_builds_one_model_per_phase():
    """The point of the cache: build once, then only re-point."""
    from edi.solvers.ipopt.sia import SIAOptions, SubproblemCache

    st = _detect(_singleton_row_model())
    problem = build_problem(st)
    cache = SubproblemCache(problem, SIAOptions())
    assert cache.usable

    cache.get(False, False)
    cache.get(False, False)
    cache.get(False, False)
    assert cache.builds == 1                 # reused, not rebuilt
    cache.get(True, False)                   # a different phase does build
    assert cache.builds == 2


def test_a_black_box_body_is_not_cacheable():
    """No conservative model exists for it, so it must be re-linearized."""
    from edi.solvers.ipopt.sia import SIAOptions, SubproblemCache
    from edi.solvers.ipopt.slcp import Constraint, Posynomial, Signomial

    n = 2
    obj = Posynomial([(1.0, [1.0, 0.0])], n)
    box = Signomial(lambda x: (float(x[0]), np.array([1.0, 0.0])), n)
    problem = build_problem(_detect(_singleton_row_model()))
    problem.constraints.append(Constraint(box, '<='))
    assert not SubproblemCache(problem, SIAOptions()).usable


# ---------------------------------------------------------------------------
# post-solve degeneracy
# ---------------------------------------------------------------------------
def test_degeneracy_finds_a_variable_the_optimum_does_not_determine():
    """`u` is real, constrained, and completely free at the optimum.

    It sits in a constraint that is slack there, so no structural check sees
    anything wrong -- which is why the post-solve test has to exist.
    """
    f = Formulation()
    x = f.Variable('x', 1.0, '', 'x', bounds=[0.1, 10.0])
    u = f.Variable('u', 1.0, '', 'u', bounds=[0.5, 2.0])
    f.Objective(x)
    f.Constraint(x >= 2.0)
    f.Constraint(x * u >= 0.2)          # slack at x=2, u anywhere in [0.5, 2]
    st = _detect(f)
    problem = build_problem(st)
    res = solve_sia(st, x0=np.array([1.0, 1.0]))

    names = [str(v) for v in st['variables']]
    free = dict(degeneracy_report(problem, res.x, names=names))
    assert 'u' in free
    assert 'x' not in free


# ---------------------------------------------------------------------------
# signomial equalities: split pair vs single condensed equality
# ---------------------------------------------------------------------------
def _sig_equality_model():
    """min z  s.t.  z == x - y,  x >= 4,  1 <= y <= 2.

    Normalizes to `(z + y)/x == 1`: a multi-term POSYNOMIAL equality, which the
    bridge splits into `p <= 1` plus a condensed `1/p <= 1`. That pair is
    dual-degenerate exactly as the ratio pair is -- on SPaircraft the two
    halves came back as 910.8 and 910.9. Minimising z drives x down and y up,
    so the optimum is x=4, y=2, z=2.
    """
    f = Formulation()
    x = f.Variable('x', 5.0, '', 'x', bounds=[0.1, 100.0])
    y = f.Variable('y', 1.5, '', 'y', bounds=[1.0, 2.0])
    z = f.Variable('z', 3.0, '', 'z', bounds=[0.1, 100.0])
    f.Objective(z)
    f.Constraint(z == x - y)
    f.Constraint(x >= 4.0)
    return f


def test_signomial_equality_is_a_ratio_with_an_equality_operator():
    """Guards the premise: this really is the case under test."""
    from edi.solvers.ipopt.slcp import CondensedEquality, PosynomialRatio

    split = build_problem(_detect(_sig_equality_model()), split_equalities=True)
    single = build_problem(_detect(_sig_equality_model()),
                           split_equalities=False)

    assert any(isinstance(c.body, PosynomialRatio) for c in split.constraints)
    assert any(isinstance(c.body, CondensedEquality)
               for c in single.constraints)
    assert not any(isinstance(c.body, PosynomialRatio)
                   for c in single.constraints)
    # the pair becomes one constraint
    assert len(single.constraints) == len(split.constraints) - 1


def test_the_single_equality_reaches_the_optimum_and_the_pair_does_not():
    """The bug and its fix, on three variables.

    Split into a pair, the step is confined to the null space of the summed
    log-Hessians and the run stalls 31% high with `y` nowhere near its bound.
    As one condensed equality the same problem solves exactly.
    """
    a = solve_sia(_detect(_sig_equality_model()), split_equalities=True)
    b = solve_sia(_detect(_sig_equality_model()), split_equalities=False)

    assert b.objective == pytest.approx(2.0, rel=1e-6)
    assert a.objective > 2.5                      # the pair does not get there
    assert b.objective < a.objective - 0.5


def test_single_equality_keeps_the_multipliers_well_conditioned():
    """The point of the change: no huge cancelling multiplier pair.

    Split into two condensed inequalities the pair is dual-degenerate, so the
    solver may return arbitrarily large multipliers whose DIFFERENCE is the
    only meaningful quantity. As one equality there is a single signed
    multiplier and nothing to cancel.
    """
    a = solve_sia(_detect(_sig_equality_model()), split_equalities=True)
    b = solve_sia(_detect(_sig_equality_model()), split_equalities=False)

    # measured: ~3611 for the pair against ~2 for the single equality
    assert np.max(np.abs(np.asarray(a.multipliers))) > 1e3
    assert np.max(np.abs(np.asarray(b.multipliers))) < 1e2


def test_condensed_equality_reports_the_true_gradient():
    """The KKT test must use the TRUE gradient, not the condensed one."""
    from edi.solvers.ipopt.slcp import CondensedEquality, Posynomial

    n = 2
    p = Posynomial([(1.0, [1.0, 0.0]), (1.0, [0.0, 1.0])], n)   # x + y
    q = Posynomial([(2.0, [1.0, 0.0])], n)                      # 2x
    ce = CondensedEquality(p, q, n)
    x = np.array([3.0, 5.0])

    assert ce(x) == pytest.approx((3.0 + 5.0) / (2 * 3.0))
    expected = p.log_grad(x) - q.log_grad(x)
    assert np.allclose(ce.log_grad(x), expected)

    # the condensed monomial matches p/q in value and gradient AT x
    c, a = ce.condensed(x)
    val = c * np.prod(x ** a)
    assert val == pytest.approx(ce(x), rel=1e-10)
    assert np.allclose(a, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# bounds that meet, and bounds that cross
# ---------------------------------------------------------------------------
def _two_sided(lo, hi):
    """min y  s.t.  x >= lo,  x <= hi,  x*y >= 1."""
    f = Formulation()
    x = f.Variable('x', 1.0, '', 'x')
    y = f.Variable('y', 1.0, '', 'y', bounds=[0.1, 10.0])
    f.Objective(y)
    f.Constraint(x >= lo)
    f.Constraint(x <= hi)
    f.Constraint(x * y >= 1.0)
    return f


def test_bounds_that_meet_become_a_fixed_variable():
    """`x >= 1` with `x <= 1` is an equality however it was written."""
    st = fold_singleton_rows(_detect(_two_sided(1.0, 1.0),
                                     bounds_as_rows=False))
    names = [str(v) for v in st['variables']]
    assert st['bounds'][names.index('x')] == (1.0, 1.0)

    _small, removed = reduce_columns(st)
    assert [(r.name, r.reason, r.value) for r in removed] == [('x', 'fixed', 1.0)]


def test_an_ordinary_range_is_left_alone():
    st = fold_singleton_rows(_detect(_two_sided(1.0, 3.0),
                                     bounds_as_rows=False))
    _small, removed = reduce_columns(st)
    assert [r.name for r in removed if r.reason == 'fixed'] == []


def test_crossed_bounds_are_reported_as_infeasible():
    """`x >= 2` with `x <= 1` has no solution, and must say so.

    Folding them naively gives "x is fixed at 2", and the solver then answers a
    different question than the one that was asked -- the same class of silent
    substitution as relaxing an equality.
    """
    with pytest.raises(InfeasibleProblem, match='no feasible point'):
        fold_singleton_rows(_detect(_two_sided(2.0, 1.0),
                                    bounds_as_rows=False))


def test_crossed_declared_bounds_are_caught_too():
    """Not just folded rows -- bounds that arrive already crossed."""
    st = _detect(_two_sided(1.0, 3.0), bounds_as_rows=False)
    names = [str(v) for v in st['variables']]
    st = dict(st)
    st['bounds'] = list(st['bounds'])
    st['bounds'][names.index('x')] = (5.0, 2.0)      # crossed
    with pytest.raises(InfeasibleProblem, match='no feasible point'):
        reduce_columns(st)


# ---------------------------------------------------------------------------
# bound propagation
# ---------------------------------------------------------------------------
def test_propagation_derives_a_bound_in_log_space():
    """`x*y >= 1` with `y <= 10` implies `x >= 0.1`."""
    f = Formulation()
    x = f.Variable('x', 1.0, '', 'x', bounds=[1e-30, 1e30])
    y = f.Variable('y', 1.0, '', 'y', bounds=[1e-30, 10.0])
    f.Objective(x)
    f.Constraint(x * y >= 1.0)
    st = fold_singleton_rows(_detect(f, bounds_as_rows=False))
    out, n = propagate_bounds(st)

    names = [str(v) for v in out['variables']]
    lo, _hi = out['bounds'][names.index('x')]
    assert n >= 1
    assert lo == pytest.approx(0.1, rel=1e-6)


def test_an_equality_uses_both_endpoints_not_just_the_minimum():
    """The regression for the bug that called SPaircraft infeasible.

    An equality bounds `v_k` above using the MINIMUM of the other terms and
    below using their MAXIMUM -- different sums. Reusing the minimum for both
    manufactures contradictions between unrelated equalities and "proves" a
    perfectly feasible model infeasible.
    """
    f = Formulation()
    a = f.Variable('a', 1.0, '', 'a', bounds=[1e-30, 1e30])
    b = f.Variable('b', 1.0, '', 'b', bounds=[1.0, 100.0])
    c = f.Variable('c', 1.0, '', 'c', bounds=[1.0, 100.0])
    f.Objective(a)
    f.Constraint(a == 2.0 * b)          # monomial equalities, sharing `a`
    f.Constraint(a == 3.0 * c)
    st = fold_singleton_rows(_detect(f, bounds_as_rows=False))

    out, _n = propagate_bounds(st)      # must not raise
    names = [str(v) for v in out['variables']]
    lo, hi = out['bounds'][names.index('a')]
    # a = 2b in [2,200] and a = 3c in [3,300}  ->  a in [3,200]
    assert lo == pytest.approx(3.0, rel=1e-6)
    assert hi == pytest.approx(200.0, rel=1e-6)


def test_propagation_works_on_a_linear_program():
    """EDI solves LPs too, and the same arithmetic applies in natural space."""
    f = Formulation()
    x = f.Variable('x', 1.0, '', 'x', bounds=[-100.0, 100.0])
    y = f.Variable('y', 1.0, '', 'y', bounds=[-100.0, 5.0])
    f.Objective(x + y)
    f.Constraint(x - y >= 2.0)
    st = _detect(f, bounds_as_rows=False)
    assert st['Linear_Program'][0]

    out, n = propagate_bounds(st)
    names = [str(v) for v in out['variables']]
    lo, _hi = out['bounds'][names.index('x')]
    assert n >= 1
    assert lo == pytest.approx(-98.0, rel=1e-6)   # -x + y <= -2 with y >= -100


def test_propagation_proves_an_lp_infeasible():
    f = Formulation()
    x = f.Variable('x', 1.0, '', 'x', bounds=[-100.0, 100.0])
    f.Objective(x)
    f.Constraint(x >= 10.0)
    f.Constraint(x <= 1.0)
    with pytest.raises(InfeasibleProblem, match='no feasible point'):
        propagate_bounds(_detect(f, bounds_as_rows=False))


def test_propagation_leaves_a_genuinely_unbounded_variable_alone():
    """It must not invent a bound where the model provides none."""
    f = Formulation()
    x = f.Variable('x', 1.0, '', 'x', bounds=[1e-30, 1e30])
    w = f.Variable('w', 1.0, '', 'w', bounds=[1e-30, 1e30])
    f.Objective(x)
    f.Constraint(x >= 2.0)
    f.Constraint(x * w >= 1.0)          # bounds w below, nothing above
    st = fold_singleton_rows(_detect(f, bounds_as_rows=False))
    out, _n = propagate_bounds(st)
    assert 'w' in presolve_report(out).unbounded_above


# ---------------------------------------------------------------------------
# monomial equality elimination
# ---------------------------------------------------------------------------
def test_a_monomial_equality_substitutes_a_variable_out():
    """`z == 3x` determines z, so z need not be solved for."""
    f = Formulation()
    x = f.Variable('x', 2.0, '', 'x', bounds=[0.1, 100.0])
    z = f.Variable('z', 1.0, '', 'z', bounds=[1e-30, 1e30])
    f.Objective(x)
    f.Constraint(z == 3.0 * x)
    f.Constraint(x * z >= 12.0)            # 3x^2 >= 12  ->  x >= 2
    st = fold_singleton_rows(_detect(f, bounds_as_rows=False))
    small, removed = eliminate_monomial_equalities(st)

    assert [(r.name, r.reason) for r in removed] == [('z', 'substituted')]
    assert len(small['variables']) == len(st['variables']) - 1
    assert small['info']['N_cons_total'] == st['info']['N_cons_total'] - 1

    res = solve_sia(small)
    assert res.objective == pytest.approx(2.0, rel=1e-6)
    names = [str(v) for v in st['variables']]
    back = restore_columns(removed, res.x, n_original=len(names))
    assert back[names.index('x')] == pytest.approx(2.0, rel=1e-5)
    assert back[names.index('z')] == pytest.approx(6.0, rel=1e-5)


def test_chained_eliminations_recover_in_the_right_order():
    """A pivot's formula may name a variable eliminated in a LATER round.

    `w == 2z` and `z == 3x`: whichever is eliminated first, its stored formula
    can reference the other, so recovery has to run backwards. Forwards it
    silently returns values off by orders of magnitude while the reduced
    problem stays exactly right -- measured on SPaircraft at 4.3e+03 relative
    error with the objective still correct to 12 figures.
    """
    f = Formulation()
    x = f.Variable('x', 2.0, '', 'x', bounds=[0.1, 100.0])
    z = f.Variable('z', 1.0, '', 'z', bounds=[1e-30, 1e30])
    w = f.Variable('w', 1.0, '', 'w', bounds=[1e-30, 1e30])
    f.Objective(x)
    f.Constraint(z == 3.0 * x)
    f.Constraint(w == 2.0 * z)
    f.Constraint(x >= 2.0)
    st = fold_singleton_rows(_detect(f, bounds_as_rows=False))
    small, removed = eliminate_monomial_equalities(st)

    assert {r.name for r in removed} == {'z', 'w'}
    res = solve_sia(small)
    names = [str(v) for v in st['variables']]
    back = restore_columns(removed, res.x, n_original=len(names))

    xv = back[names.index('x')]
    assert xv == pytest.approx(2.0, rel=1e-5)
    assert back[names.index('z')] == pytest.approx(3.0 * xv, rel=1e-5)
    assert back[names.index('w')] == pytest.approx(6.0 * xv, rel=1e-5)


def test_a_variable_with_a_real_bound_is_not_substituted_out():
    """Its bound would become a constraint on the survivors, undoing the gain."""
    f = Formulation()
    x = f.Variable('x', 2.0, '', 'x', bounds=[0.1, 100.0])
    z = f.Variable('z', 1.0, '', 'z', bounds=[1.0, 50.0])   # a real box
    f.Objective(x)
    f.Constraint(z == 3.0 * x)
    f.Constraint(x >= 2.0)
    st = fold_singleton_rows(_detect(f, bounds_as_rows=False))
    _small, removed = eliminate_monomial_equalities(st)
    assert [r.name for r in removed if r.reason == 'substituted'] == []


# ---------------------------------------------------------------------------
# sensitivities across a structural reduction
# ---------------------------------------------------------------------------
def _constant_model(k=3.0):
    """min x  s.t.  z == K*x,  x*z >= 12.   So x* = sqrt(12/K).

    `K` appears ONLY in the monomial equality that elimination consumes, which
    is the hardest case for sensitivity recovery: after the reduction that
    constraint is gone and `K` survives only inside the coefficients it was
    folded into.
    """
    f = Formulation()
    x = f.Variable('x', 2.0, '', 'x', bounds=[0.1, 100.0])
    z = f.Variable('z', 1.0, '', 'z', bounds=[1e-30, 1e30])
    K = f.Constant('K', k, '', 'fitting constant')
    f.Objective(x)
    f.Constraint(z == K * x)
    f.Constraint(x * z >= 12.0)
    return f


def test_sensitivities_survive_monomial_elimination():
    """d log f* / d log K is -1/2 analytically, with or without the reduction.

    This works because presolve transforms the detected *structure*, never the
    model, and `sensitivities` recovers duals from the primal solution on the
    original model. So long as the full primal vector is restored, the
    reduction is invisible to it.
    """
    from edi.solvers.sensitivity import sensitivities
    from edi.solvers.writeback import write_solution

    fm = _constant_model()
    st = _detect(fm)
    res = solve_sia(st)
    write_solution(st, {'x': list(res.x)}, model=fm)
    plain = sensitivities(fm)['sensitivities']['K']

    fm2 = _constant_model()
    st2 = fold_singleton_rows(_detect(fm2, bounds_as_rows=False))
    small, removed = eliminate_monomial_equalities(st2)
    assert removed, "the equality should have been eliminated"
    res2 = solve_sia(small, presolve=False)
    names = [str(v) for v in st2['variables']]
    x_full = restore_columns(removed, res2.x, n_original=len(names))
    write_solution(st2, {'x': list(x_full)}, model=fm2)
    reduced = sensitivities(fm2)['sensitivities']['K']

    assert plain == pytest.approx(-0.5, abs=1e-4)
    assert reduced == pytest.approx(plain, abs=1e-6)


# ---------------------------------------------------------------------------
# the composed pipeline and its log
# ---------------------------------------------------------------------------
def _pipeline_model():
    """A model with something for each pass to find."""
    f = Formulation()
    x = f.Variable('x', 2.0, '', 'x', bounds=[0.1, 100.0])
    z = f.Variable('z', 1.0, '', 'z', bounds=[1e-30, 1e30])   # eliminable
    k = f.Variable('k', 3.0, '', 'k', bounds=[3.0, 3.0])      # fixed
    q = f.Variable('q', 1.0, '', 'q', bounds=[1e-30, 1e30])   # disconnected
    f.Objective(x)
    f.Constraint(z == 3.0 * x)
    f.Constraint(x * z >= 12.0)
    f.Constraint(x >= k * 0.5)
    f.Constraint(q >= 1.0)
    return f


def test_the_pipeline_composes_and_round_trips():
    """Each pass renumbers, so restore must unwind them in reverse."""
    st = _detect(_pipeline_model(), bounds_as_rows=False)
    names = [str(v) for v in st['variables']]
    small, log = presolve(st)

    assert len(small['variables']) < len(names)
    removed = log.removed_variables
    assert {r.name for r in removed} >= {'z', 'k', 'q'}

    res = solve_sia(small, presolve=False)
    full = log.restore(res.x)
    assert len(full) == len(names)
    assert res.objective == pytest.approx(2.0, rel=1e-5)

    xv = full[names.index('x')]
    assert xv == pytest.approx(2.0, rel=1e-4)
    assert full[names.index('z')] == pytest.approx(3.0 * xv, rel=1e-4)
    assert full[names.index('k')] == pytest.approx(3.0, rel=1e-9)


def test_the_log_reports_what_it_did():
    st = _detect(_pipeline_model(), bounds_as_rows=False)
    _small, log = presolve(st)
    text = str(log)

    assert 'presolve:' in text
    assert 'removed' in text
    assert 'variables removed in total' in text
    # the per-variable account names names
    assert 'z' in log.detail()


def test_the_log_is_quiet_when_there_is_nothing_to_do():
    f = Formulation()
    x = f.Variable('x', 1.0, '', 'x', bounds=[0.1, 10.0])
    y = f.Variable('y', 1.0, '', 'y', bounds=[0.1, 10.0])
    f.Objective(x + y)
    f.Constraint(x * y >= 1.0)
    _small, log = presolve(_detect(f, bounds_as_rows=False))
    assert 'INFEASIBLE' not in str(log)


def test_the_log_records_infeasibility():
    f = Formulation()
    x = f.Variable('x', 1.0, '', 'x', bounds=[0.1, 100.0])
    y = f.Variable('y', 1.0, '', 'y', bounds=[0.1, 10.0])
    f.Objective(y)
    f.Constraint(x >= 5.0)
    f.Constraint(x <= 2.0)
    f.Constraint(x * y >= 1.0)
    with pytest.raises(InfeasibleProblem):
        presolve(_detect(f, bounds_as_rows=False))
