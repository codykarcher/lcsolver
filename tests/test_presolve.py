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

from edi import Formulation
from edi.presolve import (
    cancellation_report,
    degeneracy_report,
    fold_singleton_rows,
    presolve_report,
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
    reduced, removed = presolve_structures(st)

    assert reduced['bounds'] is not None
    assert [nm for _j, nm, _v, _why in removed] == ['y']


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
