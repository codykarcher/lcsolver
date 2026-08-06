#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Run a detected GP/SP through the SLCP solver.

``lcsolver.solvers.ipopt.slcp`` implements sequential log-convex programming over
its own :class:`~lcsolver.solvers.ipopt.slcp.Problem` object, which nothing in LCsolver
built. This module is the missing adapter: it turns the row form produced by
:func:`~lcsolver.presolve.structureDetector.structure_detector` into that object,
so the same formulation can be solved either way and the two compared.

The row form is already close to what SLCP wants. Rows carry
``[constraint_index, coefficient, *exponents]``; index ``i >= 1`` is
constraint *i*'s numerator, index ``-i-1`` its denominator, and index 0 is the
objective. Each group maps as:

* numerator only, all coefficients positive -> ``Posynomial``, imposed exactly
  in log space;
* numerator and denominator -> ``PosynomialRatio``, which keeps the numerator
  exact and condenses only the denominator by the arithmetic-geometric-mean
  inequality. This is the classical signomial-program treatment and is
  strictly less lossy than linearizing the whole body;
* a single-term numerator with ``==`` -> a monomial equality, which is affine
  in log space and imposed exactly.

A group with a negative coefficient cannot be represented: SLCP's
``Posynomial`` requires positive coefficients, and the detector should already
have moved negative terms into a denominator. Such a group is reported rather
than silently dropped.
"""

import numpy as np

from lcsolver.presolve.detected import as_detected
from lcsolver.solvers.ipopt.slcp import (CondensedEquality, Constraint, Options,
                                    Posynomial,
                                    PosynomialRatio, Problem, Signomial,
                                    solve as _slcp_solve)


def greybox_blocks(structures):
    """The active grey-box (black-box constraint) blocks of the detected model.

    The model is reached through ``structures['variables']``, so this sees the
    same (unit-corrected) model the rest of the structure describes.
    """
    variables = structures.get('variables') or []
    if not variables:
        return []
    try:
        from pyomo.contrib.pynumero.interfaces.external_grey_box import (
            ExternalGreyBoxBlock,
        )
    except Exception:
        return []
    model = variables[0].model()
    return list(model.component_data_objects(ExternalGreyBoxBlock,
                                             descend_into=True, active=True))


def _unwrap_vars(vars_):
    import pyomo.core.base.var as _var

    out = []
    for v in vars_:
        if isinstance(v, _var.ScalarVar):
            out.append(v)
        elif isinstance(v, _var.IndexedVar):
            out.extend(v[i] for i in v.index_set().data())
        else:
            out.append(v)
    return out


def _greybox_rows(blocks, structures, n):
    """Opaque ``Signomial`` equality rows for every grey-box block.

    Each block's output variable is tied to its black box as
    ``bb(inputs) / output == 1`` -- the ratio form every other opaque row in
    this file uses, affine-friendly in log space. Values and jacobians come
    from the block model's own ``set_input_values`` / ``evaluate_outputs`` /
    ``evaluate_jacobian_outputs``, the same standardized path (units,
    caching) the native cyipopt grey-box route evaluates through.
    """
    col = {id(v): j for j, v in enumerate(structures.get('variables') or [])}
    rows = []
    for block in blocks:
        bb = getattr(block, '_ex_model', None)
        if bb is None:
            raise ValueError(
                f'grey-box block {block.name!r} has no external model. This '
                'is the signature of a model clone made before '
                'BBList.__deepcopy__ existed; re-build the formulation.')
        ins = _unwrap_vars(bb.inputVariables_optimization)
        outs = _unwrap_vars(bb.outputVariables_optimization)
        try:
            in_idx = [col[id(v)] for v in ins]
            out_idx = [col[id(v)] for v in outs]
        except KeyError:
            raise ValueError(
                f'grey-box block {block.name!r} references a variable that '
                'the structure detector did not record; cannot map it into '
                'the problem columns.')
        in_idx = np.asarray(in_idx, dtype=int)

        def make_fn(bb, in_idx, k, iout):
            def fn(x):
                x = np.asarray(x, dtype=float)
                bb.set_input_values(x[in_idx])
                vals = np.atleast_1d(np.asarray(bb.evaluate_outputs(),
                                                dtype=float))
                jac = bb.evaluate_jacobian_outputs()
                jac = np.asarray(jac.todense() if hasattr(jac, 'todense')
                                 else jac, dtype=float)
                jac = jac.reshape(len(vals), len(in_idx))
                val = vals[k] / x[iout]
                g = np.zeros(n)
                for pos, iin in enumerate(in_idx):
                    g[iin] += jac[k, pos] / x[iout]
                g[iout] += -vals[k] / x[iout] ** 2
                return val, g
            return fn

        for k, iout in enumerate(out_idx):
            rows.append(Constraint(Signomial(make_fn(bb, in_idx, k, iout), n),
                                   '=='))
    return rows


def build_problem(structures, sp_form=True, split_equalities=False):
    """Translate a detected structure into an SLCP :class:`Problem`.

    ``sp_form`` selects how a signomial constraint ``p/q <= 1`` is handled:

    ``True`` (default)
        Build a :class:`~lcsolver.solvers.ipopt.slcp.PosynomialRatio`, which keeps
        ``p`` exact in log space and condenses only ``q`` by the AGM
        inequality. Less approximation, and conservative.
    ``False``
        Build a plain :class:`~lcsolver.solvers.ipopt.slcp.Signomial` -- a
        value/gradient callback over the same ratio -- which SLCP then
        linearizes whole, discarding ``p``'s log-convexity along with ``q``'s
        curvature. This is stock SLCP as the paper describes it, and is the
        setting to use when comparing against it.

    Turning it off also loses sub-problem caching for those constraints: a
    linearization moves every iteration, so there is nothing to cache.
    """
    st = as_detected(structures)
    if st.log_key is None:
        raise ValueError('structure is neither a GP nor an SP')

    obj_terms = [t for t in st.terms(0) if not t.denominator]
    if not obj_terms:
        raise ValueError('no objective rows found in the detected structure')
    # The DENSE row width, not the sparsity pattern: an all-zero column still
    # occupies its place, and narrowing would renumber every variable after it.
    n = max(len(r) - 2 for r in st[st.log_key][1])
    gb = greybox_blocks(structures)
    if gb:
        # A variable that appears only in a grey-box row is invisible to the
        # algebraic rows and can sit beyond their width.
        n = max(n, len(structures.get('variables') or []))

    def posynomial(terms, what):
        bad = [t.coeff for t in terms if t.coeff <= 0]
        if bad:
            raise ValueError(
                f'{what} has non-positive coefficients {bad}; SLCP needs each '
                'posynomial part to be positive, so the detector should have '
                'moved these into a denominator')
        return Posynomial(
            [(t.coeff, [t.exponents.get(j, 0.0) for j in range(n)])
             for t in terms], n)

    objective = posynomial(obj_terms, 'objective')

    constraints = []
    n_split = 0

    def unit():
        """The monomial 1, for writing ``1/p <= 1``."""
        return Posynomial([(1.0, [0.0] * n)], n)

    for idx in st.constraint_indices:
        op = st.operator(idx)
        all_terms = st.terms(idx)
        num = [t for t in all_terms if not t.denominator]
        den = [t for t in all_terms if t.denominator]

        def wrap(body):
            """Opaque the body when sp_form is off, so SLCP linearizes it."""
            if sp_form or not isinstance(body, PosynomialRatio):
                return body
            return Signomial(lambda x, r=body: (r(x), r.grad(x)), n)

        if den:
            p_ = posynomial(num, f'constraint {idx} numerator')
            q_ = posynomial(den, f'constraint {idx} denominator')
            if op == '==' and split_equalities is False:
                # One constraint, both sides condensed. See CondensedEquality
                # for why the split pair is bad on both counts -- it pins the
                # step to a null space and makes the multipliers degenerate.
                constraints.append(
                    Constraint(CondensedEquality(p_, q_, n), '=='))
                n_split += 1
                continue
            constraints.append(Constraint(wrap(PosynomialRatio(p_, q_, n)), '<='))
            if op == '==':
                # An equality is TWO inequalities. Writing only p/q <= 1
                # RELAXES the problem -- the solver is then free to drive
                # p/q below 1, which the equality forbids. The AGM
                # condensation is one-sided so it cannot represent an
                # equality directly, but the reverse direction q/p <= 1 is
                # another ratio and condenses just as well.
                constraints.append(
                    Constraint(wrap(PosynomialRatio(q_, p_, n)), '<='))
                n_split += 1
        else:
            body = posynomial(num, f'constraint {idx}')
            if op == '==' and body.is_monomial:
                # A monomial equality is affine in log space: exact as is.
                constraints.append(Constraint(body, '=='))
            elif op == '==' and split_equalities is False:
                # One condensed equality, p_hat == 1, instead of the pair.
                # This branch matters as much as the ratio one above: the pair
                # it replaces is `p <= 1` plus `1/p <= 1`, whose multipliers
                # are just as degenerate -- measured at 910.8 against 910.9 on
                # SPaircraft, differing only in the fourth figure.
                constraints.append(
                    Constraint(CondensedEquality(body, unit(), n), '=='))
                n_split += 1
            elif op == '==':
                # A multi-term posynomial equality. p <= 1 is log-convex and
                # goes in as it stands; the reverse 1/p <= 1 is NOT a
                # posynomial, so it goes in as a ratio with p underneath and
                # is condensed. Dropping it -- which is what writing only
                # p <= 1 does -- relaxes the problem. This is the same device
                # PCCP uses in cvxopt/SP.py for the identical case.
                constraints.append(Constraint(body, '<='))
                constraints.append(
                    Constraint(wrap(PosynomialRatio(unit(), body, n)), '<='))
                n_split += 1
            else:
                constraints.append(Constraint(body, '<='))

    # Carry variable bounds as bounds when the detector split them out. They
    # are padded rather than trusted blindly: `bounds` is aligned with
    # structures['variables'], and `n` comes from the widest exponent row, so
    # the two can disagree if a trailing variable appears in no row at all.
    bounds = structures.get('bounds')
    if bounds is not None:
        bounds = list(bounds[:n]) + [(None, None)] * max(0, n - len(bounds))

    if gb:
        constraints.extend(_greybox_rows(gb, structures, n))

    names = [str(v) for v in structures.get('variables', [])][:n]
    return Problem(n, objective, constraints, names=names or None,
                   bounds=bounds)


def presolve_structures(structures, verbose=False):
    """Shrink a detected structure before handing it to a solver.

    Folds rows that are really bounds into the bounds, then drops columns
    nothing refers to -- variables fixed by equal bounds, and variables that
    appear in no real constraint and no objective term. Both reductions are
    exact: the optimal objective is unchanged and every removed variable gets
    a value that is feasible for the original problem.

    Returns ``(reduced, removed)``. ``removed`` is in
    :func:`~lcsolver.presolve.reductions.reduce_columns` form and is what
    :func:`~lcsolver.presolve.reductions.restore_columns` needs to rebuild a full solution.

    Unlike ``structure_detector(bounds_as_rows=False)``, this works on a
    structure whose bounds are still rows: it synthesizes the empty bounds
    array to fold them into. That is a deliberate choice made here rather than
    in ``lcsolver.presolve.reductions``, because this is the point that knows the consumer --
    SLCP and SIA both read variable bounds -- whereas the cvxopt backends do
    not, and ``lcsolver.presolve.reductions`` refuses the conversion for exactly that reason.
    """
    from lcsolver.presolve.reductions import presolve as _presolve_pipeline

    st = dict(structures)
    if st.get('bounds') is None:
        key = ('Signomial_Program' if st['Signomial_Program'][0]
               else 'Geometric_Program')
        width = max((len(r) - 2 for r in st[key][1]), default=0)
        st['bounds'] = [(None, None)] * max(width,
                                            len(st.get('variables') or []))
    return _presolve_pipeline(st, verbose=verbose)


def _apply_presolve(structures, x0):
    """``(reduced_structures, reduced_x0, log, n_original)``."""
    n_original = len(structures.get('variables') or [])
    reduced, log = presolve_structures(structures)
    if x0 is not None:
        # Drop what each pass removed, in the space that pass ran in. The log
        # unwinds them in reverse afterwards, so no index remapping is needed
        # in either direction.
        x = list(x0)
        for _label, removed, _counts in log.steps:
            if removed:
                gone = {r.index for r in removed}
                x = [v for j, v in enumerate(x) if j not in gone]
        x0 = x
    return reduced, x0, log, n_original


def _restore(result, log, n_original):
    """Put presolved-away variables back, so callers see the original layout."""
    result.presolve = log
    result.removed = log.removed_variables if log is not None else []
    if log is not None and getattr(result, 'x', None) is not None:
        result.x = log.restore(result.x)
    return result

def solve_slcp(structures, x0=None, method='slcp', options=None,
               sp_form=True, presolve=True):
    """Solve a detected GP/SP with SLCP.

    ``x0`` is in the natural (not log) variables and must be strictly
    positive; it defaults to the current values of ``structures['variables']``.
    Returns the SLCP :class:`~lcsolver.solvers.ipopt.slcp.Result`.

    ``presolve`` (default True) shrinks the problem first via
    :func:`presolve_structures` and puts the removed variables back into
    ``result.x`` afterwards, so the result is indexed by the original variable
    ordering either way. ``result.removed`` records what was taken out.
    """
    import pyomo.environ as pyo

    if x0 is None:
        x0 = [float(pyo.value(v)) for v in structures['variables']]
    log, n_original = None, len(structures.get('variables') or [])
    if presolve and greybox_blocks(structures):
        # Presolve reasons only about the algebraic rows, so it can fold or
        # drop a column a grey-box row still references. Skip it.
        presolve = False
    if presolve:
        structures, x0, log, n_original = _apply_presolve(structures, x0)

    problem = build_problem(structures, sp_form=sp_form)
    x0 = np.asarray(x0, dtype=float)
    if len(x0) < problem.n:
        # PCCP-style slack columns, if any, start at 1.
        x0 = np.concatenate([x0, np.ones(problem.n - len(x0))])
    x0 = np.where(x0 > 0, x0, 1.0)
    result = _slcp_solve(problem, x0[:problem.n], method=method,
                         options=options)
    return _restore(result, log, n_original)


def solve_sia(structures, x0=None, options=None, sp_form=True,
              presolve=True, split_equalities=False):
    """Solve a detected GP/SP by sequential inner approximation.

    Same adapter as :func:`solve_slcp`, pointed at
    :func:`lcsolver.solvers.ipopt.sia.solve_sia`. ``sp_form`` defaults to True here
    and should stay that way -- the conservative condensation is the whole
    basis of the method, and turning it off downgrades every signomial
    constraint to a linearization that then has to be globalized.
    """
    import pyomo.environ as pyo

    from lcsolver.solvers.ipopt.sia import solve_sia as _sia_solve

    if x0 is None:
        x0 = [float(pyo.value(v)) for v in structures['variables']]
    log, n_original = None, len(structures.get('variables') or [])
    if presolve and greybox_blocks(structures):
        # Presolve reasons only about the algebraic rows, so it can fold or
        # drop a column a grey-box row still references. Skip it.
        presolve = False
    if presolve:
        structures, x0, log, n_original = _apply_presolve(structures, x0)

    problem = build_problem(structures, sp_form=sp_form,
                            split_equalities=split_equalities)
    x0 = np.asarray(x0, dtype=float)
    if len(x0) < problem.n:
        x0 = np.concatenate([x0, np.ones(problem.n - len(x0))])
    x0 = np.where(x0 > 0, x0, 1.0)
    result = _sia_solve(problem, x0[:problem.n], options=options)
    return _restore(result, log, n_original)
