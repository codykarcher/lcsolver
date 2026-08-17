#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Run a detected GP/SP through the SLCP solver.

``lcsolver.solvers.sequential.slcp`` implements sequential log-convex programming over
its own :class:`~lcsolver.solvers.sequential.slcp.Problem` object, which nothing in LCsolver
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
from lcsolver.solvers.sequential.slcp import (CondensedEquality, Constraint,
                                    GreyboxSignomial, Options,
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

    # ---- batch prefetch: give the boxes ONE look at each new iterate ----
    # A black box that can evaluate many queries in one shot (e.g.
    # lcwindturbine's airfoil_bb, whose torch surrogate batches trivially)
    # exposes a `batch_prefetch(items)` attribute on its CLASS; items is
    # [(box, x[in_idx]), ...] for every registered box sharing that hook.
    # The hook warms the box's own cache, so the per-row evaluations below
    # become cache hits.  Boxes without the attribute are untouched, and a
    # failing hook is ignored -- the scalar path computes as before, so
    # this is a pure optimization with no correctness surface.
    pairs = []                                  # (bb, in_idx), ALL blocks
    _pf_state = {'x': None}

    def _prefetch(x):
        xb = x.tobytes()
        if _pf_state['x'] == xb:
            return
        _pf_state['x'] = xb                     # set FIRST: no re-entry
        hooks = {}
        for bb_, idx_ in pairs:
            hook = getattr(type(bb_), 'batch_prefetch', None)
            if not callable(hook):
                continue
            hooks.setdefault(id(hook), (hook, []))[1].append((bb_, x[idx_]))
        for hook, items in hooks.values():
            try:
                hook(items)
            except Exception:
                pass

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
        pairs.append((bb, in_idx))

        def make_fn(bb, in_idx, k, iout, flip=False):
            """flip=False: bb/out (rows 'out >= bb' and '==').
            flip=True:  out/bb (rows 'out <= bb')."""
            def fn(x):
                x = np.asarray(x, dtype=float)
                _prefetch(x)
                bb.set_input_values(x[in_idx])
                vals = np.atleast_1d(np.asarray(bb.evaluate_outputs(),
                                                dtype=float))
                jac = bb.evaluate_jacobian_outputs()
                jac = np.asarray(jac.todense() if hasattr(jac, 'todense')
                                 else jac, dtype=float)
                jac = jac.reshape(len(vals), len(in_idx))
                g = np.zeros(n)
                if flip:
                    val = x[iout] / vals[k]
                    for pos, iin in enumerate(in_idx):
                        g[iin] += -x[iout] * jac[k, pos] / vals[k] ** 2
                    g[iout] += 1.0 / vals[k]
                else:
                    val = vals[k] / x[iout]
                    for pos, iin in enumerate(in_idx):
                        g[iin] += jac[k, pos] / x[iout]
                    g[iout] += -vals[k] / x[iout] ** 2
                return val, g
            return fn

        # Directional rows per the declared operator (Formulation records
        # them as _lc_operators; absent = all '==', the historical form).
        # 'out >= bb' -> bb/out <= 1;  'out <= bb' -> out/bb <= 1; both
        # one-sided, so no black-box equality manifold exists for those
        # rows --- the model's own pressure binds them at the optimum.
        ops = getattr(block, '_lc_operators', None)
        if not ops or len(ops) != len(out_idx):
            ops = ['=='] * len(out_idx)
        for k, iout in enumerate(out_idx):
            if ops[k] == '>=':
                rows.append(Constraint(
                    GreyboxSignomial(make_fn(bb, in_idx, k, iout), n, iout),
                    '<='))
            elif ops[k] == '<=':
                rows.append(Constraint(
                    GreyboxSignomial(make_fn(bb, in_idx, k, iout, flip=True),
                                     n, iout), '<='))
            else:
                rows.append(Constraint(
                    GreyboxSignomial(make_fn(bb, in_idx, k, iout), n, iout),
                    '=='))
    return rows


def build_problem(structures, sp_form=True, split_equalities=False,
                  pair_equalities=True):
    """Translate a detected structure into an SLCP :class:`Problem`.

    ``pair_equalities`` (default True) repairs TWO-SIDED PINS written as
    separate rows -- ``p <= q`` and ``q <= p`` appended independently,
    which is how EDI models express signomial equalities (survival/gamma
    series, exact-G, BEM operating-point pins) -- merging each
    mutual-reverse pair into ONE ``==`` row.  An equality must reach the
    solver AS an equality: without the merge SIA counts ``n_eq == 0``,
    the composite Phase I (Gauss-Newton restore onto the manifold +
    tangential steps) never engages, and the elastic L1 it degrades to
    slacks both pair sides independently and stalls on the interior-less
    pinned manifold (observed: 30+ identical phase1-L1 iterations per
    restoration cycle, the dominant cost of an 11-minute wind-turbine
    solve).

    Matching runs in a CANONICAL FRAME: any single-monomial side is
    folded into the other side first (the detector normalizes the two
    directions of a pin differently), signatures are rounded to 11
    significant figures, and reciprocal single-monomial rows (mono ==
    mono pins fold to C x^A <= 1 and (1/C) x^-A <= 1) are matched by a
    dedicated rule.  On the wind missions SP this pairs 389 of 397
    two-sided rows; the survivors are the model's GENUINE one-sided
    signomial rows (AEP, bin balances, the 3P edge).  A pair that fails
    to match merely stays split -- the previous behavior.

    The downstream requirements this relies on (all in sia.py): tau
    escalation ignores equality-row slacks/duals, Phase-I proximity is
    seed-anchored, and accepted steps are restored onto the manifold
    before the acceptance tests.

    ``sp_form`` selects how a signomial constraint ``p/q <= 1`` is handled:

    ``True`` (default)
        Build a :class:`~lcsolver.solvers.sequential.slcp.PosynomialRatio`, which keeps
        ``p`` exact in log space and condenses only ``q`` by the AGM
        inequality. Less approximation, and conservative.
    ``False``
        Build a plain :class:`~lcsolver.solvers.sequential.slcp.Signomial` -- a
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

    # ---- two-sided pin recognition (see docstring) ----------------------
    merged_eq, skip_idx = set(), set()
    if pair_equalities:
        import math as _math

        def _round_c(c):
            return float('%.11e' % c)

        def _round_e(e):
            return round(e, 11)

        def _canon(num, den):
            """Canonical (num, den) term-tuples for matching.

            The detector normalizes the two directions of a pin
            DIFFERENTLY: `p <= m` (m a monomial) often arrives with m
            folded into p's coefficients/exponents, while its reverse
            `m <= p` keeps the ratio form.  Matching therefore folds any
            SINGLE-MONOMIAL side into the other side first, so both
            directions land in the same canonical frame; signatures are
            rounded so the two fold arithmetics agree."""
            def fold(terms, mono, into_num):
                cM = float(mono.coeff)
                eM = dict(mono.exponents)
                out = []
                for t in terms:
                    ex = dict((j, float(e)) for j, e in t.exponents.items())
                    for j, e in eM.items():
                        if e:
                            ex[j] = ex.get(j, 0.0) - float(e)
                            if ex[j] == 0.0:
                                del ex[j]
                    out.append((float(t.coeff) / cM, ex))
                return out
            if len(den) == 1:
                folded = fold(num, den[0], True)
                return folded, []
            if len(num) == 1 and len(den) > 1:
                folded = fold(den, num[0], False)
                return [], folded
            return ([(float(t.coeff),
                      dict((j, float(e)) for j, e in t.exponents.items()))
                     for t in num],
                    [(float(t.coeff),
                      dict((j, float(e)) for j, e in t.exponents.items()))
                     for t in den])

        def _sig_folded(folded):
            out = tuple(sorted(
                (_round_c(c),
                 tuple(sorted((j, _round_e(e))
                              for j, e in ex.items() if e != 0.0)))
                for c, ex in folded))
            return out if out else ((1.0, ()),)     # empty side = the unit

        def _mono_fold(num, den):
            """A row that is one monomial vs one (or no) monomial folds to
            C * x^A <= 1.  Returns (A_sparse, C) or None."""
            if len(num) != 1 or len(den) > 1:
                return None
            cN, tN = float(num[0].coeff), num[0].exponents
            A = dict((j, float(e)) for j, e in tN.items() if e != 0.0)
            C = cN
            if den:
                cD, tD = float(den[0].coeff), den[0].exponents
                if cD == 0.0:
                    return None
                C = C / cD
                for j, e in tD.items():
                    if e != 0.0:
                        A[j] = A.get(j, 0.0) - float(e)
                        if A[j] == 0.0:
                            del A[j]
            if not A:
                return None                          # constant row
            return (tuple(sorted(A.items())), C)

        seen = {}
        seen_mono = {}
        for idx in st.constraint_indices:
            if st.operator(idx) != '<=':
                continue
            all_t = st.terms(idx)
            num = [t for t in all_t if not t.denominator]
            den = [t for t in all_t if t.denominator]

            # (a) general posynomial pairs: p <= q here, q <= p elsewhere,
            # matched in the canonical single-mono-folded frame
            cn, cd = _canon(num, den)
            key = (_sig_folded(cn), _sig_folded(cd))
            if key[0] != key[1]:                     # self-reverse: degenerate
                rkey = (key[1], key[0])
                j0 = seen.get(rkey)
                if (j0 is not None and j0 not in merged_eq
                        and j0 not in skip_idx):
                    merged_eq.add(j0)
                    skip_idx.add(idx)
                    seen[rkey] = None                # consume the partner
                    continue
                seen.setdefault(key, idx)

            # (b) MONOMIAL pins: the detector folds `m1 <= m2` and its
            # reverse into single rows C x^A <= 1 and C' x^-A <= 1 with
            # C C' = 1 -- reciprocal forms the (num, den) swap above can
            # never see.  These are the sin/Re/flow-angle pins of the wind
            # models: 151 of 287 pin rows on the missions SP fell through
            # the posynomial matcher for exactly this reason.
            fold = _mono_fold(num, den)
            if fold is not None:
                A, C = fold
                negA = tuple(sorted((j, -e) for j, e in A))
                hit = seen_mono.get(negA)
                if (hit is not None and hit[1] != 0.0
                        and _math.isclose(C * hit[1], 1.0, rel_tol=1e-9)
                        and hit[0] not in skip_idx
                        and hit[0] not in merged_eq):
                    merged_eq.add(hit[0])
                    skip_idx.add(idx)
                    seen_mono[negA] = (hit[0], 0.0)  # consume
                else:
                    seen_mono.setdefault(A, (idx, C))

    for idx in st.constraint_indices:
        if idx in skip_idx:
            continue
        op = '==' if idx in merged_eq else st.operator(idx)
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


def _bound_row_block(structures):
    """Constraint indices of the detector's declared-bound rows.

    The detector emits them during its variable walk, after the model's own
    constraints were declared, so they are always the trailing
    ``N_cons_bounds`` block of the numbering. Empty when bounds already came
    split out -- then no declared bound rides as a row, and every remaining
    singleton is a MODEL row that must stay one.
    """
    if structures.get('bounds') is not None:
        return frozenset()
    info = structures.get('info') or {}
    n_bound = int(info.get('N_cons_bounds') or 0)
    n_total = int(info.get('N_cons_total') or 0)
    return frozenset(range(n_total - n_bound + 1, n_total + 1))


def presolve_structures(structures, verbose=False, fold_only=None,
                        eliminate=True):
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
    return _presolve_pipeline(st, verbose=verbose, fold_only=fold_only,
                              eliminate=eliminate)


def _apply_presolve(structures, x0):
    """``(reduced_structures, reduced_x0, log, n_original)``.

    Two of the pipeline's passes are tuned down here, both on measured
    evidence from the spcomparisons b737 case (1,298 vars, 4,160 rows):

    * The fold is restricted to the declared-bound rows: a singleton MODEL
      row folded into a hard bound leaves the elastic relaxation nothing to
      put slack on, and folding them all stalled the case at the iteration
      cap (see :func:`_fold_bound_rows`).
    * Monomial-equality elimination is OFF. Substituting away 706 variables
      shrank the problem (592 vars, 3,454 rows) yet took the same solve from
      39 iterations / 136 s to 62 iterations / 2,504 s -- the early
      iterations stay cheap and the trajectory then enters an expensive
      restoration phase the unsubstituted problem never visits. Smaller is
      not faster for the sequential solvers; the pass remains available and
      default-on in :func:`lcsolver.presolve.reductions.presolve`.
    """
    n_original = len(structures.get('variables') or [])
    reduced, log = presolve_structures(
        structures, fold_only=_bound_row_block(structures), eliminate=False)
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

def _fold_bound_rows(structures):
    """Declared-bound rows -> native bounds, and nothing else.

    ``structure_detector`` defaults to ``bounds_as_rows=True`` because the
    cvxopt backends can only read bounds from rows -- but this solver family
    takes bounds natively, and carrying them as rows is pure cost: on the
    spcomparisons b737 case (1,298 vars, 4,160 real rows, 2,596 bound rows)
    the identical 38-iteration solve runs 1,338 s with bound rows and 188 s
    without, same optimum to the tenth of a pound.

    Only the DECLARED-bound rows fold -- singleton model rows (span gates and
    the like) stay as rows, because the elastic relaxation must be able to
    put slack on an active constraint mid-trajectory and a hard bound cannot
    take slack. Folding everything singleton (which the full presolve does)
    turned that same b737 case into a 200-iteration stall; folding only the
    declared bounds reproduces ``bounds_as_rows=False`` exactly, which
    converges in the same 38-39 iterations as the unfolded problem with the
    bit-equal optimum.

    The declared-bound rows are identified positionally: the detector emits
    them during its variable walk, AFTER the model's own constraints were
    declared, so they are always the trailing ``N_cons_bounds`` block of the
    constraint numbering (the constant-row drop upstream preserves order).

    No-op when bounds are already split out, none were declared, or a
    grey-box block is present (folding renumbers rows, which a grey-box row
    may reference).
    """
    if structures.get('bounds') is not None or greybox_blocks(structures):
        return structures
    info = structures.get('info') or {}
    n_bound = int(info.get('N_cons_bounds') or 0)
    n_total = int(info.get('N_cons_total') or 0)
    if not n_bound:
        return structures
    from lcsolver.presolve.reductions import fold_singleton_rows

    st = dict(structures)
    key = ('Signomial_Program' if st['Signomial_Program'][0]
           else 'Geometric_Program')
    width = max((len(r) - 2 for r in st[key][1]), default=0)
    st['bounds'] = [(None, None)] * max(width,
                                        len(st.get('variables') or []))
    return fold_singleton_rows(
        st, only=set(range(n_total - n_bound + 1, n_total + 1)))


def solve_slcp(structures, x0=None, method='slcp', options=None,
               sp_form=True, presolve=True):
    """Solve a detected GP/SP with SLCP.

    ``x0`` is in the natural (not log) variables and must be strictly
    positive; it defaults to the current values of ``structures['variables']``.
    Returns the SLCP :class:`~lcsolver.solvers.sequential.slcp.Result`.

    ``presolve`` (default True) shrinks the problem first via
    :func:`presolve_structures` and puts the removed variables back into
    ``result.x`` afterwards, so the result is indexed by the original variable
    ordering either way. ``result.removed`` records what was taken out.
    Either way, bound rows are folded into native variable bounds first
    (see :func:`_fold_bound_rows`); ``presolve=False`` disables the column
    reductions, not the fold.
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
    else:
        structures = _fold_bound_rows(structures)

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
              presolve=True, split_equalities=False, pair_equalities=True):
    """Solve a detected GP/SP by sequential inner approximation.

    Same adapter as :func:`solve_slcp`, pointed at
    :func:`lcsolver.solvers.sequential.sia.solve_sia`. ``sp_form`` defaults to True here
    and should stay that way -- the conservative condensation is the whole
    basis of the method, and turning it off downgrades every signomial
    constraint to a linearization that then has to be globalized.

    Bound rows are folded into native variable bounds whether or not
    ``presolve`` is on (see :func:`_fold_bound_rows`); ``presolve=False``
    disables the column reductions, not the fold.
    """
    import pyomo.environ as pyo

    from lcsolver.solvers.sequential.sia import solve_sia as _sia_solve

    if x0 is None:
        x0 = [float(pyo.value(v)) for v in structures['variables']]
    log, n_original = None, len(structures.get('variables') or [])
    if presolve and greybox_blocks(structures):
        # Presolve reasons only about the algebraic rows, so it can fold or
        # drop a column a grey-box row still references. Skip it.
        presolve = False
    if presolve:
        structures, x0, log, n_original = _apply_presolve(structures, x0)
    else:
        structures = _fold_bound_rows(structures)

    problem = build_problem(structures, sp_form=sp_form,
                            split_equalities=split_equalities,
                            pair_equalities=pair_equalities)
    x0 = np.asarray(x0, dtype=float)
    if len(x0) < problem.n:
        x0 = np.concatenate([x0, np.ones(problem.n - len(x0))])
    x0 = np.where(x0 > 0, x0, 1.0)
    result = _sia_solve(problem, x0[:problem.n], options=options)
    return _restore(result, log, n_original)
