#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Adapter: detected GP/SP row form -> SLCP Problem.

Rows are [constraint_index, coefficient, *exponents]; index i >= 1 is
constraint i's numerator, -i-1 its denominator, 0 the objective. Mapping:
numerator only -> Posynomial (exact in log space); numerator + denominator
-> PosynomialRatio (numerator exact, denominator AGM-condensed -- the
classical SP treatment); single-term == -> monomial equality (affine, exact).
A group with a negative coefficient can't be represented (the detector
should have moved it into a denominator) and is reported, not dropped.
"""

import numpy as np

from lcsolver.presolve.detected import as_detected
from lcsolver.solvers.sequential.slcp import (CondensedEquality, Constraint,
                                    GreyboxSignomial, Options,
                                    Posynomial,
                                    PosynomialRatio, Problem, Signomial,
                                    solve as _slcp_solve)


def greybox_blocks(structures):
    """Active grey-box blocks, reached via structures['variables'] so we see
    the same unit-corrected model the rest of the structure describes."""
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
    """Opaque Signomial rows per grey-box block: bb(inputs)/output == 1,
    evaluated through the same block-model path (units, caching) as the
    native cyipopt grey-box route."""
    col = {id(v): j for j, v in enumerate(structures.get('variables') or [])}
    rows = []

    # ---- batch prefetch: give the boxes ONE look at each new iterate ----
    # A box exposing batch_prefetch(items) on its CLASS (e.g. lcwindturbine's
    # airfoil_bb) gets [(box, x[in_idx]), ...] to warm its own cache; the
    # per-row evaluations below then hit that cache. No hook or a failing
    # hook falls back to the scalar path, so this is a pure optimization.
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
            flip=True:  out/bb (rows 'out <= bb').
            Returns (fn, fn_value): the full value+gradient callback and the
            values-only one -- line searches and violation checks go through
            fn_value, so the box is never asked for derivatives it does not
            need (on an FD box that is the whole central-difference sweep)."""
            def fn_value(x):
                x = np.asarray(x, dtype=float)
                _prefetch(x)
                bb.set_input_values(x[in_idx])
                vals = np.atleast_1d(np.asarray(bb.evaluate_outputs(),
                                                dtype=float))
                return (x[iout] / vals[k]) if flip else (vals[k] / x[iout])

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
            return fn, fn_value

        # Directional rows per the declared operator (_lc_operators; absent
        # = all '==', the historical form). 'out >= bb' -> bb/out <= 1,
        # 'out <= bb' -> out/bb <= 1; one-sided, so no black-box equality
        # manifold -- the model's own pressure binds them at the optimum.
        ops = getattr(block, '_lc_operators', None)
        if not ops or len(ops) != len(out_idx):
            ops = ['=='] * len(out_idx)
        for k, iout in enumerate(out_idx):
            flip = ops[k] == '<='
            fn, fv = make_fn(bb, in_idx, k, iout, flip=flip)
            rows.append(Constraint(
                GreyboxSignomial(fn, n, iout, fn_value=fv),
                '<=' if ops[k] in ('>=', '<=') else '=='))
    return rows


def build_problem(structures, sp_form=True, split_equalities=False,
                  pair_equalities=True):
    """Translate a detected structure into an SLCP Problem.

    pair_equalities (default True) merges mutual-reverse row pairs (p <= q
    and q <= p, how EDI models write signomial equalities) into one == row.
    SIA needs real equalities: without the merge n_eq == 0, Phase I never
    engages, and the elastic L1 slacks both sides and stalls (30+ identical
    phase1-L1 iterations per restoration cycle on the wind-turbine solve).
    Matching folds any single-monomial side into the other first, rounds to
    11 sig figs, and has a dedicated rule for reciprocal monomial pins;
    pairs 389/397 two-sided rows on the wind missions SP, and an unmatched
    pair just stays split. Relies on sia.py: tau escalation ignores
    equality slacks, Phase-I proximity is seed-anchored, accepted steps are
    restored onto the manifold first.

    sp_form=True (default): p/q <= 1 becomes a PosynomialRatio (p exact,
    q AGM-condensed; conservative). False: a plain Signomial callback that
    SLCP linearizes whole -- stock SLCP for paper comparisons; also loses
    sub-problem caching since a linearization moves every iteration.
    """
    st = as_detected(structures)
    if st.log_key is None:
        raise ValueError('structure is neither a GP nor an SP')

    obj_terms = [t for t in st.terms(0) if not t.denominator]
    if not obj_terms:
        raise ValueError('no objective rows found in the detected structure')
    # Dense row width, not the sparsity pattern: an all-zero column still
    # occupies its place, narrowing would renumber every variable after it.
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
            """Canonical (num, den) term-tuples for matching. The detector
            normalizes a pin's two directions differently, so fold any
            single-monomial side into the other side first; signatures are
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
            """Fold a mono-vs-mono (or mono-only) row to C x^A <= 1.
            Returns (A_sparse, C) or None."""
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

            # (b) monomial pins: m1 <= m2 and its reverse fold to reciprocal
            # rows C x^A <= 1 and (1/C) x^-A <= 1, which the (num, den) swap
            # above can never see. 151 of 287 pin rows on the wind missions
            # SP fell through the posynomial matcher for this reason.
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
                # One constraint, both sides condensed. The split pair pins
                # the step to a null space and makes the multipliers
                # degenerate; see CondensedEquality.
                constraints.append(
                    Constraint(CondensedEquality(p_, q_, n), '=='))
                n_split += 1
                continue
            constraints.append(Constraint(wrap(PosynomialRatio(p_, q_, n)), '<='))
            if op == '==':
                # An equality is two inequalities: p/q <= 1 alone relaxes
                # the problem. AGM condensation is one-sided, but the
                # reverse q/p <= 1 is another ratio and condenses fine.
                constraints.append(
                    Constraint(wrap(PosynomialRatio(q_, p_, n)), '<='))
                n_split += 1
        else:
            body = posynomial(num, f'constraint {idx}')
            if op == '==' and body.is_monomial:
                # A monomial equality is affine in log space: exact as is.
                constraints.append(Constraint(body, '=='))
            elif op == '==' and split_equalities is False:
                # One condensed equality, p_hat == 1, instead of the pair
                # p <= 1 plus 1/p <= 1, whose multipliers are just as
                # degenerate (910.8 vs 910.9 on SPaircraft).
                constraints.append(
                    Constraint(CondensedEquality(body, unit(), n), '=='))
                n_split += 1
            elif op == '==':
                # Multi-term posynomial equality: p <= 1 goes in as is; the
                # reverse 1/p <= 1 is not a posynomial, so it goes in as a
                # condensed ratio. Dropping it relaxes the problem. Same
                # device PCCP uses in cvxopt/SP.py.
                constraints.append(Constraint(body, '<='))
                constraints.append(
                    Constraint(wrap(PosynomialRatio(unit(), body, n)), '<='))
                n_split += 1
            else:
                constraints.append(Constraint(body, '<='))

    # Carry variable bounds as bounds when the detector split them out.
    # Padded, not trusted blindly: bounds aligns with structures['variables']
    # but n comes from the widest exponent row, and the two disagree when a
    # trailing variable appears in no row at all.
    bounds = structures.get('bounds')
    if bounds is not None:
        bounds = list(bounds[:n]) + [(None, None)] * max(0, n - len(bounds))

    if gb:
        constraints.extend(_greybox_rows(gb, structures, n))

    names = [str(v) for v in structures.get('variables', [])][:n]
    return Problem(n, objective, constraints, names=names or None,
                   bounds=bounds)


def _bound_row_block(structures):
    """Constraint indices of the declared-bound rows: the detector emits them
    after the model's own constraints, so they're the trailing N_cons_bounds
    block. Empty when bounds came split out -- then every remaining singleton
    is a model row that must stay one."""
    if structures.get('bounds') is not None:
        return frozenset()
    info = structures.get('info') or {}
    n_bound = int(info.get('N_cons_bounds') or 0)
    n_total = int(info.get('N_cons_total') or 0)
    return frozenset(range(n_total - n_bound + 1, n_total + 1))


def _greybox_protected(structures):
    """Column indices any grey-box block references, or None if unmappable.
    Column removal is safe only for columns no block references (grey-box
    rows key structures['variables'] on identity, and reduce_columns keeps
    it in sync). None means a block couldn't be mapped -- the caller must
    skip presolve, since no column is provably safe."""
    blocks = greybox_blocks(structures)
    if not blocks:
        return frozenset()
    col = {id(v): j for j, v in enumerate(structures.get('variables') or [])}
    protected = set()
    for block in blocks:
        bb = getattr(block, '_ex_model', None)
        if bb is None:
            return None
        try:
            for v in _unwrap_vars(list(bb.inputVariables_optimization)
                                  + list(bb.outputVariables_optimization)):
                protected.add(col[id(v)])
        except (KeyError, TypeError):
            return None
    return frozenset(protected)


def presolve_structures(structures, verbose=False, fold_only=None,
                        eliminate=True, fold=True, protect=None):
    """Shrink a detected structure before handing it to a solver.

    Folds bound-like rows into the bounds, then drops columns nothing refers
    to. Both reductions are exact. Returns (reduced, removed); removed is in
    reduce_columns form, what restore_columns needs to rebuild a solution.
    Works on structures whose bounds are still rows by synthesizing an empty
    bounds array -- done here because SLCP/SIA read bounds and the cvxopt
    backends don't, which is why lcsolver.presolve.reductions refuses it.
    """
    from lcsolver.presolve.reductions import presolve as _presolve_pipeline

    st = dict(structures)
    if st.get('bounds') is None:
        key = ('Signomial_Program' if st['Signomial_Program'][0]
               else 'Geometric_Program')
        width = max((len(r) - 2 for r in st[key][1]), default=0)
        st['bounds'] = [(None, None)] * max(width,
                                            len(st.get('variables') or []))
    return _presolve_pipeline(st, verbose=verbose, fold=fold,
                              fold_only=fold_only, eliminate=eliminate,
                              protect=protect)


def _apply_presolve(structures, x0):
    """Returns (reduced_structures, reduced_x0, log, n_original).

    On a grey-box model only the column peel runs, with grey-box-referenced
    columns protected -- the peel removes output-only variables and their
    defining constraints instead of letting IPOPT park them arbitrarily.
    Two passes are tuned down on b737 evidence: the fold is restricted to
    declared-bound rows (folding singleton model rows stalled the case at
    the iteration cap; see _fold_bound_rows), and monomial-equality
    elimination is off (shrinking 1,298 -> 592 vars took the solve from
    39 iters / 136 s to 62 iters / 2,504 s -- smaller is not faster here).
    """
    n_original = len(structures.get('variables') or [])
    protect = _greybox_protected(structures)
    if protect is None:
        # Unmappable grey-box block: no removal is provably safe, and the
        # fold is barred too (see _fold_bound_rows).
        return structures, x0, None, n_original
    if protect:
        # Grey-box present: column peel only, referenced columns protected.
        # No fold -- folding renumbers rows a grey-box row may reference.
        reduced, log = presolve_structures(structures, fold=False,
                                           eliminate=False, protect=protect)
    else:
        reduced, log = presolve_structures(
            structures, fold_only=_bound_row_block(structures),
            eliminate=False)
    if x0 is not None:
        # Drop what each pass removed, in the space that pass ran in; the
        # log unwinds in reverse afterwards, so no index remapping needed.
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

    This solver family takes bounds natively; carrying them as rows is pure
    cost (b737: 1,338 s with bound rows vs 188 s without, same optimum).
    Only declared-bound rows fold -- singleton model rows stay rows so the
    elastic relaxation can put slack on them; folding everything singleton
    turned b737 into a 200-iteration stall. Declared-bound rows are the
    trailing N_cons_bounds block (the detector emits them after the model's
    own constraints). No-op when bounds are already split out, none were
    declared, or a grey-box block is present (folding renumbers rows a
    grey-box row may reference).
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
    """Solve a detected GP/SP with SLCP. Returns the SLCP Result.

    x0 is in natural (not log) variables, strictly positive; defaults to the
    current variable values. presolve=True shrinks the problem and restores
    removed variables into result.x afterwards (result.removed records what
    was taken out). Bound rows fold into native bounds either way;
    presolve=False disables only the column reductions, not the fold.
    """
    import pyomo.environ as pyo

    if x0 is None:
        x0 = [float(pyo.value(v)) for v in structures['variables']]
    log, n_original = None, len(structures.get('variables') or [])
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

    Same adapter as solve_slcp, pointed at sia.solve_sia. Keep sp_form=True:
    the conservative condensation is the basis of the method. Bound rows
    fold into native bounds either way; presolve=False disables only the
    column reductions, not the fold.
    """
    import pyomo.environ as pyo

    from lcsolver.solvers.sequential.sia import solve_sia as _sia_solve

    if x0 is None:
        x0 = [float(pyo.value(v)) for v in structures['variables']]
    log, n_original = None, len(structures.get('variables') or [])
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
