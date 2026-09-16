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
                                    solve as slcp_solve)


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


def unwrap_variables(vars_):
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


class BatchPrefetch:
    """Give every black box ONE look at each new iterate before the per-row
    evaluations. A box exposing batch_prefetch(items) on its CLASS (e.g.
    lcwindturbine's airfoil_bb) gets [(box, x[in_idx]), ...] to warm its own
    cache; no hook, or a failing hook, falls back to the scalar path."""

    def __init__(self):
        self.pairs = []                         # (bb, in_idx), ALL blocks
        self.last = None

    def __call__(self, x):
        xb = x.tobytes()
        if self.last == xb:
            return
        self.last = xb                          # set FIRST: no re-entry
        hooks = {}
        for bb, idx in self.pairs:
            hook = getattr(type(bb), 'batch_prefetch', None)
            if not callable(hook):
                continue
            hooks.setdefault(id(hook), (hook, []))[1].append((bb, x[idx]))
        for hook, items in hooks.values():
            try:
                hook(items)
            except Exception:
                pass


class GreyboxRow:
    """One output of a black box as a signomial row body.

    flip=False: bb/out (rows 'out >= bb' and '=='); flip=True: out/bb (rows
    'out <= bb'). value() asks the box for values only -- line searches and
    violation checks go through it, so the box is never asked for
    derivatives it does not need (on an FD box that is the whole
    central-difference sweep); value_and_gradient() is the linearization.
    """

    def __init__(self, bb, in_idx, k, iout, flip, n, prefetch):
        self.bb = bb
        self.in_idx = in_idx
        self.k = k
        self.iout = iout
        self.flip = flip
        self.n = n
        self.prefetch = prefetch

    def outputs_at(self, x):
        self.prefetch(x)
        self.bb.set_input_values(x[self.in_idx])
        return np.atleast_1d(np.asarray(self.bb.evaluate_outputs(), dtype=float))

    def value(self, x):
        x = np.asarray(x, dtype=float)
        vals = self.outputs_at(x)
        if self.flip:
            return x[self.iout] / vals[self.k]
        return vals[self.k] / x[self.iout]

    def value_and_gradient(self, x):
        x = np.asarray(x, dtype=float)
        vals = self.outputs_at(x)
        jac = self.bb.evaluate_jacobian_outputs()
        if hasattr(jac, 'todense'):
            jac = jac.todense()
        jac = np.asarray(jac, dtype=float).reshape(len(vals), len(self.in_idx))
        k, iout = self.k, self.iout
        g = np.zeros(self.n)
        if self.flip:
            val = x[iout] / vals[k]
            for pos, iin in enumerate(self.in_idx):
                g[iin] += -x[iout] * jac[k, pos] / vals[k] ** 2
            g[iout] += 1.0 / vals[k]
        else:
            val = vals[k] / x[iout]
            for pos, iin in enumerate(self.in_idx):
                g[iin] += jac[k, pos] / x[iout]
            g[iout] += -vals[k] / x[iout] ** 2
        return val, g


def greybox_rows(blocks, structures, n):
    """Opaque Signomial rows per grey-box block: bb(inputs)/output == 1,
    evaluated through the same block-model path (units, caching) as the
    native cyipopt grey-box route."""
    col = {id(v): j for j, v in enumerate(structures.get('variables') or [])}
    rows = []
    prefetch = BatchPrefetch()

    for block in blocks:
        bb = getattr(block, '_ex_model', None)
        if bb is None:
            raise ValueError(
                f'grey-box block {block.name!r} has no external model. This '
                'is the signature of a model clone made before '
                'BBList.__deepcopy__ existed; re-build the formulation.')
        ins = unwrap_variables(bb.inputVariables_optimization)
        outs = unwrap_variables(bb.outputVariables_optimization)
        try:
            in_idx = [col[id(v)] for v in ins]
            out_idx = [col[id(v)] for v in outs]
        except KeyError:
            raise ValueError(
                f'grey-box block {block.name!r} references a variable that '
                'the structure detector did not record; cannot map it into '
                'the problem columns.')
        in_idx = np.asarray(in_idx, dtype=int)
        prefetch.pairs.append((bb, in_idx))

        # Directional rows per the declared operator (_lc_operators; absent
        # = all '==', the historical form). 'out >= bb' -> bb/out <= 1,
        # 'out <= bb' -> out/bb <= 1; one-sided, so no black-box equality
        # manifold -- the model's own pressure binds them at the optimum.
        ops = getattr(block, '_lc_operators', None)
        if not ops or len(ops) != len(out_idx):
            ops = ['=='] * len(out_idx)
        for k, iout in enumerate(out_idx):
            row = GreyboxRow(bb, in_idx, k, iout, ops[k] == '<=', n, prefetch)
            if ops[k] == '==':
                operator = '=='
            else:
                operator = '<='
            rows.append(Constraint(
                GreyboxSignomial(row.value_and_gradient, n, iout,
                                 fn_value=row.value),
                operator))
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

    objective = dense_posynomial(obj_terms, n, 'objective')

    constraints = []
    n_split = 0

    # ---- two-sided pin recognition (see docstring) ----------------------
    merged_eq, skip_idx = set(), set()
    if pair_equalities:
        merged_eq, skip_idx = find_equality_pairs(st)

    for idx in st.constraint_indices:
        if idx in skip_idx:
            continue
        if idx in merged_eq:
            op = '=='
        else:
            op = st.operator(idx)
        all_terms = st.terms(idx)
        num = [t for t in all_terms if not t.denominator]
        den = [t for t in all_terms if t.denominator]

        if den:
            p_ = dense_posynomial(num, n, f'constraint {idx} numerator')
            q_ = dense_posynomial(den, n, f'constraint {idx} denominator')
            if op == '==' and split_equalities is False:
                # One constraint, both sides condensed. The split pair pins
                # the step to a null space and makes the multipliers
                # degenerate; see CondensedEquality.
                constraints.append(
                    Constraint(CondensedEquality(p_, q_, n), '=='))
                n_split += 1
                continue
            constraints.append(Constraint(
                opaque_if_stock(PosynomialRatio(p_, q_, n), sp_form, n), '<='))
            if op == '==':
                # An equality is two inequalities: p/q <= 1 alone relaxes
                # the problem. AGM condensation is one-sided, but the
                # reverse q/p <= 1 is another ratio and condenses fine.
                constraints.append(Constraint(
                    opaque_if_stock(PosynomialRatio(q_, p_, n), sp_form, n),
                    '<='))
                n_split += 1
        else:
            body = dense_posynomial(num, n, f'constraint {idx}')
            if op == '==' and body.is_monomial:
                # A monomial equality is affine in log space: exact as is.
                constraints.append(Constraint(body, '=='))
            elif op == '==' and split_equalities is False:
                # One condensed equality, p_hat == 1, instead of the pair
                # p <= 1 plus 1/p <= 1, whose multipliers are just as
                # degenerate (910.8 vs 910.9 on SPaircraft).
                constraints.append(
                    Constraint(CondensedEquality(body, unit_monomial(n), n), '=='))
                n_split += 1
            elif op == '==':
                # Multi-term posynomial equality: p <= 1 goes in as is; the
                # reverse 1/p <= 1 is not a posynomial, so it goes in as a
                # condensed ratio. Dropping it relaxes the problem. Same
                # device PCCP uses in cvxopt/SP.py.
                constraints.append(Constraint(body, '<='))
                constraints.append(Constraint(
                    opaque_if_stock(PosynomialRatio(unit_monomial(n), body, n),
                                    sp_form, n), '<='))
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
        constraints.extend(greybox_rows(gb, structures, n))

    names = [str(v) for v in structures.get('variables', [])][:n]
    return Problem(n, objective, constraints, names=names or None,
                   bounds=bounds)


def dense_posynomial(terms, n, what):
    """Detected Terms -> a Posynomial over all n columns; a non-positive
    coefficient is the detector's mistake (it belongs in a denominator)."""
    bad = [t.coeff for t in terms if t.coeff <= 0]
    if bad:
        raise ValueError(
            f'{what} has non-positive coefficients {bad}; SLCP needs each '
            'posynomial part to be positive, so the detector should have '
            'moved these into a denominator')
    dense = []
    for t in terms:
        dense.append((t.coeff, [t.exponents.get(j, 0.0) for j in range(0, n)]))
    return Posynomial(dense, n)


def unit_monomial(n):
    """The monomial 1, for writing ``1/p <= 1``."""
    return Posynomial([(1.0, [0.0] * n)], n)


def opaque_if_stock(body, sp_form, n):
    """With sp_form off, hide a ratio behind a plain callback so SLCP
    linearizes it whole (stock SLCP, for paper comparisons)."""
    if sp_form or not isinstance(body, PosynomialRatio):
        return body
    return Signomial(lambda x, r=body: (r(x), r.grad(x)), n)


# -- two-sided pin recognition ---------------------------------------------
def round_coefficient(c):
    return float('%.11e' % c)


def round_exponent(e):
    return round(e, 11)


def fold_monomial_into(terms, mono):
    """Divide every term by a monomial: (c_t / c_m, exponents - mono's)."""
    cM = float(mono.coeff)
    eM = dict(mono.exponents)
    out = []
    for t in terms:
        ex = {}
        for j, e in t.exponents.items():
            ex[j] = float(e)
        for j, e in eM.items():
            if e:
                ex[j] = ex.get(j, 0.0) - float(e)
                if ex[j] == 0.0:
                    del ex[j]
        out.append((float(t.coeff) / cM, ex))
    return out


def canonical_sides(num, den):
    """Canonical (num, den) term-tuples for matching. The detector
    normalizes a pin's two directions differently, so fold any
    single-monomial side into the other side first."""
    if len(den) == 1:
        return fold_monomial_into(num, den[0]), []
    if len(num) == 1 and len(den) > 1:
        return [], fold_monomial_into(den, num[0])
    plain_num = []
    for t in num:
        plain_num.append((float(t.coeff),
                          {j: float(e) for j, e in t.exponents.items()}))
    plain_den = []
    for t in den:
        plain_den.append((float(t.coeff),
                          {j: float(e) for j, e in t.exponents.items()}))
    return plain_num, plain_den


def side_signature(folded):
    """A hashable, rounded signature of one side; the empty side is 1."""
    sig = []
    for c, ex in folded:
        expo = []
        for j, e in ex.items():
            if e != 0.0:
                expo.append((j, round_exponent(e)))
        sig.append((round_coefficient(c), tuple(sorted(expo))))
    sig = tuple(sorted(sig))
    if sig:
        return sig
    return ((1.0, ()),)


def monomial_row_fold(num, den):
    """Fold a mono-vs-mono (or mono-only) row to C x^A <= 1.
    Returns (A_sparse, C) or None."""
    if len(num) != 1 or len(den) > 1:
        return None
    cN, tN = float(num[0].coeff), num[0].exponents
    A = {}
    for j, e in tN.items():
        if e != 0.0:
            A[j] = float(e)
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


def find_equality_pairs(st):
    """Mutual-reverse row pairs (p <= q and q <= p, how EDI models write
    signomial equalities) -> (merged_eq, skip_idx): the first row of each
    pair becomes an ==, the second is dropped. Two matchers: (a) general
    posynomial pairs in the canonical single-mono-folded frame; (b)
    monomial pins, whose two directions fold to reciprocal rows C x^A <= 1
    and (1/C) x^-A <= 1 that the (num, den) swap can never see -- 151 of
    287 pin rows on the wind missions SP fell through (a) for that reason."""
    import math

    merged_eq, skip_idx = set(), set()
    seen = {}
    seen_mono = {}
    for idx in st.constraint_indices:
        if st.operator(idx) != '<=':
            continue
        all_t = st.terms(idx)
        num = [t for t in all_t if not t.denominator]
        den = [t for t in all_t if t.denominator]

        cn, cd = canonical_sides(num, den)
        key = (side_signature(cn), side_signature(cd))
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

        fold = monomial_row_fold(num, den)
        if fold is not None:
            A, C = fold
            negA = tuple(sorted((j, -e) for j, e in A))
            hit = seen_mono.get(negA)
            if (hit is not None and hit[1] != 0.0
                    and math.isclose(C * hit[1], 1.0, rel_tol=1e-9)
                    and hit[0] not in skip_idx
                    and hit[0] not in merged_eq):
                merged_eq.add(hit[0])
                skip_idx.add(idx)
                seen_mono[negA] = (hit[0], 0.0)  # consume
            else:
                seen_mono.setdefault(A, (idx, C))
    return merged_eq, skip_idx


def bound_row_block(structures):
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


def greybox_protected(structures):
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
            for v in unwrap_variables(list(bb.inputVariables_optimization)
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
    from lcsolver.presolve.reductions import presolve as presolve_pipeline

    st = dict(structures)
    if st.get('bounds') is None:
        key = ('Signomial_Program' if st['Signomial_Program'][0]
               else 'Geometric_Program')
        width = max((len(r) - 2 for r in st[key][1]), default=0)
        st['bounds'] = [(None, None)] * max(width,
                                            len(st.get('variables') or []))
    return presolve_pipeline(st, verbose=verbose, fold=fold,
                              fold_only=fold_only, eliminate=eliminate,
                              protect=protect)


def apply_presolve(structures, x0):
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
    protect = greybox_protected(structures)
    if protect is None:
        # Unmappable grey-box block: no removal is provably safe, and the
        # fold is barred too (see fold_bound_rows).
        return structures, x0, None, n_original
    if protect:
        # Grey-box present: column peel only, referenced columns protected.
        # No fold -- folding renumbers rows a grey-box row may reference.
        reduced, log = presolve_structures(structures, fold=False,
                                           eliminate=False, protect=protect)
    else:
        reduced, log = presolve_structures(
            structures, fold_only=bound_row_block(structures),
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


def restore_presolved(result, log, n_original):
    """Put presolved-away variables back, so callers see the original layout."""
    result.presolve = log
    result.removed = log.removed_variables if log is not None else []
    if log is not None and getattr(result, 'x', None) is not None:
        result.x = log.restore(result.x)
    return result

def fold_bound_rows(structures):
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
        structures, x0, log, n_original = apply_presolve(structures, x0)
    else:
        structures = fold_bound_rows(structures)

    problem = build_problem(structures, sp_form=sp_form)
    x0 = np.asarray(x0, dtype=float)
    if len(x0) < problem.n:
        # PCCP-style slack columns, if any, start at 1.
        x0 = np.concatenate([x0, np.ones(problem.n - len(x0))])
    x0 = np.where(x0 > 0, x0, 1.0)
    result = slcp_solve(problem, x0[:problem.n], method=method,
                         options=options)
    return restore_presolved(result, log, n_original)


def solve_sia(structures, x0=None, options=None, sp_form=True,
              presolve=True, split_equalities=False, pair_equalities=True):
    """Solve a detected GP/SP by sequential inner approximation.

    Same adapter as solve_slcp, pointed at sia.solve_sia. Keep sp_form=True:
    the conservative condensation is the basis of the method. Bound rows
    fold into native bounds either way; presolve=False disables only the
    column reductions, not the fold.
    """
    import pyomo.environ as pyo

    from lcsolver.solvers.sequential.sia import solve_sia as sia_solve

    if x0 is None:
        x0 = [float(pyo.value(v)) for v in structures['variables']]
    log, n_original = None, len(structures.get('variables') or [])
    if presolve:
        structures, x0, log, n_original = apply_presolve(structures, x0)
    else:
        structures = fold_bound_rows(structures)

    problem = build_problem(structures, sp_form=sp_form,
                            split_equalities=split_equalities,
                            pair_equalities=pair_equalities)
    x0 = np.asarray(x0, dtype=float)
    if len(x0) < problem.n:
        x0 = np.concatenate([x0, np.ones(problem.n - len(x0))])
    x0 = np.where(x0 > 0, x0, 1.0)
    result = sia_solve(problem, x0[:problem.n], options=options)
    return restore_presolved(result, log, n_original)


# older scripts in the lc* repos import these by their former names
_restore = restore_presolved
_apply_presolve = apply_presolve
