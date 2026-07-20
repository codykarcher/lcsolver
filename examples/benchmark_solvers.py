#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Compare EDI's solver backends on identical models.

Four routes are available for a problem that happens to be a geometric program:

``cvxopt (convex)``
    The default. Solves the log-transformed problem with cvxopt's cone solver.
``ipopt (convex)``
    The same log transform, but the transformed problem is declared in Pyomo and
    handed to IPOPT. Still convex, so the global-optimality guarantee holds.
``ipopt (raw NLP)``
    The model as written, in its natural variables, straight to IPOPT. This is
    *not* convex -- the guarantee is given up in exchange for not needing the
    problem to be a GP at all.
``SLCP``
    Sequential log-convex programming. Overkill for a pure GP, and included to
    show what the general machinery costs when the specialised route applies.

Because every route solves the same model, disagreement in the objective is a
correctness signal and the timing is a like-for-like cost comparison.

Usage::

    python examples/benchmark_solvers.py [--repeats N]
"""

import argparse
import statistics
import sys
import time
import warnings

sys.path.insert(0, __file__.rsplit('/', 2)[0])

warnings.filterwarnings('ignore')


def _time(fn, repeats):
    """Run ``fn`` ``repeats`` times; return (median seconds, result, error)."""
    times, result, err = [], None, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:                       # noqa: BLE001
            return float('nan'), None, f'{type(exc).__name__}: {exc}'
        times.append(time.perf_counter() - t0)
    return statistics.median(times), result, err


def bench_case(name, build_formulation, slcp_case=None, repeats=5):
    from edi.solvers.ipopt import ipopt_solve
    from edi.solvers.ipopt.convex import solve_gp_ipopt
    from edi.solvers.solver import cvxopt_solve
    from edi.structure.structureDetector import structure_detector
    from edi.units.unitCorrector import unit_corrector

    print(f'\n{"=" * 76}\n{name.upper()}\n{"=" * 76}')

    rows = []

    # --- cvxopt, log transform ------------------------------------------------
    def run_cvxopt():
        f = build_formulation()
        return cvxopt_solve(f)['primal objective']

    t, obj, err = _time(run_cvxopt, repeats)
    rows.append(('cvxopt (convex)', t, obj, err))

    # --- IPOPT, same log transform -------------------------------------------
    def run_ipopt_convex():
        f = build_formulation()
        s = structure_detector(unit_corrector(f))
        return solve_gp_ipopt(s, model=f)['primal objective']

    t, obj, err = _time(run_ipopt_convex, repeats)
    rows.append(('ipopt (convex)', t, obj, err))

    # --- IPOPT on the raw, non-convex model ----------------------------------
    def run_ipopt_raw():
        f = build_formulation()
        return ipopt_solve(f)['objective']

    t, obj, err = _time(run_ipopt_raw, repeats)
    rows.append(('ipopt (raw NLP)', t, obj, err))

    # --- SLCP -----------------------------------------------------------------
    if slcp_case is not None:
        from edi.solvers.ipopt import slcp as slcp_mod

        def run_slcp():
            problem, x0, _ = slcp_case()
            r = slcp_mod.solve(problem, x0, method='slcp',
                               options=slcp_mod.Options())
            return r.objective

        t, obj, err = _time(run_slcp, max(1, repeats // 2))
        rows.append(('SLCP', t, obj, err))

    # --- report ---------------------------------------------------------------
    ok = [r[2] for r in rows if r[2] is not None and r[3] is None]
    best = min((r[1] for r in rows if r[1] == r[1]), default=float('nan'))

    print(f'{"backend":<20s} {"time [ms]":>11s} {"vs fastest":>11s} '
          f'{"objective":>18s}  notes')
    print('-' * 76)
    for label, t, obj, err in rows:
        if err:
            print(f'{label:<20s} {"--":>11s} {"--":>11s} {"ERROR":>18s}  {err[:28]}')
            continue
        rel = f'{t / best:>10.2f}x' if best == best and best > 0 else '--'
        print(f'{label:<20s} {t * 1000:>11.2f} {rel:>11s} {obj:>18.8f}')

    if len(ok) > 1:
        spread = (max(ok) - min(ok)) / abs(max(ok, key=abs))
        verdict = 'agree' if spread < 1e-5 else 'DISAGREE'
        print(f'\nobjective spread across backends: {spread:.2e}  ({verdict})')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repeats', type=int, default=5)
    args = ap.parse_args()

    from examples.slcp_cases import hoburg, kirschen_ozturk_gp, simple_example
    from examples.slcp_formulations import hoburg_gp, ko_gp, simple_gp

    print('Every backend below solves the SAME model. Objectives must agree;')
    print('the timings are therefore a like-for-like cost comparison.')
    print(f'Median of {args.repeats} runs.')

    def ko_slcp():
        problem, x0, _ = kirschen_ozturk_gp()
        return problem, x0, None

    bench_case('simple (2 var, 1 con)', simple_gp,
               slcp_case=simple_example, repeats=args.repeats)
    bench_case('kirschen-ozturk (18 var, 17 con)', ko_gp,
               slcp_case=ko_slcp, repeats=args.repeats)
    bench_case('hoburg (61 var, 58 con)', hoburg_gp,
               slcp_case=lambda: hoburg(0), repeats=args.repeats)

    print(f'\n{"=" * 76}')
    print('Why the raw-NLP route struggles as the model grows')
    print('=' * 76)
    print("""
The natural variables of these models span many orders of magnitude -- at the
Hoburg optimum, from I_cap_bar ~ 2e-5 to Re_2 ~ 1e7, a range of 5.1e11 (11.7
decades). A general NLP solver has to cope with that conditioning directly, and
here IPOPT hits its iteration limit and fails.

The log transform maps that same range onto a span of 27 in the transformed
variables, and simultaneously makes the problem convex. Both convex backends
solve it in tens of milliseconds. That is the argument for detecting GP structure
rather than handing every model to a general solver: it is not only about the
global-optimality guarantee, it is about being solvable at all.
""".strip())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
