#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Demonstrate SLCP-on-IPOPT against the LSQP and SQP baselines.

Usage::

    python utilities/run_slcp.py            # all implemented cases
    python utilities/run_slcp.py simple     # one case
    python utilities/run_slcp.py --trend    # small multi-start trend check

Each case runs from a single starting point, which is enough to demonstrate
feasibility. The paper's published results are averages over 1000 random starts
per band, so single-start iteration counts here are indicative, not a
reproduction of the study.
"""

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 2)[0])

from lcsolver.solvers.sequential import slcp                      # noqa: E402
from slcp_cases import CASES                   # noqa: E402

METHODS = ('slcp', 'lsqp', 'sqp')


def gp_reference(problem):
    """Solve the problem exactly as a GP, when it happens to be one.

    Only valid if the objective and every constraint are posynomial/monomial.
    Gives an independent optimum rather than relying on a published number.
    """
    import math

    import pyomo.environ as pyo

    from lcsolver.solvers.sequential.slcp import Posynomial

    if not isinstance(problem.objective, Posynomial):
        return None
    if not all(c.exact_in_logspace for c in problem.constraints):
        return None

    n = problem.n
    m = pyo.ConcreteModel()
    m.J = pyo.RangeSet(0, n - 1)
    m.t = pyo.Var(m.J, initialize=0.0)

    def posy(terms):
        return sum(pyo.exp(math.log(c) + sum(a[j] * m.t[j] for j in range(n)))
                   for c, a in terms)

    m.obj = pyo.Objective(expr=posy(problem.objective.terms), sense=pyo.minimize)
    m.cons = pyo.ConstraintList()
    for con in problem.constraints:
        if con.operator == '==':
            m.cons.add(posy(con.body.terms) == 1.0)
        else:
            m.cons.add(posy(con.body.terms) <= 1.0)

    opt = pyo.SolverFactory('ipopt')
    opt.options['print_level'] = 0
    opt.options['sb'] = 'yes'
    opt.solve(m)
    x = np.array([math.exp(pyo.value(m.t[j])) for j in range(n)])
    return {'x': x, 'objective': problem.objective_value(x),
            'source': 'solved exactly as a GP'}


def run_case(name, builder):
    problem, x0, reference = builder()

    # Prefer an independently computed optimum over a transcribed one.
    truth = gp_reference(problem) or reference
    source = (truth or {}).get('source', 'unspecified')

    print(f'\n{"=" * 78}\n{name.upper()}   '
          f'({problem.n} variables, {len(problem.constraints)} constraints)\n{"=" * 78}')
    print(f'start point   f(x0) = {problem.objective_value(x0):.8f}')
    if truth:
        print(f'reference     f*    = {truth["objective"]:.8f}   [{source}]')
    print()
    print(f'{"method":<7s} {"itr":>5s} {"converged":>10s} {"objective":>16s} '
          f'{"rel. err":>10s} {"max x err":>10s} {"time":>8s}')
    print('-' * 78)

    results = {}
    for method in METHODS:
        t0 = time.time()
        try:
            r = slcp.solve(problem, x0, method=method, options=slcp.Options())
        except Exception as exc:                          # noqa: BLE001
            print(f'{method:<7s} {"--":>5s} {"ERROR":>10s}   '
                  f'{type(exc).__name__}: {exc}')
            continue
        dt = time.time() - t0
        results[method] = r

        if truth:
            ferr = abs(r.objective - truth['objective']) / abs(truth['objective'])
            xerr = float(np.max(np.abs(r.x - truth['x']) / np.abs(truth['x'])))
            errs = f'{ferr:>10.2e} {xerr:>10.2e}'
        else:
            errs = f'{"--":>10s} {"--":>10s}'

        print(f'{method:<7s} {r.iterations:>5d} {str(r.converged):>10s} '
              f'{r.objective:>16.8f} {errs} {dt:>7.2f}s')
        if not r.converged:
            print(f'        {r.status}')

    return results


def run_trend(name, builder, n_starts=12, seed=0):
    """Small multi-start check. NOT the paper's study, which uses 1000 per band."""
    problem, _, reference = builder()
    truth = gp_reference(problem) or reference
    if truth is None:
        print(f'{name}: no reference optimum, skipping trend check')
        return

    rng = np.random.default_rng(seed)
    xopt = truth['x']
    print(f'\n{name.upper()} -- {n_starts} random starts per band '
          f'(indicative only; the paper uses 1000)')
    print(f'{"band":>6s} ' + ' '.join(f'{m.upper() + " ok":>10s} {m.upper() + " itr":>9s}'
                                      for m in METHODS))
    for band in (0.10, 0.50, 0.80):
        cells = []
        for method in METHODS:
            ok, iters = 0, []
            for _ in range(n_starts):
                x0 = xopt * (1.0 + rng.uniform(-band, band, size=len(xopt)))
                try:
                    r = slcp.solve(problem, x0, method=method, options=slcp.Options())
                except Exception:                          # noqa: BLE001
                    continue
                if (r.converged and abs(r.objective - truth['objective'])
                        / abs(truth['objective']) < 1e-3):
                    ok += 1
                    iters.append(r.iterations)
            cells.append(f'{ok:>7d}/{n_starts:<2d}')
            cells.append(f'{np.median(iters):>9.1f}' if iters else f'{"--":>9s}')
        print(f'{int(band * 100):>5d}% ' + ' '.join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cases', nargs='*', default=None,
                    help=f'cases to run (default: all). Available: {", ".join(CASES)}')
    ap.add_argument('--trend', action='store_true',
                    help='also run a small multi-start trend check')
    args = ap.parse_args()

    names = args.cases or list(CASES)
    for name in names:
        if name not in CASES:
            print(f'unknown case {name!r}; available: {", ".join(CASES)}')
            return 1
        run_case(name, CASES[name])
        if args.trend:
            run_trend(name, CASES[name])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
