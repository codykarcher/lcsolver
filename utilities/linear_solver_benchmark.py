#!/usr/bin/env python3
"""Benchmark IPOPT linear solvers (ma27 / mumps / ...) on LCsolver models.

Runs the repo's examples plus a synthetic ill-conditioned GP family against
each requested linear solver, one row per (model, solver): status, wall
time, objective. Disagreements are the payload (MUMPS fails where MA27
succeeds is the argument for building IPOPT with HSL).

    python utilities/linear_solver_benchmark.py
    python utilities/linear_solver_benchmark.py --linear-solvers ma27,mumps \
        --executable ~/software/ipopt/build-mumps/bin/ipopt
    python utilities/linear_solver_benchmark.py --csv out.csv

The examples solve at import; harvesting stubs out lcsolver.solve so each
model is solved only under this script's control.
"""
import argparse
import csv
import importlib
import os
import sys
import time
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(REPO, 'examples')

# examples that build ONE formulation and call lcsolver.solve at module level;
# hoburg_blackbox and unit_circle_blackbox need derivative machinery irrelevant here
EXAMPLE_MODULES = (
    'aircraft_gp',
    'boyd',
    'cooling_loop_gp',
    'floudas',
    'gear_train_gp',
    'hoburg',
    'hydrogen_network_lp',
    'isolator_stack_qp',
    'kirschen_ozturk',
)


def harvest_formulation(module_name):
    """Import an example with lcsolver.solve stubbed; return its Formulation.
    The stub's return object absorbs whatever the example does with the result."""
    import lcsolver

    captured = []

    class _AbsorbEverything:
        def __call__(self, *a, **k):
            return self

        def __getattr__(self, name):
            return self

        def __getitem__(self, key):
            return self

        # without __iter__, a `for` over this object walks __getitem__ forever
        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

        def __format__(self, spec):
            return '0'

        def __float__(self):
            return 0.0

        def __str__(self):
            return ''

        # post-solve arithmetic in an example must absorb too
        def _binop(self, *a):
            return self
        __add__ = __radd__ = __sub__ = __rsub__ = _binop
        __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _binop
        __pow__ = __rpow__ = __neg__ = __pos__ = __abs__ = _binop
        __lt__ = __le__ = __gt__ = __ge__ = _binop

    def _stub(f, *args, **kwargs):
        captured.append(f)
        return _AbsorbEverything()

    real = lcsolver.solve
    if EXAMPLES not in sys.path:
        sys.path.insert(0, EXAMPLES)
    try:
        lcsolver.solve = _stub
        sys.modules.pop(module_name, None)
        importlib.import_module(module_name)
    finally:
        lcsolver.solve = real
        sys.modules.pop(module_name, None)
    if not captured:
        raise RuntimeError(f'{module_name} never called lcsolver.solve')
    return captured[0]


def illconditioned_gp(n=18, spread=8.0):
    """A GP whose log-space KKT system is deliberately badly scaled.

    A chain x[i+1] >= c * x[i]**p walks the variables across `spread` orders
    of magnitude, coupled to a constraint touching every variable with
    alternating-sign exponents. Conditioning grows with n and spread
    (SPaircraft-like: 1e-9 to 1e+9 in one factorization).
    """
    from lcsolver import Formulation

    f = Formulation()
    xs = []
    for i in range(n):
        guess = 10.0 ** (spread * (i / (n - 1) - 0.5))
        xs.append(f.Variable(f'x{i}', guess, '-', f'stage {i}',
                             bounds=[1e-12, 1e12]))
    f.Objective(sum(xs))
    rows = []
    p = 10.0 ** (spread / (n - 1))
    for i in range(n - 1):
        rows.append(xs[i + 1] >= p * xs[i])
    prod = 1.0
    for i, x in enumerate(xs):
        prod = prod * x if i % 2 == 0 else prod / x
    # the alternating product cannot exceed ~10**(-spread/2), so the floor must
    # sit below that or the model is infeasible; one decade of slack keeps it binding-ish
    rows.append(prod >= 10.0 ** (-spread / 2 - 1.0)
                * 10.0 ** (-spread / (2.0 * (n - 1))))
    rows.append(xs[0] >= 10.0 ** (-spread / 2))
    f.ConstraintList(rows)
    return f


def model_battery(include_synthetic=True):
    """(name, builder) pairs; builders return a fresh Formulation."""
    battery = [(name, (lambda nm=name: harvest_formulation(nm)))
               for name in EXAMPLE_MODULES]
    if include_synthetic:
        battery += [
            ('illcond_gp_mild', lambda: illconditioned_gp(12, 4.0)),
            ('illcond_gp_hard', lambda: illconditioned_gp(18, 8.0)),
            ('illcond_gp_extreme', lambda: illconditioned_gp(24, 12.0)),
        ]
    return battery


def _run_one_inprocess(model_name, linear_solver, library=None):
    """Child side of run_one: build and solve one cell, print a parsable RESULT line."""
    import json

    import lcsolver

    t0 = time.perf_counter()
    try:
        if model_name.startswith('illcond_gp'):
            n, spread = {'illcond_gp_mild': (12, 4.0),
                         'illcond_gp_hard': (18, 8.0),
                         'illcond_gp_extreme': (24, 12.0)}[model_name]
            f = illconditioned_gp(n, spread)
        else:
            f = harvest_formulation(model_name)
        from lcsolver.environment import linear_solver_library_option
        _lib = (library if library and
                linear_solver_library_option(linear_solver) else None)
        res = lcsolver.solve(f, linear_solver=linear_solver,
                             linear_solver_library=_lib)
        dt = time.perf_counter() - t0
        status = str(res.get('status', '?'))
        try:
            obj = float(res.objective)
        except Exception:
            obj = None
        out = {'ok': status in ('ok', 'optimal'), 'status': status,
               'time': dt, 'objective': obj, 'error': None}
    except Exception as exc:
        out = {'ok': False, 'status': 'raised',
               'time': time.perf_counter() - t0, 'objective': None,
               'error': f'{type(exc).__name__}: {str(exc)[:140]}'}
    print('RESULT ' + json.dumps(out))


def run_one(model_name, linear_solver, executable=None, timeout=120.0,
            library=None):
    """One (model, solver) cell in a fresh subprocess with a hard timeout.
    A divergent in-process IPOPT run is unkillable and holds memory; a child per cell can just be killed."""
    import json
    import subprocess

    env = dict(os.environ)
    if executable:
        env['LCSOLVER_IPOPT_EXECUTABLE'] = executable
    cmd = [sys.executable, os.path.abspath(__file__),
           '--_child', model_name, linear_solver]
    if library:
        cmd += ['--linear-solver-library', library]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env, cwd=REPO)
        for line in proc.stdout.splitlines():
            if line.startswith('RESULT '):
                return json.loads(line[len('RESULT '):])
        return {'ok': False, 'status': 'crashed',
                'time': time.perf_counter() - t0, 'objective': None,
                'error': (proc.stderr or proc.stdout or '')[-140:]}
    except subprocess.TimeoutExpired:
        return {'ok': False, 'status': f'timeout>{timeout:.0f}s',
                'time': timeout, 'objective': None, 'error': 'killed'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--linear-solvers', default='ma27,mumps',
                    help='comma-separated list to compare')
    ap.add_argument('--executable', default=None,
                    help='ipopt binary carrying the solvers (default: the '
                         'usual lcsolver resolution)')
    ap.add_argument('--csv', default=None, help='also write rows here')
    ap.add_argument('--no-synthetic', action='store_true')
    ap.add_argument('--models', default=None,
                    help='comma-separated subset of model names')
    ap.add_argument('--timeout', type=float, default=120.0,
                    help='hard per-cell timeout in seconds')
    ap.add_argument('--linear-solver-library', default=None,
                    help='shared library for runtime-loaded solvers '
                         '(hsllib for ma57/77/86/97, pardisolib for '
                         'pardiso); ignored for compiled-in solvers')
    ap.add_argument('--_child', nargs=2, metavar=('MODEL', 'SOLVER'),
                    help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args._child:
        _run_one_inprocess(*args._child,
                           library=args.linear_solver_library)
        return []

    solvers = [s.strip() for s in args.linear_solvers.split(',') if s.strip()]
    battery = model_battery(include_synthetic=not args.no_synthetic)
    if args.models:
        wanted = {m.strip() for m in args.models.split(',')}
        battery = [(n, b) for n, b in battery if n in wanted]

    rows = []
    width = max(len(n) for n, _ in battery)
    header = f'{"model":<{width}}  ' + '  '.join(f'{s:>18}' for s in solvers)
    print(header)
    print('-' * len(header))
    for name, builder in battery:
        cells = []
        for s in solvers:
            r = run_one(name, s, executable=args.executable,
                        timeout=args.timeout,
                        library=args.linear_solver_library)
            rows.append({'model': name, 'linear_solver': s, **r})
            cell = (f'{r["status"]}/{r["time"]:.1f}s' if r['ok']
                    else f'FAIL({r["status"]})')
            cells.append(f'{cell:>18}')
        print(f'{name:<{width}}  ' + '  '.join(cells))

    disagreements = {}
    for r in rows:
        disagreements.setdefault(r['model'], []).append(r)
    print()
    found = False
    for name, rs in disagreements.items():
        oks = {r['linear_solver']: r['ok'] for r in rs}
        if len(set(oks.values())) > 1:
            found = True
            good = [s for s, ok in oks.items() if ok]
            bad = [s for s, ok in oks.items() if not ok]
            print(f'DISAGREEMENT on {name}: {", ".join(bad)} fail where '
                  f'{", ".join(good)} succeed')
    if not found:
        print('no solver disagreements in this battery')

    if args.csv:
        with open(args.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'rows written to {args.csv}')
    return rows


if __name__ == '__main__':
    main()
