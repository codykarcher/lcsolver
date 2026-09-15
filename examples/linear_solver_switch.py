# ===========
# Description
# ===========
# Selecting IPOPT's inner linear solver, and why it matters.
#
# Every IPOPT iteration solves one KKT system, and WHICH sparse factorization
# does it -- MUMPS (shipped default in many builds), HSL MA27, Pardiso, ... --
# is invisible until it isn't: on a hard model the pivoting strategy decides
# whether IPOPT gets a usable step at all. LCsolver exposes the choice as
#
#     lcsolver.solve(f, linear_solver='ma27')      # or 'mumps', 'pardiso', ...
#
# validated by name before any backend runs and probed against the IPOPT
# build actually in use, on every route (raw NLP, log-space GP, SIA and PCCP
# sub-problems alike).
#
# Part 2 replays a real disagreement. The bundled model
# (data/d8_sia_subproblem.nl) is the log-space SIA sub-problem at iteration 2
# of the SPaircraft D8.2 deck (York, Ozturk, Burnell & Hoburg, AIAA J.,
# DOI 10.2514/1.J057020; 1172 variables) at SIA's own tolerances. Measured
# on Ipopt 3.14.20 with MUMPS 5.9.1 vs HSL MA27 (macOS arm64, 2026-09-15):
#
#     mumps:  "Converged to a point of local infeasibility"  <- FALSE: the
#             sub-problem is feasible, and the SIA run it came from then
#             stopped uncertified at 25,102 lbf -- 17% above the answer
#     ma27:   "Optimal Solution Found", and the full D8 run certifies at
#             179 iterations / 21,384.0 lbf (gpkit reference 20,859.7)
#
# The whole aircraft solve fails or succeeds on this one factorization
# choice, which is why the switch exists and why LCsolver's install
# tooling builds IPOPT with MA27.

# =================
# Import Statements
# =================
import os

import pyomo.environ as pyo

import lcsolver
from lcsolver import Formulation, units
from lcsolver.environment import ipopt_executable, linear_solver_available

HERE = os.path.dirname(os.path.abspath(__file__))
SUBPROBLEM_NL = os.path.join(HERE, 'data', 'd8_sia_subproblem.nl')

# The exact IPOPT options the SIA loop was running with when the sub-problem
# was captured. At looser tolerances MUMPS gets away with it; at the loop's
# own 1e-12 the difference in factorization accuracy surfaces.
SIA_TOLERANCES = {'tol': 1e-12, 'constr_viol_tol': 1e-12,
                  'acceptable_constr_viol_tol': 1e-10}


def part1_the_switch():
    """The API: same model, explicitly chosen linear solvers."""
    print('=' * 70)
    print('Part 1: selecting the linear solver')
    print('=' * 70)

    exe = ipopt_executable()
    for name in ('ma27', 'mumps'):
        if not linear_solver_available(name, exe):
            print(f'  {name:<6}: not in this IPOPT build, skipped')
            continue
        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='-', description='x',
                       bounds=[0.1, 10.0])
        y = f.Variable(name='y', guess=1.0, units='-', description='y',
                       bounds=[0.1, 10.0])
        f.Objective(x + y)
        f.Constraint(x * y >= 1.0 * units.dimensionless)
        res = lcsolver.solve(f, linear_solver=name)
        print(f'  {name:<6}: status={res["status"]}, '
              f'objective={float(res.objective):.6f}')


def part2_the_disagreement():
    """Replay the D8 SIA sub-problem on every available linear solver."""
    print()
    print('=' * 70)
    print('Part 2: the SPaircraft D8 sub-problem (see header comment)')
    print('=' * 70)

    exe = ipopt_executable()
    for name in ('mumps', 'ma27'):
        if not linear_solver_available(name, exe):
            print(f'  {name:<6}: not in this IPOPT build, skipped')
            continue
        opt = pyo.SolverFactory('ipopt', executable=exe)
        for k, v in SIA_TOLERANCES.items():
            opt.options[k] = v
        opt.options['linear_solver'] = name
        results = opt.solve(SUBPROBLEM_NL, tee=False, load_solutions=False)
        tc = str(results.solver.termination_condition)
        verdict = ('FALSELY declared infeasible' if tc == 'infeasible'
                   else tc)
        print(f'  {name:<6}: termination_condition={tc}  ->  {verdict}')


if __name__ == '__main__':
    part1_the_switch()
    part2_the_disagreement()
