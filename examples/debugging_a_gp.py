# ===========
# Description
# ===========
# Debugging a geometric program with the pre-solve chain, tool by tool.
#
# `lcsolver.solve(f)` runs a chain -- unit correction, structure detection,
# pre-solve checks, then a backend -- and when a model is broken, the right
# move is to run the failing stage BY ITSELF and read its report, rather than
# solving repeatedly and deciphering tracebacks. This example breaks one
# small wing-sizing GP three different ways and lets each tool catch its own
# class of mistake, then solves the fixed model and reads everything back.
#
# The tools, in the order the chain runs them:
#
#     lcsolver.unit_check(f)      units balance? (nothing else means anything
#                                 before this passes)
#     f.structure_report()        WHAT is this problem, and which constraints
#                                 stop it being something simpler?
#     f.optimization_check()      structure + presolve in one report: unused
#                                 or unbounded variables, infeasible-as-
#                                 written rows, degenerate pairs
#     lcsolver.solve(f)           quiet by default: every warning the solve
#                                 raised is in sol.messages, and the solution
#                                 object carries the sensitivities
#
# Every stage below prints what the tool says, so running this file is the
# walkthrough.

# =================
# Import Statements
# =================
import lcsolver
from lcsolver import Formulation, units
from lcsolver.presolve.unitCorrector import unit_check


def build(wrong_units=False, zero_bound=False, dangling=False,
          sp_slip=False):
    """One wing-sizing GP, with four optional mistakes to demonstrate."""
    f = Formulation()

    S    = f.Variable(name='S',    guess=10.0, units='m^2',
                      description='Wing area')
    AR   = f.Variable(name='AR',   guess=8.0,  units='-',
                      description='Aspect ratio',
                      # MISTAKE 3: a zero lower bound. A log-space variable
                      # is strictly positive; zero is not representable.
                      bounds=[0.0 if zero_bound else 1.0, 20.0])
    C_L  = f.Variable(name='C_L',  guess=0.5,  units='-',
                      description='Lift coefficient')
    C_D  = f.Variable(name='C_D',  guess=0.02, units='-',
                      description='Drag coefficient')
    D    = f.Variable(name='D',    guess=300.0, units='N',
                      description='Drag force')
    if dangling:
        # MISTAKE 2: declared, then never used in any constraint. The
        # optimizer cannot price it and presolve refuses the model.
        f.Variable(name='Re', guess=1e6, units='-',
                   description='Reynolds number (never constrained!)')

    rho  = f.Constant(name='rho', value=1.2,   units='kg/m^3',
                      description='Air density')
    V    = f.Constant(name='V',   value=40.0,  units='m/s',
                      description='Cruise speed')
    W    = f.Constant(name='W',   value=5000.0, units='N',
                      description='Weight to lift')
    CD0  = f.Constant(name='CD0', value=0.01,  units='-',
                      description='Parasite drag coefficient')
    e    = f.Constant(name='e',   value=0.9,   units='-',
                      description='Oswald efficiency')

    f.Objective(D)

    q = 0.5 * rho * V**2
    lift = (q * S * C_L if not wrong_units
            # MISTAKE 1: forgetting the dynamic pressure. The right-hand
            # side is then an AREA times a coefficient -- not a force.
            else S * C_L)
    f.ConstraintList([
        W <= lift,
        # MISTAKE 4: writing the drag build-up as an EQUALITY. A GP admits
        # only monomial equalities; a posynomial one makes the model an SP.
        # As an inequality it costs nothing -- drag never helps, so the
        # optimizer holds it tight anyway.
        (C_D == CD0 + C_L**2 / (3.14159 * e * AR)) if sp_slip
        else (C_D >= CD0 + C_L**2 / (3.14159 * e * AR)),
        D >= q * S * C_D,
        AR <= 12.0 * units.dimensionless,
    ])
    return f


# ======================================================================
# Stage 1: units. Nothing downstream is meaningful until this passes.
# ======================================================================
print('=' * 70)
print('Stage 1: unit_check on a model missing its dynamic pressure')
print('=' * 70)
check = unit_check(build(wrong_units=True), raise_on_error=False)
print(check.summary())
# The report names the constraint, prints both sides' units, and says what
# factor the short side is missing -- here, exactly the dynamic pressure
# [kg/m^3][m/s]^2 that was dropped.

# ======================================================================
# Stage 2: structure. What is this problem, and what would make it simpler?
# ======================================================================
print()
print('=' * 70)
print('Stage 2a: structure_report on a model with a posynomial equality')
print('=' * 70)
print(build(sp_slip=True).structure_report())
# The report says SP, and the blockers table names constraint_2 as a
# posynomial equality -- one relaxed operator away from a GP. This is THE
# classic GP slip: the equality costs a global-optimality guarantee and a
# convex single solve, and buys nothing the inequality does not give.

print()
print('=' * 70)
print('Stage 2b: the detector itself, on a model with a zero lower bound')
print('=' * 70)
# structure_detector is what solve() actually routes on, and its blockers
# list says WHY a class was refused. A zero lower bound is the sneakiest
# way to lose GP/SP classification -- a log-space variable is strictly
# positive, so bounds=[0.0, ...] is unrepresentable, and on a grey-box
# model the resulting reroute to the raw IPOPT path discards SIAOptions
# (solve() warns LC-W207 when that happens).
from lcsolver.presolve.structureDetector import structure_detector  # noqa: E402
from lcsolver.presolve.unitCorrector import unit_corrector          # noqa: E402
st = structure_detector(unit_corrector(build(zero_bound=True)))
print('detected as GP:', st['Geometric_Program'][0])
print('detected as SP:', st['Signomial_Program'][0])
for name, reason, _row in st['blockers']['Geometric_Program']:
    print(f'  blocked by {name}:')
    print(f'      {reason}')

# ======================================================================
# Stage 3: presolve. Structural nonsense that units and class cannot see.
# ======================================================================
print()
print('=' * 70)
print('Stage 3: optimization_check on a model with a dangling variable')
print('=' * 70)
report = build(dangling=True).optimization_check()
print(report)
# 'Re' appears in no constraint, so no finite optimum can price it; the
# presolve section says so (and would also flag missing bounds, rows that
# are infeasible as written, and degenerate constraint pairs).

# ======================================================================
# Stage 4: the fixed model -- solve, and read everything back.
# ======================================================================
print()
print('=' * 70)
print('Stage 4: the fixed model')
print('=' * 70)
f = build()
sol = lcsolver.solve(f)
print(sol.summary())
# solve() runs quiet: anything it would have warned -- holographic
# constraints that came out active, unreliable sensitivities, backend
# fallbacks -- is in sol.messages rather than lost to a scrolled terminal.
print('messages captured during the solve:', sol['messages'] or 'none')
