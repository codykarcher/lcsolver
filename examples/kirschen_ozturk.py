"""The Kirschen-Ozturk aircraft sizing problem, a signomial program.

Source
------
B. Ozturk and P. Kirschen's SimPleAC-style wing/fuel sizing model, as stated
in the SLCP paper, Equation 18. Eighteen variables sizing a wing (area,
aspect ratio, structure) and its fuel system against a range requirement.

Every constraint but one is GP-compatible. The exception is the fuel-volume
availability row

    V_f_avail <= V_f_wing + V_f_fuse

-- a monomial bounded below by a posynomial -- which is what makes this a
*signomial* program: it cannot be solved as a pure GP, but LCsolver's SP
machinery handles it directly.
"""

import numpy as np
from pyomo.environ import units

from lcsolver.objects.formulation import Formulation
from lcsolver.solvers.solver import solve

pi = np.pi

# =========
# Constants  (all SI base units)
# =========
C_Lmax, CDA0, e, g, k = 1.5, 0.06, 0.96, 9.81, 1.2
mu, N_ult, rho, rho_f = 1.78e-5, 2.5, 1.23, 804.0
R, S_wet, tau = 1.0e6, 2.05, 0.12
TSFC = 0.4 / 3600.0
V_min, W_0 = 22.0, 4940.0
W_c1, W_c2 = 8.71e-5, 45.24

# =====================
# Declare the Variables
# =====================
f = Formulation()
variables = [
    ('A',         10.0,    'aspect ratio'),
    ('C_D',       0.025,   'drag coefficient'),
    ('C_L',       0.5,     'lift coefficient'),
    ('C_f',       0.003,   'skin friction coefficient'),
    ('D',         300.0,   'drag force'),
    ('Re',        3.0e6,   'Reynolds number'),
    ('S',         10.0,    'wing area'),
    ('t',         6000.0,  'flight time'),
    ('V',         30.0,    'cruise speed'),
    ('V_f',       0.3,     'fuel volume'),
    ('V_f_avail', 0.3,     'fuel volume available'),
    ('V_f_fuse',  0.3,     'fuselage fuel volume'),
    ('V_f_wing',  0.3,     'wing fuel volume'),
    ('W',         10000.0, 'total weight'),
    ('W_f',       2500.0,  'fuel weight'),
    ('W_w',       2500.0,  'wing weight'),
    ('W_w_strc',  2500.0,  'wing structural weight'),
    ('W_w_surf',  2500.0,  'wing surface weight'),
]
for name, guess, desc in variables:
    f.Variable(name=name, guess=guess, units='', description=desc)
v = {name: f.find_component(name) for name, *_ in variables}

# =======================
# Objective / Constraints
# =======================
f.Objective(v['W_f'])

one = 1.0 * units.dimensionless
f.ConstraintList([
    # fuel burn and range
    v['W_f'] >= TSFC * v['t'] * v['D'],
    v['t'] >= R / v['V'],
    # drag
    v['D'] >= 0.5 * rho * v['S'] * v['C_D'] * v['V'] ** 2,
    v['C_D'] >= CDA0 / v['S'] + k * S_wet * v['C_f']
                + v['C_L'] ** 2 / (pi * v['A'] * e),
    v['C_f'] >= 0.074 / v['Re'] ** 0.2,
    mu * v['Re'] * v['A'] ** 0.5 / (v['V'] * v['S'] ** 0.5) <= one,
    # lift
    0.5 * rho * v['V'] ** 2 * v['S'] * v['C_L']
        >= W_0 + v['W_w'] + 0.5 * v['W_f'],
    v['W'] <= 0.5 * rho * V_min ** 2 * v['S'] * C_Lmax,
    # weight build-up
    v['W'] >= W_0 + v['W_w'] + v['W_f'],
    v['W_w'] >= v['W_w_surf'] + v['W_w_strc'],
    v['W_w_surf'] >= W_c2 * v['S'],
    v['W_w_strc'] ** 2 >= W_c1 ** 2 * N_ult ** 2 * v['A'] ** 3
        * (W_0 + v['V_f_fuse'] * rho_f * g) * v['W'] * v['S'] / tau ** 2,
    # fuel volume
    v['V_f'] <= v['V_f_avail'],
    v['V_f'] >= v['W_f'] / (g * rho_f),
    # the one signomial constraint: monomial <= posynomial
    v['V_f_avail'] <= v['V_f_wing'] + v['V_f_fuse'],
    v['V_f_wing'] ** 2 <= 0.0009 * v['S'] ** 3 * tau ** 2 / v['A'],
    v['V_f_fuse'] <= CDA0 * 10.0,
])

# =====
# Solve
# =====
solve(f)
print(f.solution)
