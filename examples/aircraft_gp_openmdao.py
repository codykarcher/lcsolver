# ===========
# Description
# ===========
# The aircraft GP of aircraft_gp.py written the way an OpenMDAO user would
# write it, and the two solved side by side on the same problem.
#
# The GP states 15 variables and 13 rows. It is ported twice:
#
#   1. The MDO formulation. A, S and V are the design variables and the rest
#      is an explicit chain: the stall row fixes W from S, level flight fixes
#      C_L, the drag build-up and the wing weight follow. Every GP inequality
#      that is tight at the optimum becomes an assignment in that chain and
#      the one that is not forced, W >= W_0 + W_w, is the single constraint.
#      Nothing is held fixed; the design space is three-dimensional because
#      the modeller already knows which rows go tight.
#   2. The literal port. All 15 variables are design variables and the 13
#      rows are constraints in the GP's own form, rhs / lhs <= 1 (== 1 for
#      the equalities), scaled by the initial guesses. This is what an
#      optimizer sees when nobody has done that reduction by hand.
#
# SLSQP from the same initial guesses reaches the lcsolver optimum both ways.
# What differs is what each solve knows about it:
#   - the GP is convex in log space, so lcsolver's answer is the global
#     optimum; SLSQP reaches a KKT point from a starting guess. Drop the
#     scaling from port 2 and it reports success at a point 1.7% above the
#     optimum with Re parked at its guess (the rows span 1e-3 to 1e6). The GP
#     has no scaling knob: the log transform is the scaling.
#   - lcsolver reads the sensitivity of the optimum to every Constant off the
#     KKT system of the one solve; here each one is a central difference over
#     two re-optimizations.
#
# Needs openmdao (pip install openmdao); scipy's SLSQP is the optimizer.

# =================
# Import Statements
# =================
import os
import io
import time
import runpy
import contextlib

import numpy as np
import openmdao.api as om

HERE = os.path.dirname(os.path.abspath(__file__))

# =====================
# The Data
# =====================
# The constants are inputs rather than hard-coded so the sensitivity study
# below can perturb them. The guesses are aircraft_gp.py's.

CONSTANTS = {
    # name      value    openmdao units
    'C_Lmax': (2.0,     None),
    'CDA0':   (0.0306,  'm**2'),
    'e':      (0.96,    None),
    'k':      (1.2,     None),
    'mu':     (1.78e-5, 'kg/(m*s)'),
    'N_ult':  (2.5,     None),
    'rho':    (1.23,    'kg/m**3'),
    'Srat':   (2.05,    None),
    'tau':    (0.12,    None),
    'V_min':  (22.0,    'm/s'),
    'W_0':    (4940.0,  'N'),
    'W_c1':   (8.71e-5, '1/m'),
    'W_c2':   (45.24,   'Pa'),
}

GUESS = {
    'A': 10.0, 'C_D': 0.025, 'C_D_fuse': 0.01, 'C_D_wpar': 0.01, 'C_D_ind': 0.01,
    'C_f': 0.003, 'C_L': 0.5, 'D': 300.0, 'Re': 3e6, 'S': 10.0, 'V': 30.0,
    'W': 10000.0, 'W_w': 2500.0, 'W_w_strc': 1500.0, 'W_w_surf': 1000.0,
}

SLSQP = dict(optimizer='SLSQP', tol=1e-12, maxiter=200, disp=False)


# =====================
# Port 1, the Analysis
# =====================
class Aircraft(om.ExplicitComponent):
    def setup(self):
        self.add_input('A', 10.0)
        self.add_input('S', 10.0, units='m**2')
        self.add_input('V', 30.0, units='m/s')
        for name, (value, unit) in CONSTANTS.items():
            self.add_input(name, value, units=unit)

        self.add_output('W',        units='N')
        self.add_output('C_L')
        self.add_output('Re')
        self.add_output('C_f')
        self.add_output('C_D_fuse')
        self.add_output('C_D_wpar')
        self.add_output('C_D_ind')
        self.add_output('C_D')
        self.add_output('D',        units='N')
        self.add_output('W_w_strc', units='N')
        self.add_output('W_w_surf', units='N')
        self.add_output('W_w',      units='N')
        self.add_output('weight_margin', units='N')   # W - W_0 - W_w >= 0

        self.declare_partials('*', '*', method='cs')

    def compute(self, i, o):
        A, S, V = i['A'], i['S'], i['V']

        # stall speed sets the weight the wing must carry, level flight the C_L
        o['W']   = 0.5 * i['rho'] * S * i['C_Lmax'] * i['V_min']**2
        o['C_L'] = o['W'] / (0.5 * i['rho'] * S * V**2)

        # drag build-up
        o['Re']       = (i['rho'] / i['mu']) * V * (S / A)**0.5
        o['C_f']      = 0.074 / o['Re']**0.2
        o['C_D_fuse'] = i['CDA0'] / S
        o['C_D_wpar'] = i['k'] * o['C_f'] * i['Srat']
        o['C_D_ind']  = o['C_L']**2 / (np.pi * A * i['e'])
        o['C_D']      = o['C_D_fuse'] + o['C_D_wpar'] + o['C_D_ind']
        o['D']        = 0.5 * i['rho'] * S * o['C_D'] * V**2

        # wing weight, and the closure the optimizer must respect
        o['W_w_strc'] = i['W_c1'] * i['N_ult'] * A**1.5 * (i['W_0'] * o['W'] * S)**0.5 / i['tau']
        o['W_w_surf'] = i['W_c2'] * S
        o['W_w']      = o['W_w_strc'] + o['W_w_surf']
        o['weight_margin'] = o['W'] - i['W_0'] - o['W_w']


def build_reduced():
    p = om.Problem(reports=False)
    p.model.add_subsystem('aircraft', Aircraft(), promotes=['*'])

    p.model.add_design_var('A', lower=1.0,  upper=50.0,  ref=10.0)
    p.model.add_design_var('S', lower=1.0,  upper=100.0, ref=10.0)
    p.model.add_design_var('V', lower=10.0, upper=200.0, ref=30.0)
    p.model.add_objective('D', ref=300.0)
    p.model.add_constraint('weight_margin', lower=0.0, ref=CONSTANTS['W_0'][0])

    p.driver = om.ScipyOptimizeDriver(**SLSQP)
    p.setup(force_alloc_complex=True)
    return p


# =====================
# Port 2, the Rows
# =====================
# Each GP row lhs >= rhs (or ==) becomes the output rhs / lhs, constrained
# <= 1 (or == 1). Written in the order of aircraft_gp.py.

ROWS = [
    # name        kind   lhs         rhs
    ('drag',      '>=', 'D',        lambda v, c: 0.5 * c['rho'] * v['S'] * v['C_D'] * v['V']**2),
    ('cd',        '>=', 'C_D',      lambda v, c: v['C_D_fuse'] + v['C_D_wpar'] + v['C_D_ind']),
    ('cd_fuse',   '>=', 'C_D_fuse', lambda v, c: c['CDA0'] / v['S']),
    ('cd_wpar',   '>=', 'C_D_wpar', lambda v, c: c['k'] * v['C_f'] * c['Srat']),
    ('cd_ind',    '>=', 'C_D_ind',  lambda v, c: v['C_L']**2 / (np.pi * v['A'] * c['e'])),
    ('re',        '==', 'Re',       lambda v, c: (c['rho'] / c['mu']) * v['V'] * (v['S'] / v['A'])**0.5),
    ('cf',        '==', 'C_f',      lambda v, c: 0.074 / v['Re']**0.2),
    ('ww',        '>=', 'W_w',      lambda v, c: v['W_w_surf'] + v['W_w_strc']),
    ('ww_strc',   '>=', 'W_w_strc', lambda v, c: c['W_c1'] * c['N_ult'] * v['A']**1.5
                                                 * (c['W_0'] * v['W'] * v['S'])**0.5 / c['tau']),
    ('ww_surf',   '>=', 'W_w_surf', lambda v, c: c['W_c2'] * v['S']),
    ('w',         '>=', 'W',        lambda v, c: c['W_0'] + v['W_w']),
    ('level',     '==', 'W',        lambda v, c: 0.5 * c['rho'] * v['S'] * v['C_L'] * v['V']**2),
    ('stall',     '==', 'W',        lambda v, c: 0.5 * c['rho'] * v['S'] * c['C_Lmax'] * c['V_min']**2),
]


class AircraftRows(om.ExplicitComponent):
    def setup(self):
        for name, guess in GUESS.items():
            self.add_input(name, guess)
        for name, (value, unit) in CONSTANTS.items():
            self.add_input(name, value)
        for name, kind, lhs, rhs in ROWS:
            self.add_output(name)
        self.declare_partials('*', '*', method='cs')

    def compute(self, i, o):
        for name, kind, lhs, rhs in ROWS:
            o[name] = rhs(i, i) / i[lhs]


def build_literal(scaled=True):
    p = om.Problem(reports=False)
    p.model.add_subsystem('rows', AircraftRows(), promotes=['*'])

    for name, guess in GUESS.items():
        p.model.add_design_var(name, lower=1e-3 * guess, upper=1e3 * guess,
                               ref=guess if scaled else 1.0)
    p.model.add_objective('D', ref=GUESS['D'] if scaled else 1.0)
    for name, kind, lhs, rhs in ROWS:
        if kind == '>=':
            p.model.add_constraint(name, upper=1.0)
        else:
            p.model.add_constraint(name, equals=1.0)

    p.driver = om.ScipyOptimizeDriver(**SLSQP)
    p.setup(force_alloc_complex=True)
    return p


# =====================
# Solving
# =====================
def optimize(p):
    with contextlib.redirect_stdout(io.StringIO()):
        result = p.run_driver()
    if not result.success:
        raise RuntimeError('SLSQP did not converge')
    return float(p.get_val('D')[0])


def log_sensitivities(p, D_star, h=1e-3):
    """d(log D*)/d(log c) for each constant, central difference over two
    re-optimizations, warm-started from the optimum."""
    sens = {}
    for name, (value, unit) in CONSTANTS.items():
        p.set_val(name, value * (1 + h), units=unit)
        D_plus = optimize(p)
        p.set_val(name, value * (1 - h), units=unit)
        D_minus = optimize(p)
        p.set_val(name, value, units=unit)
        sens[name] = (D_plus - D_minus) / (2 * h * D_star)
    optimize(p)   # back to the nominal optimum
    return sens


t0 = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    sol = runpy.run_path(os.path.join(HERE, 'aircraft_gp.py'))['sol']
t_lcs = time.perf_counter() - t0

t0 = time.perf_counter()
reduced = build_reduced()
D_star = optimize(reduced)
t_reduced = time.perf_counter() - t0

t0 = time.perf_counter()
literal = build_literal()
optimize(literal)
t_literal = time.perf_counter() - t0

t0 = time.perf_counter()
om_sens = log_sensitivities(reduced, D_star)
t_sens = time.perf_counter() - t0

# =====================
# Compare
# =====================
lcs_vars = {k: float(getattr(v, 'magnitude', v)) for k, v in sol.variables().items()}
lcs_sens = {k: float(v) for k, v in sol.sensitivities().items()}

print(f'lcsolver            {t_lcs:6.3f} s  (build, presolve, solve, sensitivities)')
print(f'OpenMDAO, 3 DVs     {t_reduced:6.3f} s  SLSQP, {reduced.driver.iter_count} evaluations; '
      f'+ {t_sens:.3f} s for {2 * len(CONSTANTS)} re-optimizations (sensitivities)')
print(f'OpenMDAO, 15 DVs    {t_literal:6.3f} s  SLSQP, {literal.driver.iter_count} evaluations')
print()
print(f'{"variable":10s} {"lcsolver":>12s} {"3 DVs":>12s} {"rel":>8s} {"15 DVs":>12s} {"rel":>8s}')
for name in lcs_vars:
    a = lcs_vars[name]
    b = float(reduced.get_val(name)[0])
    c = float(literal.get_val(name)[0])
    print(f'{name:10s} {a:12.6g} {b:12.6g} {abs(a - b) / abs(a):8.1e} '
          f'{c:12.6g} {abs(a - c) / abs(a):8.1e}')
print()
print(f'{"constant":10s} {"lcsolver":>10s} {"OpenMDAO":>10s}   (d log D* / d log c)')
for name in sorted(lcs_sens, key=lambda n: -abs(lcs_sens[n])):
    print(f'{name:10s} {lcs_sens[name]:+10.4f} {om_sens[name]:+10.4f}')
