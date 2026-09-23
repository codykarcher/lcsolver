# ===========
# Description
# ===========
# The aircraft GP of aircraft_gp.py written the way an OpenMDAO user would
# write it, and the two solved side by side on the same problem.
#
# The GP states 15 variables and 13 rows. The MDO formulation keeps the three
# quantities the designer chooses -- A, S, V -- as design variables and
# computes the rest in an explicit chain: the stall row fixes W from S, level
# flight fixes C_L, the drag build-up and the wing weight follow. Every GP
# inequality that is tight at the optimum becomes an assignment in that chain
# and the one that is not forced, W >= W_0 + W_w, is the single constraint.
# The objective is the drag.
#
# SLSQP from the same initial guesses reaches the lcsolver optimum. What
# differs is what each solve knows about it:
#   - the GP is convex in log space, so lcsolver's answer is the global
#     optimum; SLSQP reaches a KKT point from a starting guess.
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
# The Analysis
# =====================
# The constants are inputs rather than hard-coded so the sensitivity study
# below can perturb them.

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


# =====================
# The Optimization
# =====================
def build_problem():
    p = om.Problem(reports=False)
    p.model.add_subsystem('aircraft', Aircraft(), promotes=['*'])

    p.model.add_design_var('A', lower=1.0,  upper=50.0,  ref=10.0)
    p.model.add_design_var('S', lower=1.0,  upper=100.0, ref=10.0)
    p.model.add_design_var('V', lower=10.0, upper=200.0, ref=30.0)
    p.model.add_objective('D', ref=300.0)
    p.model.add_constraint('weight_margin', lower=0.0, ref=CONSTANTS['W_0'][0])

    p.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-12, maxiter=200, disp=False)
    p.setup(force_alloc_complex=True)
    return p


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


# =====================
# Solve, Both Ways
# =====================
t0 = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    sol = runpy.run_path(os.path.join(HERE, 'aircraft_gp.py'))['sol']
t_lcs = time.perf_counter() - t0

t0 = time.perf_counter()
p = build_problem()
D_star = optimize(p)
t_om = time.perf_counter() - t0
iters = p.driver.iter_count

t0 = time.perf_counter()
om_sens = log_sensitivities(p, D_star)
t_om_sens = time.perf_counter() - t0

# =====================
# Compare
# =====================
lcs_vars = {k: float(getattr(v, 'magnitude', v)) for k, v in sol.variables().items()}
lcs_sens = {k: float(v) for k, v in sol.sensitivities().items()}

print(f'lcsolver   {t_lcs:6.3f} s  (build, presolve, solve, sensitivities)')
print(f'OpenMDAO   {t_om:6.3f} s  SLSQP, {iters} evaluations; '
      f'+ {t_om_sens:.3f} s for {2 * len(CONSTANTS)} re-optimizations (sensitivities)')
print()
print(f'{"variable":10s} {"lcsolver":>14s} {"OpenMDAO":>14s} {"rel diff":>10s}')
for name in lcs_vars:
    a = lcs_vars[name]
    b = float(p.get_val(name)[0])
    print(f'{name:10s} {a:14.6g} {b:14.6g} {abs(a - b) / abs(a):10.1e}')
print()
print(f'{"constant":10s} {"lcsolver":>10s} {"OpenMDAO":>10s}   (d log D* / d log c)')
for name in sorted(lcs_sens, key=lambda n: -abs(lcs_sens[n])):
    print(f'{name:10s} {lcs_sens[name]:+10.4f} {om_sens[name]:+10.4f}')
