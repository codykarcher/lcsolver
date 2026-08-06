"""The Hoburg UAV with profile drag supplied by a black box.

Source
------
W. Hoburg and P. Abbeel, "Geometric Programming for Aircraft Design
Optimization", AIAA Journal 52(11), 2014 -- and the SLCP paper, Sections
7.5-7.6, which black-box the drag fit.

The model is the full 61-variable UAV sizing problem: steady level flight
over three segments (outbound cruise, return cruise, sprint), landing,
drag build-up, propulsive efficiency, Breguet range, weight build-up, and
wing structure. In the pure-GP statement each segment carries an explicit
five-term posynomial fit for profile drag:

    1 >= sum_k a_k C_L^b_k tau^c_k Re^d_k C_Dp^e_k

Here that posynomial is *hidden inside a black box*: an opaque function
``C_Dp = f(C_L, Re, tau)`` that solves the same fit implicitly and returns
the result with derivatives, exactly the way an external analysis code
(XFOIL, a neural surrogate, CFD) would plug in. The optimizer never sees
the algebra -- it must treat the constraint like any opaque analysis.

Because the box solves the same relation the explicit posynomial states,
the optimum is unchanged: W_fuel = 5911.3 N. Only the solver's visibility
into the structure changes -- which is precisely LCsolver's black-box
capability: swap any posynomial for any analysis code without touching the
rest of the model.
"""

import numpy as np
import pyomo.environ as pyo
from pyomo.environ import units

from lcsolver.objects.formulation import Formulation
from lcsolver.objects.blackBoxFunctionModel import BlackBoxFunctionModel
from lcsolver.solvers.solver import solve

pi = np.pi
S3 = range(3)

# =========
# Constants  (SI; atmosphere at 3000 m and sea level)
# =========
A_prop, CDA0, C_Lmax, e = 0.785, 0.05, 1.5, 0.95
eta_eng, eta_v, f_wadd, g = 0.35, 0.85, 2.0, 9.81
h_fuel, k_ew = 46.0e6, 0.0372
rho, rho_SL = 0.909122, 1.225
mu_air = rho * 1.71e-5
N_lift, r_h = 6.0, 0.75
rho_cap = rho_web = 2700.0
sigma_max, sigma_max_shear = 310.0e6, 167.0e6
w_bar, W_fixed = 0.5, 14700.0

# ==========================================
# The black box: C_Dp = f(C_L, Re, tau)
# ==========================================
# Coefficients of the profile-drag fit (paper Eq. "dragFit"):
#   1 >= sum_k a_k C_L^b_k tau^c_k Re^d_k C_Dp^e_k
_DRAG_FIT = [
    (2.56,     5.88, -3.32, -1.54, -2.26),
    (3.80e-9, -0.92,  6.23, -1.38, -9.57),
    (2.20e-3, -0.01,  0.03,  0.14, -0.73),
    (1.19e4,   9.78,  1.76, -1.00, -0.91),
    (6.14e-6,  6.53, -0.52, -0.99, -5.19),
]


class ProfileDrag(BlackBoxFunctionModel):
    """Profile drag coefficient from an implicit solve of the drag fit.

    Stands in for any external drag analysis. Every exponent on C_Dp in the
    fit is negative, so the residual is strictly decreasing in C_Dp and the
    root is unique; derivatives come from implicit differentiation.
    """

    def __init__(self):
        super().__init__()
        self.description = 'C_Dp = f(C_L, Re, tau) by implicit solve of the drag fit'
        self.inputs.append(name='C_L', units='', description='lift coefficient')
        self.inputs.append(name='Re', units='', description='Reynolds number')
        self.inputs.append(name='tau', units='', description='thickness to chord')
        self.outputs.append(name='C_Dp', units='', description='profile drag coefficient')
        self.availableDerivative = 1
        self.post_init_setup(len(self.inputs))

    def BlackBox(self, C_L, Re, tau):
        from scipy.optimize import brentq

        C_L, Re, tau = self.sanitizeInputs(C_L, Re, tau, strip_units=True)

        def F(cdp):
            return sum(a * C_L ** b * tau ** c * Re ** d * cdp ** e
                       for a, b, c, d, e in _DRAG_FIT) - 1.0

        # F is strictly decreasing in C_Dp (every exponent on C_Dp in the fit
        # is negative), so a sign change brackets the unique root. The bracket
        # is widened adaptively because the optimizer evaluates this box at
        # intermediate iterates far from the optimum.
        lo, hi = 1e-8, 1.0
        while F(hi) > 0.0 and hi < 1e6:
            hi *= 10.0
        while F(lo) < 0.0 and lo > 1e-30:
            lo /= 10.0
        C_Dp = brentq(F, lo, hi, xtol=1e-14, rtol=1e-14)

        # Implicit differentiation: dC_Dp/dz = -(dF/dz) / (dF/dC_Dp)
        dF_dcdp = dF_dCL = dF_dRe = dF_dtau = 0.0
        for a, b, c, d, e in _DRAG_FIT:
            t = a * C_L ** b * tau ** c * Re ** d * C_Dp ** e
            dF_dcdp += e * t / C_Dp
            dF_dCL += b * t / C_L
            dF_dRe += d * t / Re
            dF_dtau += c * t / tau

        return self.packOutputs(C_Dp, [-dF_dCL / dF_dcdp,
                                       -dF_dRe / dF_dcdp,
                                       -dF_dtau / dF_dcdp])


# =====================
# Declare the Variables
# =====================
f = Formulation()
scalars = [
    ('AR', 12.0, 'aspect ratio'),
    ('I_cap_bar', 1.0e-2, 'normalized spar cap area moment'),
    ('M_r_bar', 1.0e4, 'root bending moment per unit chord'),
    ('nu', 0.8, 'taper placeholder'),
    ('p', 2.0, 'dummy variable 1 + 2 lam'),
    ('P_max', 1.0e5, 'maximum engine power'),
    ('q', 1.5, 'dummy variable 1 + lam'),
    ('R', 5.0e6, 'single segment range'),
    ('S', 40.0, 'wing area'),
    ('t_cap_bar', 0.05, 'spar cap thickness per chord'),
    ('t_web_bar', 0.05, 'spar web thickness per chord'),
    ('tau', 0.12, 'airfoil thickness to chord'),
    ('V_stall', 35.0, 'stall speed'),
    ('W_cap', 600.0, 'spar cap weight'),
    ('W_eng', 5000.0, 'engine weight'),
    ('W_fuel_out', 3000.0, 'fuel weight, outbound'),
    ('W_fuel_ret', 3000.0, 'fuel weight, return'),
    ('W_MTO', 40000.0, 'maximum takeoff weight'),
    ('W_pay', 4905.0, 'payload weight'),
    ('W_tilde', 25000.0, 'dry weight less wing'),
    ('W_web', 400.0, 'spar web weight'),
    ('W_wing', 2000.0, 'wing weight'),
    ('W_zfw', 30000.0, 'zero fuel weight'),
]
for name, guess, desc in scalars:
    f.Variable(name=name, guess=guess, units='', description=desc)

vectors = {
    'V': [60.0, 60.0, 150.0], 'C_L': [0.6, 0.6, 0.2],
    'C_D': [0.03, 0.03, 0.03], 'C_Dfuse': [1.5e-3] * 3,
    'C_Dp': [0.01] * 3, 'C_Di': [0.01, 0.01, 0.005],
    'T': [1500.0, 1500.0, 4000.0], 'W': [3.0e4] * 3,
    'Re': [3.0e6, 3.0e6, 8.0e6], 'eta_i': [0.8] * 3,
    'eta_prop': [0.7] * 3, 'eta_0': [0.25] * 3,
}
for base, guesses in vectors.items():
    for i, gv in enumerate(guesses):
        f.Variable(name=f'{base}_{i}', guess=gv, units='',
                   description=f'{base}, segment {i}')
for i in range(2):
    f.Variable(name=f'z_bre_{i}', guess=0.3, units='',
               description=f'Breguet factor, segment {i}')

v = {name: f.find_component(name) for name, *_ in scalars}
for base in vectors:
    for i in S3:
        v[f'{base}_{i}'] = f.find_component(f'{base}_{i}')
for i in range(2):
    v[f'z_bre_{i}'] = f.find_component(f'z_bre_{i}')

# ==========================
# Objective and Constraints
# ==========================
f.Objective(v['W_fuel_out'] + v['W_fuel_ret'])

one = 1.0 * units.dimensionless
C = []

# --- steady level flight ---------------------------------------------------
for i in S3:
    C.append(0.5 * rho * v[f'V_{i}'] ** 2 * v[f'C_L_{i}'] * v['S']
             / v[f'W_{i}'] == one)
    C.append(0.5 * rho * v[f'V_{i}'] ** 2 * v[f'C_D_{i}'] * v['S']
             / v[f'T_{i}'] <= one)
    C.append(rho / mu_air * v[f'V_{i}'] * v['S'] ** 0.5
             / (v['AR'] ** 0.5 * v[f'Re_{i}']) == one)

# --- landing and sprint ----------------------------------------------------
C.append(0.5 * rho_SL * C_Lmax * v['V_stall'] ** 2 * v['S']
         / v['W_MTO'] == one)
C.append(v['V_stall'] / 38.0 <= one)
C.append(v['T_2'] * v['V_2'] / (v['eta_0_2'] * v['P_max']) <= one)
C.append(150.0 / v['V_2'] <= one)

# --- drag ------------------------------------------------------------------
for i in S3:
    C.append(CDA0 / (v['S'] * v[f'C_Dfuse_{i}']) == one)
    C.append(v[f'C_L_{i}'] ** 2
             / (pi * e * v['AR'] * v[f'C_Di_{i}']) == one)
    C.append((v[f'C_Dfuse_{i}'] + v[f'C_Dp_{i}'] + v[f'C_Di_{i}'])
             / v[f'C_D_{i}'] <= one)

# --- propulsion ------------------------------------------------------------
for i in S3:
    C.append(eta_eng * v[f'eta_prop_{i}'] / v[f'eta_0_{i}'] == one)
    C.append(eta_v * v[f'eta_i_{i}'] / v[f'eta_prop_{i}'] == one)
    C.append(v[f'eta_i_{i}']
             + v[f'T_{i}'] * v[f'eta_i_{i}'] ** 2
             / (4.0 * 0.5 * rho * A_prop * v[f'V_{i}'] ** 2) <= one)

# --- range -----------------------------------------------------------------
C.append(5.0e6 / v['R'] <= one)
for i in range(2):
    C.append(g / h_fuel * v['R'] * v[f'T_{i}']
             / (v[f'eta_0_{i}'] * v[f'W_{i}'] * v[f'z_bre_{i}']) == one)
for i, wf in ((0, 'W_fuel_out'), (1, 'W_fuel_ret')):
    z, W = v[f'z_bre_{i}'], v[f'W_{i}']
    C.append((z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24) * W
             / v[wf] <= one)

# --- weight ----------------------------------------------------------------
C += [
    500.0 * g / v['W_pay'] <= one,
    (W_fixed + v['W_pay'] + v['W_eng']) / v['W_tilde'] <= one,
    (v['W_tilde'] + v['W_wing']) / v['W_zfw'] <= one,
    k_ew * v['P_max'] ** 0.803 / v['W_eng'] <= one,
    f_wadd * (v['W_web'] + v['W_cap']) / v['W_wing'] <= one,
    (v['W_zfw'] + v['W_fuel_ret']) / v['W_0'] <= one,
    (v['W_0'] + v['W_fuel_out']) / v['W_MTO'] <= one,
    v['W_zfw'] / v['W_1'] <= one,
    v['W_0'] / v['W_2'] == one,
]

# --- wing structure --------------------------------------------------------
C += [
    (1.0 + v['p']) / (2 * v['q']) <= one,
    1.9 / v['p'] <= one,
    v['tau'] / 0.15 <= one,
    v['W_tilde'] * v['AR'] * v['p'] / (24.0 * v['M_r_bar']) == one,
    (0.92 * w_bar * v['tau'] * v['t_cap_bar'] ** 2 + v['I_cap_bar'])
    / (0.92 ** 2 / 2 * w_bar * v['tau'] ** 2 * v['t_cap_bar']) <= one,
    N_lift * v['M_r_bar'] * v['AR'] * v['q'] ** 2 * v['tau']
    / (8.0 * v['S'] * v['I_cap_bar'] * sigma_max) == one,
    v['AR'] * v['W_tilde'] * N_lift * v['q'] ** 3
    / (12.0 * v['tau'] * v['S'] * v['t_web_bar'] * sigma_max_shear) == one,
    (0.86 * v['p'] ** -2.38 + 0.14 * v['p'] ** 0.56)
    / v['nu'] ** 3.94 <= one,
    8.0 * rho_cap * g * w_bar * v['t_cap_bar'] * v['S'] ** 1.5 * v['nu']
    / (3.0 * v['AR'] ** 0.5 * v['W_cap']) <= one,
    8.0 * rho_web * g * r_h * v['tau'] * v['t_web_bar'] * v['S'] ** 1.5
    * v['nu'] / (3.0 * v['AR'] ** 0.5 * v['W_web']) <= one,
]

# --- profile drag: the black box, one per segment --------------------------
# Where the pure-GP model writes the five-term posynomial fit explicitly,
# each segment's C_Dp is here supplied by the opaque analysis above.
for i in S3:
    C.append([v[f'C_Dp_{i}'], '==',
              [v[f'C_L_{i}'], v[f'Re_{i}'], v['tau']], ProfileDrag()])

f.ConstraintList(C)

# =====
# Solve
# =====
# Plain solve(): the router sees the black-box constraints and sends the
# model to SIA, which imposes each box through its linearization inside the
# trust-region loop while keeping every algebraic constraint exact.
if __name__ == '__main__':
    solve(f)
    print(f.solution)
