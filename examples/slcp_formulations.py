#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""The SLCP paper's problems written as EDI ``Formulation`` objects.

``slcp_cases`` states these problems in the low-level ``Problem`` form the SLCP
driver consumes. This module states the *geometric-program* ones the way a user
would actually write them -- unit-annotated ``Formulation`` objects -- so they
can be routed through EDI's ordinary solver machinery.

That serves two purposes: it is what the documentation shows, and it lets the
same model be handed to every available backend so their costs can be compared
on identical input.
"""

import numpy as np
import pyomo.environ as pyo
from pyomo.environ import units

from edi import Formulation


# ---------------------------------------------------------------------------
def simple_gp():
    """Paper Equation 16, the two-variable example.

    .. math::

        \\begin{aligned}
        \\underset{x,y}{\\text{minimize}} \\quad
          & x^{-0.1} + 15x^{0.01} + y^{-0.1} + 15y^{0.01} \\\\
        \\text{subject to} \\quad
          & 0.01x^{-1.1} + x^{0.1} + y \\le 1
        \\end{aligned}

    Optimum :math:`f^* = 31.8115934` at :math:`(0.0593208, 0.0224878)`.
    """
    f = Formulation()
    f.Variable(name='x', guess=0.3, units='', description='x')
    f.Variable(name='y', guess=0.05, units='', description='y')
    f.Objective(f.x ** -0.1 + 15 * f.x ** 0.01
                + f.y ** -0.1 + 15 * f.y ** 0.01)
    f.ConstraintList([
        0.01 * f.x ** -1.1 + f.x ** 0.1 + f.y <= 1.0 * units.dimensionless,
    ])
    return f


# ---------------------------------------------------------------------------
def hoburg_gp():
    """The Hoburg UAV problem as a geometric program, paper Section 7.

    61 variables, 58 constraints. Every constraint is GP-compatible, so this is
    the largest model in the set that any convex backend can take directly.

    Optimum :math:`W_{fuel} = 5911.314` N, validated against the paper's Table 3.
    """
    f = Formulation()
    pi = np.pi
    S3 = range(3)

    # --- constants (SI; atmosphere at 3000 m and sea level) ------------------
    A_prop, CDA0, C_Lmax, e = 0.785, 0.05, 1.5, 0.95
    eta_eng, eta_v, f_wadd, g = 0.35, 0.85, 2.0, 9.81
    h_fuel, k_ew = 46.0e6, 0.0372
    rho, rho_SL = 0.909122, 1.225
    mu_air = rho * 1.71e-5
    N_lift, r_h = 6.0, 0.75
    rho_cap = rho_web = 2700.0
    sigma_max, sigma_max_shear = 310.0e6, 167.0e6
    w_bar, W_fixed = 0.5, 14700.0

    # --- variables ------------------------------------------------------------
    scalars = [
        ('AR', 12.0, '', 'aspect ratio'),
        ('I_cap_bar', 1.0e-2, '', 'normalized spar cap area moment'),
        ('M_r_bar', 1.0e4, '', 'root bending moment per unit chord'),
        ('nu', 0.8, '', 'taper placeholder'),
        ('p', 2.0, '', 'dummy variable 1 + 2 lam'),
        ('P_max', 1.0e5, '', 'maximum engine power'),
        ('q', 1.5, '', 'dummy variable 1 + lam'),
        ('R', 5.0e6, '', 'single segment range'),
        ('S', 40.0, '', 'wing area'),
        ('t_cap_bar', 0.05, '', 'spar cap thickness per chord'),
        ('t_web_bar', 0.05, '', 'spar web thickness per chord'),
        ('tau', 0.12, '', 'airfoil thickness to chord'),
        ('V_stall', 35.0, '', 'stall speed'),
        ('W_cap', 600.0, '', 'spar cap weight'),
        ('W_eng', 5000.0, '', 'engine weight'),
        ('W_fuel_out', 3000.0, '', 'fuel weight, outbound'),
        ('W_fuel_ret', 3000.0, '', 'fuel weight, return'),
        ('W_MTO', 40000.0, '', 'maximum takeoff weight'),
        ('W_pay', 4905.0, '', 'payload weight'),
        ('W_tilde', 25000.0, '', 'dry weight less wing'),
        ('W_web', 400.0, '', 'spar web weight'),
        ('W_wing', 2000.0, '', 'wing weight'),
        ('W_zfw', 30000.0, '', 'zero fuel weight'),
    ]
    for name, guess, unit, desc in scalars:
        f.Variable(name=name, guess=guess, units=unit, description=desc)

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

    one = 1.0 * units.dimensionless
    C = []

    # --- steady level flight --------------------------------------------------
    for i in S3:
        C.append(0.5 * rho * v[f'V_{i}'] ** 2 * v[f'C_L_{i}'] * v['S']
                 / v[f'W_{i}'] == one)
        C.append(0.5 * rho * v[f'V_{i}'] ** 2 * v[f'C_D_{i}'] * v['S']
                 / v[f'T_{i}'] <= one)
        C.append(rho / mu_air * v[f'V_{i}'] * v['S'] ** 0.5
                 / (v['AR'] ** 0.5 * v[f'Re_{i}']) == one)

    # --- landing and sprint ---------------------------------------------------
    C.append(0.5 * rho_SL * C_Lmax * v['V_stall'] ** 2 * v['S']
             / v['W_MTO'] == one)
    C.append(v['V_stall'] / 38.0 <= one)
    C.append(v['T_2'] * v['V_2'] / (v['eta_0_2'] * v['P_max']) <= one)
    C.append(150.0 / v['V_2'] <= one)

    # --- drag -----------------------------------------------------------------
    for i in S3:
        C.append(CDA0 / (v['S'] * v[f'C_Dfuse_{i}']) == one)
        C.append(v[f'C_L_{i}'] ** 2
                 / (pi * e * v['AR'] * v[f'C_Di_{i}']) == one)
        C.append((v[f'C_Dfuse_{i}'] + v[f'C_Dp_{i}'] + v[f'C_Di_{i}'])
                 / v[f'C_D_{i}'] <= one)

    # --- propulsion -----------------------------------------------------------
    for i in S3:
        C.append(eta_eng * v[f'eta_prop_{i}'] / v[f'eta_0_{i}'] == one)
        C.append(eta_v * v[f'eta_i_{i}'] / v[f'eta_prop_{i}'] == one)
        C.append(v[f'eta_i_{i}']
                 + v[f'T_{i}'] * v[f'eta_i_{i}'] ** 2
                 / (4.0 * 0.5 * rho * A_prop * v[f'V_{i}'] ** 2) <= one)

    # --- range ----------------------------------------------------------------
    C.append(5.0e6 / v['R'] <= one)
    for i in range(2):
        C.append(g / h_fuel * v['R'] * v[f'T_{i}']
                 / (v[f'eta_0_{i}'] * v[f'W_{i}'] * v[f'z_bre_{i}']) == one)
    for i, wf in ((0, 'W_fuel_out'), (1, 'W_fuel_ret')):
        z, W = v[f'z_bre_{i}'], v[f'W_{i}']
        C.append((z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24) * W
                 / v[wf] <= one)

    # --- weight ---------------------------------------------------------------
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

    # --- wing structure -------------------------------------------------------
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

    # --- profile drag: the five-term posynomial fit ---------------------------
    for i in S3:
        cl, re, cdp = v[f'C_L_{i}'], v[f'Re_{i}'], v[f'C_Dp_{i}']
        C.append(
            2.56 * cl ** 5.88 * v['tau'] ** -3.32 * re ** -1.54 * cdp ** -2.26
            + 3.80e-9 * cl ** -0.92 * v['tau'] ** 6.23 * re ** -1.38 * cdp ** -9.57
            + 2.20e-3 * cl ** -0.01 * v['tau'] ** 0.03 * re ** 0.14 * cdp ** -0.73
            + 1.19e4 * cl ** 9.78 * v['tau'] ** 1.76 * re ** -1.00 * cdp ** -0.91
            + 6.14e-6 * cl ** 6.53 * v['tau'] ** -0.52 * re ** -0.99 * cdp ** -5.19
            <= one)

    f.Objective(v['W_fuel_out'] + v['W_fuel_ret'])
    f.ConstraintList(C)
    return f


# ---------------------------------------------------------------------------
def problem_to_formulation(problem, x0=None):
    """Convert a GP-compatible ``slcp_cases.Problem`` into an EDI ``Formulation``.

    Only valid when the objective and every constraint are posynomial or
    monomial. Useful for putting an existing ``Problem`` through EDI's ordinary
    solver machinery -- for instance to benchmark backends on it -- without
    transcribing it a second time by hand.

    The hand-written builders above are what the documentation shows, because
    they read the way a user would actually write the model. This is the
    mechanical path.
    """
    from edi.solvers.ipopt.slcp import Posynomial

    if not isinstance(problem.objective, Posynomial):
        raise ValueError('objective must be a posynomial')
    if not all(c.exact_in_logspace for c in problem.constraints):
        raise ValueError('every constraint must be posynomial or monomial')

    x0 = np.asarray(x0 if x0 is not None else np.ones(problem.n), dtype=float)

    f = Formulation()
    for j, name in enumerate(problem.names):
        f.Variable(name=name, guess=float(x0[j]), units='', description=name)
    v = [f.find_component(name) for name in problem.names]

    def build(terms):
        total = 0
        for c, a in terms:
            term = float(c)
            for j in range(problem.n):
                if a[j] != 0:
                    term = term * v[j] ** float(a[j])
            total = total + term
        return total

    f.Objective(build(problem.objective.terms))
    one = 1.0 * units.dimensionless
    f.ConstraintList([
        (build(c.body.terms) == one) if c.operator == '=='
        else (build(c.body.terms) <= one)
        for c in problem.constraints
    ])
    return f


def ko_gp():
    """Kirschen-Ozturk in its all-inequality (mode 1) form: a pure GP.

    18 variables, sitting between the 2-variable example and the 61-variable
    Hoburg model, which is where the backend crossover shows up.
    """
    from examples.slcp_cases import KO_X0, kirschen_ozturk_gp

    problem, _, _ = kirschen_ozturk_gp()
    return problem_to_formulation(problem, KO_X0)


FORMULATIONS = {
    'simple': simple_gp,
    'ko': ko_gp,
    'hoburg': hoburg_gp,
}
