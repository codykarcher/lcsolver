"""Hoburg's UAV, as an EDI Formulation.

Source
------
W. Hoburg and P. Abbeel, "Geometric Programming for Aircraft Design
Optimization", AIAA Journal 52(11), 2014, doi:10.2514/1.J052732, Section VII.
Transcribed from the same notebook as ``examples/slcp_cases.hoburg``, which
carries the identical model at the raw ``Problem``/``Constraint`` level; this
file is the readable counterpart -- units, names and descriptions -- and it
solves to the same optimum.

Three mission segments: outbound, return, and a sprint condition that sizes the
powerplant. Minimize fuel.

The wing box
------------
The structural model is Hoburg's and was already here; it is simply written
non-dimensionally. ``t_cap_bar`` and ``t_web_bar`` are thicknesses over chord,
``I_cap_bar`` is a second moment over chord^4, ``M_r_bar`` a root moment over
its own scale. Those are the right variables to *optimize* -- they are what the
constraints are naturally posed in, and they keep the problem scale-free -- and
the wrong ones to read, draw, or hand to anybody.

So this file adds the dimensional recovery: span, chords, taper ratio, spar box,
root moment, and the actual cap and web thicknesses in metres. That is ten more
variables and they cost nothing, because every one is *defined* by an equality
rather than constrained by one. The presolve removes them and back-substitutes
afterwards, so they never reach the optimizer. Measured, by running the presolve
chain on ``build(recover=False)`` and ``build(recover=True)``:

    recover=False    61 declared -> 48 at the solver
    recover=True     71 declared -> 48 at the solver

Same 48 columns either way, same optimum to ten digits, and in the printed
solution the recovered quantities are indistinguishable from the optimized ones
-- which is the point. There is no reason to make a model unreadable to keep it
small.

Two ways of being free
----------------------
The ten split across the two mechanisms EDI has, and the split is instructive.

Nine are **monomial equalities**, eliminated by Gaussian elimination on the
exponent matrix. ``c_tip == c_root * lambda`` counts: a product of two variables
is still a monomial.

The taper ratio is not. ``lambda = q - 1`` is a *sum*, and Hoburg's (p, q)
substitution exists precisely to keep the taper terms GP compatible, which costs
monomial invertibility. Written the way round that a GP admits, ``q == 1 +
lambda``, it is a posynomial equality, and the presolve removes it anyway under
the weaker and more general condition: ``lambda`` appears in exactly one live
constraint, is absent from the objective, and is unbounded in the direction that
would matter, so nothing it touches is restricted by it. That is an
**output-only** variable. Order matters and the two mechanisms feed each other:
eliminating c_tip also removes the constraint that defined it, and only then
does lambda sit in a single live constraint and become output-only.

The modelling lesson is that neither mechanism is something to design around.
Write the quantity you want to read, in whichever direction the algebra is
natural. Whether it is free is EDI's problem.
"""
from __future__ import annotations

import numpy as np
from pyomo.environ import units

from edi import Formulation

N_SEG = 3                       # outbound, return, sprint
OUT, RET, SPRINT = 0, 1, 2


def build(n_seg: int = N_SEG, recover: bool = True) -> Formulation:
    """Hoburg's UAV. Returns an EDI ``Formulation`` ready to solve.

    ``recover=False`` drops the dimensional-recovery block, for measuring
    what it costs. The answer is nothing; see the module docstring.
    """
    f = Formulation()
    pi = np.pi
    S3 = range(n_seg)

    V_ = lambda n, g, u, d: f.Variable(name=n, guess=g, units=u, description=d)
    Vn = lambda n, g, u, d: f.Variable(name=n, guess=g, units=u, description=d,
                                       size=n_seg)
    C_ = lambda n, v, u, d: f.Constant(name=n, value=v, units=u, description=d)

    # ---- constants ---------------------------------------------------------
    # Atmosphere at 3000 m and sea level, from the paper's table.
    rho     = C_('rho',      0.909122, 'kg/m^3',  'air density at altitude')
    rho_SL  = C_('rho_SL',   1.225,    'kg/m^3',  'air density at sea level')
    nu_air  = C_('nu_air',   1.71e-5,  'm^2/s',   'kinematic viscosity')
    g_acc   = C_('g',        9.81,     'm/s^2',   'gravitational acceleration')
    CDA0    = C_('CDA0',     0.05,     'm^2',     'fuselage drag area')
    e_osw   = C_('e',        0.95,     '-',       'Oswald efficiency')
    C_Lmax  = C_('C_Lmax',   1.5,      '-',       'max lift coefficient, landing')
    A_prop  = C_('A_prop',   0.785,    'm^2',     'propeller disc area')
    eta_eng = C_('eta_eng',  0.35,     '-',       'engine efficiency')
    eta_v   = C_('eta_v',    0.85,     '-',       'propeller viscous efficiency')
    h_fuel  = C_('h_fuel',   46.0e6,   'J/kg',    'fuel specific energy')
    # An empirical fit with a fractional exponent has no honest unit of its
    # own -- 'N/W^0.803' is a bookkeeping fiction. Non-dimensionalise it: the
    # coefficient is a pure number and the fit is written against a reference
    # power, which is what the regression actually meant.
    k_ew    = C_('k_ew',     0.0372,   '-',  'engine weight fit coefficient')
    P_ref   = C_('P_ref',    1.0,      'W',  'reference power for the fit')
    W_ref   = C_('W_ref',    1.0,      'N',  'reference weight for the fit')
    f_wadd  = C_('f_wadd',   2.0,      '-',       'wing added-weight fraction')
    W_fixed = C_('W_fixed',  14700.0,  'N',       'fixed airframe weight')
    m_pay   = C_('m_pay',    500.0,    'kg',      'payload mass')
    R_req   = C_('R_req',    5.0e6,    'm',       'required range')
    V_stall_max = C_('V_stall_max', 38.0, 'm/s',  'max stall speed')
    V_sprint    = C_('V_sprint',   150.0, 'm/s',  'required sprint speed')

    N_lift   = C_('N_lift',   6.0,       '-',   'ultimate load factor')
    r_h      = C_('r_h',      0.75,      '-',   'web height / airfoil thickness')
    w_bar    = C_('w_bar',    0.5,       '-',   'box width / chord')
    rho_cap  = C_('rho_cap',  2700.0, 'kg/m^3', 'spar cap density')
    rho_web  = C_('rho_web',  2700.0, 'kg/m^3', 'shear web density')
    sig_max  = C_('sigma_max',       310.0e6, 'Pa', 'allowable tensile stress')
    sig_shr  = C_('sigma_max_shear', 167.0e6, 'Pa', 'allowable shear stress')
    tau_max  = C_('tau_max',  0.15,      '-',   'max thickness/chord')

    # ---- variables ---------------------------------------------------------
    S    = V_('S',    30.0,  'm^2', 'wing area')
    AR   = V_('AR',   20.0,  '-',   'aspect ratio')
    tau  = V_('tau',  0.14,  '-',   'airfoil thickness/chord')
    p    = V_('p',    1.9,   '-',   'taper substitution 1 + 2*lambda')
    q    = V_('q',    1.45,  '-',   'taper substitution 1 + lambda')

    Vel   = Vn('V',        50.0,  'm/s', 'flight speed')
    C_L   = Vn('C_L',      0.5,   '-',   'lift coefficient')
    C_D   = Vn('C_D',      0.05,  '-',   'drag coefficient')
    C_Dfu = Vn('C_Dfuse',  0.002, '-',   'fuselage drag coefficient')
    C_Dp  = Vn('C_Dp',     0.01,  '-',   'profile drag coefficient')
    C_Di  = Vn('C_Di',     0.01,  '-',   'induced drag coefficient')
    Thr   = Vn('T',        1000., 'N',   'thrust')
    Wseg  = Vn('W',        30000.,'N',   'segment weight')
    Re    = Vn('Re',       1.5e6, '-',   "Reynolds number")
    eta_i = Vn('eta_i',    0.9,   '-',   'propeller inviscid efficiency')
    eta_p = Vn('eta_prop', 0.8,   '-',   'propeller efficiency')
    eta_0 = Vn('eta_0',    0.28,  '-',   'overall efficiency')

    V_stall = V_('V_stall', 35.0,   'm/s', 'stall speed')
    P_max   = V_('P_max',   1.0e5,  'W',   'maximum engine power')
    Rng     = V_('R',       5.0e6,  'm',   'range')
    z_out   = V_('z_bre_0', 0.1,    '-',   'Breguet parameter, outbound')
    z_ret   = V_('z_bre_1', 0.1,    '-',   'Breguet parameter, return')

    W_MTO   = V_('W_MTO',      35000., 'N', 'max takeoff weight')
    W_zfw   = V_('W_zfw',      30000., 'N', 'zero-fuel weight')
    W_tilde = V_('W_tilde',    25000., 'N', 'weight less wing')
    W_pay   = V_('W_pay',      4900.,  'N', 'payload weight')
    W_eng   = V_('W_eng',      1000.,  'N', 'engine weight')
    W_wing  = V_('W_wing',     5000.,  'N', 'wing weight')
    W_cap   = V_('W_cap',      600.,   'N', 'spar cap weight')
    W_web   = V_('W_web',      400.,   'N', 'shear web weight')
    W_f_out = V_('W_fuel_out', 2000.,  'N', 'fuel burned outbound')
    W_f_ret = V_('W_fuel_ret', 2000.,  'N', 'fuel burned on return')

    # Hoburg's non-dimensional wing box. These are what the constraints are
    # posed in and what the optimizer sees.
    t_cap_b = V_('t_cap_bar', 0.004, '-', 'spar cap thickness / chord')
    t_web_b = V_('t_web_bar', 0.001, '-', 'shear web thickness / chord')
    I_cap_b = V_('I_cap_bar', 1e-5,  '-', 'cap second moment / chord^4')
    M_r_bar = V_('M_r_bar',   1e3,   'N', 'root bending moment / chord')
    nu_t    = V_('nu',        0.8,   '-', 'taper integration factor')

    f.Objective(W_f_out + W_f_ret)

    cons = []

    # ---- steady level flight -----------------------------------------------
    for i in S3:
        cons += [
            Wseg[i] == 0.5 * rho * Vel[i] ** 2 * C_L[i] * S,
            Thr[i] >= 0.5 * rho * Vel[i] ** 2 * C_D[i] * S,
            Re[i] == Vel[i] * (S / AR) ** 0.5 / nu_air,
        ]

    # ---- landing and sprint -------------------------------------------------
    cons += [
        W_MTO == 0.5 * rho_SL * V_stall ** 2 * C_Lmax * S,
        V_stall <= V_stall_max,
        P_max >= Thr[SPRINT] * Vel[SPRINT] / eta_0[SPRINT],
        Vel[SPRINT] >= V_sprint,
    ]

    # ---- drag ---------------------------------------------------------------
    for i in S3:
        cons += [
            C_Dfu[i] == CDA0 / S,
            C_Di[i] == C_L[i] ** 2 / (pi * e_osw * AR),
            C_D[i] >= C_Dfu[i] + C_Dp[i] + C_Di[i],
            # Hoburg's five-term fit to the XFOIL profile drag. This is an
            # *implicit* softmax-affine fit: C_Dp carries a different exponent
            # in every term, so it cannot be pulled out to the left as a single
            # power. The GP-compatible form is the whole fit <= 1, with C_Dp in
            # the denominators.
            1.0 >= (
                2.56 * C_L[i] ** 5.88 / (tau ** 3.32 * Re[i] ** 1.54
                                         * C_Dp[i] ** 2.26)
                + 3.80e-9 * tau ** 6.23 / (C_L[i] ** 0.92 * Re[i] ** 1.38
                                           * C_Dp[i] ** 9.57)
                + 2.20e-3 * tau ** 0.03 * Re[i] ** 0.14 / (C_L[i] ** 0.01
                                                           * C_Dp[i] ** 0.73)
                + 1.19e4 * C_L[i] ** 9.78 * tau ** 1.76 / (Re[i] ** 1.00
                                                           * C_Dp[i] ** 0.91)
                + 6.14e-6 * C_L[i] ** 6.53 / (tau ** 0.52 * Re[i] ** 0.99
                                              * C_Dp[i] ** 5.19)),
        ]

    # ---- propulsion ---------------------------------------------------------
    for i in S3:
        cons += [
            eta_0[i] == eta_eng * eta_p[i],
            eta_p[i] == eta_v * eta_i[i],
            # actuator disc: 4 eta_i + T eta_i^2 / (1/2 rho V^2 A) <= 4
            eta_i[i] + Thr[i] * eta_i[i] ** 2
                / (4.0 * 0.5 * rho * Vel[i] ** 2 * A_prop) <= 1.0,
        ]

    # ---- range and fuel -----------------------------------------------------
    cons += [Rng >= R_req]
    for i, (z, wf) in enumerate(((z_out, W_f_out), (z_ret, W_f_ret))):
        cons += [
            z == g_acc * Rng * Thr[i] / (h_fuel * eta_0[i] * Wseg[i]),
            # Breguet, as the first four terms of the series
            wf / Wseg[i] >= z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24,
        ]

    # ---- weight buildup -----------------------------------------------------
    cons += [
        W_pay >= m_pay * g_acc,
        W_tilde >= W_fixed + W_pay + W_eng,
        W_zfw >= W_tilde + W_wing,
        W_eng >= k_ew * W_ref * (P_max / P_ref) ** 0.803,
        W_wing / f_wadd >= W_web + W_cap,
        # The weight carried by each segment is its weight at the END of that
        # segment, because that is the W the Breguet series above is written
        # against: W_initial/W_final = exp(z), so W_fuel = W_final (e^z - 1).
        # So Wseg[OUT] is the weight at the end of the outbound leg, which is
        # also the weight at the start of the return leg -- one quantity, not
        # two -- and the sprint is flown at that same weight.
        Wseg[OUT] >= W_zfw + W_f_ret,
        W_MTO >= Wseg[OUT] + W_f_out,
        Wseg[RET] >= W_zfw,
        Wseg[SPRINT] == Wseg[OUT],
    ]

    # ---- wing structure, non-dimensional ------------------------------------
    cons += [
        2 * q >= 1 + p,
        p >= 1.9,
        tau <= tau_max,
        M_r_bar == W_tilde * AR * p / 24,
        # cap bending: 0.92 w_bar tau t_cap^2 + I_cap <= 0.92^2/2 w_bar tau^2 t_cap
        0.92 * w_bar * tau * t_cap_b ** 2 + I_cap_b
            <= 0.92 ** 2 / 2 * w_bar * tau ** 2 * t_cap_b,
        8 * I_cap_b * sig_max / (N_lift * M_r_bar * AR * q ** 2 * tau) == 1 / S,
        12 * t_web_b * sig_shr / (N_lift * AR * W_tilde * q ** 3 / tau) == 1 / S,
        nu_t ** 3.94 >= 0.86 * p ** -2.38 + 0.14 * p ** 0.56,
        W_cap >= 8 * rho_cap * g_acc * w_bar * t_cap_b * S ** 1.5 * nu_t
                 / (3 * AR ** 0.5),
        W_web >= 8 * rho_web * g_acc * r_h * tau * t_web_b * S ** 1.5 * nu_t
                 / (3 * AR ** 0.5),
    ]

    # ---- dimensional recovery ----------------------------------------------
    # These exist so the answer can be read, drawn and checked against an
    # aeroplane, and none of them costs anything at solve time: each is
    # *defined* by an equality rather than constrained by one, so the presolve
    # removes it and back-substitutes afterwards.
    #
    # Two kinds appear here, and the difference is worth knowing. Most are
    # monomial equalities, eliminated by Gaussian elimination on the exponent
    # matrix. The taper ratio is not: lambda = q - 1 is a *sum*, and Hoburg's
    # (p, q) substitution exists precisely to keep the taper terms GP
    # compatible, which costs monomial invertibility. Written as q == 1 + lambda
    # it is a posynomial equality, and the presolve removes it anyway -- as an
    # output-only variable, which is the weaker and more general condition:
    # lambda appears in exactly one live constraint, is absent from the
    # objective, and is otherwise unbounded, so nothing it touches is
    # restricted by it -- but only once c_tip's monomial elimination has taken
    # `c_tip == c_root*lambda` away with it. Nine go the first way, lambda the
    # second.
    #
    # So write the quantity you want to read. Whether it is free is EDI's
    # problem, not the modeller's.
    if not recover:
        f.ConstraintList(cons)
        return f

    b      = V_('b',      25.0,  'm',   'wing span')
    c_bar  = V_('c_bar',  1.2,   'm',   'mean chord')
    c_root = V_('c_root', 1.6,   'm',   'root chord')
    t_cap  = V_('t_cap',  0.005, 'm',   'spar cap thickness')
    t_web  = V_('t_web',  0.002, 'm',   'shear web thickness')
    h_spar = V_('h_spar', 0.13,  'm',   'spar box height')
    w_spar = V_('w_spar', 0.60,  'm',   'spar box width')
    M_root = V_('M_root', 1.0e4, 'N*m', 'root bending moment')
    lam    = V_('lambda', 0.45,  '-',   'taper ratio, c_tip / c_root')
    c_tip  = V_('c_tip',  0.72,  'm',   'tip chord')

    cons += [
        b == (S * AR) ** 0.5,
        c_bar == (S / AR) ** 0.5,
        c_root == 2 * c_bar / q,          # q = 1 + lambda
        t_cap == t_cap_b * c_bar,
        t_web == t_web_b * c_bar,
        h_spar == r_h * tau * c_bar,      # web height = r_h * (t/c) * chord
        w_spar == w_bar * c_bar,          # box width  = w_bar * chord
        M_root == M_r_bar * c_bar,
        q == 1 + lam,                     # the (p, q) substitution, inverted
        c_tip == c_root * lam,
    ]

    f.ConstraintList(cons)
    return f


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')
    from edi.solvers.solver import solve

    f = build()
    solve(f)                       # sensitivities are on by default
    print(f.solution.summary(top=12))
