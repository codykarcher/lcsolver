#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Test problems from the SLCP paper.

Each builder returns ``(problem, x0, reference)`` where ``reference`` holds the
published optimum where one exists.

Source: Karcher & Haimes, "A Method of Sequential Log-Convex Programming for
Engineering Design", Optimization and Engineering (2022),
doi:10.1007/s11081-022-09750-3, Sections 6 and 7; transcribed from the notebooks in
``publications/2021/03_slcpTheory/ipyNotebooks/``.
"""

import numpy as np

from lcsolver.solvers.sequential.slcp import Constraint, Posynomial, Problem, Signomial


# ---------------------------------------------------------------------------
# Transcription helpers
# ---------------------------------------------------------------------------
def _exps(spec, names):
    """Turn ``{'S': 1, 'V': -2}`` into a dense exponent vector."""
    a = np.zeros(len(names))
    for k, v in spec.items():
        a[names.index(k)] = v
    return a


def posy(terms, names):
    """Posynomial from ``[(coeff, {var: exponent}), ...]``."""
    return Posynomial([(c, _exps(s, names)) for c, s in terms], len(names))


def signomial(terms, names):
    """Signomial (coefficients may be negative) with an analytic gradient."""
    packed = [(float(c), _exps(s, names)) for c, s in terms]
    n = len(names)

    def fn(x):
        x = np.asarray(x, dtype=float)
        v = 0.0
        g = np.zeros(n)
        for c, a in packed:
            t = c * np.prod(x ** a)
            v += t
            g += t * a / x
        return v, g

    return Signomial(fn, n)


def con(terms, names, operator='<='):
    """Constraint ``sum_k c_k prod_j x_j^a_kj  <op>  1``.

    Dispatches on the coefficients and the operator: a form that is
    GP-compatible becomes a :class:`Posynomial` so SLCP can impose it exactly;
    anything else becomes a :class:`Signomial` and gets linearized. This is the
    split that the whole algorithm turns on, so it is done here once rather than
    being asserted by hand at each constraint.
    """
    gp_compatible = all(c > 0 for c, _ in terms) and (
        operator == '<=' or len(terms) == 1)
    if gp_compatible:
        return Constraint(posy(terms, names), operator)
    return Constraint(signomial(terms, names), operator)


# ---------------------------------------------------------------------------
def simple_example():
    """The two-variable example of paper Equation 16.

    .. math::

        \\begin{aligned}
        \\underset{x,y}{\\text{minimize}} \\quad
          & x^{-0.1} + 15 x^{0.01} + y^{-0.1} + 15 y^{0.01} \\\\
        \\text{subject to} \\quad
          & 0.01 x^{-1.1} + x^{0.1} + y \\le 1
        \\end{aligned}

    This is a pure geometric program, so the optimum can be obtained
    independently and exactly rather than taken on faith.

    The paper starts from :math:`(x_0, y_0) = (0.3, 0.05)` and reports that LSQP
    overshoots the constraint at iteration 6 and needs 4 more iterations than
    SLCP to recover.
    """
    n = 2
    objective = Posynomial([
        (1.0,  [-0.1,  0.0]),
        (15.0, [0.01,  0.0]),
        (1.0,  [0.0,  -0.1]),
        (15.0, [0.0,   0.01]),
    ], n)

    constraints = [
        Constraint(Posynomial([
            (0.01, [-1.1, 0.0]),
            (1.0,  [0.1,  0.0]),
            (1.0,  [0.0,  1.0]),
        ], n), '<='),
    ]

    problem = Problem(n, objective, constraints, names=['x', 'y'])
    return problem, np.array([0.3, 0.05]), None


# ---------------------------------------------------------------------------
def floudas():
    """Floudas heat-exchanger design, paper Equation 17.

    Eight variables, six constraints, only one of which is a posynomial -- the
    rest carry negative coefficients and so are genuine signomials. That mix is
    what makes it a useful discriminator between the algorithms.

    Constraints are stated in the paper in a form with negative terms, e.g.

    .. math:: \\frac{833.33252 x_4}{x_2 x_6} + \\frac{100}{x_6}
              - \\frac{83333.333}{x_1 x_6} \\le 1

    A signomial cannot be a :class:`Posynomial`, so each is supplied as a
    callback with an analytic gradient.
    """
    n = 8

    # objective: x1 + x2 + x3 (a posynomial)
    objective = Posynomial([
        (1.0, [1, 0, 0, 0, 0, 0, 0, 0]),
        (1.0, [0, 1, 0, 0, 0, 0, 0, 0]),
        (1.0, [0, 0, 1, 0, 0, 0, 0, 0]),
    ], n)

    def _signomial(terms):
        """Build a value/gradient callback for sum_k c_k prod_j x_j^a_kj.

        Unlike :class:`Posynomial` this permits negative ``c_k``.
        """
        terms = [(float(c), np.asarray(a, dtype=float)) for c, a in terms]

        def fn(x):
            x = np.asarray(x, dtype=float)
            v = 0.0
            g = np.zeros(len(x))
            for c, a in terms:
                t = c * np.prod(x ** a)
                v += t
                g += t * a / x
            return v, g

        return fn

    from lcsolver.solvers.sequential.slcp import Signomial

    constraints = [
        # 833.33252 x4/(x1 x6) + 100/x6 - 83333.333/(x1 x6) <= 1
        #
        # NOTE: paper Equation 17 prints the first term's denominator as x2 x6,
        # but both the notebook (`833.33252/x1*x4/x6`) and the standard Floudas
        # statement of this problem use x1. With x2 the published optimum is not
        # even feasible, so the paper has a typographical error here; the code
        # that produced the published results used x1.
        Constraint(Signomial(_signomial([
            (833.33252,  [-1, 0, 0, 1, 0, -1, 0, 0]),
            (100.0,      [0,  0, 0, 0, 0, -1, 0, 0]),
            (-83333.333, [-1, 0, 0, 0, 0, -1, 0, 0]),
        ]), n), '<='),
        # 1250 x5/(x2 x7) + x4/x7 - 1250 x4/(x2 x7) <= 1
        Constraint(Signomial(_signomial([
            (1250.0,  [0, -1, 0, 0, 1, 0, -1, 0]),
            (1.0,     [0,  0, 0, 1, 0, 0, -1, 0]),
            (-1250.0, [0, -1, 0, 1, 0, 0, -1, 0]),
        ]), n), '<='),
        # 1250000/(x3 x8) + x5/x8 - 2500 x5/(x3 x8) <= 1
        Constraint(Signomial(_signomial([
            (1250000.0, [0, 0, -1, 0, 0, 0, 0, -1]),
            (1.0,       [0, 0,  0, 0, 1, 0, 0, -1]),
            (-2500.0,   [0, 0, -1, 0, 1, 0, 0, -1]),
        ]), n), '<='),
        # 0.0025 x4 + 0.0025 x6 <= 1   (a genuine posynomial)
        Constraint(Posynomial([
            (0.0025, [0, 0, 0, 1, 0, 0, 0, 0]),
            (0.0025, [0, 0, 0, 0, 0, 1, 0, 0]),
        ], n), '<='),
        # -0.0025 x4 + 0.0025 x5 + 0.0025 x7 <= 1
        Constraint(Signomial(_signomial([
            (-0.0025, [0, 0, 0, 1, 0, 0, 0, 0]),
            (0.0025,  [0, 0, 0, 0, 1, 0, 0, 0]),
            (0.0025,  [0, 0, 0, 0, 0, 0, 1, 0]),
        ]), n), '<='),
        # -0.01 x5 + 0.01 x8 <= 1
        Constraint(Signomial(_signomial([
            (-0.01, [0, 0, 0, 0, 1, 0, 0, 0]),
            (0.01,  [0, 0, 0, 0, 0, 0, 0, 1]),
        ]), n), '<='),
    ]

    problem = Problem(n, objective, constraints,
                      names=[f'x{i + 1}' for i in range(n)])

    # Published optimum (Floudas, via the notebook's `optimum` dict).
    reference = {
        'x': np.array([579.307, 1359.971, 5109.971, 182.018,
                       295.601, 217.982, 286.417, 395.601]),
        'objective': 579.307 + 1359.971 + 5109.971,
        'source': 'published value (Floudas)',
    }
    # The notebook's guess, which is close to the optimum.
    x0 = np.array([580.0, 1360.0, 5110.0, 181.0, 290.0, 220.0, 290.0, 400.0])
    return problem, x0, reference


# ---------------------------------------------------------------------------
KO_NAMES = ['A', 'C_D', 'C_L', 'C_f', 'D', 'Re', 'S', 't', 'V', 'V_f',
            'V_f_avail', 'V_f_fuse', 'V_f_wing', 'W', 'W_f', 'W_w',
            'W_w_strc', 'W_w_surf']

# The notebook's declared guesses, converted to SI (time in seconds, volumes m^3).
KO_X0 = np.array([
    10.0,     # A
    0.025,    # C_D
    0.5,      # C_L
    0.003,    # C_f
    300.0,    # D
    3.0e6,    # Re
    10.0,     # S
    6000.0,   # t
    30.0,     # V
    0.3,      # V_f
    0.3,      # V_f_avail
    0.3,      # V_f_fuse
    0.3,      # V_f_wing
    10000.0,  # W
    2500.0,   # W_f
    2500.0,   # W_w
    2500.0,   # W_w_strc
    2500.0,   # W_w_surf
])


def _ko_fuel_volume(names):
    """``V_f_avail / (V_f_wing + V_f_fuse) <= 1``, paper Equation 18.

    A ratio, not a sum of monomials, so it needs its own value/gradient callback
    rather than the generic :func:`signomial` helper.
    """
    ia = names.index('V_f_avail')
    iw = names.index('V_f_wing')
    ifu = names.index('V_f_fuse')
    n = len(names)

    def fn(x):
        x = np.asarray(x, dtype=float)
        denom = x[iw] + x[ifu]
        g = np.zeros(n)
        g[ia] = 1.0 / denom
        g[iw] = g[ifu] = -x[ia] / denom ** 2
        return x[ia] / denom, g

    return Constraint(Signomial(fn, n), '<=')


def _ko_constraints(mode):
    """Kirschen-Ozturk constraints in ``body <op> 1`` form.

    Three variants, and the difference between them matters a great deal:

    ``mode=1``
        All inequalities, with the fuel-volume constraint replaced by the
        notebook's monomial surrogate. This is a pure geometric program and so
        supplies an exact reference optimum.
    ``mode=2``
        The notebook's comparison form: five constraints tightened to
        equalities. Every multi-term posynomial thereby becomes a signomial
        equality, leaving only monomials GP-compatible. Since a monomial is
        affine in log space, linearizing it is exact -- so on this variant SLCP
        and LSQP are *mathematically identical*, exactly as the paper's remark
        that the SLCP sub-problem "reverts to the LSQP sub-problem in the absence
        of posynomial constraints" would predict. Useful as a consistency check,
        useless as a comparison.
    ``mode=3`` (default)
        Paper Equation 18 as printed: all inequalities, with the true
        ``V_f_avail <= V_f_wing + V_f_fuse``. That is a monomial bounded below by
        a posynomial, which is *not* GP-compatible -- it is the single constraint
        that makes the problem a signomial program, as the paper states. It
        leaves the five multi-term posynomials intact for SLCP to impose
        exactly, so this is the variant on which the two algorithms can differ.

    All quantities are in SI base units (m, kg, s, N), so the unit-bearing
    constants of the notebook are converted here.
    """
    N = KO_NAMES
    pi = np.pi

    # Constants, converted to SI.
    C_Lmax, CDA0, e, g, k = 1.5, 0.06, 0.96, 9.81, 1.2
    mu, N_ult, rho, rho_f = 1.78e-5, 2.5, 1.23, 804.0      # kg/L -> kg/m^3
    R, S_wet, tau = 1.0e6, 2.05, 0.12                      # 1000 km -> m
    TSFC = 0.4 / 3600.0                                    # 1/hour -> 1/s
    V_min, W_0 = 22.0, 4940.0
    W_c1, W_c2 = 8.71e-5, 45.24

    eq = '==' if mode == 2 else '<='  # modes 1 and 3 keep inequalities

    return [
        # W_f >= TSFC t D
        con([(TSFC, {'t': 1, 'D': 1, 'W_f': -1})], N),
        # t >= R / V
        con([(R, {'V': -1, 't': -1})], N),
        # D >= 1/2 rho S C_D V^2
        con([(0.5 * rho, {'S': 1, 'C_D': 1, 'V': 2, 'D': -1})], N),
        # C_D >= CDA0/S + k C_f S_wet + C_L^2/(pi A e)
        con([(CDA0, {'S': -1, 'C_D': -1}),
             (k * S_wet, {'C_f': 1, 'C_D': -1}),
             (1.0 / (pi * e), {'C_L': 2, 'A': -1, 'C_D': -1})], N, eq),
        # C_f >= 0.074 / Re^0.2
        con([(0.074, {'Re': -0.2, 'C_f': -1})], N),
        # Re <= rho V (S/A)^0.5 / mu
        con([(mu, {'Re': 1, 'A': 0.5, 'V': -1, 'S': -0.5})], N),
        # 1/2 rho V^2 S C_L >= W_0 + W_w + W_f/2
        con([(W_0 / (0.5 * rho), {'V': -2, 'S': -1, 'C_L': -1}),
             (1.0 / (0.5 * rho), {'W_w': 1, 'V': -2, 'S': -1, 'C_L': -1}),
             (0.5 / (0.5 * rho), {'W_f': 1, 'V': -2, 'S': -1, 'C_L': -1})], N, eq),
        # 1/2 rho V_min^2 S C_Lmax >= W
        con([(1.0 / (0.5 * rho * V_min ** 2 * C_Lmax), {'W': 1, 'S': -1})], N),
        # W >= W_0 + W_w + W_f
        con([(W_0, {'W': -1}), (1.0, {'W_w': 1, 'W': -1}),
             (1.0, {'W_f': 1, 'W': -1})], N, eq),
        # W_w >= W_w_surf + W_w_strc
        con([(1.0, {'W_w_surf': 1, 'W_w': -1}),
             (1.0, {'W_w_strc': 1, 'W_w': -1})], N, eq),
        # W_w_surf >= W_c2 S
        con([(W_c2, {'S': 1, 'W_w_surf': -1})], N),
        # W_w_strc^2 >= W_c1^2 N_ult^2 A^3 (W_0 + V_f_fuse rho_f g) W S / tau^2
        con([(W_c1 ** 2 * N_ult ** 2 * W_0 / tau ** 2,
              {'A': 3, 'W': 1, 'S': 1, 'W_w_strc': -2}),
             (W_c1 ** 2 * N_ult ** 2 * rho_f * g / tau ** 2,
              {'A': 3, 'V_f_fuse': 1, 'W': 1, 'S': 1, 'W_w_strc': -2})], N, eq),
        # V_f <= V_f_avail
        con([(1.0, {'V_f': 1, 'V_f_avail': -1})], N),
        # V_f >= W_f / (g rho_f)
        con([(1.0 / (g * rho_f), {'W_f': 1, 'V_f': -1})], N),
        # Fuel volume available. Paper Equation 18 states
        #     V_f_avail <= V_f_wing + V_f_fuse,
        # a monomial bounded below by a posynomial, which is NOT GP-compatible;
        # this is the constraint that makes the problem a signomial program. The
        # notebook substitutes a monomial surrogate to obtain a pure GP for
        # reference purposes. Mode 3 uses the paper's form, modes 1 and 2 the
        # notebook's.
        (_ko_fuel_volume(N) if mode == 3
         else con([(1.0e-4, {'V_f_avail': 1, 'V_f_wing': -1, 'V_f_fuse': -1})], N)),
        # V_f_wing^2 <= 0.0009 S^3 tau^2 / A
        con([(1.0 / (0.0009 * tau ** 2), {'V_f_wing': 2, 'S': -3, 'A': 1})], N),
        # V_f_fuse <= CDA0 * 10 m
        con([(1.0 / (CDA0 * 10.0), {'V_f_fuse': 1})], N),
    ]


def _solve_as_gp(problem, x0):
    """Solve a pure GP exactly, in log space. Used to generate references."""
    import math

    import pyomo.environ as pyo

    n = problem.n
    m = pyo.ConcreteModel()
    m.J = pyo.RangeSet(0, n - 1)
    m.t = pyo.Var(m.J, initialize=lambda _m, j: math.log(x0[j]))

    def P(terms):
        return sum(pyo.exp(math.log(c) + sum(a[j] * m.t[j] for j in range(n)))
                   for c, a in terms)

    m.obj = pyo.Objective(expr=P(problem.objective.terms), sense=pyo.minimize)
    m.cons = pyo.ConstraintList()
    for c in problem.constraints:
        if not isinstance(c.body, Posynomial):
            raise ValueError('_solve_as_gp requires every constraint to be a posynomial')
        m.cons.add(P(c.body.terms) == 1.0 if c.operator == '=='
                   else P(c.body.terms) <= 1.0)

    opt = pyo.SolverFactory('ipopt')
    opt.options['print_level'] = 0
    opt.options['sb'] = 'yes'
    opt.solve(m)
    x = np.array([math.exp(pyo.value(m.t[j])) for j in range(n)])
    return {'x': x, 'objective': problem.objective_value(x),
            'source': 'solved exactly as a GP'}


def ko_reference(mode=3):
    """Reference optimum for the Kirschen-Ozturk problem.

    Modes 1 and 2 share the notebook's monomial fuel-volume surrogate and hence
    the same optimum, obtained exactly from the mode-1 GP. Mode 3 uses the
    paper's true posynomial fuel constraint, which is a *different* problem with
    a different optimum -- comparing a mode-3 solve against the mode-1 number
    reports a spurious 0.2% error. Mode 3 is therefore referenced against a
    direct IPOPT solve of the full nonlinear program, which is independent of
    the SLCP code under test.
    """
    if mode in (1, 2):
        gp_problem, x0, _ = kirschen_ozturk_gp()
        return _solve_as_gp(gp_problem, x0)

    import math

    import pyomo.environ as pyo

    N = KO_NAMES
    problem = Problem(len(N), posy([(1.0, {'W_f': 1})], N),
                      _ko_constraints(3), names=N)
    n = problem.n
    m = pyo.ConcreteModel()
    m.J = pyo.RangeSet(0, n - 1)
    m.t = pyo.Var(m.J, initialize=lambda _m, j: math.log(KO_X0[j]))
    x = [pyo.exp(m.t[j]) for j in range(n)]

    ia, iw, ifu = (N.index('V_f_avail'), N.index('V_f_wing'), N.index('V_f_fuse'))

    def val(body):
        if isinstance(body, Posynomial):
            return sum(c * np.prod([x[j] ** a[j] for j in range(n)])
                       for c, a in body.terms)
        return x[ia] / (x[iw] + x[ifu])          # the one signomial constraint

    m.obj = pyo.Objective(expr=val(problem.objective), sense=pyo.minimize)
    m.cons = pyo.ConstraintList()
    for c in problem.constraints:
        m.cons.add(val(c.body) == 1.0 if c.operator == '=='
                   else val(c.body) <= 1.0)

    opt = pyo.SolverFactory('ipopt')
    opt.options['print_level'] = 0
    opt.options['sb'] = 'yes'
    opt.solve(m)
    xs = np.array([math.exp(pyo.value(m.t[j])) for j in range(n)])
    return {'x': xs, 'objective': problem.objective_value(xs),
            'source': 'direct IPOPT solve of the full NLP'}


def kirschen_ozturk(mode=3):
    """Kirschen-Ozturk aircraft sizing, paper Equation 18.

    Minimize fuel weight over 18 variables. Attribution in the paper is to
    Kirschen (2018), with the problem originating in Hoburg (2014).

    The reference optimum is obtained by solving ``mode=1`` exactly as a
    geometric program, so it does not depend on a transcribed number.
    """
    N = KO_NAMES
    objective = posy([(1.0, {'W_f': 1})], N)
    problem = Problem(len(N), objective, _ko_constraints(mode), names=N)
    x0 = KO_X0.copy()
    return problem, x0, ko_reference(mode)


def kirschen_ozturk_gp():
    """The all-inequality (mode 1) form: a pure GP, used as the reference."""
    N = KO_NAMES
    objective = posy([(1.0, {'W_f': 1})], N)
    problem = Problem(len(N), objective, _ko_constraints(1), names=N)
    return problem, KO_X0.copy(), None


# ---------------------------------------------------------------------------
# Hoburg UAV problem
# ---------------------------------------------------------------------------
_HOB_SEG = 3

HOBURG_NAMES = (
    ['AR', 'I_cap_bar', 'M_r_bar', 'nu', 'p', 'P_max', 'q', 'R', 'S',
     't_cap_bar', 't_web_bar', 'tau', 'V_stall', 'W_cap', 'W_eng',
     'W_fuel_out', 'W_fuel_ret', 'W_MTO', 'W_pay', 'W_tilde', 'W_web',
     'W_wing', 'W_zfw']
    + [f'{b}_{i}' for b in ('V', 'C_L', 'C_D', 'C_Dfuse', 'C_Dp', 'C_Di',
                            'T', 'W', 'Re', 'eta_i', 'eta_prop', 'eta_0')
       for i in range(_HOB_SEG)]
    + ['z_bre_0', 'z_bre_1']
)

# The notebook also declares C_f_i and z_bre_2, which no constraint references.
# corsairlite keeps them alive with a blanket `>= 1e-12` bound; here they are
# simply omitted, since an unreferenced variable makes the Lagrangian Hessian
# singular without changing the solution.

# Guesses, chosen to be physically sensible (the notebook's blanket 1.0 for every
# vector variable is far outside the feasible set).
HOBURG_X0 = np.array(
    [12.0, 1.0e-2, 1.0e4, 0.8, 2.0, 1.0e5, 1.5, 5.0e6, 40.0,
     0.05, 0.05, 0.12, 35.0, 600.0, 5000.0,
     3000.0, 3000.0, 40000.0, 4905.0, 25000.0, 400.0,
     2000.0, 30000.0]
    + [60.0, 60.0, 150.0]            # V
    + [0.6, 0.6, 0.2]                # C_L
    + [0.03, 0.03, 0.03]             # C_D
    + [1.5e-3, 1.5e-3, 1.5e-3]       # C_Dfuse
    + [0.01, 0.01, 0.01]             # C_Dp
    + [0.01, 0.01, 0.005]            # C_Di
    + [1500.0, 1500.0, 4000.0]       # T
    + [3.0e4, 3.0e4, 3.0e4]          # W
    + [3.0e6, 3.0e6, 8.0e6]          # Re
    + [0.8, 0.8, 0.8]                # eta_i
    + [0.7, 0.7, 0.7]                # eta_prop
    + [0.25, 0.25, 0.25]             # eta_0
    + [0.3, 0.3]                     # z_bre
)


def _hoburg_constraints(names):
    """Hoburg UAV constraints, paper Section 7.3, in ``body <op> 1`` form.

    Three mission segments: outbound, return, and a sprint condition that sizes
    the powerplant. SI base units throughout.

    This is the ``mode=0`` variant of the notebook -- the profile-drag model is
    the five-term posynomial fit, so the whole problem is GP-compatible. That
    makes it the most favourable case for SLCP, which can impose every
    constraint exactly, and it is the variant of paper Section 7.4 ("Hoburg
    Problem as Formulated").
    """
    N = names
    pi = np.pi
    S3 = range(_HOB_SEG)

    # Constants (SI). Atmosphere at 3000 m and sea level read from the exact
    # table rows in standardAtmosphere.py: rho=0.909122, nu=1.71e-5.
    A_prop, CDA0, C_Lmax, e = 0.785, 0.05, 1.5, 0.95
    eta_eng, eta_v, f_wadd, g = 0.35, 0.85, 2.0, 9.81
    h_fuel, k_ew = 46.0e6, 0.0372
    rho, rho_SL = 0.909122, 1.225
    mu_air = rho * 1.71e-5
    N_lift, r_h = 6.0, 0.75
    rho_cap = rho_web = 2700.0
    sigma_max, sigma_max_shear = 310.0e6, 167.0e6
    w_bar, W_fixed = 0.5, 14700.0

    C = []

    # --- steady level flight ------------------------------------------------
    for i in S3:
        # W_i == 1/2 rho V_i^2 C_L_i S
        C.append(con([(0.5 * rho, {f'V_{i}': 2, f'C_L_{i}': 1, 'S': 1,
                                   f'W_{i}': -1})], N, '=='))
    for i in S3:
        # T_i >= 1/2 rho V_i^2 C_D_i S
        C.append(con([(0.5 * rho, {f'V_{i}': 2, f'C_D_{i}': 1, 'S': 1,
                                   f'T_{i}': -1})], N))
    for i in S3:
        # Re_i == rho V_i sqrt(S) / (sqrt(AR) mu)
        C.append(con([(rho / mu_air, {f'V_{i}': 1, 'S': 0.5, 'AR': -0.5,
                                      f'Re_{i}': -1})], N, '=='))

    # --- landing -------------------------------------------------------------
    C.append(con([(0.5 * rho_SL * C_Lmax, {'V_stall': 2, 'S': 1, 'W_MTO': -1})],
                 N, '=='))
    C.append(con([(1.0 / 38.0, {'V_stall': 1})], N))

    # --- sprint --------------------------------------------------------------
    C.append(con([(1.0, {'T_2': 1, 'V_2': 1, 'eta_0_2': -1, 'P_max': -1})], N))
    C.append(con([(150.0, {'V_2': -1})], N))

    # --- drag model ----------------------------------------------------------
    for i in S3:
        C.append(con([(CDA0, {'S': -1, f'C_Dfuse_{i}': -1})], N, '=='))
    for i in S3:
        C.append(con([(1.0 / (pi * e), {f'C_L_{i}': 2, 'AR': -1,
                                        f'C_Di_{i}': -1})], N, '=='))
    for i in S3:
        # C_D_i >= C_Dfuse_i + C_Dp_i + C_Di_i   (posynomial)
        C.append(con([(1.0, {f'C_Dfuse_{i}': 1, f'C_D_{i}': -1}),
                      (1.0, {f'C_Dp_{i}': 1, f'C_D_{i}': -1}),
                      (1.0, {f'C_Di_{i}': 1, f'C_D_{i}': -1})], N))

    # --- propulsive efficiency ----------------------------------------------
    for i in S3:
        C.append(con([(eta_eng, {f'eta_prop_{i}': 1, f'eta_0_{i}': -1})], N, '=='))
    for i in S3:
        C.append(con([(eta_v, {f'eta_i_{i}': 1, f'eta_prop_{i}': -1})], N, '=='))
    for i in S3:
        # 4 eta_i + T eta_i^2 / (1/2 rho V^2 A_prop) <= 4   (posynomial)
        C.append(con([(1.0, {f'eta_i_{i}': 1}),
                      (1.0 / (4.0 * 0.5 * rho * A_prop),
                       {f'T_{i}': 1, f'eta_i_{i}': 2, f'V_{i}': -2})], N))

    # --- range ---------------------------------------------------------------
    C.append(con([(5.0e6, {'R': -1})], N))
    for i in range(_HOB_SEG - 1):
        # z_bre_i == g R T_i / (h_fuel eta_0_i W_i)
        C.append(con([(g / h_fuel, {'R': 1, f'T_{i}': 1, f'eta_0_{i}': -1,
                                    f'W_{i}': -1, f'z_bre_{i}': -1})], N, '=='))
    # Breguet series: W_fuel/W >= z + z^2/2 + z^3/6 + z^4/24
    for i, wf in ((0, 'W_fuel_out'), (1, 'W_fuel_ret')):
        z = f'z_bre_{i}'
        C.append(con([(1.0, {z: 1, f'W_{i}': 1, wf: -1}),
                      (0.5, {z: 2, f'W_{i}': 1, wf: -1}),
                      (1.0 / 6.0, {z: 3, f'W_{i}': 1, wf: -1}),
                      (1.0 / 24.0, {z: 4, f'W_{i}': 1, wf: -1})], N))

    # --- weight --------------------------------------------------------------
    C += [
        con([(500.0 * g, {'W_pay': -1})], N),
        con([(W_fixed, {'W_tilde': -1}), (1.0, {'W_pay': 1, 'W_tilde': -1}),
             (1.0, {'W_eng': 1, 'W_tilde': -1})], N),
        con([(1.0, {'W_tilde': 1, 'W_zfw': -1}),
             (1.0, {'W_wing': 1, 'W_zfw': -1})], N),
        con([(k_ew, {'P_max': 0.803, 'W_eng': -1})], N),
        con([(f_wadd, {'W_web': 1, 'W_wing': -1}),
             (f_wadd, {'W_cap': 1, 'W_wing': -1})], N),
        con([(1.0, {'W_zfw': 1, 'W_0': -1}),
             (1.0, {'W_fuel_ret': 1, 'W_0': -1})], N),
        con([(1.0, {'W_0': 1, 'W_MTO': -1}),
             (1.0, {'W_fuel_out': 1, 'W_MTO': -1})], N),
        con([(1.0, {'W_zfw': 1, 'W_1': -1})], N),
        con([(1.0, {'W_0': 1, 'W_2': -1})], N, '=='),
    ]

    # --- wing structure ------------------------------------------------------
    C += [
        con([(0.5, {'q': -1}), (0.5, {'p': 1, 'q': -1})], N),      # 2q >= 1 + p
        con([(1.9, {'p': -1})], N),
        con([(1.0 / 0.15, {'tau': 1})], N),
        con([(1.0 / 24.0, {'W_tilde': 1, 'AR': 1, 'p': 1, 'M_r_bar': -1})],
            N, '=='),
        # 0.92 w_bar tau t_cap^2 + I_cap <= 0.92^2/2 w_bar tau^2 t_cap
        con([(0.92 / (0.92 ** 2 / 2), {'t_cap_bar': 1, 'tau': -1}),
             (1.0 / (0.92 ** 2 / 2 * w_bar), {'I_cap_bar': 1, 'tau': -2,
                                              't_cap_bar': -1})], N),
        con([(N_lift / (8.0 * sigma_max),
              {'M_r_bar': 1, 'AR': 1, 'q': 2, 'tau': 1, 'S': -1,
               'I_cap_bar': -1})], N, '=='),
        con([(N_lift / (12.0 * sigma_max_shear),
              {'AR': 1, 'W_tilde': 1, 'q': 3, 'tau': -1, 'S': -1,
               't_web_bar': -1})], N, '=='),
        # nu^3.94 >= 0.86 p^-2.38 + 0.14 p^0.56
        con([(0.86, {'p': -2.38, 'nu': -3.94}),
             (0.14, {'p': 0.56, 'nu': -3.94})], N),
        con([(8.0 * rho_cap * g * w_bar / 3.0,
              {'t_cap_bar': 1, 'S': 1.5, 'nu': 1, 'AR': -0.5, 'W_cap': -1})], N),
        con([(8.0 * rho_web * g * r_h / 3.0,
              {'tau': 1, 't_web_bar': 1, 'S': 1.5, 'nu': 1, 'AR': -0.5,
               'W_web': -1})], N),
    ]

    # --- profile drag: the five-term posynomial fit --------------------------
    for i in S3:
        cl, re, cdp = f'C_L_{i}', f'Re_{i}', f'C_Dp_{i}'
        C.append(con([
            (2.56,    {cl: 5.88, 'tau': -3.32, re: -1.54, cdp: -2.26}),
            (3.80e-9, {cl: -0.92, 'tau': 6.23, re: -1.38, cdp: -9.57}),
            (2.20e-3, {cl: -0.01, 'tau': 0.03, re: 0.14, cdp: -0.73}),
            (1.19e4,  {cl: 9.78, 'tau': 1.76, re: -1.00, cdp: -0.91}),
            (6.14e-6, {cl: 6.53, 'tau': -0.52, re: -0.99, cdp: -5.19}),
        ], N))

    return C


# Coefficients of the profile-drag fit, paper Equation "dragFit":
#   1 >= sum_k  a_k  C_L^b_k  tau^c_k  Re^d_k  C_Dp^e_k
_DRAG_FIT = [
    (2.56,     5.88, -3.32, -1.54, -2.26),
    (3.80e-9, -0.92,  6.23, -1.38, -9.57),
    (2.20e-3, -0.01,  0.03,  0.14, -0.73),
    (1.19e4,   9.78,  1.76, -1.00, -0.91),
    (6.14e-6,  6.53, -0.52, -0.99, -5.19),
]


def _profile_drag_blackbox(C_L, Re, tau):
    """``C_Dp = f(C_L, Re, tau)``, by implicitly solving the drag fit.

    Returns ``(C_Dp, dC_Dp/dC_L, dC_Dp/dRe, dC_Dp/dtau)``.

    This is the black box of paper Sections 7.5 and 7.6. It deliberately returns
    *exactly* what the explicit five-term posynomial constraint would give at its
    active bound -- the physics is unchanged and so is the optimum. All that
    changes is that the posynomial is now hidden from SLCP, which must linearize
    it like any other opaque analysis code. That is what makes this a clean
    measurement of what exact posynomial handling is worth.

    Every exponent on ``C_Dp`` is negative, so ``F`` is strictly decreasing in
    ``C_Dp`` and the root is unique. A bracketed root find is used rather than
    the reference implementation's ``scipy.optimize.minimize`` on a squared
    residual, which is both slower and less reliable for a scalar monotone root.
    """
    from scipy.optimize import brentq

    def F(cdp):
        return sum(a * C_L ** b * tau ** c * Re ** d * cdp ** e
                   for a, b, c, d, e in _DRAG_FIT) - 1.0

    C_Dp = brentq(F, 1e-8, 1.0, xtol=1e-14, rtol=1e-14)

    # Implicit differentiation: df/dz = -(dF/dz) / (dF/dC_Dp)
    dF_dcdp = dF_dCL = dF_dRe = dF_dtau = 0.0
    for a, b, c, d, e in _DRAG_FIT:
        t = a * C_L ** b * tau ** c * Re ** d * C_Dp ** e
        dF_dcdp += e * t / C_Dp
        dF_dCL += b * t / C_L
        dF_dRe += d * t / Re
        dF_dtau += c * t / tau

    return (C_Dp, -dF_dCL / dF_dcdp, -dF_dRe / dF_dcdp, -dF_dtau / dF_dcdp)


def _hoburg_blackbox_constraint(names, seg):
    """``f(C_L_i, Re_i, tau) / C_Dp_i <= 1``, paper Equation "cdpCon"."""
    icl = names.index(f'C_L_{seg}')
    ire = names.index(f'Re_{seg}')
    ita = names.index('tau')
    icd = names.index(f'C_Dp_{seg}')
    n = len(names)

    def fn(x):
        x = np.asarray(x, dtype=float)
        cdp_bb, d_cl, d_re, d_tau = _profile_drag_blackbox(x[icl], x[ire], x[ita])
        value = cdp_bb / x[icd]
        g = np.zeros(n)
        g[icl] = d_cl / x[icd]
        g[ire] = d_re / x[icd]
        g[ita] = d_tau / x[icd]
        g[icd] = -cdp_bb / x[icd] ** 2
        return value, g

    return Constraint(Signomial(fn, n), '<=')


def hoburg(n_blackbox=0):
    """Hoburg UAV conceptual design, paper Section 7.

    ``n_blackbox`` selects the variant:

    ``0``
        Section 7.4, "as formulated". Every profile-drag constraint is the
        explicit five-term posynomial, so the whole model is GP-compatible and
        SLCP can impose all of it exactly.
    ``1``
        Section 7.5. The sprint-segment drag constraint is replaced by a black
        box. The paper reports the SLCP advantage shrinking to 6-8%.
    ``3``
        Section 7.6. All three drag constraints are black-boxed, shrinking the
        gap further -- and, being fully opaque, allowing a higher-fidelity model
        to be swapped in.

    The optimum is identical in all three, because the black box solves the same
    fit implicitly. Only the algorithm's visibility into the structure changes.
    """
    if n_blackbox not in (0, 1, 3):
        raise ValueError('n_blackbox must be 0, 1, or 3')

    N = HOBURG_NAMES
    objective = posy([(1.0, {'W_fuel_out': 1}), (1.0, {'W_fuel_ret': 1})], N)
    constraints = _hoburg_constraints(N)

    if n_blackbox:
        # The drag-fit constraints are the last three appended.
        segments = [2] if n_blackbox == 1 else [0, 1, 2]
        constraints = constraints[:-_HOB_SEG] + [
            (_hoburg_blackbox_constraint(N, i) if i in segments
             else constraints[-_HOB_SEG + i])
            for i in range(_HOB_SEG)
        ]

    problem = Problem(len(N), objective, constraints, names=N)
    return problem, HOBURG_X0.copy(), None


def hoburg_bb1():
    """Hoburg with one black-boxed constraint, paper Section 7.5."""
    return hoburg(1)


def hoburg_bb3():
    """Hoburg with three black-boxed constraints, paper Section 7.6."""
    return hoburg(3)


CASES = {
    'simple': simple_example,
    'floudas': floudas,
    'ko': kirschen_ozturk,
    'hoburg': hoburg,
    'hoburg_bb1': hoburg_bb1,
    'hoburg_bb3': hoburg_bb3,
}
