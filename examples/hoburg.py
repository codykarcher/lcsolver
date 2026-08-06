# ===========
# Description
# ===========
# The Hoburg UAV sizing problem, formulated as a Geometric Program
# From:  Hoburg and Abbeel
#        Geometric Programming for Aircraft Design Optimization
#        AIAA Journal
#        2014
#
# A three segment mission -- outbound cruise, return cruise, sprint -- sized
# for minimum fuel: steady level flight, landing, drag build-up, propulsive
# efficiency, Breguet range, weight build-up and wing structure.
#
# Profile drag is the paper's five term posynomial fit, one row covering all
# three segments at once,
#
#     1 >= sum_k a_k C_L^b_k tau^c_k Re^d_k C_D_p^e_k
#
# See hoburg_blackbox.py for the same model with that row replaced by an
# opaque analysis code. The two are identical apart from that one constraint,
# and reach the same optimum.
#
# The wing structure is written in the paper's non-dimensional variables, so
# the last block recovers the dimensional geometry -- span, chords, spar
# thicknesses -- from them. That block costs the model its GP label: q == 1 +
# lambda is a posynomial equality, and a GP admits only monomial ones. So the
# statement is a Signomial Program AS WRITTEN, and the Report says so.
#
# It is a Geometric Program AS SOLVED. The presolve removes every row that
# blocked it, leaving something convex in log space -- one convex solve, global
# optimum. The router dispatches on the as-written class, which is why the
# Report also names SIA: that loop is handed a GP and converges in a couple of
# iterations. `structure_report` states both classifications side by side.

# =================
# Import Statements
# =================
import numpy as np

import lcsolver
from lcsolver import Formulation, units

# ===================
# Declare Formulation
# ===================
f = Formulation()

# =================
# Declare Variables
# =================
# The three mission segments are the elements of each size 3 vector variable:
# 0 = outbound cruise, 1 = return cruise, 2 = sprint. The two size 2 vectors,
# W_fuel and z_bre, cover the cruise segments only; the sprint burns no fuel.

AR        = f.Variable(name="AR"       , guess = 12.0                    , units = "-"  ,         description="Aspect ratio")
b         = f.Variable(name="b"        , guess = 20.0                    , units = "m"  ,         description="Wing span")
c_bar     = f.Variable(name="c_bar"    , guess = 1.5                     , units = "m"  ,         description="Mean chord")
C_D       = f.Variable(name="C_D"      , guess = [0.03, 0.03, 0.03]      , units = "-"  , size=3, description="Total drag coefficient")
C_D_fuse  = f.Variable(name="C_D_fuse" , guess = 1.5e-3                  , units = "-"  , size=3, description="Fuselage drag coefficient")
C_D_ind   = f.Variable(name="C_D_ind"  , guess = [0.01, 0.01, 0.005]     , units = "-"  , size=3, description="Induced drag coefficient")
C_D_p     = f.Variable(name="C_D_p"    , guess = 0.01                    , units = "-"  , size=3, description="Profile drag coefficient")
C_L       = f.Variable(name="C_L"      , guess = [0.6, 0.6, 0.2]         , units = "-"  , size=3, description="Lift coefficient")
c_root    = f.Variable(name="c_root"   , guess = 2.0                     , units = "m"  ,         description="Root chord")
c_tip     = f.Variable(name="c_tip"    , guess = 0.9                     , units = "m"  ,         description="Tip chord")
eta_0     = f.Variable(name="eta_0"    , guess = 0.25                    , units = "-"  , size=3, description="Overall propulsive efficiency")
eta_i     = f.Variable(name="eta_i"    , guess = 0.8                     , units = "-"  , size=3, description="Propeller inviscid efficiency")
eta_prop  = f.Variable(name="eta_prop" , guess = 0.7                     , units = "-"  , size=3, description="Propeller efficiency")
h_spar    = f.Variable(name="h_spar"   , guess = 0.15                    , units = "m"  ,         description="Spar box height")
I_cap_bar = f.Variable(name="I_cap_bar", guess = 1.0e-2                  , units = "-"  ,         description="Normalized spar cap area moment")
lam       = f.Variable(name="lambda"   , guess = 0.5                     , units = "-"  ,         description="Taper ratio, c_tip / c_root")
M_r_bar   = f.Variable(name="M_r_bar"  , guess = 1.0e4                   , units = "N"  ,         description="Root bending moment per unit chord")
M_root    = f.Variable(name="M_root"   , guess = 1.0e4                   , units = "N*m",         description="Root bending moment")
nu        = f.Variable(name="nu"       , guess = 0.8                     , units = "-"  ,         description="Taper integration factor")
p         = f.Variable(name="p"        , guess = 2.0                     , units = "-"  ,         description="Taper dummy variable, p = 1 + 2*lambda")
P_max     = f.Variable(name="P_max"    , guess = 1.0e5                   , units = "W"  ,         description="Maximum engine power")
q         = f.Variable(name="q"        , guess = 1.5                     , units = "-"  ,         description="Taper dummy variable, q = 1 + lambda")
R         = f.Variable(name="R"        , guess = 5.0e6                   , units = "m"  ,         description="Single segment range")
Re        = f.Variable(name="Re"       , guess = [3.0e6, 3.0e6, 8.0e6]   , units = "-"  , size=3, description="Reynold's number")
S         = f.Variable(name="S"        , guess = 40.0                    , units = "m^2",         description="Wing area")
T         = f.Variable(name="T"        , guess = [1500.0, 1500.0, 4000.0], units = "N"  , size=3, description="Thrust")
t_cap     = f.Variable(name="t_cap"    , guess = 0.01                    , units = "m"  ,         description="Spar cap thickness")
t_cap_bar = f.Variable(name="t_cap_bar", guess = 0.05                    , units = "-"  ,         description="Spar cap thickness per chord")
t_web     = f.Variable(name="t_web"    , guess = 0.01                    , units = "m"  ,         description="Spar web thickness")
t_web_bar = f.Variable(name="t_web_bar", guess = 0.05                    , units = "-"  ,         description="Spar web thickness per chord")
tau       = f.Variable(name="tau"      , guess = 0.12                    , units = "-"  ,         description="Airfoil thickness to chord ratio")
V         = f.Variable(name="V"        , guess = [60.0, 60.0, 150.0]     , units = "m/s", size=3, description="Flight speed")
V_stall   = f.Variable(name="V_stall"  , guess = 35.0                    , units = "m/s",         description="Stall speed")
W         = f.Variable(name="W"        , guess = 3.0e4                   , units = "N"  , size=3, description="Segment start weight")
W_cap     = f.Variable(name="W_cap"    , guess = 600.0                   , units = "N"  ,         description="Spar cap weight")
W_eng     = f.Variable(name="W_eng"    , guess = 5000.0                  , units = "N"  ,         description="Engine weight")
W_fuel    = f.Variable(name="W_fuel"   , guess = 3000.0                  , units = "N"  , size=2, description="Fuel burned")
W_MTO     = f.Variable(name="W_MTO"    , guess = 40000.0                 , units = "N"  ,         description="Maximum takeoff weight")
W_pay     = f.Variable(name="W_pay"    , guess = 4905.0                  , units = "N"  ,         description="Payload weight")
w_spar    = f.Variable(name="w_spar"   , guess = 0.75                    , units = "m"  ,         description="Spar box width")
W_tilde   = f.Variable(name="W_tilde"  , guess = 25000.0                 , units = "N"  ,         description="Dry weight less wing")
W_web     = f.Variable(name="W_web"    , guess = 400.0                   , units = "N"  ,         description="Spar web weight")
W_wing    = f.Variable(name="W_wing"   , guess = 2000.0                  , units = "N"  ,         description="Wing weight")
W_zfw     = f.Variable(name="W_zfw"    , guess = 30000.0                 , units = "N"  ,         description="Zero fuel weight")
z_bre     = f.Variable(name="z_bre"    , guess = 0.3                     , units = "-"  , size=2, description="Breguet factor")

# =================
# Declare Constants
# =================
A_prop      = f.Constant(name="A_prop"     , value=0.785    , units="m^2"   , description="Propeller disk area")
C_Lmax      = f.Constant(name="C_Lmax"     , value=1.5      , units="-"     , description="Max CL at landing")
CDA0        = f.Constant(name="CDA0"       , value=0.05     , units="m^2"   , description="Fuselage drag area")
e           = f.Constant(name="e"          , value=0.95     , units="-"     , description="Oswald efficiency factor")
eta_eng     = f.Constant(name="eta_eng"    , value=0.35     , units="-"     , description="Engine efficiency")
eta_v       = f.Constant(name="eta_v"      , value=0.85     , units="-"     , description="Propeller viscous efficiency")
f_wadd      = f.Constant(name="f_wadd"     , value=2.0      , units="-"     , description="Wing added weight fraction")
g           = f.Constant(name="g"          , value=9.81     , units="m/s^2" , description="Gravitational acceleration")
h_fuel      = f.Constant(name="h_fuel"     , value=46.0e6   , units="J/kg"  , description="Fuel specific energy")
k_ew        = f.Constant(name="k_ew"       , value=0.0372   , units="N"     , description="Engine weight coefficient")
m_pay       = f.Constant(name="m_pay"      , value=500.0    , units="kg"    , description="Required payload mass")
mu          = f.Constant(name="mu"         , value=1.5546e-5, units="kg/m/s", description="Viscosity of air at 3000 m")
N_lift      = f.Constant(name="N_lift"     , value=6.0      , units="-"     , description="Ultimate load factor")
p_min       = f.Constant(name="p_min"      , value=1.9      , units="-"     , description="Taper limit, p = 1 + 2*lambda")
P_ref       = f.Constant(name="P_ref"      , value=1.0      , units="W"     , description="Reference power of the engine weight fit")
r_h         = f.Constant(name="r_h"        , value=0.75     , units="-"     , description="Spar web height per thickness")
R_min       = f.Constant(name="R_min"      , value=5.0e6    , units="m"     , description="Required segment range")
rho         = f.Constant(name="rho"        , value=0.909122 , units="kg/m^3", description="Density of air at 3000 m")
rho_cap     = f.Constant(name="rho_cap"    , value=2700.0   , units="kg/m^3", description="Density of spar cap material")
rho_SL      = f.Constant(name="rho_SL"     , value=1.225    , units="kg/m^3", description="Density of air at sea level")
rho_web     = f.Constant(name="rho_web"    , value=2700.0   , units="kg/m^3", description="Density of spar web material")
sigma_max   = f.Constant(name="sigma_max"  , value=310.0e6  , units="Pa"    , description="Allowable tensile stress, spar cap")
sigma_shear = f.Constant(name="sigma_shear", value=167.0e6  , units="Pa"    , description="Allowable shear stress, spar web")
tau_max     = f.Constant(name="tau_max"    , value=0.15     , units="-"     , description="Maximum airfoil thickness to chord ratio")
V_sprint    = f.Constant(name="V_sprint"   , value=150.0    , units="m/s"   , description="Required sprint speed")
V_stall_max = f.Constant(name="V_stall_max", value=38.0     , units="m/s"   , description="Maximum allowed stall speed")
w_bar       = f.Constant(name="w_bar"      , value=0.5      , units="-"     , description="Spar box width per chord")
W_fixed     = f.Constant(name="W_fixed"    , value=14700.0  , units="N"     , description="Fixed weight")

# =====================
# Declare the Objective
# =====================
f.Objective(f.sum(W_fuel))

# =======================
# Declare the Constraints
# =======================
f.ConstraintList([
        # Steady Level Flight, all three segments at once
        W == 0.5 * rho * V**2 * C_L * S,
        T >= 0.5 * rho * V**2 * C_D * S,
        Re == rho * V * (S / AR) ** 0.5 / mu,

        # Landing
        W_MTO == 0.5 * rho_SL * V_stall**2 * C_Lmax * S,
        V_stall <= V_stall_max,

        # Sprint
        V[2] >= V_sprint,
        P_max >= T[2] * V[2] / eta_0[2],

        # Drag Model
        C_D >= C_D_fuse + C_D_p + C_D_ind,
        C_D_fuse == CDA0 / S,
        C_D_ind == C_L**2 / (np.pi * e * AR),

        # Propulsive Efficiency
        eta_0 == eta_eng * eta_prop,
        eta_prop == eta_v * eta_i,
        eta_i + T * eta_i**2 / (2 * rho * A_prop * V**2) <= 1.0 * units.dimensionless,

        # Breguet Range, over the two cruise segments
        R >= R_min,
        z_bre == g * R * T[:2] / (h_fuel * eta_0[:2] * W[:2]),
        W_fuel >= W[:2] * (z_bre + z_bre**2 / 2 + z_bre**3 / 6 + z_bre**4 / 24),

        # Weight Model
        W_pay >= m_pay * g,
        W_eng >= k_ew * (P_max / P_ref) ** 0.803,
        W_tilde >= W_fixed + W_pay + W_eng,
        W_wing >= f_wadd * (W_web + W_cap),
        W_zfw >= W_tilde + W_wing,

        # Weight at the start of each segment
        W[0] >= W_zfw + W_fuel[1],
        W[1] >= W_zfw,
        W[2] == W[0],
        W_MTO >= W[0] + W_fuel[0],

        # Wing Structure
        tau <= tau_max,
        p >= p_min,
        q >= (1 + p) / 2,
        0.86 * p**-2.38 + 0.14 * p**0.56 <= nu**3.94,
        M_r_bar == W_tilde * AR * p / 24,
        0.92 * w_bar * tau * t_cap_bar**2 + I_cap_bar
            <= 0.92**2 / 2 * w_bar * tau**2 * t_cap_bar,
        I_cap_bar == N_lift * M_r_bar * AR * q**2 * tau / (8 * S * sigma_max),
        t_web_bar == AR * W_tilde * N_lift * q**3 / (12 * tau * S * sigma_shear),
        W_cap >= 8 * rho_cap * g * w_bar * t_cap_bar * S**1.5 * nu / (3 * AR**0.5),
        W_web >= 8 * rho_web * g * r_h * tau * t_web_bar * S**1.5 * nu / (3 * AR**0.5),

        # Dimensional Recovery, from the non-dimensional structure above. None
        # of these feed back into the sizing; they only put the answer in
        # metres. q == 1 + lambda is the posynomial equality that makes the
        # statement an SP -- see the note at the top.
        b == (S * AR) ** 0.5,
        c_bar == (S / AR) ** 0.5,
        q == 1 + lam,
        c_root == 2 * c_bar / q,
        c_tip == c_root * lam,
        t_cap == t_cap_bar * c_bar,
        t_web == t_web_bar * c_bar,
        h_spar == r_h * tau * c_bar,
        w_spar == w_bar * c_bar,
        M_root == M_r_bar * c_bar,

        # Profile Drag, the paper's five term fit. C_L and Re are per segment
        # and tau is the one airfoil they share, so this single row applies to
        # all three segments.
          2.56    * C_L** 5.88 * tau**-3.32 * Re**-1.54 * C_D_p**-2.26
        + 3.80e-9 * C_L**-0.92 * tau** 6.23 * Re**-1.38 * C_D_p**-9.57
        + 2.20e-3 * C_L**-0.01 * tau** 0.03 * Re** 0.14 * C_D_p**-0.73
        + 1.19e4  * C_L** 9.78 * tau** 1.76 * Re**-1.00 * C_D_p**-0.91
        + 6.14e-6 * C_L** 6.53 * tau**-0.52 * Re**-0.99 * C_D_p**-5.19
            <= 1.0 * units.dimensionless,
    ])

# ===========
# Solve Model
# ===========
# `solve` detects the class, eliminates the posynomial equality in the presolve,
# and hands the SP loop what is left, which is convex in log space -- so the
# answer is a global optimum rather than a local one. It also runs the
# structural checks and computes the sensitivity of the optimum to every
# Constant.
sol = lcsolver.solve(f)

# The summary prints the objective, every variable with its units and
# description, then the sensitivity of the optimum to each Constant.
print(sol.summary())
