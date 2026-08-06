# ===========
# Description
# ===========
# The Kirschen-Ozturk aircraft sizing problem, a Signomial Program
# From:  Ozturk and Kirschen's SimPleAC-style wing and fuel sizing model,
#        as stated in the SLCP paper, Equation 18
#
# Every constraint but one is GP-compatible. The exception is the fuel
# volume availability row,
#
#     V_f_avail <= V_f_wing + V_f_fuse
#
# a monomial bounded below by a posynomial, which is what makes this a
# signomial program: it cannot be solved as a pure GP, and the router sends
# it to SIA.
#
# This statement carries full units (the range in km, the TSFC in 1/hr) and
# the dimensionally consistent Reynolds relation Re == rho V sqrt(S/A) / mu.
# It solves to W_f = 870.80 N. The SLCP paper apparatus in
# utilities/slcp_cases.py keeps the paper's dimensionless transcription
# (whose Re row omits rho) and lands at 892.68 N against its own reference.

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

A         = f.Variable(name="A"        , guess = 10.0    , units = "-"    , description="Aspect ratio")
C_D       = f.Variable(name="C_D"      , guess = 0.025   , units = "-"    , description="Total drag coefficient")
C_L       = f.Variable(name="C_L"      , guess = 0.5     , units = "-"    , description="Lift coefficient of wing")
C_f       = f.Variable(name="C_f"      , guess = 0.003   , units = "-"    , description="Skin friction coefficient")
D         = f.Variable(name="D"        , guess = 300.0   , units = "N"    , description="Total drag force")
Re        = f.Variable(name="Re"       , guess = 3e6     , units = "-"    , description="Reynold's number")
S         = f.Variable(name="S"        , guess = 10.0    , units = "m^2"  , description="Total wing area")
t         = f.Variable(name="t"        , guess = 6000.0  , units = "s"    , description="Flight time")
V         = f.Variable(name="V"        , guess = 30.0    , units = "m/s"  , description="Cruise speed")
V_f       = f.Variable(name="V_f"      , guess = 0.3     , units = "m^3"  , description="Fuel volume")
V_f_avail = f.Variable(name="V_f_avail", guess = 0.3     , units = "m^3"  , description="Fuel volume available")
V_f_fuse  = f.Variable(name="V_f_fuse" , guess = 0.3     , units = "m^3"  , description="Fuselage fuel volume")
V_f_wing  = f.Variable(name="V_f_wing" , guess = 0.3     , units = "m^3"  , description="Wing fuel volume")
W         = f.Variable(name="W"        , guess = 10000.0 , units = "N"    , description="Total aircraft weight")
W_f       = f.Variable(name="W_f"      , guess = 2500.0  , units = "N"    , description="Fuel weight")
W_w       = f.Variable(name="W_w"      , guess = 2500.0  , units = "N"    , description="Wing weight")
W_w_strc  = f.Variable(name="W_w_strc" , guess = 2500.0  , units = "N"    , description="Wing structural weight")
W_w_surf  = f.Variable(name="W_w_surf" , guess = 2500.0  , units = "N"    , description="Wing surface weight")

# =================
# Declare Constants
# =================
C_Lmax    = f.Constant( name="C_Lmax" , value=1.5     , units="-"      , description="Max CL with flaps down")
CDA0      = f.Constant( name="CDA0"   , value=0.06    , units="m^2"    , description="Fuselage drag area")
e         = f.Constant( name="e"      , value=0.96    , units="-"      , description="Oswald efficiency factor")
g         = f.Constant( name="g"      , value=9.81    , units="m/s^2"  , description="Gravitational acceleration")
k         = f.Constant( name="k"      , value=1.2     , units="-"      , description="Form factor")
mu        = f.Constant( name="mu"     , value=1.78e-5 , units="kg/m/s" , description="Viscosity of air")
N_ult     = f.Constant( name="N_ult"  , value=2.5     , units="-"      , description="Ultimate load factor")
rho       = f.Constant( name="rho"    , value=1.23    , units="kg/m^3" , description="Density of air")
rho_f     = f.Constant( name="rho_f"  , value=804.0   , units="kg/m^3" , description="Density of fuel")
R         = f.Constant( name="R"      , value=1000.0  , units="km"     , description="Required range")
S_wetratio= f.Constant( name="Srat"   , value=2.05    , units="-"      , description="Wetted area ratio")
tau       = f.Constant( name="tau"    , value=0.12    , units="-"      , description="Airfoil thickness to chord ratio")
TSFC      = f.Constant( name="TSFC"   , value=0.4     , units="1/hr"   , description="Thrust specific fuel consumption")
V_min     = f.Constant( name="V_min"  , value=22.0    , units="m/s"    , description="Takeoff speed")
W_0       = f.Constant( name="W_0"    , value=4940.0  , units="N"      , description="Aircraft weight excluding wing")
W_c1      = f.Constant( name="W_c1"   , value=8.71e-5 , units="1/m"    , description="Wing weight coefficient 1")
W_c2      = f.Constant( name="W_c2"   , value=45.24   , units="Pa"     , description="Wing weight coefficient 2")
l_fuse    = f.Constant( name="l_fuse" , value=10.0    , units="m"      , description="Fuselage fuel tank length")

# =====================
# Declare the Objective
# =====================
f.Objective(W_f)

# =======================
# Declare the Constraints
# =======================
pi = np.pi

f.ConstraintList([
    # Fuel burn and range
    W_f >= TSFC * t * D,
    t >= R / V,

    # Drag build-up
    D >= 0.5 * rho * S * C_D * V**2,
    C_D >= CDA0 / S + k * C_f * S_wetratio + C_L**2 / (pi * A * e),
    C_f >= 0.074 / Re**0.2,
    Re == (rho / mu) * V * (S / A)**0.5,

    # Lift and landing
    0.5 * rho * V**2 * S * C_L >= W_0 + W_w + 0.5 * W_f,
    W <= 0.5 * rho * V_min**2 * S * C_Lmax,

    # Weight build-up
    W >= W_0 + W_w + W_f,
    W_w >= W_w_surf + W_w_strc,
    W_w_surf >= W_c2 * S,
    W_w_strc**2 >= W_c1**2 * N_ult**2 * A**3 * (W_0 + V_f_fuse * rho_f * g) * W * S / tau**2,

    # Fuel volume -- the third row is the one signomial constraint
    V_f <= V_f_avail,
    V_f >= W_f / (g * rho_f),
    V_f_avail <= V_f_wing + V_f_fuse,
    V_f_wing**2 <= 0.0009 * S**3 * tau**2 / A,
    V_f_fuse <= CDA0 * l_fuse,
    ])

# ===========
# Solve Model
# ===========
sol = lcsolver.solve(f)
print(sol.summary())
