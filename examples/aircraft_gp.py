# ===========
# Description
# ===========
# A simple aircraft sizing problem, formulated as a Geometric Program
# From:  Hoburg and Abbeel
#        Geometric Programming for Aircraft Design Optimization
#        AIAA Journal
#        2014

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

A         = f.Variable(name="A"        , guess = 10.0     , units = "-"   , description="Aspect ratio")
C_D       = f.Variable(name="C_D"      , guess = 0.025    , units = "-"   , description="Total drag coefficient")
C_D_fuse  = f.Variable(name="C_D_fuse" , guess = 0.01     , units = "-"   , description="Fuselage drag coefficient")
C_D_wpar  = f.Variable(name="C_D_wpar" , guess = 0.01     , units = "-"   , description="Wing parasite drag coefficient")
C_D_ind   = f.Variable(name="C_D_ind"  , guess = 0.01     , units = "-"   , description="Induced drag coefficient")
C_f       = f.Variable(name="C_f"      , guess = 0.003    , units = "-"   , description="Skin friction coefficient")
C_L       = f.Variable(name="C_L"      , guess = 0.5      , units = "-"   , description="Lift coefficient of wing")
D         = f.Variable(name="D"        , guess = 300      , units = "N"   , description="Total drag force")
Re        = f.Variable(name="Re"       , guess = 3e6      , units = "-"   , description="Reynold's number")
S         = f.Variable(name="S"        , guess = 10.0     , units = "m^2" , description="Total wing area")
V         = f.Variable(name="V"        , guess = 30.0     , units = "m/s" , description="Cruise speed")
W         = f.Variable(name="W"        , guess = 10000.0  , units = "N"   , description="Total aircraft weight")
W_w       = f.Variable(name="W_w"      , guess = 2500     , units = "N"   , description="Wing weight")
W_w_strc  = f.Variable(name="W_w_strc" , guess = 1500     , units = "N"   , description="Wing structural weight")
W_w_surf  = f.Variable(name="W_w_surf" , guess = 1000     , units = "N"   , description="Wing surface weight")

# =================
# Declare Constants
# =================
C_Lmax      = f.Constant( name="C_Lmax"  , value=2.0     , units="-"      , description="Max CL with flaps down")
CDA0        = f.Constant( name="CDA0"    , value=0.0306  , units="m^2"    , description="Fuselage drag area")
e           = f.Constant( name="e"       , value=0.96    , units="-"      , description="Oswald efficiency factor")
k           = f.Constant( name="k"       , value=1.2     , units="-"      , description="Form factor")
mu          = f.Constant( name="mu"      , value=1.78e-5 , units="kg/m/s" , description="Viscosity of air")
N_ult       = f.Constant( name="N_ult"   , value=2.5     , units="-"      , description="Ultimate load factor")
rho         = f.Constant( name="rho"     , value=1.23    , units="kg/m^3" , description="Density of air")
S_wetratio  = f.Constant( name="Srat"    , value=2.05    , units="-"      , description="Wetted area ratio")
tau         = f.Constant( name="tau"     , value=0.12    , units="-"      , description="Airfoil thickness to chord ratio")
V_min       = f.Constant( name="V_min"   , value=22      , units="m/s"    , description="Takeoff speed")
W_0         = f.Constant( name="W_0"     , value=4940.0  , units="N"      , description="Aircraft weight excluding wing")
W_W_coeff1  = f.Constant( name="W_c1"    , value=8.71e-5 , units="1/m"    , description="Wing Weight Coefficient 1")
W_W_coeff2  = f.Constant( name="W_c2"    , value=45.24   , units="Pa"     , description="Wing Weight Coefficient 2" )

# =====================
# Declare the Objective
# =====================
f.Objective(D)

# =======================
# Declare the Constraints
# =======================
f.ConstraintList([
        # Drag Model
        D   >= 0.5 * rho * S * C_D * V**2,
        C_D >= C_D_fuse + C_D_wpar + C_D_ind,
        C_D_fuse >= CDA0 / S,
        C_D_wpar >= k * C_f * S_wetratio,
        C_D_ind >= C_L**2 / (np.pi * A * e),

        # Reference Quantities
        Re  == (rho / mu) * V * (S / A) ** 0.5,
        C_f == 0.074 / Re**0.2,

        # Weight Model
        W_w >= W_w_surf + W_w_strc,
        W_w_strc >= W_W_coeff1 * (N_ult * A**1.5 * (W_0 * W * S) ** 0.5) / tau,
        W_w_surf >= W_W_coeff2 * S,
        W  >= W_0 + W_w,

        # Level Flight Constraint
        W  == 0.5 * rho * S * C_L * V**2,

        # Stall Speed sets wing area
        W  == 0.5 * rho * S * C_Lmax * V_min**2,
    ])

# =======================
# Print Model (inherited from pyomo)
# =======================
# f.pprint()

# `solve` detects that this is a geometric program and solves it in log space,
# where it is convex, so the answer is a global optimum rather than a local
# one. It also runs the structural checks and computes the sensitivity of the
# optimum to every Constant.
sol = lcsolver.solve(f)

# Alternative method for solution extraction, more akin to base pyomo
# lcsolver.solve(f)
# sol = f.solution()

# The solution prints itself: objective, every variable with its units and
# description, then the sensitivities. An indexed variable prints one row per
# element.
print(sol.summary())

# Extract design variables
AR_solved = sol.variables('A')
S_solved  = sol.variables('S')

solved_var_dict = sol.variables()

# Extract constants used in the solve
e_solved = sol.constants('e')
tau_solved = sol.constants('tau')

solved_constants = sol.constants()

# Extract sensitivities percent change in objective vs percent change in value
e_sens   = sol.sensitivities('e')
tau_sens = sol.sensitivities('tau')

solved_sens_dict = sol.sensitivities()

# Extract dimensioned sensitivities delta objective vs delta value
e_dsens   = sol.dimensioned_sensitivities('e')
tau_dsens = sol.dimensioned_sensitivities('tau')

solved_dsens_dict = sol.dimensioned_sensitivities()
