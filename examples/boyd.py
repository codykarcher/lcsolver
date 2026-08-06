# ===========
# Description
# ===========
# The box design problem, formulated as a Geometric Program
# From:  Boyd, Kim, Vandenberghe, and Hassibi
#        A Tutorial on Geometric Programming
#        Optimization and Engineering
#        2007
#
# Maximize the volume of a box (stated as minimizing the inverse volume)
# subject to a wall area limit, a floor area limit, and aspect ratio limits
# on both the height and the depth. Used as the introductory test problem in
# the SLCP paper; the optimum is 5.196e-3 1/m^3 (a volume of 192.45 m^3) at
# w = 5.774 m, h = 2.887 m, d = 11.547 m.

# =================
# Import Statements
# =================
import lcsolver
from lcsolver import Formulation, units

# ===================
# Declare Formulation
# ===================
f = Formulation()

# =================
# Declare Variables
# =================
w = f.Variable(name="w", guess = 5.0, units = "m", description="Box width")
h = f.Variable(name="h", guess = 5.0, units = "m", description="Box height")
d = f.Variable(name="d", guess = 5.0, units = "m", description="Box depth")

# =================
# Declare Constants
# =================
Aflr  = f.Constant( name="Aflr" , value=1000.0 , units="m^2" , description="Maximum floor area")
Awall = f.Constant( name="Awall", value=100.0  , units="m^2" , description="Maximum wall area")
alpha = f.Constant( name="alpha", value=0.5    , units="-"   , description="Minimum height aspect ratio h/w")
beta  = f.Constant( name="beta" , value=2.0    , units="-"   , description="Maximum height aspect ratio h/w")
gamma = f.Constant( name="gamma", value=0.5    , units="-"   , description="Minimum depth aspect ratio d/w")
delta = f.Constant( name="delta", value=2.0    , units="-"   , description="Maximum depth aspect ratio d/w")

# =====================
# Declare the Objective
# =====================
f.Objective(1 / (h * w * d))

# =======================
# Declare the Constraints
# =======================
f.ConstraintList([
    2 * (h * w + h * d) <= Awall,
    w * d <= Aflr,
    alpha <= h / w,
    h / w <= beta,
    gamma <= d / w,
    d / w <= delta,
    ])

# ===========
# Solve Model
# ===========
sol = lcsolver.solve(f)
print(sol.summary())
