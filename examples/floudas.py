# ===========
# Description
# ===========
# The heat exchanger design problem, a Signomial Program
# From:  Floudas and Pardalos
#        A Collection of Test Problems for Constrained Global Optimization
#        Algorithms (heat exchanger design), via the SLCP paper, Equation 17
#
# Three heat exchangers in series: minimize total exchanger area subject to
# the heat balance across each unit. Five of the six constraints carry
# negative coefficients, so they are genuine signomials rather than
# posynomials -- only one variable pair short of a GP, and a useful
# discriminator between sequential algorithms. The statement is
# dimensionless, as published.
#
# The published optimum is 7049.249 at
# x = (579.31, 1359.97, 5109.97, 182.02, 295.60, 217.98, 286.42, 395.60).
#
# Note: the SLCP paper's Equation 17 prints the first constraint's leading
# denominator as x2*x6, but the code that produced the published results
# (and the standard statement of this problem) uses x1*x6; with x2 the
# published optimum is not even feasible. The x1 form is used here.

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
x1 = f.Variable(name="x1", guess = 580.0  , units = "-", bounds = [100.0 , 10000.0], description="Exchanger 1 area")
x2 = f.Variable(name="x2", guess = 1360.0 , units = "-", bounds = [1000.0, 10000.0], description="Exchanger 2 area")
x3 = f.Variable(name="x3", guess = 5110.0 , units = "-", bounds = [1000.0, 10000.0], description="Exchanger 3 area")
x4 = f.Variable(name="x4", guess = 181.0  , units = "-", bounds = [10.0  , 1000.0 ], description="Stream temperature")
x5 = f.Variable(name="x5", guess = 290.0  , units = "-", bounds = [10.0  , 1000.0 ], description="Stream temperature")
x6 = f.Variable(name="x6", guess = 220.0  , units = "-", bounds = [10.0  , 1000.0 ], description="Stream temperature")
x7 = f.Variable(name="x7", guess = 290.0  , units = "-", bounds = [10.0  , 1000.0 ], description="Stream temperature")
x8 = f.Variable(name="x8", guess = 400.0  , units = "-", bounds = [10.0  , 1000.0 ], description="Stream temperature")

# =====================
# Declare the Objective
# =====================
f.Objective(x1 + x2 + x3)

# =======================
# Declare the Constraints
# =======================
one = 1.0 * units.dimensionless

f.ConstraintList([
    # Heat balance across each exchanger (signomial: negative terms)
    833.33252 * x4 / (x1 * x6) + 100.0 / x6 - 83333.333 / (x1 * x6) <= one,
    1250.0 * x5 / (x2 * x7) + x4 / x7 - 1250.0 * x4 / (x2 * x7) <= one,
    1250000.0 / (x3 * x8) + x5 / x8 - 2500.0 * x5 / (x3 * x8) <= one,

    # Temperature feasibility
    0.0025 * x4 + 0.0025 * x6 <= one,                  # the one posynomial
    -0.0025 * x4 + 0.0025 * x5 + 0.0025 * x7 <= one,
    -0.01 * x5 + 0.01 * x8 <= one,
    ])

# ===========
# Solve Model
# ===========
sol = lcsolver.solve(f)
print(sol.summary())
