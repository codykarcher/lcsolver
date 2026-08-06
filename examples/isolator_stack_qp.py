# ===========
# Description
# ===========
# An isolated instrument stack under sustained acceleration, formulated as a
# Quadratic Program
#
# Five shelves are stacked on vibration isolators, the bottom one mounted to
# structure, and the assembly is pulled through a 6 g axial load. A snubber
# above the top shelf stops it after a fixed clearance.
#
# The optimization here is not a design search. It is the equilibrium itself:
# a linear elastic assembly comes to rest at the deflection that minimizes
# total potential energy,
#
#     V(x) = 1/2 sum_i k_i (x_i - x_{i-1})^2  -  sum_i m_i a x_i
#
# strain energy stored in the isolators, less the work done by the load. Set
# the gradient of that to zero and you have K x = F, the stiffness equation.
# Written as a minimization instead, the snubber is just one more constraint,
# and its contact force falls out as the constraint's dual -- which is the
# whole reason to pose it this way rather than solve K x = F and check
# afterwards whether the answer violated the stop.
#
# V is a convex quadratic form in x and the constraint is affine, so the
# detector calls this a Quadratic Program: one convex solve, a global optimum,
# and exact duals -- which is what makes the last line of this file mean
# anything.
#
# The Hessian of V is the stiffness matrix K, and it is positive definite here
# because isolator 0 is grounded. It does not have to be: the detector asks
# for a convex objective, which is positive SEMI-definiteness, so a model
# where some variable is absent from the objective -- a control problem
# penalizing only its inputs, say -- is a QP too.

# =================
# Import Statements
# =================
import lcsolver
from lcsolver import Formulation, units

# ===================
# Declare Formulation
# ===================
f = Formulation()

N = 5

# =================
# Declare Variables
# =================
# One deflection per shelf, measured from the unloaded position, positive in
# the direction of the load. Shelf 0 sits on the mounting structure, so
# isolator 0 stretches by x[0] and isolator i by x[i] - x[i-1].
#
# The bounds are the isolators' travel limit -- an elastic model has nothing
# to say beyond it -- and they are also what satisfies the pre-solve gate,
# which refuses to solve a model whose variables are unbounded either way.

x = f.Variable(name="x", guess = 0.002, units = "m", size=N, bounds=[-0.05, 0.05], description="Shelf deflection under load")

# =================
# Declare Constants
# =================
a       = f.Constant( name="a"      , value=58.86                                        , units="m/s^2",         description="Sustained axial acceleration, 6 g")
gap     = f.Constant( name="gap"    , value=0.008                                        , units="m"    ,         description="Snubber clearance above the top shelf")
k       = f.Constant( name="k"      , value=[3.0e5, 2.5e5, 2.0e5, 1.5e5, 1.2e5], units="N/m"  , size=N, description="Isolator axial stiffness")
m_shelf = f.Constant( name="m_shelf", value=[4.0, 4.0, 4.0, 4.0, 6.0]          , units="kg"   , size=N, description="Shelf mass, instruments included")

# =====================
# Declare the Objective
# =====================
# `x[1:] - x[:-1]` is the extension of isolators 1 through 4, as one
# expression rather than a loop: slicing an LCsolver vector gives back the
# model's own variables, and subtracting two slices lines them up offset by
# one. Isolator 0 is grounded and so is written out separately.
#
# The load term is negative, which is fine for a QP -- it is the linear part
# of the quadratic form. (A geometric program would refuse it: a posynomial
# has no negative terms.)
stretch = x[1:] - x[:-1]

f.Objective(
    0.5 * k[0] * x[0]**2 + 0.5 * f.sum(k[1:] * stretch**2) - f.sum(m_shelf * a * x)
)

# =======================
# Declare the Constraints
# =======================
# One row. Without it the answer is the free equilibrium, which puts the top
# shelf at 19.5 mm; the snubber catches it at 8 mm and the shelves below
# redistribute.
f.ConstraintList([
    x[-1] <= gap,
    ])

# ==================
# Inspect the Model
# ==================
print(f.structure_report())

# ===========
# Solve Model
# ===========
sol = lcsolver.solve(f)
print(sol.summary(ndecimal=5))

# =====================
# Read the Answer Back
# =====================
# Deflections run 2.9 mm at the bottom shelf to 8.6 mm at the fourth, and the
# top shelf rests on the snubber at 8.0 mm. Isolator 4 is therefore in
# compression -- the stop is pushing back down on the shelf above it.
print("top shelf:", sol.variables("x[4]").to("mm"))

# The contact force, without ever having modelled a contact force.
#
# d(V*)/d(gap) is the rate at which the equilibrium energy changes as the stop
# is moved, which by the envelope theorem is minus the multiplier on that
# constraint: the force the snubber applies. It comes back dimensioned,
# -422.25 N, from the same duals the sensitivity table is built on. Finite
# differencing the solve -- two more solves, and a step size to choose --
# agrees to eight figures.
print("snubber contact force:", -sol.dimensioned_sensitivities("gap"))
