# ===========
# Description
# ===========
# A three-stage speed reducer for a robot joint, formulated as a Geometric
# Program
#
# A servo motor delivers 2.5 N m and the joint needs 90:1 of reduction. Split
# across three stages, minimizing gearbox mass. The question the model answers
# is where in the train to put the reduction, and the answer is not "evenly":
# torque rises stage by stage, a stage's mass is driven by the torque it
# carries, so reduction taken early is paid for by every stage after it.
#
# Three pieces of vector machinery carry the model:
#
#     T[1:] == eta * T[:-1] * r[:-1]      the torque recurrence, one line
#     m >= m_ref * (T/T_ref)**0.6 * r**0.3    a fit applied elementwise
#     f.prod(r) >= R_req                  the overall ratio
#
# `f.prod` is what makes this a natural GP rather than a contrivance: the
# product of the stage ratios is a monomial in the ratios, and a monomial
# bounded below by a constant is exactly the form a GP wants.
#
# The stage ratio limit is declared HOLOGRAPHIC. It is not a design rule; it
# is the largest ratio in the data the mass fit was made from, and an answer
# that leans on it is an answer extrapolated past the fit. Two of the three
# stages end up sitting on it, and LCsolver says so rather than leaving it to
# be noticed.

# =================
# Import Statements
# =================
import lcsolver
from lcsolver import Formulation, units

# ===================
# Declare Formulation
# ===================
f = Formulation()

N = 3

# =================
# Declare Variables
# =================
# One entry per stage, input side to output side. T[i] is the torque going IN
# to stage i, which is what sizes its gears.

m = f.Variable(name="m", guess = 1.0 , units = "kg"  , size=N, bounds=[0.01, 1.0e3], description="Stage mass")
r = f.Variable(name="r", guess = 4.5 , units = "-"   , size=N, bounds=[1.0, 20.0]  , description="Stage reduction ratio")
T = f.Variable(name="T", guess = 20.0, units = "N*m" , size=N, bounds=[0.1, 1.0e5] , description="Torque entering the stage")

# =================
# Declare Constants
# =================
eta   = f.Constant( name="eta"  , value=0.97  , units="-"   , description="Mesh efficiency of one stage")
m_ref = f.Constant( name="m_ref", value=1.8   , units="kg"  , description="Mass of the fit's reference stage")
r_max = f.Constant( name="r_max", value=6.0   , units="-"   , description="Largest stage ratio in the fit data")
R_req = f.Constant( name="R_req", value=90.0  , units="-"   , description="Required overall reduction")
T_in  = f.Constant( name="T_in" , value=2.5   , units="N*m" , description="Motor output torque")
T_ref = f.Constant( name="T_ref", value=100.0 , units="N*m" , description="Torque of the fit's reference stage")

# =====================
# Declare the Objective
# =====================
f.Objective(f.sum(m))

# =======================
# Declare the Constraints
# =======================
# The recurrence reads the way it is written on paper. `T[1:]` is stages 1
# and 2, `T[:-1]` is stages 0 and 1, and comparing the two slices offsets them
# by one -- two constraints, no loop and no index arithmetic to get wrong.
#
# The mass fit is stated against a reference stage rather than with a
# dimensional coefficient. A coefficient on T**0.6 would carry units of
# kg/(N m)**0.6, which is meaningless to look at and easy to mistype; m_ref
# and T_ref are quantities somebody can check against the data sheet.

f.ConstraintList([
    # Torque through the train
    T[0] == T_in,
    T[1:] == eta * T[:-1] * r[:-1],

    # Stage mass, from a monomial fit to a catalogue of planetary stages
    m >= m_ref * (T / T_ref)**0.6 * r**0.3,

    # The joint has to turn at the commanded rate
    f.prod(r) >= R_req,
    ])

# =====================================
# Declare the Model's Limits of Validity
# =====================================
# Imposed exactly like any other constraint. The difference is that LCsolver
# checks it at the solution and reports it, because an optimum resting on the
# edge of a fit is not an optimum -- it is the solver saying it wanted to go
# somewhere there is no data.
f.HolographicConstraintList([
    r <= r_max,
    ])

# ==================
# Inspect the Model
# ==================
print(f.structure_report())

# ===========
# Solve Model
# ===========
sol = lcsolver.solve(f)

# The summary carries a `holographic constraints` block above the
# sensitivities: two of three stages are pinned at r = 6, and the answer
# (2.48 kg, ratios 2.5 / 6 / 6) is only as good as the fit is at its edge.
#
# What to do about it is in the sensitivity table. `r_max` at -0.47 says a 1%
# wider fit is worth 0.47% of gearbox mass -- more than the efficiency of the
# meshes, and second only to the mass of the reference stage itself. That is
# an argument for measuring a few higher-ratio stages, and it is not an
# argument the model could make if the limit had been written as a bare
# number in a constraint.
print(sol.summary())

# The warning is captured rather than printed, so a script can act on it.
for message in sol.messages:
    print(message)
