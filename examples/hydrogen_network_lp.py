# ===========
# Description
# ===========
# A hydrogen delivery network, formulated as a Linear Program
#
# Three electrolyzer plants supply four refueling stations by tube trailer.
# Every station's daily demand has to be met, no plant can ship more than it
# makes, and the trucking energy spent doing it is minimized.
#
# The point of this file is the SHAPE of the model rather than its physics.
# The decision is a matrix -- how much hydrogen moves on each of the twelve
# plant-to-station routes -- and every constraint is a statement about one of
# its axes:
#
#     f.sum(X, axis=1) <= cap     one row per plant     (what a plant ships)
#     f.sum(X, axis=0) >= dem     one row per station   (what a station gets)
#     X <= f.broadcast_rows(...)  one row per route     (per-station limit)
#     X <= f.broadcast_cols(...)  one row per route     (per-plant limit)
#
# Four lines, twenty-three constraints. Written index by index this is four
# nested loops, and the loops are where the axis mistakes live.
#
# Objective and constraints are all affine, so the detector calls this a
# Linear Program: one convex solve, a global optimum, and exact duals -- which
# is what makes the sensitivity table at the bottom worth reading.
#
# Compression energy is left out of the objective on purpose. Every kilogram
# is compressed the same way whichever plant it comes from, so it adds a
# constant to the objective and cannot move the optimum.

# =================
# Import Statements
# =================
from pyomo.environ import NonNegativeReals

import lcsolver
from lcsolver import Formulation, units

# ===================
# Declare Formulation
# ===================
f = Formulation()

N_plant = 3
N_stn = 4

# =================
# Declare Variables
# =================
# One variable per route. A matrix quantity is declared with a list size, and
# `X.shape` is (3, 4) from then on.
#
# `domain` rather than `bounds` for the floor: a delivery cannot be negative,
# and NonNegativeReals says so once instead of repeating 0.0 in a bounds pair.
# The pre-solve gate wants every variable bounded both ways before it will
# solve; the route limits below supply the ceiling.

X = f.Variable(name="X", guess = 100.0, units = "kg/day", size=[N_plant, N_stn], domain=NonNegativeReals, description="Hydrogen moved from each plant to each station")

# =================
# Declare Constants
# =================
# A matrix Constant takes a nested list, or a numpy array, laid out the way
# numpy lays one out: the outer level is the first index, so `_distance[i][j]`
# is plant i to station j. The shape has to match the declared size exactly --
# a 3x4 array is not accepted for a size=[4, 3] declaration, because a
# transposed model is a different model and it would solve without complaint.
_distance = [[  15.0,  40.0,  90.0, 130.0],     # plant 0 -> each station
             [  70.0,  25.0,  20.0,  60.0],     # plant 1 -> each station
             [ 140.0,  95.0,  45.0,  15.0]]     # plant 2 -> each station

cap     = f.Constant( name="cap"    , value=[1200.0, 900.0, 1500.0]       , units="kg/day"  , size=N_plant         , description="Plant production capacity")
d       = f.Constant( name="d"      , value=_distance                     , units="km"      , size=[N_plant, N_stn], description="Road distance, plant to station")
dem     = f.Constant( name="dem"    , value=[400.0, 700.0, 600.0, 500.0]  , units="kg/day"  , size=N_stn           , description="Station daily demand")
e_haul  = f.Constant( name="e_haul" , value=0.012                         , units="MJ/kg/km",                        description="Round-trip diesel energy per kg per km")
intake  = f.Constant( name="intake" , value=[350.0, 500.0, 500.0, 400.0]  , units="kg/day"  , size=N_stn           , description="Most one station will accept from one supplier")
trailer = f.Constant( name="trailer", value=[300.0, 500.0, 700.0]         , units="kg/day"  , size=N_plant         , description="Most one plant will commit to one route")

# =====================
# Declare the Objective
# =====================
# `d * X` is elementwise over the whole matrix; `f.sum` with no axis reduces
# it to a single expression. Plain `sum(X)` would add the index keys, and
# LCsolver raises rather than let that pass.
f.Objective(e_haul * f.sum(d * X))

# =======================
# Declare the Constraints
# =======================
# `f.sum(X, axis=1)` is a length-3 vector -- the total leaving each plant --
# and comparing it against a length-3 Constant builds three constraints, one
# per element.
#
# The two per-route caps are the reason `broadcast_rows` and `broadcast_cols`
# exist. `X <= intake` is refused: `intake` is length 4 and X is (3, 4), and
# numpy would spread it across the rows as readily as down the columns.
# Whichever the author meant, the other reading is a different model that
# solves without complaint, so LCsolver makes you name the axis.

f.ConstraintList([
    # Supply: no plant ships more than it makes
    f.sum(X, axis=1) <= cap,

    # Demand: every station is served
    f.sum(X, axis=0) >= dem,

    # Route limits: the station's unloading bay, and the plant's trailer pool
    X <= f.broadcast_rows(intake, N_plant),
    X <= f.broadcast_cols(trailer, N_stn),
    ])

# ==================
# Inspect the Model
# ==================
# What kind of problem this is, before solving it. On a model of this shape
# the answer is worth having: a stray product of two variables would make it a
# non-convex NLP, and the difference is a global optimum against a local one.
print(f.structure_report())

# ===========
# Solve Model
# ===========
sol = lcsolver.solve(f)
print(sol.summary())

# =====================
# Read the Answer Back
# =====================
# Indexed quantities are reached by their printed name.
shipped = sol.variables()
print("plant 1 -> station 2:", shipped["X[1,2]"])

# The duals are exact here, so the sensitivities are the shadow prices of a
# linear program: `cap[1]` at -0.35 says a 1% increase in plant 1's capacity
# buys 0.35% off the haulage bill, and the twelve constants ranked below 1e-08
# are the routes and limits with slack in them. `e_haul` at exactly +1.0 is
# the sanity check -- it multiplies the whole objective and nothing else.
#
# Dimensioned, the same number is -0.30 MJ/kg: one more kg/day of capacity at
# plant 1 saves 0.30 MJ/day of diesel.
print("marginal value of plant 1 capacity:",
      sol.dimensioned_sensitivities("cap[1]"))
