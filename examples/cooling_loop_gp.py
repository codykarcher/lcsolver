# ===========
# Description
# ===========
# Liquid cooling for a power-electronics cabinet, formulated as a Geometric
# Program and assembled from three named subsystems
#
# 5 kW comes off the semiconductors into a cold plate, a pump moves
# water-glycol from the plate to a radiator, and a fan pushes 45 C air through
# the radiator. Sizing minimizes what the cooling system costs the vehicle:
# pump power, fan power, and the power equivalent of carrying the radiator
# around.
#
# Everything here is a monomial or a posynomial with positive coefficients --
# turbulent pressure drop as a power law in flow, plate resistance as another,
# fan power as a third, and one thermal budget that has to fit inside the
# junction temperature limit -- so the detector calls this a Geometric Program
# and solves it to a global optimum in log space.
#
# What this file is really about is `f.group()`. Each subsystem is built by
# its own function, and each function opens a group:
#
#     loop = f.group('loop')
#     loop.Variable('mdot', 0.1, 'kg/s', 'coolant mass flow')     -> loop_mdot
#
# Names stay flat -- `loop_mdot` is an ordinary component of the model, not a
# Pyomo sub-block -- so the detector, the unit walker and write-back are
# unaffected. What the group buys is that a builder no longer has to take a
# prefix argument and thread it through every declaration, that `f.loop.mdot`
# reaches the quantity afterwards (which is how the builders below refer to
# each other's variables, without arguments or return values), and that the
# solution prints `loop.mdot` rather than `loop_mdot`.
#
# The cold plate's resistance is declared in mK/W while every temperature in
# the model is in K. That is deliberate: the thermal budget only balances
# because the unit corrector reconciles the two before the detector ever sees
# the model.

# =================
# Import Statements
# =================
import lcsolver
from lcsolver import Formulation, units


# ==========================
# Declare the Subsystems
# ==========================
# One function per subsystem, each taking only the formulation. A builder that
# needs a quantity from another subsystem reads it off the group that owns it.

def add_coolant_loop(f):
    """Pump, plumbing, and the coolant's own temperature rise."""
    loop = f.group('loop')

    dp   = loop.Variable("dp"  , 4.0e4, "Pa"  , "Loop pressure drop")
    dT   = loop.Variable("dT"  , 14.0 , "K"   , "Coolant temperature rise")
    mdot = loop.Variable("mdot", 0.1  , "kg/s", "Coolant mass flow")
    P    = loop.Variable("P"   , 15.0 , "W"   , "Pump electrical power")

    cp       = loop.Constant("cp"      , 3600.0, "J/kg/K", "Coolant specific heat")
    dp_ref   = loop.Constant("dp_ref"  , 4.0e4 , "Pa"    , "Pressure drop at the reference flow")
    eta      = loop.Constant("eta"     , 0.35  , "-"     , "Pump wire-to-fluid efficiency")
    mdot_ref = loop.Constant("mdot_ref", 0.1   , "kg/s"  , "Reference coolant flow")
    rho      = loop.Constant("rho"     , 1050.0, "kg/m^3", "Coolant density")

    loop.ConstraintList([
        # Turbulent pressure drop, fitted about the reference flow
        dp >= dp_ref * (mdot / mdot_ref)**1.75,

        # Hydraulic power in, electrical power out
        P >= mdot * dp / (rho * eta),

        # Sensible heating of the coolant
        dT >= f.Q / (mdot * cp),
        ])
    return loop


def add_cold_plate(f):
    """The semiconductor-to-coolant resistance, which falls with flow."""
    plate = f.group('plate')

    R = plate.Variable("R", 4.0, "mK/W", "Plate thermal resistance")

    R_ref = plate.Constant("R_ref", 4.0, "mK/W", "Resistance at the reference flow")

    # `f.loop.mdot` is the flow the loop builder declared. No prefix string was
    # passed in and nothing was returned; the group is the shared name.
    plate.ConstraintList([
        R >= R_ref * (f.loop.mdot / f.loop.mdot_ref)**-0.8,
        ])
    return plate


def add_radiator(f):
    """Radiator and fan. Frontal area is the design's main free choice."""
    rad = f.group('rad')

    A    = rad.Variable("A"   , 0.09 , "m^2" , "Radiator frontal area")
    dTa  = rad.Variable("dTa" , 16.0 , "K"   , "Air temperature rise")
    ma   = rad.Variable("ma"  , 0.3  , "kg/s", "Air mass flow")
    mass = rad.Variable("mass", 2.5  , "kg"  , "Radiator mass, coolant included")
    P    = rad.Variable("P"   , 40.0 , "W"   , "Fan electrical power")
    UA   = rad.Variable("UA"  , 300.0, "W/K" , "Radiator conductance")

    A_ref  = rad.Constant("A_ref" , 0.09  , "m^2"   , "Reference frontal area")
    cp_a   = rad.Constant("cp_a"  , 1005.0, "J/kg/K", "Air specific heat")
    m_ref  = rad.Constant("m_ref" , 2.5   , "kg"    , "Mass at the reference area")
    ma_ref = rad.Constant("ma_ref", 0.3   , "kg/s"  , "Reference air flow")
    P_ref  = rad.Constant("P_ref" , 40.0  , "W"     , "Fan power at the reference point")
    UA_ref = rad.Constant("UA_ref", 300.0 , "W/K"   , "Conductance at the reference point")

    rad.ConstraintList([
        # More air and more area both buy conductance
        UA <= UA_ref * (ma / ma_ref)**0.8 * (A / A_ref),

        # Fan power goes as flow cubed, and falls as the core is opened out
        P >= P_ref * (ma / ma_ref)**3 * (A_ref / A)**2,

        # Sensible heating of the air
        dTa >= f.Q / (ma * cp_a),

        # Mass scales with frontal area at fixed core depth
        mass >= m_ref * (A / A_ref),
        ])
    return rad


# ===================
# Declare Formulation
# ===================
f = Formulation()

# =================
# Declare Constants
# =================
# The three quantities every subsystem needs stay at the top level, ungrouped.
Q       = f.Constant( name="Q"      , value=5000.0 , units="W"    , description="Heat rejected by the electronics")
T_air   = f.Constant( name="T_air"  , value=318.15 , units="K"    , description="Cooling air inlet temperature")
T_max   = f.Constant( name="T_max"  , value=398.15 , units="K"    , description="Junction temperature limit")
k_carry = f.Constant( name="k_carry", value=5.0    , units="W/kg" , description="Power equivalent of carried mass")

# ==================
# Build the Model
# ==================
loop  = add_coolant_loop(f)
plate = add_cold_plate(f)
rad   = add_radiator(f)

# =====================
# Declare the Objective
# =====================
f.Objective(loop.P + rad.P + k_carry * rad.mass)

# =======================
# Declare the Constraints
# =======================
# The one constraint that belongs to no subsystem, and the one that decides
# the design: the temperature rise from cooling air to junction, accumulated
# across all three, has to fit under the limit. `plate.R` is in mK/W and every
# other term is in K.
f.ConstraintList([
    T_air + Q / rad.UA + rad.dTa + loop.dT + Q * plate.R <= T_max,
    ])

# ==================
# Inspect the Model
# ==================
print(f.structure_report())

# ===========
# Solve Model
# ===========
sol = lcsolver.solve(f)

# The summary names quantities by group -- `loop.mdot`, `rad.A` -- and sorts
# them that way, so a fifty-variable model reads as its subsystems rather than
# as one alphabetical list.
#
# The answer costs 30.6 W: the radiator grows to 0.13 m^2, both flows drop
# below their reference values, and the thermal budget is tight. The
# sensitivity table says the same thing far more sharply than the variable
# values do -- T_max at -6.2 and T_air at +4.9 dwarf everything a designer
# controls, which is the model reporting that this cabinet is thermally
# limited and that the useful conversations are about the air inlet and the
# junction rating, not about the pump.
print(sol.summary())

# A group also reads like the dictionary of quantities it replaces, which is
# what makes it convenient to hand a whole subsystem to a reporting function.
print("radiator area:", sol.variables("rad.A"))
print("radiator quantities:", sorted(k for k in sol.variables() if k.startswith("rad.")))
