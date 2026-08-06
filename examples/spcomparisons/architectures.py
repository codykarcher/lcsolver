"""The five propulsion / airframe architectures.

Airframe versus propulsion
--------------------------
Only ``d8`` changes the *airframe*. The other four fly a conventional
single-bubble tube with podded engines, so that rows 1, 3, 4 and 5 isolate
the effect of the energy carrier and rows 1 and 2 isolate the effect of the
airframe. Mixing both at once -- a hydrogen D8, say -- is a legitimate design
but a useless comparison, and it is not run here.

What each one actually changes
------------------------------
``conventional``  Jet-A turbofan, single bubble, no BLI. The datum.
``d8``            Jet-A, double-bubble fuselage, boundary-layer ingestion,
                  rear-mounted engines. SPaircraft's optimalD8, which is the
                  one configuration its own CI exercises.
``h2burn``        The same turbofan cycle burning hydrogen: LHV 120 MJ/kg
                  against Jet-A's 43.003, everything else identical. Adds a
                  fuselage LH2 tank, which lengthens the shell, and dries the
                  wing -- which costs the fuel's bending relief.
``h2fc``          PEM stack driving electric ducted fans. Adds the tank AND
                  the stack; deletes the core.
``battery``       Pack driving the same electric fans. No fuel burn at all,
                  so the aircraft does not get lighter as it flies.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Architecture:
    key: str
    label: str
    # --- airframe ---
    double_bubble: bool          # w_db free, or pinned to ~0
    BLI: bool                    # boundary-layer ingestion drag credit
    rear_engines: bool
    # --- energy carrier ---
    fuel: str                    # "jeta" | "lh2" | "electric"
    engine: str                  # turbofan variant key, or "" for electric
    cryo_tank: bool
    fuel_cell: bool
    battery: bool
    wet_wing: bool               # fuel in the wing -> bending relief
    note: str
    # Cruise Mach pinned to the class's real design Mach instead of optimised.
    lock_mach: bool = False
    # Strut-braced wing (TASOPT iwplan=2): root moment pinned to the break
    # loads, tension strut + its profile drag added. An airframe-axis flag
    # like double_bubble (defaulted because it postdates the original five),
    # so it composes with any energy carrier the same way.
    strut: bool = False


ARCHS = {
    "conventional": Architecture(
        key="conventional", label="Conventional (Jet-A)",
        double_bubble=False, BLI=False, rear_engines=False,
        fuel="jeta", engine="D82_SPaircraft", cryo_tank=False,
        fuel_cell=False, battery=False, wet_wing=True,
        note="Datum. Single bubble, podded underwing engines, wet wing."),
    "d8": Architecture(
        key="d8", label="D8 (Jet-A, BLI)",
        double_bubble=True, BLI=True, rear_engines=True,
        fuel="jeta", engine="D82_SPaircraft", cryo_tank=False,
        fuel_cell=False, battery=False, wet_wing=True,
        note="Double bubble, BLI, rear engines. Airframe change only."),
    "strut": Architecture(
        key="strut", label="Strut-braced wing (Jet-A)",
        double_bubble=False, BLI=False, rear_engines=False, strut=True,
        fuel="jeta", engine="D82_SPaircraft", cryo_tank=False,
        fuel_cell=False, battery=False, wet_wing=True,
        note="Conventional tube and engines, wing root moment relieved by a "
             "tension strut at the planform break. Airframe change only, "
             "like d8 -- rows 1 and 6 isolate the brace."),
    "h2burn": Architecture(
        key="h2burn", label="H2-burning turbofan",
        double_bubble=False, BLI=False, rear_engines=False,
        fuel="lh2", engine="D82_LH2", cryo_tank=True,
        fuel_cell=False, battery=False, wet_wing=False,
        note="Same cycle, hf 120 vs 43.003 MJ/kg. Fuselage tank, dry wing."),
    "h2fc": Architecture(
        key="h2fc", label="H2 fuel cell + electric fans",
        double_bubble=False, BLI=False, rear_engines=False,
        fuel="electric", engine="", cryo_tank=True,
        fuel_cell=True, battery=False, wet_wing=False,
        note="PEM stack, electric ducted fans, LH2 tank. No core."),
    "battery": Architecture(
        key="battery", label="Battery-electric",
        double_bubble=False, BLI=False, rear_engines=False,
        fuel="electric", engine="", cryo_tank=False,
        fuel_cell=False, battery=True, wet_wing=False,
        note="Pack + electric fans. Constant weight -- no fuel burn, so the "
             "weight-decrement chain disappears entirely."),
}

# ---------------------------------------------------------------------------
# Mixed rows: D8 airframe under electric energy carriers. Outside the
# canonical five because they change BOTH axes at once -- legitimate designs,
# and exactly the reason the axes are separate flags -- but read them against
# their single-axis parents (battery, h2fc), not against the datum. The
# electric powertrain takes the BLI inflow defect through f_BLI_V in
# add_powertrain (the actuator-disc form of the turbofan's branch), so these
# rows fly with the wake credit AND the fan paying for it, like the d8.
ARCHS["battery_d8"] = Architecture(
    key="battery_d8", label="Battery-electric D8 (BLI)",
    double_bubble=True, BLI=True, rear_engines=True,
    fuel="electric", engine="", cryo_tank=False,
    fuel_cell=False, battery=True, wet_wing=False,
    note="D8 airframe, pack + BLI electric fans. The airframe the electric "
         "propulsor arguably wants: no core to distort, short inlets.")
ARCHS["h2fc_d8"] = Architecture(
    key="h2fc_d8", label="H2 fuel cell D8 (BLI)",
    double_bubble=True, BLI=True, rear_engines=True,
    fuel="electric", engine="", cryo_tank=True,
    fuel_cell=True, battery=False, wet_wing=False,
    note="D8 airframe, PEM stack + BLI electric fans + LH2 tank. Radiator "
         "cooling drag charged inside the BLI drag row.")

# ---------------------------------------------------------------------------
# Mach-locked twins.
#
# With Mach free and fuel as the objective the optimiser flies slow -- 0.60
# for the Citation against its real 0.90 -- because nothing in the objective
# values speed. A slower aircraft needs less sweep, less structure and less
# thrust, so part of the empty-weight shortfall against the real aircraft is
# not a modelling error at all: it is the optimiser buying lightness with
# cruise speed, which a real programme is not free to do because speed is
# part of the product.
#
# These twins pin cruise Mach to the class's actual design value, which
# separates the two effects. Whatever weight gap survives at the real Mach is
# a genuine modelling shortfall; whatever closes was design point.
from dataclasses import replace as _replace

LOCKED = {f"{k}_M": _replace(v, key=f"{k}_M", lock_mach=True,
                             label=f"{v.label} @ design M")
          for k, v in ARCHS.items()}
ARCHS.update(LOCKED)

ORDER = ["conventional", "d8", "strut", "h2burn", "h2fc", "battery"]
ORDER_LOCKED = [f"{k}_M" for k in ORDER]
ORDER_ALL = ORDER + ORDER_LOCKED
