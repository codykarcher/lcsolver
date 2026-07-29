"""DAPCA IV airframe cost, as v3 uses it.

The correlations behind ``cost_val.jl``, with inputs made into arguments.
See the package docstring for why that is necessary and what is disclaimed.

The model
---------
DAPCA IV (Raymer) estimates four labour pools -- engineering, tooling,
manufacturing and quality control -- as power laws in a **weighted empty
weight**, a maximum speed and the production quantity, then multiplies by
wrap rates in dollars per hour. Development support, flight test and
materials are separate dollar correlations.

The weighting is the interesting part. Structure is not costed by mass but by
mass times a complexity factor, and the factors differ by nearly a factor of
twenty::

    fuselage 1.49   tails 2.42   wing 0.82
    engines  0.40   gear  0.12   systems 1.59

so a kilogram of empennage costs six times what a kilogram of engine does,
and twenty times what a kilogram of landing gear does. That is what makes the
model say anything at all beyond "heavier is dearer".

Everything is inflated by a producer price index ratio; the reference carries
two, for 1999 and 2007 bases.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["COST_WARNING", "COMPLEXITY", "PPI_1999", "PPI_2007",
           "WRAP_RATES", "AirframeCost", "dapca_hours", "airframe_cost",
           "engine_cost", "cost_val_baseline"]

COST_WARNING = (
    "TASOPT.jl's cost model is marked 'Unused and Unvetted' and 'not "
    "endorsed by the current dev team' by its own authors. Treat any number "
    "from it accordingly.")

#: Complexity factors on component weight, from ``CostVal``.
COMPLEXITY = {
    "fuselage": 1.49, "tails": 2.42, "wing": 0.82,
    "engines": 0.40, "gear": 0.12, "systems": 1.59,
}

#: Aerospace producer price index ratios (US BLS).
PPI_1999 = 242.8 / 144.8        # 1999 -> 2020
PPI_2007 = 242.8 / 186.8        # 2007 -> 2020

#: Labour wrap rates, $/hr: engineering, tooling, quality, manufacturing.
WRAP_RATES = {"RE": 86.0, "RT": 88.0, "RQ": 81.0, "RM": 73.0}

#: Flight-test aircraft. Fixed at the upper end "due to novelty of
#: propulsion system tech".
FTA = 6


@dataclass(frozen=True)
class AirframeCost:
    """The pieces of an airframe unit cost, all in dollars."""
    HE: float        # engineering hours
    HT: float        # tooling hours
    HM: float        # manufacturing hours
    HQ: float        # quality control hours
    CD: float        # development support
    CF: float        # flight test
    CM: float        # materials
    CI: float        # interior
    Wmod: float      # the weighted weight the correlations use, kg
    unit_cost: float  # per aircraft, inflated and adjusted


def weighted_weight(*, fuselage: float, tails: float, wing: float,
                    engines: float, gear: float, systems: float) -> float:
    """The complexity-weighted weight DAPCA is correlated against, kg.

    Weights in kg. The factors differ by a factor of twenty between
    empennage and landing gear -- see the module docstring.
    """
    c = COMPLEXITY
    return (c["fuselage"] * fuselage + c["tails"] * tails
            + c["wing"] * wing + c["engines"] * engines
            + c["gear"] * gear + c["systems"] * systems)


def dapca_hours(Wmod: float, Vmax: float, prod_Q: float) -> tuple:
    """``(HE, HT, HM, HQ)`` labour hours.

    ``Wmod`` in kg, ``Vmax`` in km/h, ``prod_Q`` the production run.

    The learning-curve exponents on ``prod_Q`` are what differ between
    pools: manufacturing scales as ``Q^0.641`` -- so its *per-aircraft* cost
    falls steeply with volume -- while engineering is ``Q^0.163``, nearly
    fixed, because it is largely done once.
    """
    HE = 5.18 * Wmod ** 0.777 * Vmax ** 0.894 * prod_Q ** 0.163
    HT = 7.22 * Wmod ** 0.777 * Vmax ** 0.696 * prod_Q ** 0.263
    HM = 10.5 * Wmod ** 0.82 * Vmax ** 0.484 * prod_Q ** 0.641
    HQ = 0.133 * HM
    return HE, HT, HM, HQ


def airframe_cost(Wmod: float, Vmax: float, prod_Q: float, Npax: int, *,
                  fta: int = FTA, ppi: float = PPI_1999,
                  avionics_adj: float = 4.0 / 3.0) -> AirframeCost:
    """Airframe cost per aircraft, dollars.

    ``avionics_adj`` is the reference's ``cA``, "such that avionics costs are
    around 25% of flyaway cost" -- i.e. it is a markup chosen to make a
    number come out, not a correlation.
    """
    HE, HT, HM, HQ = dapca_hours(Wmod, Vmax, prod_Q)
    r = WRAP_RATES

    CD = 48.7 * Wmod ** 0.630 * Vmax ** 1.3
    CF = 1408.0 * Wmod ** 0.325 * Vmax ** 0.822 * fta ** 1.21
    CM = 22.6 * Wmod ** 0.921 * Vmax ** 0.621 * prod_Q ** 0.799
    CI = 2500.0 * Npax

    total = (HE * r["RE"] + HT * r["RT"] + HM * r["RM"] + HQ * r["RQ"]
             + CD + CF + CM + CI)
    return AirframeCost(HE=HE, HT=HT, HM=HM, HQ=HQ, CD=CD, CF=CF, CM=CM,
                        CI=CI, Wmod=Wmod,
                        unit_cost=ppi * total * avionics_adj / prod_Q)


def engine_cost(Fmax: float, Mmax: float, Tin: float,
                ppi: float = PPI_1999) -> float:
    """Gas-turbine cost, dollars -- Birkler's model as published in Raymer.

    ``Fmax`` maximum thrust (kN), ``Mmax`` maximum Mach, ``Tin`` turbine
    inlet temperature (K). Note the ``-2228`` constant: the correlation goes
    *negative* for a small enough engine, which is a reminder that it is a
    regression over a fleet rather than a physical model.
    """
    return 2251.0 * (9.66 * Fmax + 243.25 * Mmax + 1.74 * Tin - 2228.0) \
        * 1.2 * ppi


def cost_val_baseline(prod_Q: float) -> tuple:
    """``CostVal(prod_Q)`` -- the reference's hard-wired 737 MAX9 case.

    Every input is a literal in the reference's body, so this is a function
    of the production quantity alone. Kept exactly as written, because it is
    the only thing there is to verify the correlations against.
    """
    Wmod = weighted_weight(fuselage=20183.2, tails=1202.0 + 793.3,
                           wing=11813.0, engines=7995.9, gear=1012.6,
                           systems=920.5)
    Vmax, Npax = 855.0, 220
    Fmax, Mmax, Tin = 130.4, 1.0, 1804.0

    af = airframe_cost(Wmod, Vmax, prod_Q, Npax)
    CTS = engine_cost(Fmax, Mmax, Tin)

    cost_airframe = af.unit_cost
    cost_prop = 2.0 * CTS            # two engines; CEC and CPE are zero here

    HE, HT, HM, HQ = af.HE, af.HT, af.HM, af.HQ
    r = WRAP_RATES
    cost_dev = ((HE * r["RE"] + HT * r["RT"] + af.CD + af.CF)
                * (4.0 / 3.0) * PPI_1999 / prod_Q)
    cost_prod = ((HM * r["RM"] + HQ * r["RQ"] + af.CM + af.CI)
                 * (4.0 / 3.0) * PPI_1999 / prod_Q)
    return cost_dev, cost_prod, cost_prop
