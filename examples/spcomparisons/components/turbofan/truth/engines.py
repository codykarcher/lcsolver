"""The two anchor engines, with provenance for every number.

Provenance discipline follows validation.py's rule: every figure says where
it came from, and anything soft is flagged approximate so it cannot quietly
become an acceptance bound. The CFM56-class spec is pycycle's own validated
regression case and is EXACT; the GEnx-class spec is assembled from public
data and the repo's own tech-level table, and its soft numbers say so.

Rating temperatures
-------------------
The takeoff/cruise T4 split mirrors the aircraft model's multi-rating
structure (Tt4_TO / Tt4_CR in components/turbofan/model.py SUBS):

* CFM56-class: Tt4_TO = 1833 K = 3299.4 R and Tt4_CR = 1587 K = 2857 R are
  the TASOPT 737.tas deck values; 2857 R is also exactly the pycycle
  example's design T4_MAX, which is why the example IS the cruise-rated
  CFM56-class machine.
* GEnx-class: Tt4_TO = 1900 K = 3420 R, the modern-technology rating the
  handoff names as the point of doing equilibrium chemistry (dissociation is
  a %-level effect there); Tt4 design/climb = 1620 K = 2916 R, scaled from
  the CFM56 TO/CR ratio (approximate).
"""
from __future__ import annotations

from .hbtf import EngineSpec, ODPoint


def _ref(value, units, source, approximate=False):
    return {"value": value, "units": units, "source": source,
            "approximate": approximate}


CFM56_CLASS = EngineSpec(
    name="cfm56_class",
    # Design point: pycycle example_cycles/high_bypass_turbofan.py, verbatim.
    # benchmark_hbtf.py pins the solution (W 344.303 lbm/s, OPR 30.094,
    # FAR 0.0249199, TSFC 0.63072), which makes the design point a regression
    # test of THIS file against pycycle upstream.
    MN_des=0.8, alt_des_ft=35000.0, Fn_des_lbf=5900.0, T4_des_R=2857.0,
    BPR=5.105, FPR=1.685, LPC_PR=1.935, HPC_PR=9.369,
    eff_fan=0.8948, eff_lpc=0.9243, eff_hpc=0.8707,
    eff_hpt=0.8888, eff_lpt=0.8996,
    LP_Nmech=4666.1, HP_Nmech=14705.7,
    W_guess=320.0,
    od_points=(
        # Static sea level at the takeoff rating -- the fan-sizing and
        # noise-relevant point, and TASOPT's Tt4TO condition.
        ODPoint("TO",  MN=0.001, alt_ft=0.0,     mode="T4", T4_R=3299.4),
        # Rolling takeoff, TASOPT's other low-speed rating point.
        ODPoint("RTO", MN=0.25,  alt_ft=0.0,     mode="T4", T4_R=3299.4),
        # Top of climb at the aircraft model's cruise Mach (the mission flies
        # 0.785, not the example's 0.8), max-climb = cruise rating.
        ODPoint("TOC", MN=0.785, alt_ft=35000.0, mode="T4", T4_R=2857.0),
        # Part-power cruise, hung off TOC the way the example hangs
        # OD_part_pwr off OD_full_pwr.
        ODPoint("CRZ", MN=0.785, alt_ft=35000.0, mode="PC", PC=0.92,
                pc_of="TOC"),
    ),
    refs={
        "TO.perf.Fn": _ref(27300.0, "lbf",
                           "CFM56-7B27 takeoff rating, EASA TCDS E.004 "
                           "(flat-rated to ISA+15; our point is ISA SLS, "
                           "uninstalled)", approximate=True),
        "TO.splitter.BPR": _ref(5.3, "-",
                                "CFM56-7B27 SLS bypass ratio, CFM public "
                                "data", approximate=True),
        "CRZ.perf.TSFC": _ref(0.627, "lbm/(lbf*h)",
                              "CFM56-7B27 cruise SFC, M0.8/35kft, "
                              "widely-quoted brochure figure",
                              approximate=True),
        "TO.perf.OPR": _ref(32.8, "-",
                            "CFM56-7B27 takeoff overall pressure ratio, "
                            "CFM public data", approximate=True),
    },
    notes="pycycle's validated HBTF example; the 737-800 anchor.",
)


# Adiabatic efficiencies below are the repo's ETAS['GE90'] polytropic row
# (0.9245, 0.9127, 0.9339, 0.9212, 0.9320 -- the deck that 'stands in for a
# GEnx', model.py:86) converted poly->adiabatic at each component's design
# pressure ratio (gamma 1.4 cold side, 1.3 hot side).
GENX_CLASS = EngineSpec(
    name="genx_class",
    # Design at top-of-climb-like max cruise: M0.85/35kft. Fn_des sized so
    # the engine covers a 787-8 top-of-climb (~14 klbf/engine at 480 klbf,
    # L/D ~18.5, 300 fpm residual -- derived, approximate).
    MN_des=0.85, alt_des_ft=35000.0, Fn_des_lbf=13000.0, T4_des_R=2916.0,
    # OPR = 1.55 * 1.9 * 15.6 = 45.9 against GEnx-1B74/75 max-climb 46.3
    # (EASA TCDS E.011). BPR 9.0 at cruise design; public SLS quotes ~9.1-9.6.
    BPR=9.0, FPR=1.55, LPC_PR=1.9, HPC_PR=15.6,
    eff_fan=0.920, eff_lpc=0.905, eff_hpc=0.905,
    eff_hpt=0.932, eff_lpt=0.945,
    # GEnx-1B redlines: N1 2694, N2 11377 rpm (EASA TCDS E.011); design point
    # set near 90% of both.
    LP_Nmech=2400.0, HP_Nmech=10600.0,
    W_guess=900.0,
    params={
        # The 787 is bledless -- no customer bleed; shaft power extraction
        # instead (~500 hp per engine in cruise, approximate).
        "hpc.cust:frac_W": 0.001,
        ("hp_shaft.HPX", "hp"): 500.0,
    },
    od_points=(
        ODPoint("TO",  MN=0.001, alt_ft=0.0,     mode="T4", T4_R=3420.0),
        ODPoint("RTO", MN=0.25,  alt_ft=0.0,     mode="T4", T4_R=3420.0),
        ODPoint("TOC", MN=0.85,  alt_ft=35000.0, mode="T4", T4_R=2916.0),
        ODPoint("CRZ", MN=0.85,  alt_ft=35000.0, mode="PC", PC=0.90,
                pc_of="TOC"),
    ),
    refs={
        "TO.perf.Fn": _ref(69800.0, "lbf",
                           "GEnx-1B70 takeoff rating, EASA TCDS E.011",
                           approximate=True),
        "TO.splitter.BPR": _ref(9.1, "-",
                                "GEnx-1B public bypass ratio (quotes range "
                                "9.0-9.6)", approximate=True),
        "CRZ.perf.TSFC": _ref(0.514, "lbm/(lbf*h)",
                              "GEnx-1B cruise SFC, M0.85/35kft, "
                              "widely-quoted figure", approximate=True),
        "TOC.perf.OPR": _ref(46.3, "-",
                             "GEnx-1B74/75 max-climb OPR, EASA TCDS E.011",
                             approximate=True),
    },
    notes="GEnx-class assembled from public data + repo ETAS; the 787 anchor.",
)


ENGINES = {s.name: s for s in (CFM56_CLASS, GENX_CLASS)}
