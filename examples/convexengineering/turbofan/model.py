"""The SP turbofan engine — a 1D core + fan flowpath cycle model.

Source model
------------
``turbofan/`` in https://github.com/convexengineering/turbofan, principally
``engine_validation.py`` (the ``Engine`` class and its performance models),
with the component models in ``compressor.py``, ``combustor.py``,
``turbine.py`` and ``maps.py``.

Paper
-----
    M. York, W. Hoburg and M. Drela, "Turbofan Engine Sizing and Tradeoff
    Analysis via Signomial Programming", AIAA J. Aircraft, DOI 10.2514/1.C034463

What the model is
-----------------
A full one-dimensional core and fan flowpath, solved simultaneously across
several operating points, with no on-design/off-design distinction: the same
engine geometry (fan area ``A_2``, HPC area ``A_2.5``, nozzle areas ``A_5``
and ``A_7``, and the map design mass flows) is shared by every flight segment,
while the cycle state is per-segment. That coupling is the whole point of the
formulation — it lets the optimizer *choose* the design point rather than
having it specified.

Stations follow the TASOPT numbering: 0 free stream, 1.8 diffuser exit, 2 fan
inlet, 2.1 fan exit, 2.5 LPC exit, 3 HPC exit, 4 combustor exit, 4.1 turbine
inlet (after cooling flow mixes in), 4.5 HPT exit, 4.9 LPT exit, 5 core nozzle
exit, 6 core exhaust, 7 fan nozzle exit, 8 fan exhaust.

Why it is a signomial program, not a GP
---------------------------------------
Three places, all of them "a sum appears on the greater side":

* ``fp1 == f + 1``, the fuel-air ratio plus one. The source writes this as a
  ``SignomialEquality`` and its comment records that relaxing it makes the
  problem hit the iteration limit.
* ``alpha_p1 == alpha + 1``, likewise for the bypass ratio.
* ``F <= F_6 + F_8``, total thrust bounded by the sum of core and fan thrust,
  and the two momentum balances feeding it.

Everything else is GP-compatible by construction, which is what the paper's
component-by-component derivation is for.

Verification
------------
``reference.json`` is a snapshot of the gpkit model, solved with IPOPT
underneath gpkit's sequential-GP loop (see ``reference.py`` — cvxopt cannot
solve this model and the repo's own TESTCONFIG says so). It also carries the
published engine data and the paper's own Table 9/12 values.

Known discrepancies (see DISCREPANCIES.md §16, §17)
---------------------------------------------------
Top-of-climb TSFC runs ~10% high and every other operating point ~5% low,
consistently across two unrelated engines. Note also that ``Engine.Ttmax`` is
``True`` and ``T_{t_{4.1_{max}}}`` is declared and substituted at 1400 K in
every one of the four substitution sets — but it appears in no constraint
anywhere in the source. The turbine inlet temperature limit is dead: this
rebuild reproduces that (faithfully), and the solved ``T_{t_{4.1}}`` duly
exceeds 1400 K at top of climb.
"""
from __future__ import annotations

import numpy as np
from pyomo.environ import units

from edi import Formulation

# ---------------------------------------------------------------------------
# Component efficiencies and gas properties, from Engine.setvals()
# ---------------------------------------------------------------------------

# goption=1 (the default). gamma for: fan, LPC, HPC, combustor/cooling,
# LPT, HPT, station 6, station 8.
GAMMAS = dict(fgamma=1.401, lpcgamma=1.398, hpcgamma=1.354, ccgamma=1.313,
              lptgamma=1.354, hptgamma=1.318, sta6gamma=1.4, sta8gamma=1.387)

# Polytropic efficiencies: fan, LPC, HPC, HPT, LPT.
ETAS = {
    "CFM56":         (0.9005, 0.9306, 0.9030, 0.8731, 0.8851),
    "TASOPT_737800": (0.8948, 0.8800, 0.8700, 0.8990, 0.8890),
    "GE90":          (0.9153, 0.9037, 0.9247, 0.9121, 0.9228),
    "D82":           (0.9300, 0.9200, 0.8900, 0.9100, 0.9200),
}


def exponents(engine: str) -> dict:
    """The polytropic exponents, exactly as Engine.setvals() forms them."""
    g = GAMMAS
    faneta, lpceta, hpceta, hpteta, lpteta = ETAS[engine]
    return dict(
        fexp1=(g["fgamma"] - 1) / (faneta * g["fgamma"]),
        lpcexp1=(g["lpcgamma"] - 1) / (lpceta * g["lpcgamma"]),
        hpcexp1=(g["hpcgamma"] - 1) / (hpceta * g["hpcgamma"]),
        # Note the sign: ccexp1 is negative, ccexp2 positive.
        ccexp1=g["ccgamma"] / (1 - g["ccgamma"]),
        ccexp2=-g["ccgamma"] / (1 - g["ccgamma"]),
        lptexp1=g["lptgamma"] * lpteta / (g["lptgamma"] - 1),
        hptexp1=g["hptgamma"] * hpteta / (g["hptgamma"] - 1),
        fanexexp=(g["sta8gamma"] - 1) / g["sta8gamma"],
        turbexexp=(g["sta6gamma"] - 1) / g["sta6gamma"],
    )


# ---------------------------------------------------------------------------
# Per-engine constants, from turbofan/subs.py
# ---------------------------------------------------------------------------

M4A = 0.1025
HOLD4A = 1 + 0.5 * (1.313 - 1) * M4A ** 2

# Note alpha_OD: the GE90 substitution set does not define it, so on-design
# bypass ratio is a *free variable* for that engine and a constant for the
# others. That is not an oversight to be tidied up -- it is what lets the
# GE90's fan-size bracket float, and fixing it makes the model infeasible.
SUBS = {
    "CFM56": dict(pi_f_D=1.685, pi_lc_D=1.935, pi_hc_D=9.369,
                  alpha_OD=5.105, alpha_max=5.105, hf=40.8, OPR_max=32.0,
                  eta_B=0.9827, r_uc=0.01, alpha_c=0.19036, M_takeoff=0.9556,
                  pi_tn=0.98, pi_d=0.98, pi_fn=0.98),
    "TASOPT_737800": dict(pi_f_D=1.685, pi_lc_D=4.744, pi_hc_D=3.75,
                          alpha_OD=5.105, alpha_max=5.105, hf=43.003,
                          OPR_max=32.0, eta_B=0.9827, r_uc=0.01,
                          alpha_c=0.19036, M_takeoff=0.9556,
                          pi_tn=0.98, pi_d=0.98, pi_fn=0.98),
    "GE90": dict(pi_f_D=1.58, pi_lc_D=1.26, pi_hc_D=20.033,
                 alpha_OD=None, alpha_max=8.7877, hf=43.003, OPR_max=40.0,
                 eta_B=0.9970, r_uc=0.1, alpha_c=0.14, M_takeoff=0.955,
                 pi_tn=0.98, pi_d=0.98, pi_fn=0.98),
    "D82": dict(pi_f_D=1.60474, pi_lc_D=4.98, pi_hc_D=35. / 8.,
                alpha_OD=6.97, alpha_max=6.97, hf=43.003, OPR_max=32.0,
                eta_B=0.9827, r_uc=0.01, alpha_c=0.19036, M_takeoff=0.9556,
                pi_tn=0.995, pi_d=0.995, pi_fn=0.985),
}

# On-design mass flow anchors, from the `onDest` blocks of Engine.setup().
# (Tt_HPT, Pt_HPT, Tt_LPT, Pt_LPT, Tt_lpc, Pt_lpc, Tt_hpc, Pt_hpc,
#  fan_lo, fan_hi, Pt_fan)
ONDESIGN = {
    "CFM56":         (1400.0, 1527.0, 1038.8, 589.2, 292.57, 84.25,
                      362.47, 163.02, 0.7, 1.3, 50.0),
    "TASOPT_737800": (1400.0, 1498.0, 1144.8, 788.5, 294.5, 84.25,
                      482.7, 399.682, 0.7, 1.3, 50.0),
    "GE90":          (1400.0, 1527.0, 1038.8, 589.2, 292.57, 84.25,
                      362.47, 163.02, 0.3, 1.7, 50.0),
    "D82":           (1400.0, 1598.32, 1142.6, 835.585, 289.77, 80.237,
                      481.386, 399.58, 0.7, 1.3, 50.0),
}

# ---------------------------------------------------------------------------
# Flight segments, from turbofan/test_missions.py
# ---------------------------------------------------------------------------
# (thrust_N, P_atm_kPa, T_atm_K, M0, M2, M25)
LBF_N = 4.4482216152605

MISSIONS = {
    # Both segments sit at 36k ft / M0.8; only thrust differs. Note the source
    # writes the thrust as `5496.4 * 4.4 * units('N')` -- a lbf-to-N
    # conversion using 4.4 rather than 4.448, so these are not quite the round
    # lbf figures they look like. Reproduced as written.
    "CFM56": [
        (5496.4 * 4.4, 23.84, 218.0, 0.8, 0.6, 0.6),
        (5961.9 * 4.4, 23.84, 218.0, 0.8, 0.6, 0.6),
    ],
    # Segment 0 is cruise and segment 1 top of climb -- the reverse of the
    # CFM56 ordering, and the two differ in Mach number as well as thrust.
    # Here the thrusts really are in lbf.
    "GE90": [
        (16408.4 * LBF_N, 23.84, 218.0, 0.65, 0.60, 0.45),
        (19600.4 * LBF_N, 23.84, 218.0, 0.85, 0.65, 0.60),
    ],
}
# Engine weight caps, from the mission models (the GE90's is written in N).
WCAP_N = {"CFM56": 5216.0 * LBF_N, "GE90": 77399.0}

# gamma used to form the hold_{2} / hold_{2.5} stagnation factors.
G2, G25 = 1.398, 1.354


def build(engine: str = "CFM56") -> Formulation:
    """Build the standalone EDI turbofan for one of the validated engines.

    This is the engine on its own test stand: it creates its own ambient
    state, pins the operating points from ``MISSIONS``, applies the published
    engine-weight cap, and sets the TSFC-weighted objective. For the engine as
    a *component* of an aircraft -- where ambient conditions, Mach number and
    thrust all come from the mission -- call ``add_engine`` directly.
    """
    if engine not in MISSIONS:
        raise ValueError(f"no mission defined for {engine!r}; "
                         f"have {sorted(MISSIONS)}")
    segs = MISSIONS[engine]
    N = len(segs)
    f = Formulation()

    Vn = lambda n, g, u, d: f.Variable(name=n, guess=g, units=u,
                                       description=d, size=N)
    state = {
        "P_atm": Vn("P_atm", 23.84, "kPa", "ambient static pressure"),
        "T_atm": Vn("T_atm", 218.0, "K", "ambient static temperature"),
        "a": Vn("a", 300.0, "m/s", "ambient speed of sound"),
        "V": Vn("V", 240.0, "m/s", "flight speed"),
        "M": Vn("M", 0.8, "-", "flight Mach number"),
    }
    v, cons = add_engine(f, N, state, engine=engine)

    # The test-stand mission: prescribed operating points, weight cap, and the
    # TSFC-weighted objective from engine_validation.__main__.
    R = v["R"]
    for i in range(N):
        thrust_N, P_kPa, T_K, m0, m2, m25 = segs[i]
        cons += [
            state["P_atm"][i] == P_kPa * units.kPa,
            state["T_atm"][i] == T_K * units.K,
            state["M"][i] == m0,
            state["a"][i] == (1.4 * R * state["T_atm"][i]) ** 0.5,
            state["V"][i] == state["M"][i] * state["a"][i],
            v["M_2"][i] == m2,
            v["M_25"][i] == m25,
            v["c1"][i] == 1 + 0.5 * 0.401 * m0 ** 2,
            v["hold_2"][i] == 1 + 0.5 * (G2 - 1) * m2 ** 2,
            v["hold_25"][i] == 1 + 0.5 * (G25 - 1) * m25 ** 2,
            v["F"][i] == thrust_N * units.N,
        ]
    cons += [v["W_engine"] <= WCAP_N[engine] * units.N]

    w = [10.0] + [1.0] * (N - 1)
    f.Objective(v["W_engine"] ** 0.5
                * sum(w[i] * v["TSFC"][i] for i in range(N)))
    f.ConstraintList(cons)
    return f


def add_engine(f, N, state, *, engine: str = "CFM56", BLI: bool = False,
               prefix: str = ""):
    """Add one turbofan to ``f``, sharing geometry across ``N`` segments.

    ``state`` supplies the per-segment freestream as ``P_atm``, ``T_atm``,
    ``a``, ``V`` and ``M``. Everything else -- the cycle, the maps, the
    areas, the weight -- is created here. Returns ``(vars, constraints)``.

    With ``BLI=True`` the inlet ingests boundary layer: free-stream stagnation
    pressure is reduced by ``f_{BLI_P}`` and the fan-thrust momentum balance
    sees an inlet velocity reduced by ``f_{BLI_V}``. Those two factors are the
    entire boundary-layer-ingestion model as far as the engine is concerned.
    """
    exp = exponents(engine)
    sub = SUBS[engine]
    P = prefix
    V = lambda n, g, u, d: f.Variable(name=f"{P}{n}", guess=g, units=u,
                                      description=d)
    Vn = lambda n, g, u, d: f.Variable(name=f"{P}{n}", guess=g, units=u,
                                       description=d, size=N)
    C = lambda n, v, u, d: f.Constant(name=f"{P}{n}", value=v, units=u,
                                      description=d)

    # ---- gas properties (Cp values are the source's, at the stated temps) --
    R      = C("R", 287.0, "J/kg/K", "air gas constant")
    Cpair  = C("Cp_air", 1003.0, "J/kg/K", "Cp of air at 250 K")
    Cp1    = C("Cp_1", 1008.0, "J/kg/K", "Cp of air at 350 K")
    Cp2    = C("Cp_2", 1099.0, "J/kg/K", "Cp of air at 800 K")
    Cpc    = C("Cp_c", 1216.0, "J/kg/K", "Cp of fuel/air mix in combustor")
    Cpfuel = C("Cp_fuel", 2010.0, "J/kg/K", "Cp of kerosene")
    Cpt1   = C("Cp_t1", 1280.0, "J/kg/K", "Cp of combustion products, HPT")
    Cpt2   = C("Cp_t2", 1184.0, "J/kg/K", "Cp of combustion products, LPT")
    Cptex  = C("Cp_tex", 1029.0, "J/kg/K", "Cp of core exhaust at 500 K")
    Cpfex  = C("Cp_fex", 1005.0, "J/kg/K", "Cp of fan exhaust at 300 K")
    hf     = C("h_f", sub["hf"], "MJ/kg", "heat of combustion of jet fuel")
    g0     = C("g", 9.81, "m/s^2", "gravitational acceleration")

    # ---- pressure ratios and efficiencies ---------------------------------
    pid   = C("pi_d", sub["pi_d"], "-", "diffuser pressure ratio")
    pifn  = C("pi_fn", sub["pi_fn"], "-", "fan duct pressure loss ratio")
    pib   = C("pi_b", 0.94, "-", "burner pressure ratio")
    pitn  = C("pi_tn", sub["pi_tn"], "-", "turbine nozzle pressure ratio")
    etaB  = C("eta_B", sub["eta_B"], "-", "burner efficiency")
    etaHP = C("eta_HPshaft", 0.97, "-", "HP shaft transmission efficiency")
    etaLP = C("eta_LPshaft", 0.97, "-", "LP shaft transmission efficiency")
    Mtakeoff = C("M_takeoff", sub["M_takeoff"], "-",
                 "1 - bleed mass flow fraction")
    Tref  = C("T_ref", 288.15, "K", "reference stagnation temperature")
    Pref  = C("P_ref", 101.325, "kPa", "reference stagnation pressure")

    # ---- cooling flow ------------------------------------------------------
    alpha_c = C("alpha_c", sub["alpha_c"], "-",
                "total cooling flow bypass ratio")
    ruc   = C("r_uc", sub["r_uc"], "-", "cooling flow velocity ratio")
    hold4a = C("hold_4a", HOLD4A, "-", "1 + (gamma-1)/2 M_4a^2")
    M4a   = C("M_4a", M4A, "-", "station 4a Mach number")
    Ttf   = C("T_t_f", 435.0, "K", "incoming fuel total temperature")

    # ---- design pressure ratios and geometry ratios -----------------------
    piFanD = C("pi_f_D", sub["pi_f_D"], "-", "fan on-design pressure ratio")
    pilcD  = C("pi_lc_D", sub["pi_lc_D"], "-", "LPC on-design pressure ratio")
    pihcD  = C("pi_hc_D", sub["pi_hc_D"], "-", "HPC on-design pressure ratio")
    alphaOD = (C("alpha_OD", sub["alpha_OD"], "-", "on-design bypass ratio")
               if sub["alpha_OD"] is not None else
               V("alpha_OD", 8.0, "-", "on-design bypass ratio (free)"))
    alphamax = C("alpha_max", sub["alpha_max"], "-", "maximum bypass ratio")
    Gf     = C("G_f", 1.0, "-", "fan/LPC gear ratio")
    OPRmax = C("OPR_max", sub["OPR_max"], "-",
               "maximum overall pressure ratio")
    HTRfS  = C("HTR_f_SUB", 1 - 0.3 ** 2, "-", "1 - HTR_fan^2")
    fBLIP  = C("f_BLI_P", 0.9627, "-", "BLI stagnation pressure loss ratio")
    fBLIV  = C("f_BLI_V", 0.927288, "-", "BLI velocity loss ratio")
    HTRlS  = C("HTR_lpc_SUB", 1 - 0.6 ** 2, "-", "1 - HTR_lpc^2")

    # ---- engine-level free variables (shared across all segments) ---------
    W_engine = V("W_engine", 1e4, "N", "weight of a single turbofan")
    df   = V("d_f", 1.5, "m", "fan diameter")
    dlpc = V("d_LPC", 0.8, "m", "LPC diameter")
    A2   = V("A_2", 1.5, "m^2", "fan area")
    A25  = V("A_25", 0.3, "m^2", "HPC area")
    A5   = V("A_5", 0.3, "m^2", "core exhaust nozzle area")
    A7   = V("A_7", 1.0, "m^2", "fan exhaust nozzle area")
    mhtD = V("m_htD", 10.0, "kg/s", "design HPT corrected mass flow")
    mltD = V("m_ltD", 20.0, "kg/s", "design LPT corrected mass flow")
    mCoreD = V("m_coreD", 30.0, "kg/s", "estimated on-design core mass flow")
    mFanD = V("mbar_fan_D", 200.0, "kg/s", "fan on-design corrected mass flow")
    mlcD = V("m_lc_D", 30.0, "kg/s", "LPC on-design corrected mass flow")
    mhcD = V("m_hc_D", 15.0, "kg/s", "HPC on-design corrected mass flow")

    # ---- per-segment free variables ---------------------------------------
    # Ambient state is supplied by the caller; c1 is the engine's own.
    Patm, Tatm = state["P_atm"], state["T_atm"]
    a, Vinf, M0 = state["a"], state["V"], state["M"]
    c1 = Vn("c1", 1.13, "-", "1 + (gamma-1)/2 M_0^2")

    # stagnation states through the machine
    Pt0  = Vn("P_t_0", 36.0, "kPa", "free stream stagnation pressure")
    Tt0  = Vn("T_t_0", 246.0, "K", "free stream stagnation temperature")
    ht0  = Vn("h_t_0", 2.5e5, "J/kg", "free stream stagnation enthalpy")
    Pt18 = Vn("P_t_18", 35.0, "kPa", "diffuser exit stagnation pressure")
    Tt18 = Vn("T_t_18", 246.0, "K", "diffuser exit stagnation temperature")
    ht18 = Vn("h_t_18", 2.5e5, "J/kg", "diffuser exit stagnation enthalpy")
    Pt2  = Vn("P_t_2", 35.0, "kPa", "fan inlet stagnation pressure")
    Tt2  = Vn("T_t_2", 246.0, "K", "fan inlet stagnation temperature")
    ht2  = Vn("h_t_2", 2.5e5, "J/kg", "fan inlet stagnation enthalpy")
    Pt21 = Vn("P_t_21", 59.0, "kPa", "fan exit stagnation pressure")
    Tt21 = Vn("T_t_21", 288.0, "K", "fan exit stagnation temperature")
    ht21 = Vn("h_t_21", 2.9e5, "J/kg", "fan exit stagnation enthalpy")
    Pt25 = Vn("P_t_25", 114.0, "kPa", "LPC exit stagnation pressure")
    Tt25 = Vn("T_t_25", 350.0, "K", "LPC exit stagnation temperature")
    ht25 = Vn("h_t_25", 3.5e5, "J/kg", "LPC exit stagnation enthalpy")
    Pt3  = Vn("P_t_3", 1000.0, "kPa", "HPC exit stagnation pressure")
    Tt3  = Vn("T_t_3", 700.0, "K", "HPC exit stagnation temperature")
    ht3  = Vn("h_t_3", 7.7e5, "J/kg", "HPC exit stagnation enthalpy")
    Pt4  = Vn("P_t_4", 950.0, "kPa", "combustor exit stagnation pressure")
    Tt4  = Vn("T_t_4", 1500.0, "K", "combustor exit stagnation temperature")
    ht4  = Vn("h_t_4", 1.8e6, "J/kg", "combustor exit stagnation enthalpy")
    Pt41 = Vn("P_t_41", 950.0, "kPa", "turbine inlet stagnation pressure")
    Tt41 = Vn("T_t_41", 1400.0, "K", "turbine inlet stagnation temperature")
    ht41 = Vn("h_t_41", 1.7e6, "J/kg", "turbine inlet stagnation enthalpy")
    Pt45 = Vn("P_t_45", 250.0, "kPa", "HPT exit stagnation pressure")
    Tt45 = Vn("T_t_45", 1000.0, "K", "HPT exit stagnation temperature")
    ht45 = Vn("h_t_45", 1.3e6, "J/kg", "HPT exit stagnation enthalpy")
    Pt49 = Vn("P_t_49", 60.0, "kPa", "LPT exit stagnation pressure")
    Tt49 = Vn("T_t_49", 750.0, "K", "LPT exit stagnation temperature")
    ht49 = Vn("h_t_49", 8.9e5, "J/kg", "LPT exit stagnation enthalpy")
    Pt5  = Vn("P_t_5", 59.0, "kPa", "core nozzle exit stagnation pressure")
    Tt5  = Vn("T_t_5", 750.0, "K", "core nozzle exit stagnation temperature")
    ht5  = Vn("h_t_5", 8.9e5, "J/kg", "core nozzle exit stagnation enthalpy")
    Pt7  = Vn("P_t_7", 58.0, "kPa", "fan nozzle exit stagnation pressure")
    Tt7  = Vn("T_t_7", 288.0, "K", "fan nozzle exit stagnation temperature")
    ht7  = Vn("h_t_7", 2.9e5, "J/kg", "fan nozzle exit stagnation enthalpy")
    Pt8  = Vn("P_t_8", 58.0, "kPa", "fan exhaust stagnation pressure")
    Tt8  = Vn("T_t_8", 288.0, "K", "fan exhaust stagnation temperature")
    ht8  = Vn("h_t_8", 2.9e5, "J/kg", "fan exhaust stagnation enthalpy")
    Pt6  = Vn("P_t_6", 59.0, "kPa", "core exhaust stagnation pressure")
    Tt6  = Vn("T_t_6", 750.0, "K", "core exhaust stagnation temperature")
    ht6  = Vn("h_t_6", 7.7e5, "J/kg", "core exhaust stagnation enthalpy")

    # static states
    T2   = Vn("T_2", 230.0, "K", "static temperature at fan face")
    P2   = Vn("P_2", 28.0, "kPa", "static pressure at fan face")
    rho2 = Vn("rho_2", 0.42, "kg/m^3", "static density at fan face")
    u2   = Vn("u_2", 180.0, "m/s", "axial speed at fan face")
    h2   = Vn("h_2", 2.3e5, "J/kg", "static enthalpy at fan face")
    M2v  = Vn("M_2", 0.6, "-", "fan face axial Mach number")
    T25  = Vn("T_25", 330.0, "K", "static temperature at HPC face")
    P25  = Vn("P_25", 90.0, "kPa", "static pressure at HPC face")
    rho25 = Vn("rho_25", 0.95, "kg/m^3", "static density at HPC face")
    u25  = Vn("u_25", 220.0, "m/s", "axial speed at HPC face")
    h25  = Vn("h_25", 3.6e5, "J/kg", "static enthalpy at HPC face")
    M25v = Vn("M_25", 0.6, "-", "HPC face axial Mach number")
    hold2  = Vn("hold_2", 1 + 0.5 * (G2 - 1) * 0.36, "-", "1+(g-1)/2 M_2^2")
    hold25 = Vn("hold_25", 1 + 0.5 * (G25 - 1) * 0.36, "-", "1+(g-1)/2 M_25^2")

    T5   = Vn("T_5", 700.0, "K", "static temperature at core nozzle exit")
    P5   = Vn("P_5", 30.0, "kPa", "static pressure at core nozzle exit")
    rho5 = Vn("rho_5", 0.15, "kg/m^3", "static density at core nozzle exit")
    u5   = Vn("u_5", 400.0, "m/s", "core nozzle exit velocity")
    a5   = Vn("a_5", 530.0, "m/s", "speed of sound at station 5")
    M5   = Vn("M_5", 0.8, "-", "station 5 Mach number")
    T7   = Vn("T_7", 270.0, "K", "static temperature at fan nozzle exit")
    P7   = Vn("P_7", 30.0, "kPa", "static pressure at fan nozzle exit")
    rho7 = Vn("rho_7", 0.39, "kg/m^3", "static density at fan nozzle exit")
    u7   = Vn("u_7", 280.0, "m/s", "fan nozzle exit velocity")
    a7   = Vn("a_7", 330.0, "m/s", "speed of sound at station 7")
    M7   = Vn("M_7", 0.85, "-", "station 7 Mach number")
    T6   = Vn("T_6", 700.0, "K", "core exhaust static temperature")
    h6   = Vn("h_6", 7.2e5, "J/kg", "core exhaust static enthalpy")
    u6   = Vn("u_6", 400.0, "m/s", "core exhaust velocity")
    T8   = Vn("T_8", 270.0, "K", "fan exhaust static temperature")
    h8   = Vn("h_8", 2.7e5, "J/kg", "fan exhaust static enthalpy")
    u8   = Vn("u_8", 290.0, "m/s", "fan exhaust velocity")

    # cooling-flow mixing station 4a
    u41  = Vn("u_41", 200.0, "m/s", "flow velocity at station 4.1")
    T41  = Vn("T_41", 1380.0, "K", "static temperature at station 4.1")
    u4a  = Vn("u_4a", 80.0, "m/s", "flow velocity at station 4a")
    uc   = Vn("u_c", 1.0, "m/s", "cooling airflow speed at station 4a")
    P4a  = Vn("P_4a", 940.0, "kPa", "static pressure at station 4a")

    # pressure ratios, speeds, mass flows
    pif  = Vn("pi_f", 1.6, "-", "fan pressure ratio")
    pilc = Vn("pi_lc", 1.9, "-", "LPC pressure ratio")
    pihc = Vn("pi_hc", 9.0, "-", "HPC pressure ratio")
    pihpt = Vn("pi_HPT", 0.3, "-", "HPT pressure ratio")
    pilpt = Vn("pi_LPT", 0.25, "-", "LPT pressure ratio")
    OPR  = Vn("OPR", 25.0, "-", "overall pressure ratio")
    Nf   = Vn("N_f", 1.0, "-", "fan speed")
    N1   = Vn("N_1", 1.0, "-", "LPC speed")
    N2   = Vn("N_2", 1.0, "-", "HPC speed")
    mf   = Vn("m_f", 200.0, "kg/s", "fan corrected mass flow")
    mlc  = Vn("m_lc", 40.0, "kg/s", "LPC corrected mass flow")
    mhc  = Vn("m_hc", 15.0, "kg/s", "HPC corrected mass flow")
    mtildf  = Vn("m_tild_f", 1.0, "-", "fan normalized mass flow")
    mtildlc = Vn("m_tild_lc", 1.0, "-", "LPC normalized mass flow")
    mtildhc = Vn("m_tild_hc", 1.0, "-", "HPC normalized mass flow")
    mCore = Vn("m_core", 30.0, "kg/s", "core mass flow")
    mFan  = Vn("m_fan", 150.0, "kg/s", "fan mass flow")
    mtot  = Vn("m_total", 180.0, "kg/s", "total engine mass flow")

    # combustion and thrust
    fuel  = Vn("f", 0.02, "-", "fuel/air mass flow fraction")
    fp1   = Vn("fp1", 1.02, "-", "f + 1")
    alpha = Vn("alpha", 5.0, "-", "bypass ratio")
    alphap1 = Vn("alpha_p1", 6.0, "-", "1 + bypass ratio")
    F   = Vn("F", 24000.0, "N", "total thrust")
    F6  = Vn("F_6", 6000.0, "N", "core thrust")
    F8  = Vn("F_8", 18000.0, "N", "fan thrust")
    Fsp = Vn("F_sp", 0.5, "-", "specific net thrust")
    Isp = Vn("I_sp", 5000.0, "s", "specific impulse")
    TSFC = Vn("TSFC", 0.7, "1/hr", "thrust specific fuel consumption")

    cons = []
    pi = np.pi

    # ---- scalar (engine geometry) -----------------------------------------
    cons += [
        A5 + A7 <= A2,
        df == (4 * A2 / (pi * HTRfS)) ** 0.5,
        dlpc == (4 * A25 / (pi * HTRlS)) ** 0.5,
    ]

    # On-design mass-flow anchors. These are +/-30% (or +/-70% on the fan for
    # the GE90) brackets tying the map design flows to m_coreD; they are what
    # pins the engine's absolute size.
    (TtH, PtH, TtL, PtL, Ttlc, Ptlc, Tthc, Pthc,
     flo, fhi, Ptfan) = ONDESIGN[engine]
    for i in range(N):
        cons += [
            mhtD <= 1.3 * fp1[i] * Mtakeoff * mCoreD * (TtH / 288) ** 0.5 / (PtH / 101.325),
            mhtD >= 0.7 * fp1[i] * Mtakeoff * mCoreD * (TtH / 288) ** 0.5 / (PtH / 101.325),
            mltD <= 1.3 * fp1[i] * Mtakeoff * mCoreD * (TtL / 288) ** 0.5 / (PtL / 101.325),
            mltD >= 0.7 * fp1[i] * Mtakeoff * mCoreD * (TtL / 288) ** 0.5 / (PtL / 101.325),
        ]
    cons += [
        mlcD >= 0.7 * mCoreD * (Ttlc / 288) ** 0.5 / (Ptlc / 101.325),
        mlcD <= 1.3 * mCoreD * (Ttlc / 288) ** 0.5 / (Ptlc / 101.325),
        mhcD >= 0.7 * mCoreD * (Tthc / 288) ** 0.5 / (Pthc / 101.325),
        mhcD <= 1.3 * mCoreD * (Tthc / 288) ** 0.5 / (Pthc / 101.325),
        mFanD >= flo * alphaOD * mCoreD * (250.0 / 288) ** 0.5 / (Ptfan / 101.325),
        mFanD <= fhi * alphaOD * mCoreD * (250.0 / 288) ** 0.5 / (Ptfan / 101.325),
    ]

    # ---- per-segment -------------------------------------------------------
    for i in range(N):
        # diffuser (station 0 -> 1.8). With BLI the free-stream stagnation
        # pressure is knocked down by f_BLI_P before the diffuser sees it.
        cons += [
            Tt0[i] == Tatm[i] * c1[i],
            ht0[i] == Cpair * Tt0[i],
            (Pt0[i] == fBLIP * Patm[i] * c1[i] ** 3.5) if BLI else
            (Pt0[i] == Patm[i] * c1[i] ** 3.5),
            Pt18[i] == pid * Pt0[i],
            Tt18[i] == Tt0[i],
            ht18[i] == ht0[i],
        ]

        # fan (2 -> 2.1 -> 7)
        cons += [
            Tt2[i] == Tt18[i], ht2[i] == ht18[i], Pt2[i] == Pt18[i],
            Pt21[i] == pif[i] * Pt2[i],
            Tt21[i] == Tt2[i] * pif[i] ** exp["fexp1"],
            ht21[i] == Cpair * Tt21[i],
            Pt7[i] == pifn * Pt21[i],
            Tt7[i] == Tt21[i], ht7[i] == ht21[i],
        ]

        # LPC (-> 2.5) and HPC (-> 3)
        cons += [
            Pt25[i] == pilc[i] * pif[i] * Pt2[i],
            Tt25[i] == Tt2[i] * (pif[i] * pilc[i]) ** exp["lpcexp1"],
            ht25[i] == Tt25[i] * Cp1,
            Pt3[i] == pihc[i] * Pt25[i],
            Tt3[i] == Tt25[i] * pihc[i] ** exp["hpcexp1"],
            ht3[i] == Cp2 * Tt3[i],
        ]

        # combustor (-> 4) and cooling-flow mixing (-> 4.1)
        cons += [
            ht4[i] == Cpc * Tt4[i],
            ht41[i] == Cpc * Tt41[i],
            fp1[i] == fuel[i] + 1,                                   # [SP]
            Pt4[i] == pib * Pt3[i],
            etaB * fuel[i] * hf >= (1 - alpha_c) * ht4[i]
                                   - (1 - alpha_c) * ht3[i]
                                   + Cpfuel * fuel[i] * (Tt4[i] - Ttf),
            ht41[i] * fp1[i] <= ((1 - alpha_c + fuel[i]) * ht4[i]
                                 + alpha_c * ht3[i]),
            fp1[i] * u41[i] == (u4a[i] * fp1[i] * alpha_c * uc[i]) ** 0.5,
            T41[i] <= Tt41[i] - 0.5 * u41[i] ** 2 / Cpc,
            Pt41[i] == P4a[i] * (Tt41[i] / T41[i]) ** exp["ccexp1"],
            u4a[i] == M4a * ((1.313 * R * Tt4[i]) ** 0.5) / hold4a,
            uc[i] == ruc * u4a[i],
            P4a[i] == Pt4[i] * hold4a ** exp["ccexp2"],
        ]

        # turbines (4.1 -> 4.5 -> 4.9 -> 5)
        cons += [
            ht45[i] == Cpt1 * Tt45[i],
            Pt45[i] == pihpt[i] * Pt41[i],
            pihpt[i] == (Tt45[i] / Tt41[i]) ** exp["hptexp1"],
            Pt49[i] == pilpt[i] * Pt45[i],
            pilpt[i] == (Tt49[i] / Tt45[i]) ** exp["lptexp1"],
            ht49[i] == Cpt2 * Tt49[i],
            Pt5[i] == pitn * Pt49[i],
            Tt5[i] == Tt49[i], ht5[i] == ht49[i],
        ]

        # shaft power balances
        cons += [
            Mtakeoff * etaHP * fp1[i] * (ht41[i] - ht45[i]) >= ht3[i] - ht25[i],
            Mtakeoff * etaLP * fp1[i] * (ht45[i] - ht49[i])
                >= ht25[i] - ht18[i] + alpha[i] * (ht21[i] - ht2[i]),
        ]

        # component maps. The 1.7 and 26 are the pressure ratios the E3 map
        # fits were normalized about; dividing by the design ratio re-centres
        # each map on this engine.
        cons += [
            pif[i] * (1.7 / piFanD) == (1.05 * Nf[i] ** 0.0871) ** 10,
            pif[i] * (1.7 / piFanD) <= 1.1 * (1.06 * mtildf[i] ** 0.137) ** 10,
            pif[i] * (1.7 / piFanD) >= 0.9 * (1.06 * mtildf[i] ** 0.137) ** 10,
            mf[i] == mFan[i] * ((Tt2[i] / Tref) ** 0.5) / (Pt2[i] / Pref),
            mtildf[i] == mf[i] / mFanD,
            pif[i] >= 1,

            pilc[i] * (26 / pilcD) == (1.38 * N1[i] ** 0.566) ** 10,
            pilc[i] * (26 / pilcD) <= 1.1 * (1.38 * mtildlc[i] ** 0.122) ** 10,
            pilc[i] * (26 / pilcD) >= 0.9 * (1.38 * mtildlc[i] ** 0.122) ** 10,
            mlc[i] == mCore[i] * ((Tt2[i] / Tref) ** 0.5) / (Pt2[i] / Pref),
            mtildlc[i] == mlc[i] / mlcD,
            pilc[i] >= 1,

            pihc[i] * (26 / pihcD) == (1.38 * N2[i] ** 0.566) ** 10,
            pihc[i] * (26 / pihcD) <= 1.1 * (1.38 * mtildhc[i] ** 0.122) ** 10,
            pihc[i] * (26 / pihcD) >= 0.9 * (1.38 * mtildhc[i] ** 0.122) ** 10,
            mhc[i] == mCore[i] * ((Tt25[i] / Tref) ** 0.5) / (Pt25[i] / Pref),
            mtildhc[i] == mhc[i] / mhcD,
            pihc[i] >= 1,

            OPR[i] == pihc[i] * pilc[i] * pif[i],
            OPR[i] <= OPRmax,
        ]

        # spool speed residuals
        cons += [
            Nf[i] * Gf == N1[i],
            N1[i] <= 1.1,
            N2[i] <= 1.1,
        ]

        # turbine corrected mass flow residuals (2 and 3)
        cons += [
            mhtD == fp1[i] * mhc[i] * Mtakeoff * (Pt25[i] / Pt41[i])
                    * (Tt41[i] / Tt25[i]) ** 0.5,
            fp1[i] * mlc[i] * Mtakeoff * (Pt18[i] / Pt45[i])
                    * (Tt45[i] / Tt18[i]) ** 0.5 == mltD,
        ]

        # nozzle residuals (4 and 5)
        cons += [
            (P7[i] / Pt7[i]) == (T7[i] / Tt7[i]) ** 3.5,
            (T7[i] / Tt7[i]) ** -1 >= 1 + 0.2 * M7[i] ** 2,
            (P5[i] / Pt5[i]) == (T5[i] / Tt5[i]) ** 3.583979,
            (T5[i] / Tt5[i]) ** -1 >= 1 + 0.2 * M5[i] ** 2,
        ]

        # mass flux and areas
        cons += [
            Mtakeoff * mCore[i] == rho5[i] * A5 * u5[i] / fp1[i],
            mFan[i] == rho7[i] * A7 * u7[i],
            mtot[i] >= mFan[i] + mCore[i],
            P2[i] == Pt2[i] * hold2[i] ** (-3.512),
            T2[i] == Tt2[i] * hold2[i] ** -1,
            A2 == mFan[i] / (rho2[i] * u2[i]),
            P25[i] == Pt25[i] * hold25[i] ** (-3.824857),
            T25[i] == Tt25[i] * hold25[i] ** -1,
            A25 == mCore[i] / (rho25[i] * u25[i]),
        ]

        # nozzle and face static states
        cons += [
            P7[i] >= Patm[i], M7[i] <= 1,
            a7[i] == (1.4 * R * T7[i]) ** 0.5,
            a7[i] * M7[i] == u7[i],
            rho7[i] == P7[i] / (R * T7[i]),
            P5[i] >= Patm[i], M5[i] <= 1,
            a5[i] == (1.387 * R * T5[i]) ** 0.5,
            a5[i] * M5[i] == u5[i],
            rho5[i] == P5[i] / (R * T5[i]),
            h2[i] == Cp1 * T2[i],
            rho2[i] == P2[i] / (R * T2[i]),
            # The 781 J/kg/K is Cp - R for the fan-face gas; u = M sqrt(gamma R T)
            # rewritten so gamma need not be a separate variable.
            u2[i] == M2v[i] * (Cp1 * R * T2[i] / (781.0 * units.J / units.kg / units.K)) ** 0.5,
            h25[i] == Cp2 * T25[i],
            rho25[i] == P25[i] / (R * T25[i]),
            u25[i] == M25v[i] * (Cp2 * R * T25[i] / (781.0 * units.J / units.kg / units.K)) ** 0.5,
        ]

        # exhausts and thrust
        cons += [
            Pt8[i] == Pt7[i], Tt8[i] == Tt7[i],
            Pt6[i] == Pt5[i], Tt6[i] == Tt5[i],
            # P_8 and P_6 are both the ambient static pressure, so they are
            # substituted directly rather than carried as variables.
            h8[i] == Cpfex * T8[i],
            ht8[i] == Cpfex * Tt8[i],
            u8[i] ** 2 + 2 * h8[i] <= 2 * ht8[i],
            (Patm[i] / Pt8[i]) ** exp["fanexexp"] == T8[i] / Tt8[i],
            h6[i] == Cptex * T6[i],
            ht6[i] == Cptex * Tt6[i],
            u6[i] ** 2 + 2 * h6[i] <= 2 * ht6[i],
            (Patm[i] / Pt6[i]) ** exp["turbexexp"] == T6[i] / Tt6[i],
            alpha[i] == mFan[i] / mCore[i],
            alphap1[i] == alpha[i] + 1,                              # [SP]
            alpha[i] <= alphamax,
            F[i] <= F6[i] + F8[i],                                   # [SP]
            F6[i] / (Mtakeoff * mCore[i]) + fp1[i] * Vinf[i] <= fp1[i] * u6[i],
            (F8[i] / (alpha[i] * mCore[i]) + Vinf[i] * fBLIV <= u8[i]) if BLI
            else (F8[i] / (alpha[i] * mCore[i]) + Vinf[i] <= u8[i]),
            Fsp[i] == F[i] / (alphap1[i] * mCore[i] * a[i]),
            Isp[i] == Fsp[i] * a[i] * alphap1[i] / (fuel[i] * g0),
            TSFC[i] == 1 / Isp[i],
        ]

        # engine weight (TASOPT data fit, relaxed to an inequality)
        cons += [
            W_engine / units.kg >= (mtot[i] / alphap1[i])
                * ((1 / (100 * units.lb / units.s)) * 9.81 * units.m / units.s ** 2)
                * (1684.5 + 17.7 * (pif[i] * pilc[i] * pihc[i]) / 30
                   + 1662.2 * (alpha[i] / 5) ** 1.2),
        ]

    out = dict(W_engine=W_engine, TSFC=TSFC, F=F, F_6=F6, F_8=F8, R=R,
               M_2=M2v, M_25=M25v, hold_2=hold2, hold_25=hold25, c1=c1,
               A_2=A2, A_25=A25, A_5=A5, A_7=A7, d_f=df, d_LPC=dlpc,
               alpha=alpha, alpha_p1=alphap1, OPR=OPR, T_t_41=Tt41,
               T_t_4=Tt4, m_core=mCore, m_fan=mFan, m_total=mtot, f=fuel,
               pi_f=pif, pi_lc=pilc, pi_hc=pihc, N_1=N1, N_2=N2, I_sp=Isp)
    return out, cons


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from harness import solve_edi, feasibility, load_reference

    name = sys.argv[1] if len(sys.argv) > 1 else "CFM56"
    fm = build(name)
    sol, obj, note = solve_edi(fm)
    ref = load_reference(Path(__file__).with_name("reference.json"))[name]

    # gpkit's reference keys -> how to read the same quantity out of the EDI
    # solution. W_engine is declared in newtons here and reported in lbf there.
    N_SEG = len(MISSIONS[name])
    LBF = LBF_N

    def rebuilt(key):
        if key == "W_engine_lbf":
            return sol["W_engine"] / LBF
        head, _, idx = key.partition("[")
        i = int(idx.rstrip("]"))
        if head == "T_t4.1":
            return sol[f"T_t_41[{i}]"]
        if head == "cooling_dT":
            return sol[f"T_t_4[{i}]"] - sol[f"T_t_41[{i}]"]
        return sol[f"{head}[{i}]"]

    print(f"\n{name}: objective = {obj:.6g}  (gpkit {ref['cost']:.6g}, "
          f"{100 * (obj - ref['cost']) / ref['cost']:+.3f}%)")
    print(f"{'quantity':16} {'rebuilt':>12} {'gpkit':>12} {'vs gpkit':>10} "
          f"{'published':>12}")
    pub = ref["published"]
    worst_rel = 0.0
    for key in sorted(ref["checked"]):
        exp = ref["checked"][key]
        try:
            got = rebuilt(key)
        except KeyError:
            print(f"{key:16} {'--':>12} {exp:12.5g}")
            continue
        rel = abs(got - exp) / max(abs(exp), 1e-30)
        worst_rel = max(worst_rel, rel)
        print(f"{key:16} {got:12.5g} {exp:12.5g} {rel:10.2e} "
              f"{pub.get(key, float('nan')):12.5g}")
    print(f"worst relative difference vs gpkit: {worst_rel:.2e}")
    nv, worst, where = feasibility(fm)
    print(f"feasibility: {nv} violated, worst rel {worst:.2e}"
          + (f" at {where}" if where else ""))
    if note:
        print(f"note: {note}")
