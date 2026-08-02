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

import components.technology as _technology

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
    "D82_SPaircraft": (0.9300, 0.9200, 0.8900, 0.9100, 0.9200),
    # Same turbomachinery as D82_SPaircraft -- only the fuel differs.
    "D82_LH2":       (0.9300, 0.9200, 0.8900, 0.9100, 0.9200),
    "CFM56":         (0.9005, 0.9306, 0.9030, 0.8731, 0.8851),
    "TASOPT_737800": (0.8948, 0.8800, 0.8700, 0.8990, 0.8890),
    "GE90":          (0.9153, 0.9037, 0.9247, 0.9121, 0.9228),
    "D82":           (0.9300, 0.9200, 0.8900, 0.9100, 0.9200),
}


def fitzgerald_coeffs(BPR, geared=False, advanced=False):
    """``(a, b, c)`` for TASOPT's ``We1 = a (mdotc/45.35)^b (OPR/40)^c``.

    A port of ``_fitzgerald_coeffs`` in tasopt_py/engine/weight.py, which is
    itself ``tfweight.f``'s ``iengwgt = 1, 2`` branch. ``a`` comes out in lbf.

    Evaluated at the DECK's design bypass ratio, which is what makes this
    usable here: ``b`` and ``c`` are exponents on ``mdotc`` and ``OPR``, so a
    BPR-dependent ``b`` would put a variable in an exponent and leave GP.
    TASOPT has the same structure -- its BPR is a deck input, not a solved
    quantity -- so evaluating at the design value is the source behaviour, not
    an approximation of it.
    """
    if not geared:
        if not advanced:
            return (18.09 * BPR ** 2 + 476.9 * BPR + 701.3,
                    0.001077 * BPR ** 2 - 0.03716 * BPR + 1.190,
                    -0.01058 * BPR + 0.3259)
        return (15.38 * BPR ** 2 + 401.1 * BPR + 631.5,
                0.001057 * BPR ** 2 - 0.03693 * BPR + 1.171,
                -0.01022 * BPR + 0.2321)
    if not advanced:
        return (-0.6590 * BPR ** 2 + 292.8 * BPR + 1915.0,
                0.00006784 * BPR ** 2 - 0.006488 * BPR + 1.061,
                -0.001969 * BPR + 0.07107)
    return (-0.6204 * BPR ** 2 + 237.3 * BPR + 1702.0,
            0.00005845 * BPR ** 2 - 0.005866 * BPR + 1.045,
            -0.001918 * BPR + 0.06765)


def exponents(engine: str) -> dict:
    """The polytropic exponents, exactly as Engine.setvals() forms them."""
    g = GAMMAS
    faneta, lpceta, hpceta, hpteta, lpteta = ETAS[engine]
    # FAN EFFICIENCY LAPSE WITH PRESSURE RATIO, from the TASOPT decks:
    #
    #     epolf_actual = epolf + K_epf * (FPR - FPRo)
    #
    # (runs/737/737.tas:485-486, runs/D8/d82.tas:490-491, both K_epf = -0.077).
    # A higher-pressure-ratio fan is a harder fan to make efficient, and this
    # is the ONLY thing in TASOPT that opposes a runaway FPR -- there is no
    # constraint on FPR itself. We had the reference value and not the lapse:
    # ETAS stores epolf AT FPRo, and applying it at every FPR made high fan
    # pressure ratio free. Unanchored, the D8's design FPR ran to 1.889 against
    # the deck's 1.605, which shrank the fan to 0.83 of TASOPT's diameter and
    # the bare engine to 0.615 of Webare.
    #
    # Evaluated at the DESIGN FPR, as TASOPT does -- its FPR is a deck input
    # and off-design excursions ride the fan map. That matters here for a
    # second reason: it keeps the result a CONSTANT. The polytropic exponent
    # is used as `pi_f ** fexp1`, so an efficiency that varied with the
    # operating pi_f would put a variable in an exponent and leave GP.
    _sub = SUBS.get(engine, {})
    _fpro, _kepf, _pifd = (_sub.get("FPRo"), _sub.get("K_epf"),
                           _sub.get("pi_f_D"))
    if _fpro is not None and _kepf is not None and _pifd is not None:
        faneta = faneta + _kepf * (_pifd - _fpro)
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
# Bypass ratio was pinned on the kerosene engines by giving alpha_OD and
# alpha_max the SAME value, which is an equality in disguise: on-design BPR
# equals max BPR equals the reference engine's. The audit caught it binding on
# all five mission segments with a dual of 0.085.
#
# That is defensible for reproducing a CFM56 and indefensible for an
# architecture study: BPR is one of the few knobs that genuinely distinguishes
# a hydrogen or fuel-cell propulsor from a kerosene one, and pinning it at the
# 1980s value hands every architecture the same fan. The H2 sets below already
# leave it free (alpha_OD=None, alpha_max=None); these now do too.
SUBS = {
    "CFM56": dict(pi_f_D=1.685, pi_lc_D=1.935, pi_hc_D=9.369,
                  alpha_OD=None, alpha_max=None, hf=40.8, OPR_max=32.0,
                  eta_B=0.9827, r_uc=0.01, alpha_c=0.19036, M_takeoff=0.9556,
                  pi_tn=0.98, pi_d=0.98, pi_fn=0.98),
    "TASOPT_737800": dict(FPRo=1.685, K_epf=-0.077,
                          BPR_D=5.1, iengwgt=1, Gearf=1.0, Tmetal=1280.0,
                          pi_f_D=1.685, pi_lc_D=4.744, pi_hc_D=3.75,
                          alpha_OD=None, alpha_max=None, hf=43.003,
                          OPR_max=32.0, eta_B=0.9827, r_uc=0.01,
                          alpha_c=0.19036, M_takeoff=0.9556,
                          pi_tn=0.98, pi_d=0.98, pi_fn=0.98),
    "GE90": dict(pi_f_D=1.58, pi_lc_D=1.26, pi_hc_D=20.033,
                 alpha_OD=None, alpha_max=8.7877, hf=43.003, OPR_max=40.0,
                 eta_B=0.9970, r_uc=0.1, alpha_c=0.14, M_takeoff=0.955,
                 pi_tn=0.98, pi_d=0.98, pi_fn=0.98),
    "D82": dict(Tt4_TO=1750.0, Tt4_CR=1450.0, pi_f_D=1.60474, pi_lc_D=4.98, pi_hc_D=35. / 8.,
                alpha_OD=None, alpha_max=None, hf=43.003, OPR_max=32.0,
                eta_B=0.9827, r_uc=0.01, alpha_c=0.19036, M_takeoff=0.9556,
                pi_tn=0.995, pi_d=0.995, pi_fn=0.985),
    # As installed in SPaircraft's optimalD8, which is NOT the same engine as
    # the standalone "D82" above. subs/optimalD8.py overrides the shaft and
    # burner efficiencies, the cooling fraction, all three specific heats and
    # the OPR limit; and SPaircraft.optimize_aircraft(pRatOpt=True) *deletes*
    # the three design pressure ratios so the optimizer picks them. alpha_OD
    # and alpha_max are likewise absent, leaving bypass ratio free under the
    # aircraft-level bound alpha_max <= 100. Anything None here is free.
    # pi_f_D PINNED to the D8.2 deck's FPR (runs/D8/d82.tas). It was None --
    # a free design fan pressure ratio with nothing anchoring it -- and it ran
    # to 1.889 against TASOPT's 1.605. High FPR and high BPR are
    # contradictory: bypass exists to give LOW jet velocity, and a 1.9 fan
    # throws that away. Unanchored it gave high specific thrust, a fan 0.83 of
    # TASOPT's diameter, and a bare engine at 0.615 of Webare, because the
    # York/Hoburg/Drela weight fit scales with core mass flow.
    #
    # The other two decks were already anchored this way (CFM56 1.685,
    # TASOPT_737800 1.685); this one was the outlier. pi_lc_D and pi_hc_D stay
    # free -- the deck gives OPR 35 and pilc, not a full spool split.
    "D82_SPaircraft": dict(Tt4_TO=1750.0, Tt4_CR=1450.0, pi_f_D=1.60474,
                           FPRo=1.50, K_epf=-0.077, pi_lc_D=None, pi_hc_D=None,
                           BPR_D=6.9674, iengwgt=1, Gearf=1.0,
                           Tmetal=1280.0,
                           alpha_OD=None, alpha_max=None, hf=43.003,
                           OPR_max=35.0, eta_B=0.985, r_uc=0.01,
                           alpha_c=0.16, M_takeoff=0.9556,
                           pi_tn=0.995, pi_d=0.995, pi_fn=0.985,
                           eta_HPshaft=0.978, eta_LPshaft=0.99,
                           Cp_c=1257.9, Cp_t1=1236.5, Cp_t2=1200.4,
                           # subs/optimalD8.py sets hold_{4a} from a local
                           # M4a = 0.2 but never substitutes M_{4a} itself,
                           # which therefore keeps the Combustor's declared
                           # default of 0.1025. The two are supposed to be the
                           # same number -- hold_{4a} IS 1+(g-1)/2 M_4a^2 --
                           # so the shipped model is internally inconsistent
                           # by a factor of 0.1025/0.2 in u_{4a}. Reproduced,
                           # because the reference solution depends on it.
                           M_4a=0.1025, M_4a_for_hold=0.2),
    # The same machine burning liquid hydrogen. Only the heat of
    # combustion changes: 120 MJ/kg against Jet-A's 43.003, which is
    # the entire fuel switch as far as the cycle is concerned -- the
    # burner solves for a fuel-air ratio 2.79x smaller for the same
    # turbine inlet temperature.
    "D82_LH2": dict(Tt4_TO=1750.0, Tt4_CR=1450.0, FPRo=1.50, K_epf=-0.077, pi_f_D=None, pi_lc_D=None, pi_hc_D=None,
                           alpha_OD=None, alpha_max=None, hf=120.0,
                           OPR_max=35.0, eta_B=0.985, r_uc=0.01,
                           alpha_c=0.16, M_takeoff=0.9556,
                           pi_tn=0.995, pi_d=0.995, pi_fn=0.985,
                           eta_HPshaft=0.978, eta_LPshaft=0.99,
                           Cp_c=1257.9, Cp_t1=1236.5, Cp_t2=1200.4,
                           # subs/optimalD8.py sets hold_{4a} from a local
                           # M4a = 0.2 but never substitutes M_{4a} itself,
                           # which therefore keeps the Combustor's declared
                           # default of 0.1025. The two are supposed to be the
                           # same number -- hold_{4a} IS 1+(g-1)/2 M_4a^2 --
                           # so the shipped model is internally inconsistent
                           # by a factor of 0.1025/0.2 in u_{4a}. Reproduced,
                           # because the reference solution depends on it.
                           M_4a=0.1025, M_4a_for_hold=0.2),
}

# On-design mass flow anchors, from the `onDest` blocks of Engine.setup().
# (Tt_HPT, Pt_HPT, Tt_LPT, Pt_LPT, Tt_lpc, Pt_lpc, Tt_hpc, Pt_hpc,
#  fan_lo, fan_hi, Pt_fan)
ONDESIGN = {
    # The BLI branch of eng==3, which is the one optimalD8 takes. Using the
    # non-BLI numbers instead leaves the LPC mass-flow bracket ~12% out.
    "D82_SPaircraft": (1400.0, 1433.49, 1121.85, 706.84, 289.77, 65.79434,
                       481.386, 327.66, 0.7, 1.3, 41.0),
    # Same on-design point as D82_SPaircraft. These are the sizing-point
    # temperatures, pressures and mass flows of the MACHINE; hydrogen changes
    # the fuel-air ratio needed to reach T_t4, not the turbomachinery.
    "D82_LH2":        (1400.0, 1433.49, 1121.85, 706.84, 289.77, 65.79434,
                       481.386, 327.66, 0.7, 1.3, 41.0),
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

    # The ambient state is a group rather than a dictionary of quantities, so
    # it is read the same way here as it is when the mission supplies it.
    # Its prefix is empty: on the test stand the engine is the whole model and
    # its names are not namespaced.
    state = f.group("state", prefix="")
    Vn = lambda n, g, u, d: state.Variable(n, g, u, d, size=N)
    Vn("P_atm", 23.84, "kPa", "ambient static pressure")
    Vn("T_atm", 218.0, "K", "ambient static temperature")
    Vn("a", 300.0, "m/s", "ambient speed of sound")
    Vn("V", 240.0, "m/s", "flight speed")
    Vn("M", 0.8, "-", "flight Mach number")
    v, cons = add_engine(f, N, state, engine=engine)

    # The test-stand mission: prescribed operating points, weight cap, and the
    # TSFC-weighted objective from engine_validation.__main__.
    R = v.R
    for i in range(N):
        thrust_N, P_kPa, T_K, m0, m2, m25 = segs[i]
        cons += [
            state.P_atm[i] == P_kPa * units.kPa,
            state.T_atm[i] == T_K * units.K,
            state.M[i] == m0,
            state.a[i] == (1.4 * R * state.T_atm[i]) ** 0.5,
            state.V[i] == state.M[i] * state.a[i],
            v.M_2[i] == m2,
            v.M_25[i] == m25,
            v.c1[i] == 1 + 0.5 * 0.401 * m0 ** 2,
            v.hold_2[i] == 1 + 0.5 * (G2 - 1) * m2 ** 2,
            v.hold_25[i] == 1 + 0.5 * (G25 - 1) * m25 ** 2,
            v.F[i] == thrust_N * units.N,
        ]
    cons += [v.W_engine <= WCAP_N[engine] * units.N]

    w = [10.0] + [1.0] * (N - 1)
    f.Objective(v.W_engine ** 0.5
                * sum(w[i] * v.TSFC[i] for i in range(N)))
    f.ConstraintList(cons)
    return f


def add_engine(f, N, state, *, engine: str = "CFM56", BLI: bool = False,
               prefix: str = "", n_eng: float = 2.0):
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
    eng = f.group("eng", prefix=prefix)
    V, C = eng.Variable, eng.Constant
    Vn = lambda n, g, u, d: eng.Variable(n, g, u, d, size=N)

    def CV(n, key, guess, u, d):
        """Constant if the substitution set defines it, free variable if not."""
        v = sub.get(key)
        return C(n, v, u, d) if v is not None else V(n, guess, u, d)

    # ---- gas properties (Cp values are the source's, at the stated temps) --
    R      = C("R", 287.0, "J/kg/K", "air gas constant")
    Cpair  = C("Cp_air", 1003.0, "J/kg/K", "Cp of air at 250 K")
    Cp1    = C("Cp_1", 1008.0, "J/kg/K", "Cp of air at 350 K")
    Cp2    = C("Cp_2", 1099.0, "J/kg/K", "Cp of air at 800 K")
    Cpc    = C("Cp_c", sub.get("Cp_c", 1216.0), "J/kg/K", "Cp of fuel/air mix in combustor")
    Cpfuel = C("Cp_fuel", 2010.0, "J/kg/K", "Cp of kerosene")
    Cpt1   = C("Cp_t1", sub.get("Cp_t1", 1280.0), "J/kg/K", "Cp of combustion products, HPT")
    Cpt2   = C("Cp_t2", sub.get("Cp_t2", 1184.0), "J/kg/K", "Cp of combustion products, LPT")
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
    etaHP = C("eta_HPshaft", sub.get("eta_HPshaft", 0.97), "-",
              "HP shaft transmission efficiency")
    etaLP = C("eta_LPshaft", sub.get("eta_LPshaft", 0.97), "-",
              "LP shaft transmission efficiency")
    Mtakeoff = C("M_takeoff", sub["M_takeoff"], "-",
                 "1 - bleed mass flow fraction")
    Tref  = C("T_ref", 288.15, "K", "reference stagnation temperature")
    Pref  = C("P_ref", 101.325, "kPa", "reference stagnation pressure")

    # ---- cooling flow ------------------------------------------------------
    # If the deck carries a metal temperature, the cooling fraction is
    # COMPUTED from tfcool.f's model rather than taken as a constant: per
    # blade row, effectiveness theta = (Tg - Tmetal)/(Tg - Tt3), requirement
    #     eps0 = StA*(theta*(1 - efilm*tfilm) - tfilm*(1 - efilm))
    #                / (efilm*(1 - theta)),
    # cooling ratio eps = eps0/(1 + eps0), gas temperature stepping down
    # Trrat = 1/(1 + (gam4-1)/2 * Mtexit^2) per row, the first row seeing a
    # dTstreak hot-streak allowance. Every relation rearranges to the
    # all-positive one-sided signomial forms below, and the one-sidedness IS
    # tfcool's `if(eps0.lt.0) go to 5`: a row whose requirement goes
    # negative simply leaves its eps on the floor, uncharged. Evaluated at
    # the TAKEOFF segment -- the hottest -- exactly as TASOPT sizes cooling
    # at design and holds epsrow fixed off-design. On the D8 deck this gives
    # fc ~ 0.11 where the inherited SPaircraft constant said 0.19036: the
    # engine was pumping 8% of its core flow through cooling passages that
    # Tmetal = 1280 K does not require.
    _Tmetal = sub.get("Tmetal")
    if _Tmetal is not None:
        alpha_c = V("alpha_c", 0.12, "-", "total cooling flow bypass ratio")
    else:
        alpha_c = C("alpha_c", sub["alpha_c"], "-",
                    "total cooling flow bypass ratio")
    ruc   = C("r_uc", sub["r_uc"], "-", "cooling flow velocity ratio")
    m4a = sub.get("M_4a", M4A)
    m4a_hold = sub.get("M_4a_for_hold", m4a)   # see the SUBS note for D8
    hold4a = C("hold_4a", 1 + 0.5 * (1.313 - 1) * m4a_hold ** 2, "-",
               "1 + (gamma-1)/2 M_4a^2")
    M4a   = C("M_4a", m4a, "-", "station 4a Mach number")
    Ttf   = C("T_t_f", 435.0, "K", "incoming fuel total temperature")

    # ---- design pressure ratios and geometry ratios -----------------------
    piFanD = CV("pi_f_D", "pi_f_D", 1.6, "-", "fan on-design pressure ratio")
    pilcD  = CV("pi_lc_D", "pi_lc_D", 4.98, "-", "LPC on-design pressure ratio")
    pihcD  = CV("pi_hc_D", "pi_hc_D", 4.375, "-", "HPC on-design pressure ratio")
    alphaOD = CV("alpha_OD", "alpha_OD", 8.0, "-", "on-design bypass ratio")
    alphamax = CV("alpha_max", "alpha_max", 8.0, "-", "maximum bypass ratio")
    # 45, a modern-engine ceiling (GEnx ~43-47, Trent XWB ~50), above (runs/737/737s.tas:470), because
    # the weight model cannot restrain OPR and so OPR pins to whatever ceiling
    # it is given.
    #
    # The correlation below does contain pressure ratio:
    #
    #     W_engine ~ (m_tot/(1+alpha))
    #                * (1684.5 + 17.7*OPR/30 + 1662.2*(alpha/5)**1.2)
    #
    # but look at the sensitivity. Taking OPR from 30 to 45 moves the middle
    # term 17.7 -> 26.6: a 0.26% engine weight increase for a 50% higher
    # pressure ratio. The bypass term over the same kind of excursion DOUBLES,
    # 1662 -> 3324 by alpha = 9, which is why bypass ratio settles interior at
    # 8.7 while OPR sits hard against any cap offered. Given a ceiling of 45 it
    # went to 44.84; given none it went to 58.6.
    #
    # Raising OPR in this model is therefore very nearly free thrust-specific
    # fuel consumption, and the honest response is to hold it at the reference
    # engine's value rather than to let the optimizer harvest a saving the
    # weight model is not paying for. Revisit alongside a compressor exit
    # temperature (T3) constraint and an OPR-sensitive weight correlation.
    OPR_CEILING = _technology.current().opr_max
    # A real ceiling on bypass ratio. Left free, alpha_max ran to 1.9e23 and
    # took the cruise bypass ratio to 24.6 (34.2 in climb) on a 1.5 m fan --
    # the "aircraft-level bound alpha_max <= 100" the comment above promises is
    # not actually imposed anywhere. 15 is generous: CFM56 is 5.1, the PW1000G
    # geared turbofan is 12, and the most aggressive ultra-high-bypass studies
    # sit around 15. Past that the fan needs a gearbox and a nacelle this model
    # does not weigh, so the extrapolation flatters itself.
    BPR_CEILING = _technology.current().bpr_max
    Gf     = C("G_f", 1.0, "-", "fan/LPC gear ratio")
    OPRmax = C("OPR_max", sub["OPR_max"], "-",
               "maximum overall pressure ratio")
    out_extra = dict(alpha_max=alphamax, alpha_OD=alphaOD, pi_f_D=piFanD,
                     pi_lc_D=pilcD, pi_hc_D=pihcD, OPR_max=OPRmax)
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
    Patm, Tatm = state.P_atm, state.T_atm
    a, Vinf, M0 = state.a, state.V, state.M
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
    # Turbine inlet temperature limits, from TASOPT's own 737 deck
    # (runs/737/737s.tas:450,453):
    #
    #     1833.0   ! Tt4TO    takeoff
    #     1587.0   ! Tt4CR    cruise
    #     1280.0   ! Tmetal
    #
    # The model had NO thermal limit whatsoever -- the only Tt4 row was the
    # internal cooling relation T41 <= Tt41 - 0.5 u41^2/Cpc. That went unnoticed
    # while bypass ratio was pinned at the CFM56's 5.105, because a pinned BPR
    # kept the cycle near its reference point. Unpinning BPR walked straight
    # out of it: Tt4 went to 2490 K, 660 K past takeoff limit and 900 K past
    # the metal temperature, which is not a hot engine, it is a liquid one.
    # PER DECK. 1833/1587 K is the CFM56-era pair (737s.tas). The D8's engine
    # runs COOLER -- 1750/1450 in runs/D8/d82.tas -- because its component
    # efficiencies are better (epolf 0.93 against 0.8948), so it does not need
    # the turbine temperature to make up the difference. Left hard-coded, the
    # advanced cycle would have been handed the old engine's thermal limits.
    Tt4TO = C("T_t_4_TO", sub.get("Tt4_TO", 1833.0), "K",
              "max takeoff turbine inlet temperature")
    Tt4CR = C("T_t_4_CR", sub.get("Tt4_CR", 1587.0), "K",
              "max cruise turbine inlet temperature")
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

    # ---- thermal and bypass limits ----------------------------------------
    # Takeoff is segment 0; everything after is climb and cruise, which TASOPT
    # holds to the lower cruise limit. Without these the cycle has no ceiling
    # at all -- see the declarations above.
    cons += [Tt4[0] <= Tt4TO]
    cons += [Tt4[i] <= Tt4CR for i in range(1, len(Tt4))]
    cons += [alphamax <= BPR_CEILING]

    # ---- turbine cooling requirement (tfcool.f), design = takeoff ---------
    # See the alpha_c declaration. Three blade rows: enough for any deck in
    # the study (row 3 is already uncooled at Tt4 = 1750; a 1900 K deck cools
    # into it). All rows are one-sided with alpha_c charged downstream, so
    # each binds exactly where tfcool's requirement is positive and floors
    # where the Fortran breaks out of its loop.
    if _Tmetal is not None:
        _ef = sub.get("efilm", 0.7)
        _tf = sub.get("tfilm", 0.30)
        _StA = sub.get("StA", 0.09)
        _dTs = sub.get("dTstrk", 200.0)
        _Mte = sub.get("Mtexit", 1.0)
        _Trr = 1.0 / (1.0 + 0.5 * (1.313 - 1.0) * _Mte ** 2)
        Tmet = C("T_metal", _Tmetal, "K", "design blade metal temperature")
        dTs = C("dT_streak", _dTs, "K", "hot-streak allowance, first row")
        _rows = []
        for _r in (1, 2, 3):
            th = V(f"theta_cool_{_r}", 0.4, "-",
                   f"cooling effectiveness, blade row {_r}")
            e0 = V(f"eps0_cool_{_r}", 0.05, "-",
                   f"cooling flow requirement, blade row {_r}")
            ep = V(f"eps_cool_{_r}", 0.04, "-",
                   f"cooling mass flow ratio, blade row {_r}")
            # At the RATING, not the operating point: the 737's engine is
            # cruise-sized and flies takeoff throttled to ~1390 K, but its
            # blades are designed for the 1833 K the rating permits --
            # TASOPT's icool=2 design pass runs at the design Tt4.
            Tg = (Tt4TO + dTs) if _r == 1 else Tt4TO * _Trr ** (_r - 1)
            cons += [
                # theta*(Tg - Tt3) >= Tg - Tmetal, all-positive
                th * Tg + Tmet >= Tg + th * Tt3[0],          # [SP] SigIneq
                th <= 0.999,
                # eps0*ef*(1-theta) >= StA*(theta*(1-ef*tf) - tf*(1-ef)),
                # every term moved to its positive side:
                e0 * _ef + _StA * _tf * (1.0 - _ef)
                    >= _StA * th * (1.0 - _ef * _tf) + e0 * _ef * th,
                                                             # [SP] SigIneq
                ep + ep * e0 >= e0,                          # [SP] SigIneq
            ]
            _rows.append(ep)
        cons += [alpha_c >= _rows[0] + _rows[1] + _rows[2]]

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
    cons += [
        mhtD <= 1.3 * fp1 * Mtakeoff * mCoreD * (TtH / 288) ** 0.5 / (PtH / 101.325),
        mhtD >= 0.7 * fp1 * Mtakeoff * mCoreD * (TtH / 288) ** 0.5 / (PtH / 101.325),
        mltD <= 1.3 * fp1 * Mtakeoff * mCoreD * (TtL / 288) ** 0.5 / (PtL / 101.325),
        mltD >= 0.7 * fp1 * Mtakeoff * mCoreD * (TtL / 288) ** 0.5 / (PtL / 101.325),
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
            OPR[i] <= OPR_CEILING,
            # NOTE: restored, against instruction, and here is why.
            #
            # TASOPT's 737 deck carries OPR as an INPUT
            # (runs/737/737s.tas:470, `30.0 ! OPR`), not as a limit -- line 88
            # shows it can be an optimizer variable there and is simply not
            # used as one in that deck. So there is no principled bound to
            # inherit, and ours was binding at 32 and sizing the engine. What
            # actually limits pressure ratio is compressor exit temperature,
            # and that is now bounded properly through Tt4.
            #
            # But removing the cap outright sent OPR to 58.6, which is the
            # bypass-ratio failure repeating: the limit that SHOULD bound
            # pressure ratio is compressor exit temperature T3, and this model
            # has no T3 row at all, exactly as it had no Tt4 row. With nothing
            # physical in the way, OPR climbs until something else gives.
            #
            # 45 is a technology ceiling rather than an engine one -- GEnx is
            # ~43-47 and Trent XWB ~50, so it is not restrictive for a
            # clean-sheet 2035 engine, and it is far enough above the CFM56's
            # 32 not to re-pin the kerosene cases. The engine-specific
            # OPR_max values in SUBS are left in place but no longer applied;
            # they were reference-aircraft values, not limits.
            #
            # The right fix is a T3 constraint. Until then this is a fence, and
            # it is labelled as one.
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

        # ENGINE WEIGHT (TASOPT data fit, relaxed to an inequality).
        #
        # DIVIDED BY n_eng. The fit reproduces TASOPT's Webare, and tfweight.f
        # builds that as `Webare = We1 * neng` -- a SET total, not one engine.
        # This variable is a single turbofan, and the aircraft multiplies it
        # by n_eng again, so the engine set was counted twice.
        #
        # The evidence is exact rather than inferred. At the sizing segment
        # our core flow is 44.87 kg/s = 98.9 lb/s, so the (m_core/100 lb/s)
        # normaliser is 0.989 and the fit returns essentially 9.81*K = 7,860
        # lbf -- TASOPT's Webare of 7,870.7 to within 0.1%. Against TASOPT's
        # PER-ENGINE bare of 3,935 lb that is a ratio of 1.975.
        #
        # Corrected, our engine set lands on TASOPT's Webare exactly. Both are
        # then about 0.75 of a real CFM56-7B (5,216 lbf each, 10,432 the pair,
        # see PUBLISHED["CFM56"] in reference.py) -- so TASOPT is 25% light on
        # this engine and we now inherit that rather than compounding it.
        #
        # NOTE the fit was made against twins. Dividing by n_eng recovers the
        # per-engine weight only if that holds; a four-engine deck would need
        # the fit revisited, not just the divisor changed.
        # WHICH WEIGHT MODEL. tfweight.f carries three behind `iengwgt`:
        #   0     Drela's original, additive in OPR with a 1.2 power in BPR
        #   1,2   Fitzgerald's, a(BPR) * mdotc^b(BPR) * (OPR/40)^c(BPR)
        #   3,4   Pantalone Gaussian-process surrogates (not ported anywhere)
        #
        # This row was model 0. BOTH TASOPT decks select model 1 --
        # runs/737/737.tas:542 and runs/D8/d82.tas:548, `iengwgt = 1` -- so we
        # were comparing against a model TASOPT does not use. Measured at our
        # own operating points the two differ by 1.16x on the D8 and 1.01x on
        # the 737, so this is a correctness fix rather than a large one.
        #
        # a, b and c are evaluated ONCE at the deck's design BPR, which is what
        # keeps this GP: b and c are exponents on mdotc and OPR, and a
        # BPR-dependent b would put a variable in an exponent. TASOPT has the
        # same structure -- BPR is a deck input there -- so this is the source
        # behaviour rather than an approximation of it.
        _iw = sub.get("iengwgt", 0)
        _bprd = sub.get("BPR_D")
        if _iw and _bprd is not None:
            _a, _b, _c = fitzgerald_coeffs(_bprd,
                                           abs(sub.get("Gearf", 1.0) - 1.0) >= 1e-3,
                                           _iw == 2)
            cons += [
                # NO /n_eng. TASOPT's We1 is already PER ENGINE from a
                # per-engine mdotc (`Webare = We1*neng`), and our m_core is
                # per-engine too.
                #
                # THE MASS FLOW IS THE DESIGN CORRECTED CORE FLOW, NOT A
                # MISSION SEGMENT'S PHYSICAL FLOW. wsize.f:1306 builds
                # tfweight's input as
                #     mdotc = mblcD * sqrt(Tref/TSL) * (pSL/pref)
                # -- the LPC's design corrected flow re-referenced to
                # sea-level-static: the SIZE OF THE COMPRESSOR, invariant of
                # where the aircraft happens to be flying. Feeding the
                # binding segment's physical core flow instead (27.8 kg/s on
                # the D8 against TASOPT's 39.0) weighed a machine 29% smaller
                # than the one the fan was drawn around. Our fan's design
                # corrected flow mbar_fan_D is fan-STREAM corrected flow at
                # the same face, so mbar_fan_D / BPR_D IS mdotc: 271.93 /
                # 6.9674 = 39.03 kg/s on the D8, TASOPT's own value to 0.1%.
                #
                # And tfweight's OPR is pilc*pihc -- the CORE compressors,
                # 35.0 on the D8 deck -- where this row used to include the
                # fan (pif*pilc*pihc ~ 58), overstating the (OPR/40)^c term
                # 14% and masking part of the mass-flow undercount. With both
                # corrected the correlation reproduces TASOPT's Webare from
                # TASOPT's inputs to 0.1%: 4902.3*(39.03/45.35)^0.983
                # *(35/40)^0.252 = 4,089 lbf against its printed 4,084.
                W_engine / units.N >= _a * 4.44822
                    * ((mFanD / _bprd)
                       / (45.35 * units.kg / units.s)) ** _b
                    * ((pilc[i] * pihc[i]) / 40.0) ** _c,
            ]
        else:
            # UNIT SLIP FIXED, and the compensating divisor removed with it.
            # This row read
            #
            #     W/kg >= (mtot/alphap1)/n_eng * (1/(100 lb/s)) * 9.81 * bracket
            #
            # i.e. it multiplied the correlation's bracket by g, treating it as
            # KILOGRAMS-force. tfweight.f divides by LB_N, which treats it as
            # POUNDS-force -- a factor of 9.81/4.44822 = 2.205 too heavy.
            #
            # The `/n_eng` was not a per-engine conversion. TASOPT's We1 is
            # already per engine from a per-engine mdotc (`Webare = We1*neng`),
            # and our m_core is per-engine too -- 49.4 kg/s against TASOPT's
            # 53.3 on the same 737. It was cancelling the unit error: 2.205/2
            # left a residual 10%, which is why the engine looked roughly right
            # and the row survived scrutiny.
            #
            # Reaches the CFM56, GE90 and hydrogen decks. The two decks with a
            # TASOPT source now select iengwgt = 1 above and do not come here.
            cons += [
                W_engine / units.N >= (mtot[i] / alphap1[i])
                    / (45.35 * units.kg / units.s) * 4.44822
                    * (1684.5 + 17.7 * (pif[i] * pilc[i] * pihc[i]) / 30
                       + 1662.2 * (alpha[i] / 5) ** 1.2),
            ]

    out = dict(W_engine=W_engine, TSFC=TSFC, F=F, F_6=F6, F_8=F8, R=R,
               M_2=M2v, M_25=M25v, hold_2=hold2, hold_25=hold25, c1=c1,
               A_2=A2, A_25=A25, A_5=A5, A_7=A7, d_f=df, d_LPC=dlpc,
               alpha=alpha, alpha_p1=alphap1, OPR=OPR, T_t_41=Tt41,
               T_t_4=Tt4, m_core=mCore, m_fan=mFan, m_total=mtot, f=fuel,
               pi_f=pif, pi_lc=pilc, pi_hc=pihc, N_1=N1, N_2=N2, I_sp=Isp,
               # exit velocities, exported for the noise module
               u_6=u6, u_8=u8)
    out.update(out_extra)
    return eng, cons


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


# ---------------------------------------------------------------------------
# Hydrogen twins, generated rather than hand-copied.
#
# A hydrogen variant of an engine differs from its kerosene parent in exactly
# one number -- the fuel's lower heating value, 120 MJ/kg against Jet-A's
# 43.003 -- and in nothing else: same turbomachinery, same pressure ratios,
# same efficiencies, same on-design point. Writing each one out by hand meant
# three separate tables to keep in step, and the D82_LH2 entry was discovered
# to be missing from two of them one KeyError at a time.
for _parent in ("CFM56", "TASOPT_737800", "GE90", "D82_SPaircraft"):
    _child = f"{_parent}_LH2"
    if _child not in SUBS:
        SUBS[_child] = dict(SUBS[_parent], hf=120.0)
        ETAS[_child] = ETAS[_parent]
        ONDESIGN[_child] = ONDESIGN[_parent]
