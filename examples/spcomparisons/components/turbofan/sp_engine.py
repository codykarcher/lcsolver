"""The SP-native RUBBER engine, packaged for the aircraft.

Milestone 4 of HANDOFF_ENGINE.md: the validated multipoint cycle of
``sp_cycle.py`` wearing the aircraft's engine interface. The user sets a
TECHNOLOGY LEVEL (component efficiencies, metal temperature, OPR bound) and
the Tt4 ratings; the cycle variables -- FPR, LPC PR, HPC PR, BPR, corrected
flows -- are FREE design variables that settle inside the aircraft
optimization. No deck pins.

Structure
---------
One DESIGN point at the first cruise segment (TASOPT's convention: the
engine is sized so cruise at the cruise rating produces exactly the thrust
the aircraft needs there) plus one off-design point per remaining mission
segment, each running the full station chain -- equilibrium burner
included -- against the scaled-map surfaces, with the thrust the mission
demands and the Tt4 rating as a one-sided cap (takeoff rating on segment 0,
cruise rating elsewhere; required thrust pushes T4 up, the cap holds it
down, so the row binds exactly when the rating does).

Kept from the existing engine model, per the handoff:

* Fitzgerald/TASOPT weight (tfweight.f iengwgt=1) on the DESIGN CORRECTED
  CORE FLOW -- here that is simply the LPC's design corrected flow, which
  the map rows already carry -- and the CORE pressure ratio pilc*pihc.
* tfcool.f's three-row cooling requirement at the takeoff rating, as a
  FEASIBILITY row against the cycle's fixed cooling budget: the bleed
  fractions stay at the validated truth-cycle values, and the tfcool rows
  say whether the tech level's metal temperature supports the Tt4 rating
  with that budget. OPR raises Tt3 and tightens it -- the coupling that
  matters for the free-cycle optimum.
* Nozzle-area, fan-diameter and noise-interface exports (u_6, u_8, A_5,
  A_7, m_fan), and the aircraft-facing per-segment F / TSFC / M_2 / hold
  variables the airframe rows pin.

Units: the cycle is dimensionless-SI inside (sp_cycle convention); this
module owns the unit-ful interface variables and the conversion rows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pyomo.environ as pyo
from pyomo.environ import units

from . import sp_cycle as SC
from . import sp_maps as MAPS


@dataclass(frozen=True)
class SPTech:
    """A technology level: everything the user states about the engine."""
    name: str
    # design ADIABATIC efficiencies at the reference pressure ratios (the
    # anchors' values; polytropic-exact compression is a noted refinement)
    eff_fan0: float
    eff_lpc: float
    eff_hpc: float
    eff_hpt: float
    eff_lpt: float
    #: fan efficiency lapse d(eff)/d(FPR) about FPRo -- the coupling whose
    #: absence once ran a freed FPR to 1.889 against TASOPT's 1.605.
    K_eff_fan: float
    FPRo: float
    Tt4_TO_K: float
    Tt4_CR_K: float
    T_metal_K: float
    OPR_max: float
    #: Fitzgerald coefficients are evaluated at a REFERENCE bypass ratio
    #: (exponents cannot carry a variable); TASOPT has the same structure.
    BPR_ref: float
    geared: bool = False
    advanced: bool = False
    #: map-scalar spool speeds, rpm (arbitrary up to the s_Nc scaling)
    LP_Nmech: float = 4666.1
    HP_Nmech: float = 14705.7
    #: hub-to-tip ratios for the diameter rows
    HTR_fan: float = 0.30
    HTR_lpc: float = 0.60
    #: tfcool.f film/technology parameters (TASOPT defaults)
    efilm: float = 0.7
    tfilm: float = 0.30
    StA: float = 0.09
    dT_streak_K: float = 200.0
    M_t_exit: float = 1.0
    #: shaft power extraction, W
    HPX_W: float = 186425.0
    cust_frac_W: float = 0.0445
    #: guesses for the free design cycle (starting point, not pins)
    FPR_g: float = 1.65
    LPC_PR_g: float = 1.9
    HPC_PR_g: float = 9.5
    BPR_g: float = 5.2


TECHS = {
    # CFM56-era: the validated cfm56_class anchor's efficiencies and the
    # TASOPT 737.tas ratings/metal temperature.
    "cfm56_era": SPTech(
        name="cfm56_era",
        eff_fan0=0.8948, eff_lpc=0.9243, eff_hpc=0.8707,
        eff_hpt=0.8888, eff_lpt=0.8996,
        K_eff_fan=0.077, FPRo=1.685,
        Tt4_TO_K=1833.0, Tt4_CR_K=1587.0, T_metal_K=1280.0,
        OPR_max=32.0, BPR_ref=5.1,
        FPR_g=1.65, LPC_PR_g=1.9, HPC_PR_g=9.5, BPR_g=5.2),
    # GEnx-era: the genx_class anchor's efficiencies (ETAS GE90 row
    # converted poly->adiabatic), 1900 K takeoff rating, modern OPR bound.
    "genx_era": SPTech(
        name="genx_era",
        eff_fan0=0.920, eff_lpc=0.905, eff_hpc=0.905,
        eff_hpt=0.932, eff_lpt=0.945,
        K_eff_fan=0.077, FPRo=1.55,
        Tt4_TO_K=1900.0, Tt4_CR_K=1620.0, T_metal_K=1400.0,
        OPR_max=47.0, BPR_ref=9.0,
        LP_Nmech=2400.0, HP_Nmech=10600.0,
        HPX_W=372850.0, cust_frac_W=0.001,
        FPR_g=1.55, LPC_PR_g=1.9, HPC_PR_g=15.0, BPR_g=9.0),
}


class _EnginePins:
    """Duck-typed CyclePins whose PR/eff/BPR entries are pyomo VARIABLES.

    sp_cycle only reads attributes; everything a guess or a scale factor
    must stay a float (P0_Pa, Fn_N, T4_K, spool speeds), everything a
    design pin may be a variable.
    """
    def __init__(self, tech: SPTech, FPR, LPC_PR, HPC_PR, BPR, eff_fan,
                 P0_g, Fn_g):
        d = SC.CyclePins.__dataclass_fields__  # defaults for the rest
        base = SC.CyclePins(
            name=f"rubber_{tech.name}", T0_K=218.8, P0_Pa=P0_g, MN=0.785,
            Fn_N=Fn_g, T4_K=tech.Tt4_CR_K,
            FPR=1.0, LPC_PR=1.0, HPC_PR=1.0, BPR=1.0,
            eff_fan=1.0, eff_lpc=tech.eff_lpc, eff_hpc=tech.eff_hpc,
            eff_hpt=tech.eff_hpt, eff_lpt=tech.eff_lpt,
            LP_Nmech=tech.LP_Nmech, HP_Nmech=tech.HP_Nmech,
            HPX_W=tech.HPX_W,
            cust=SC.Bleed(tech.cust_frac_W, 0.5, 0.5))
        for f_ in d:
            setattr(self, f_, getattr(base, f_))
        self.FPR, self.LPC_PR, self.HPC_PR = FPR, LPC_PR, HPC_PR
        self.BPR, self.eff_fan = BPR, eff_fan


def add_engine_sp(f, N, state, *, tech: SPTech, prefix="Eng_", n_eng=2.0,
                  Nclimb=3, seg_choked=None, debug_float_conds=None,
                  debug_skip=frozenset()):
    """Add the rubber engine. Returns ``(group, cons)`` with the same
    attribute surface the aircraft reads from the deck engine."""
    eng = f.group("eng_sp", prefix=prefix)
    Vu = eng.Variable          # unit-ful interface variables
    Cu = eng.Constant
    Vn = lambda n, g, u, d: eng.Variable(n, g, u, d, size=N)
    cons = []

    # dimensionless cycle variables share the group namespace, prefixed
    # cyc_ so they cannot collide with the unit-ful interface names.
    # Bounds become ROWS -- the aircraft build path drops variable bounds
    # (see sp_cycle._point).
    def V(n, gs, d, bounds=None):
        v = eng.Variable(f"cyc_{n}", gs, "-", d)
        if bounds is not None:
            lo, hi = bounds
            if lo is not None and lo > 0:
                cons.append(v >= lo)
            if hi is not None:
                cons.append(v <= hi)
        return v
    i_des = Nclimb             # first cruise segment sizes the engine

    # ---- free design cycle variables --------------------------------------
    FPR = V("pi_f_D", tech.FPR_g, "fan design pressure ratio, FREE",
            bounds=(1.2, 2.2))
    PIlc = V("pi_lc_D", tech.LPC_PR_g, "LPC design pressure ratio, FREE",
             bounds=(1.2, 4.0))
    PIhc = V("pi_hc_D", tech.HPC_PR_g, "HPC design pressure ratio, FREE",
             bounds=(4.0, 25.0))
    BPRD = V("BPR_D", tech.BPR_g, "design bypass ratio, FREE",
             bounds=(2.0, 14.0))
    eff_fan = V("eff_fan_D", tech.eff_fan0, "fan design adiabatic eff")
    cons += [
        # fan efficiency lapse about FPRo -- linearized K_epf. Mixed signs:
        # written with every term positive on its side.
        eff_fan + tech.K_eff_fan * FPR
            == tech.eff_fan0 + tech.K_eff_fan * tech.FPRo,  # [SP] SigEq
        # technology OPR bound (pif*pilc*pihc, the cap the deck engine used)
        FPR * PIlc * PIhc <= tech.OPR_max,
    ]

    # ---- per-segment conditions -------------------------------------------
    _bridge = {}

    def cond_for(i, mode):
        T0g = float(pyo.value(state.T_atm[i]))
        # state.P_atm is declared in kPa, but a SEEDED build carries
        # SI-corrected magnitudes (Pa) in the same variable -- 75 MPa fan
        # guesses came from assuming kPa. Detect by magnitude: no
        # atmosphere is below 2 kPa or above 200 kPa.
        P0g_raw = float(pyo.value(state.P_atm[i]))
        P0g = P0g_raw * 1e3 if P0g_raw < 2000.0 else P0g_raw
        # FS_V is in KNOTS (house rule 8; declared "kts" and NOT rescaled
        # by the unit corrector, unlike kPa->Pa which is). Feeding the
        # knots magnitude as m/s put Tt0 at 321 K and was the entire
        # "spurious attractor" the coupled solves kept finding.
        KT2MS = 0.514444
        u0g = float(pyo.value(state.V[i])) * KT2MS
        MNg = float(pyo.value(state.M[i]))
        ch_core, ch_byp = (seg_choked[i] if seg_choked is not None
                           else ((True, True) if i >= Nclimb
                                 else (False, True)))
        if debug_float_conds is not None:
            # bisection hook: pin conditions as plain numbers, exactly the
            # validated standalone structure
            fc = debug_float_conds[i]
            c = dict(T0=fc['T0'], P0=fc['P0'], V0=fc['V'], MN=fc['M'],
                     mode=mode, choked_core=ch_core, choked_byp=ch_byp)
            c['u0'] = c['V0']
            return c
        # BRIDGE VARIABLES: the cycle's polynomial/fit rows must touch
        # only dimensionless variables (the unit-carrying expressions
        # produced nan gradients in the detector -- house rule 3's
        # mistranslation, measured in the mini harness). Three trivial
        # conversion rows per segment carry all the units. Memoized per
        # segment: the design anchor and the design-conditions OD point
        # share one bridge (building it twice is a duplicate component).
        if i in _bridge:
            T0v, P0v, u0v = _bridge[i]
        else:
            T0v = V(f"T0_{i}", T0g, f"segment {i} ambient T bridge, K")
            P0v = V(f"P0_{i}", P0g, f"segment {i} ambient P bridge, Pa")
            u0v = V(f"u0_{i}", u0g, f"segment {i} airspeed bridge, m/s")
            cons.extend([
                T0v * units.K == state.T_atm[i],
                P0v * units.Pa == state.P_atm[i],
                u0v * (units.m / units.s) == state.V[i] * KT2MS
                    * (units.m / units.s) / units.kts,
            ])
            _bridge[i] = (T0v, P0v, u0v)
        c = dict(T0=T0v, P0=P0v, V0=u0v,
                 MN=MNg,
                 T0_g=T0g, P0_g=P0g, u0_g=u0g, MN_g=MNg,
                 mode=mode,
                 choked_core=ch_core, choked_byp=ch_byp)
        c['u0'] = c['V0']
        return c

    # thrust the aircraft demands, per segment, per engine (interface var)
    F = Vn("F", 24000.0, "N", "total thrust")
    Fn_des_g = 24000.0
    # magnitude heuristic as in cond_for: a SEEDED aircraft build carries
    # SI-corrected Pa magnitudes in the kPa-declared state variable, and
    # blindly multiplying by 1e3 put pins.P0_Pa at 2.4e7 -- every sc-scaled
    # guess and guard band in the cycle then sat 1000x low, phase 1 crushed
    # the seeded fuel flows against Wf <= 0.15 ceilings, and every coupled
    # solve died inside the engine block (measured via the GP seed audit).
    _P0raw = float(pyo.value(state.P_atm[i_des]))
    pins = _EnginePins(tech, FPR, PIlc, PIhc, BPRD, eff_fan,
                       P0_g=_P0raw * 1e3 if _P0raw < 2000.0 else _P0raw,
                       Fn_g=Fn_des_g)

    out_by_tag = {}
    seg_out = [None] * N

    # design point: a pure SIZING ANCHOR at cruise conditions, T4 PINNED at
    # the cruise rating, thrust FREE. The earlier thrust-driven design
    # (Fn == F with T4 free under the rating cap) left two dofs (T4, W)
    # against one row whose optimum sits exactly where the cap goes active
    # -- a degenerate corner at which the SIA's conservative sub-problem
    # goes infeasible within 2-8 iterations in EVERY free-design run
    # (measured; with T4 pinned the same model converges in 4). Every
    # mission segment, including the design-conditions one, is now an
    # ordinary off-design point with its own thrust row and rating cap, so
    # nothing is lost: the SIZE still settles where the binding climb
    # rating closes (TASOPT's worst-case sizing, as optimization
    # pressure), and segment performance comes from part-power off-design
    # physics rather than from a point running flat-out at its rating.
    # T4 == Tt4_CR AND Fn == F[i_des]: square in (T4, W), no free-T4-at-
    # active-cap corner (the degeneracy that killed every earlier free
    # solve) and no flat W direction (the anchor's size is determined by
    # the thrust it must make at its rating, measured: the free-thrust
    # anchor drifted W 127.5 -> 140.8 along a zero-gradient valley and
    # ground 105 iterations before an internal solver error).
    cond = cond_for(i_des, 'T4')
    cond['T4'] = tech.Tt4_CR_K
    cond['T4_cap'] = None
    cond['V0'] = cond['u0']
    cond['F'] = F[i_des] / units.N
    des = SC._point(V, cons, "", cond, pins, None, out_by_tag)
    out_by_tag[""] = des

    # off-design points: EVERY segment, thrust-driven, rating-capped
    for i in range(N):
        cond = cond_for(i, 'F')
        cond['F'] = F[i] / units.N
        cond['T4'] = None
        # CLIMB carries the takeoff/climb rating, cruise the cruise rating:
        # a cruise-sized engine meets ~98% of its design CORRECTED thrust
        # in mid-climb but at hotter Tt0, which needs Tt4 above the cruise
        # cap (measured: 1654 K required vs 1587) -- TASOPT's own climb Tt4
        # profile ramps from Tt4TO for the same reason.
        # Climb keeps the takeoff rating as a genuine cap (inactive with
        # margin at the solution, ~1654 K vs 1833). Cruise segments get NO
        # cap: the anchor row (rating thrust == cruise demand) already
        # sizes the machine so cruise T4 lands at/below the rating through
        # the thrust rows -- a cruise cap would sit EXACTLY active at zero
        # margin on the design-conditions segment, the degenerate corner
        # this restructure exists to remove.
        cond['T4_cap'] = tech.Tt4_TO_K if i < Nclimb else None
        cond['PC'] = cond['pc_of'] = None
        tag = f"s{i}_"
        seg_out[i] = SC._point(V, cons, tag, cond, pins, des['shared'],
                               out_by_tag)
        out_by_tag[tag] = seg_out[i]

    if "iface" in debug_skip:
        return eng, cons
    # ---- interface: per-segment -------------------------------------------
    TSFC = Vn("TSFC", 0.65, "1/hr", "thrust specific fuel consumption")
    u6 = Vn("u_6", 400.0, "m/s", "core exhaust velocity")
    u8 = Vn("u_8", 290.0, "m/s", "fan exhaust velocity")
    mFan = Vn("m_fan", 150.0, "kg/s", "fan (total) mass flow")
    Tt4x = Vn("T_t_4", 1500.0, "K", "burner exit stagnation temperature")
    # aircraft-pinned cycle-convention variables (consumed only by the
    # airframe's V2 row; harmless free variables here)
    M2v = Vn("M_2", 0.6, "-", "fan face Mach number")
    M25v = Vn("M_25", 0.6, "-", "HPC face Mach number")
    hold2 = Vn("hold_2", 1.07, "-", "fan face stagnation factor")
    hold25 = Vn("hold_25", 1.06, "-", "HPC face stagnation factor")
    c1v = Vn("c1", 1.1, "-", "freestream stagnation factor")

    # TSFC is WEIGHT-specific (TASOPT 1/hr): Wf*g/Fn.
    _TSFC_CONV = 9.80665 * 3600.0
    for i in range(N):
        o = seg_out[i]
        tag = f"s{i}_"
        # group attribute lookup by full internal name
        g_ = lambda nme: getattr(eng, f"cyc_{tag}{nme}")
        cons += [
            TSFC[i] * units.hr == g_("TSFC") * _TSFC_CONV,
            u6[i] == g_("V_core") * units.m / units.s,
            u8[i] == g_("V_byp") * units.m / units.s,
            mFan[i] == g_("W") * units.kg / units.s,
            Tt4x[i] == g_("Tt4") * units.K,
        ]

    if "scalars" in debug_skip:
        return eng, cons
    # ---- interface: engine-level scalars ----------------------------------
    W_engine = Vu("W_engine", 1e4, "N", "weight of a single turbofan")
    df = Vu("d_f", 1.5, "m", "fan diameter")
    dlpc = Vu("d_LPC", 0.8, "m", "LPC diameter")
    A2 = Vu("A_2", 1.5, "m^2", "fan area")
    A25 = Vu("A_25", 0.3, "m^2", "HPC area")
    A5 = Vu("A_5", 0.3, "m^2", "core exhaust nozzle area")
    A7 = Vu("A_7", 1.0, "m^2", "fan exhaust nozzle area")
    mFanD = Vu("mbar_fan_D", 300.0, "kg/s", "fan design corrected flow")

    # Fitzgerald weight on the design corrected core flow == the LPC's
    # design corrected flow (map rows carry it), OPR = pilc*pihc.
    from .model import fitzgerald_coeffs
    _a, _b, _c = fitzgerald_coeffs(tech.BPR_ref, tech.geared, tech.advanced)
    if "fitz" not in debug_skip:
        cons += [
            W_engine / units.N >= _a * 4.44822
                * (eng.cyc_lpc_Wc / 45.35) ** _b
                * ((PIlc * PIhc) / 40.0) ** _c,
            # W_engine's only other pressure is the airframe weight chain,
            # which is absent from the restoration phases' feasibility
            # subproblems -- leaving a one-sided row's up-direction as a
            # free ray (the fitz-only bisection crash). Physical ceiling:
            # no single-aisle turbofan weighs 300 kN.
            W_engine <= 3.0e5 * units.N,
            W_engine >= 2.0e3 * units.N,
        ]
    if "areas" not in debug_skip:
        cons += [
            mFanD == eng.cyc_fan_Wc * units.kg / units.s,
            A2 * (_corr_flow_per_area(0.60)
                  * units.kg / units.s / units.m**2) == mFanD,
            A25 * (_corr_flow_per_area(0.55)
                   * units.kg / units.s / units.m**2)
                == eng.cyc_hpc_Wc * units.kg / units.s,
            A5 == eng.cyc_A_core * units.m**2,
            A7 == eng.cyc_A_byp * units.m**2,
            A5 + A7 <= A2,
        ]
    if "diam" not in debug_skip:
        cons += [
            df == (4.0 * A2 / (math.pi * (1.0 - tech.HTR_fan**2))) ** 0.5,
            dlpc == (4.0 * A25 / (math.pi
                                  * (1.0 - tech.HTR_lpc**2))) ** 0.5,
        ]

    if "cool" in debug_skip:
        return eng, cons
    # ---- tfcool.f at the takeoff rating, against the fixed budget ---------
    # Total cooling budget as a fraction of core flow, from the validated
    # cycle's bleed fractions (cool1+cool2 of core, cool3+cool4 of W3):
    p_ = pins
    _budget = (p_.cool1.frac_W + p_.cool2.frac_W
               + (1.0 - p_.cool1.frac_W - p_.cool2.frac_W
                  - tech.cust_frac_W)
               * (p_.frac_cool3 + p_.frac_cool4))
    alpha_c = Cu("alpha_cool_budget", _budget, "-",
                 "cooling budget, fraction of core flow")
    _Trr = 1.0 / (1.0 + 0.5 * (1.313 - 1.0) * tech.M_t_exit ** 2)
    Tmet = Cu("T_metal", tech.T_metal_K, "K", "design blade metal temp")
    _eps = []
    Tt3_TO = getattr(eng, "cyc_s0_hpc_Tt")  # segment 0 = the rating point
    for _r in (1, 2, 3):
        th = V(f"theta_cool_{_r}", 0.4, f"cooling effectiveness, row {_r}")
        e0 = V(f"eps0_cool_{_r}", 0.05, f"cooling requirement, row {_r}")
        ep = V(f"eps_cool_{_r}", 0.04, f"cooling flow ratio, row {_r}")
        Tg = (tech.Tt4_TO_K + tech.dT_streak_K if _r == 1
              else tech.Tt4_TO_K * _Trr ** (_r - 1))
        _ef, _tf, _StA = tech.efilm, tech.tfilm, tech.StA
        cons += [
            th * Tg + tech.T_metal_K >= Tg + th * Tt3_TO,   # [SP] SigIneq
            th <= 0.999,
            e0 * _ef + _StA * _tf * (1.0 - _ef)
                >= _StA * th * (1.0 - _ef * _tf) + e0 * _ef * th,
                                                            # [SP] SigIneq
            ep + ep * e0 >= e0,                             # [SP] SigIneq
        ]
        _eps.append(ep)
    cons += [_budget >= _eps[0] + _eps[1] + _eps[2]]

    return eng, cons


def _corr_flow_per_area(M):
    """SLS corrected mass flow per unit area at face Mach ``M``, kg/s/m^2.
    Standard-day air, gamma 1.4."""
    T = 288.15 / (1.0 + 0.2 * M * M)
    p = 101325.0 * (T / 288.15) ** 3.5
    rho = p / (287.05 * T)
    a = (1.4 * 287.05 * T) ** 0.5
    return rho * a * M
