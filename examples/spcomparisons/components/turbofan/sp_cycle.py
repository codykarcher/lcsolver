"""SP-native turbofan cycle: pycycle physics as signomial rows.

Milestones 2-3 of HANDOFF_ENGINE.md: the two-spool separate-flow turbofan of
``truth/hbtf.py`` -- pycycle's validated HBTF -- rewritten as
signomial-program rows. A NEW module; nothing in ``model.py`` is touched.
Stations are validated against ``truth/data/*.json`` by
``truth/validate_sp.py`` (design) and ``truth/validate_sp_od.py``
(off-design).

Fidelity ingredients (constants from ``sp_thermo.py`` / ``sp_maps.py``,
both generated from the same sources pycycle reads):

* Variable-cp enthalpies: cp-first cubic/quartic fits, signomial equalities.
* Isentropes as monomials: P2/P1 == psi(T2)/psi(T1) with psi the fitted
  entropy exponential; the vitiated psi carries the FULL equilibrium entropy
  (mixing term included) so turbine expansion tracks pycycle's shifting
  equilibrium, worth 0.5% of HPT work even at cruise T4.
* The burner is a mass-action equilibrium block: 11 species, 6 monomial
  reaction rows against fitted 1/Kp posynomials, 5 element balances, mole
  sum, and absolute-enthalpy conservation with the fuel stream at CEA-zero
  (pycycle's convention, measured in the truth data).
* Off-design: the NPSS-heritage maps as signomial surfaces (sp_maps.py) with
  pycycle's exact scalar structure (s_PR on PR-1, others direct ratios), and
  the OD closure pycycle uses -- W balances the core nozzle throat area, BPR
  the bypass nozzle area, spool speeds the two shaft power balances, FAR the
  T4 rating (or a thrust fraction of a named full-power point).
* Bleed and turbine-cooling bookkeeping copied line for line from pycycle's
  elements (arithmetic frac_P interpolation; cooling enters the turbine at
  the TURBINE's frac_P pressure; mass-weighted remix; per-stream ideal
  expansions).

Conventions
-----------
Dimensionless variables with SI magnitudes (K, Pa, J/kg, kg/s, N, W).
Enthalpies are CEA-absolute SHIFTED by ``H_SHIFT`` so cold-side stations
stay GP-positive; mass-conserving rows cancel the shift and the burner's
fuel-addition row carries the explicit ``far*H_SHIFT`` term. Species mole
numbers are scaled by ``N_SCALE`` (trace radicals sit below the solver's
positivity floor unscaled); the scale cancels in the mass-action rows
provided the pressure factor uses the SCALED mole sum. Nozzle choking is a
per-point branch fixed a priori (TASOPT ichoke5/7 style). Off-design point
variables are prefixed ``<tag>_``; design variables are unprefixed.
"""
from __future__ import annotations

from dataclasses import dataclass

from edi import Formulation

from . import sp_thermo as TH
from . import sp_maps as MAPS

R_UNIV = TH.R_UNIV            # J/(mol K)
H_SHIFT = 1.5e6               # J/kg; see module docstring
# REGENERATION TRAPS: conventions hard-coded into rows below (row_G's
# '+1.0' assumes the c0 boundary fit is INVERSE; the enthalpy shift
# must match the generated fits). Fail loudly if a refit flips them.
assert abs(TH.G_VIT.get('H_SHIFT', 1.5e6) - H_SHIFT) < 1e-6
assert abs(TH.AIR_H_SMA.get('H_SHIFT', 1.5e6) - H_SHIFT) < 1e-6
assert TH.G_VIT.get('c0_inv', True) is True, 'row_G needs INVERSE c0'

N_SCALE = 1.0e6               # species mole-number scaling; see burner block
P_REF_PA = TH.P_REF_BAR * 1e5
T_STD = 288.15                # map corrected-flow reference, K (518.67 R)
P_STD = 101325.0              # and Pa (14.696 psi)
LB2KG = 0.45359237            # map Wc tables are lbm/s

_N_AIR = sum(TH.AIR_COMPOSITION_KMOL_KG.values())
R_AIR = R_UNIV * 1000.0 * _N_AIR
_FUEL = TH.FUEL
#: kmol of gas added per kg of fuel by complete combustion of C12H23.
_N_FUEL_ADD = (_FUEL['elements']['H'] / 4.0) / _FUEL['wt']


# ---------------------------------------------------------------------------
# fitted-property expressions (pyomo-compatible)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ZERO-CANCELLATION thermo forms (SMA / posynomial; fits.py emits them).
#
# The original free-sign signomial fits carried 65-410x internal
# cancellation, and the free-design isolation showed the SIA can VERIFY
# such rows but not NAVIGATE them. Every thermo equality now enters the
# model as monomial == posynomial: for an SMA fit with softness a,
#     y == (sum_k m_k(x))^(1/a)   becomes   y**a == sum_k m_k(x)**a,
# no negative terms anywhere. The float-callable evaluators below keep the
# ORIGINAL function contracts so warm_start and all guess arithmetic stay
# consistent with the rows to fit precision.
# ---------------------------------------------------------------------------
_PA = TH.AIR_PSI_SMA
_HA = TH.AIR_H_SMA
_PV = TH.VIT_PSI_SMA
_GV = TH.G_VIT
_CPA = TH.CP_AIR_POSY


def _sma_posy(fit, *xs):
    """sum_k m_k**a -- the posynomial side of an SMA equality."""
    a = fit['a1']
    out = 0.0
    for c, e in zip(fit['c'], fit['e']):
        term = c**a
        for x, ei in zip(xs, e):
            term = term * x ** (a * ei)
        out = out + term
    return out


def psi_air_posy(T):
    return _sma_posy(_PA, T / _PA['T_ref_K'])


def h_air_posy(T):
    return _sma_posy(_HA, T / _HA['T_ref_K'])


def psi_vit_posy(T, far):
    return _sma_posy(_PV, T / _PV['T_ref_K'], far)


def cp_air_posy(T):
    t = T / _CPA['T_ref_K']
    p = sum(c * t**e[0] for c, e in zip(_CPA['c'], _CPA['e']))
    return (1.0 / p) if _CPA['inv'] else p


def g_vit_core(T, far):
    """The integrated-cp part of G_vit (small negative T_lo constants,
    cancellation <= 2 like the species enthalpies)."""
    Tr, Tlo = _GV['T_ref_K'], _GV['T_lo_K']
    t, tlo = T / Tr, Tlo / Tr
    return sum(c * Tr / (e[0] + 1.0)
               * (t ** (e[0] + 1.0) - tlo ** (e[0] + 1.0)) * far ** e[1]
               for c, e in zip(_GV['cp_c'], _GV['cp_e']))


def c0_posy(far):
    """C0 is fitted INVERSE (1/posy); rows multiply through by this."""
    return sum(c * far ** e[0] for c, e in zip(_GV['c0_c'], _GV['c0_e']))


# row helpers: each returns one monomial == posynomial constraint
def row_psi_air(psi, T):
    return psi ** _PA['a1'] == psi_air_posy(T)


def row_h_air(ht_shifted, T):
    return ht_shifted ** _HA['a1'] == h_air_posy(T)


def row_psi_vit(psi, T, far):
    return psi ** _PV['a1'] == psi_vit_posy(T, far)


def row_G(h_shifted, far, T):
    """(1+far)*h_shifted == G_vit(T,far), multiplied through by the C0
    posynomial so both sides stay near-posynomial."""
    c0p = c0_posy(far)
    return ((1.0 + far) * h_shifted * c0p
            == g_vit_core(T, far) * c0p + 1.0)


def row_cp_vit(cp, far, T):
    """(1+far)*cp == dG/dT, the fitted cp posynomial directly."""
    Tr = _GV['T_ref_K']
    rhs = sum(c * (T / Tr) ** e[0] * far ** e[1]
              for c, e in zip(_GV['cp_c'], _GV['cp_e']))
    return (1.0 + far) * cp == rhs


def row_cp_air(cp, T):
    t = T / _CPA['T_ref_K']
    p = sum(c * t**e[0] for c, e in zip(_CPA['c'], _CPA['e']))
    if _CPA['inv']:
        return cp * p == 1.0
    return cp == p


# float-callable evaluators with the ORIGINAL contracts
def h_air(T):
    return h_air_posy(T) ** (1.0 / _HA['a1']) - _HA['H_SHIFT']


def cp_air(T):
    return cp_air_posy(T)


def psi_air(T):
    return psi_air_posy(T) ** (1.0 / _PA['a1'])


def h_vit(T, far):
    """(1+far) * h_abs of vitiated gas, J/kg (original contract)."""
    G = g_vit_core(T, far) + 1.0 / c0_posy(far)
    return G - (1.0 + far) * _GV['H_SHIFT']


def cp_vit_times_1pf(T, far):
    Tr = _GV['T_ref_K']
    return sum(c * (T / Tr) ** e[0] * far ** e[1]
               for c, e in zip(_GV['cp_c'], _GV['cp_e']))


def psi_vit(T, far):
    return psi_vit_posy(T, far) ** (1.0 / _PV['a1'])


def h_molar(sp, T):
    c, Tr = TH.SPECIES_H[sp]['c'], TH.SPECIES_H[sp]['T_ref_K']
    return sum(ck * (T / Tr)**k for k, ck in enumerate(c))


def kp_inv(sp, T):
    d = TH.KP[sp]
    assert d['kind'] == 'posy_inv', d['kind']
    return sum(ck * (T / d['T_ref_K'])**a for ck, a in d['terms'])


def map2d(terms, x, y):
    return sum(c * x**a * y**b for c, a, b in terms)


def sma_map_posy(fit, x1, x2):
    """posynomial side of an SMA map equality."""
    a = fit['a1']
    return sum((c**a) * x1 ** (a * e[0]) * x2 ** (a * e[1])
               for c, e in zip(fit['c'], fit['e']))


def row_sma_map(var, fit, x1, x2):
    """var == SMA(x1,x2) as monomial == posynomial (inverse-fit aware)."""
    a = fit['a1'] * (-1.0 if fit['inv'] else 1.0)
    return var ** a == sma_map_posy(fit, x1, x2)


def sma_map_val(fit, x1, x2):
    v = sma_map_posy(fit, x1, x2) ** (1.0 / fit['a1'])
    return (1.0 / v) if fit['inv'] else v


# ---------------------------------------------------------------------------
# pins
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Bleed:
    frac_W: float
    frac_P: float = 0.0
    frac_work: float = 0.0


@dataclass(frozen=True)
class CyclePins:
    """Design-point inputs. All SI."""
    name: str
    T0_K: float
    P0_Pa: float
    MN: float
    Fn_N: float
    T4_K: float
    FPR: float
    LPC_PR: float
    HPC_PR: float
    BPR: float
    eff_fan: float
    eff_lpc: float
    eff_hpc: float
    eff_hpt: float
    eff_lpt: float
    LP_Nmech: float = 4666.1     # rpm; sets the map corrected-speed scalars
    HP_Nmech: float = 14705.7
    ram_recovery: float = 0.9990
    dP_duct4: float = 0.0048
    dP_duct6: float = 0.0101
    dP_burner: float = 0.0540
    dP_duct11: float = 0.0051
    dP_duct13: float = 0.0107
    dP_duct15: float = 0.0149
    Cv_core: float = 0.9933
    Cv_byp: float = 0.9939
    frac_bypBld: float = 0.005
    cool1: Bleed = Bleed(0.050708, 0.5, 0.5)    # HPC -> LPT
    cool2: Bleed = Bleed(0.020274, 0.55, 0.5)   # HPC -> LPT
    cust: Bleed = Bleed(0.0445, 0.5, 0.5)       # HPC -> overboard
    frac_cool3: float = 0.067214                # bld3 -> HPT inlet
    frac_cool4: float = 0.101256                # bld3 -> HPT exit
    HPX_W: float = 186425.0                     # 250 hp
    choked_core: bool = False
    choked_byp: bool = True
    #: Freestream velocity, m/s. None -> computed from MN with fit-derived
    #: gamma(T0); the aircraft state supplies V directly, and the fit's cp
    #: slope at the cold end is the least accurate part of the cubic, so
    #: prefer pinning it.
    V0_m_s: float | None = None


@dataclass(frozen=True)
class ODPins:
    """One off-design operating point."""
    name: str
    T0_K: float
    P0_Pa: float
    MN: float
    V0_m_s: float
    #: 'T4' throttles to T4_K; 'PC' to PC * (net thrust of point pc_of);
    #: 'F' to a required net thrust F_N with the rating cap T4_cap_K.
    mode: str
    T4_K: float | None = None
    PC: float | None = None
    pc_of: str | None = None
    F_N: float | None = None
    T4_cap_K: float | None = None
    choked_core: bool = False
    choked_byp: bool = True


SPECIES = list(TH.SPECIES_H)
_ELEM_ORDER = ['C', 'H', 'N', 'O', 'Ar']


def _complete_combustion(far):
    """Complete-combustion kmol/kg-mix -- initial guesses for the species
    block (house rule 7: guesses within a decade)."""
    air = TH.AIR_COMPOSITION_KMOL_KG
    w_air, w_fuel = 1.0 / (1 + far), far / (1 + far)
    bC = w_air * air['CO2'] + w_fuel * _FUEL['elements']['C'] / _FUEL['wt']
    bH = w_fuel * _FUEL['elements']['H'] / _FUEL['wt']
    nCO2, nH2O = bC, bH / 2.0
    nO2 = w_air * air['O2'] - (nCO2 - w_air * air['CO2']) - nH2O / 2.0
    return {'N2': w_air * air['N2'], 'O2': max(nO2, 1e-6),
            'Ar': w_air * air['Ar'], 'CO2': nCO2, 'H2O': nH2O,
            'CO': 3e-7, 'H2': 1e-7, 'OH': 3e-6, 'NO': 3e-5,
            'O': 3e-8, 'H': 1e-9}


def _gval(x, fallback):
    """A float for a GUESS: ``x`` itself when it is a number, else the
    fallback -- the rubber engine hands variables in as design pins, and a
    pyomo variable is not a guess."""
    return x if isinstance(x, (int, float)) else fallback


def _u0_from(T0, MN):
    cp0 = float(cp_air(T0))
    gam0 = cp0 / (cp0 - R_AIR)
    return MN * (gam0 * R_AIR * T0) ** 0.5


# ---------------------------------------------------------------------------
# one operating point
# ---------------------------------------------------------------------------

def _point(V, cons, tag, cond, pins, shared, out_by_tag, warm=None):
    """Add one operating point's rows. ``tag`` prefixes variable names ("",
    "TO_", ...). ``shared`` is None for the design point (PRs/effs/BPR come
    from ``pins`` and this point DEFINES the shared map scalars and nozzle
    areas); off-design points receive the design's shared dict. ``warm``
    maps full variable names to initial guesses that override the builder's
    defaults -- the phase-1 machinery chokes when a cold multipoint start
    is far from feasible, and the validation warm-starts from truth the
    same way the aircraft integration will warm-start from mission
    continuity."""
    design = shared is None
    P = lambda n: f"{tag}{n}"
    # Guards are emitted as explicit inequality ROWS, not variable bounds:
    # the aircraft build path (unit_corrector on a unit-ful formulation)
    # DROPS declared variable bounds entirely -- measured: 0 of 1589
    # bounds survived into the detected structure, and every
    # restoration-ray blowup the guards had killed came straight back
    # inside the aircraft. Rows survive every path. (Blanket guards on
    # every variable were separately tried and broke the solver; the set
    # stays TARGETED on the observed ray families: nozzle block, thrust,
    # turbine expansion, compressor block, burner species/far, map
    # coordinates.)
    V_raw = V

    def V(n, gs, d, bounds=None):
        v = V_raw(n, warm.get(n, gs) if warm else gs, d)
        if bounds is not None:
            lo, hi = bounds
            if lo is not None and lo > 0:
                cons.append(v >= lo)
            if hi is not None:
                cons.append(v <= hi)
        return v

    # Conditions may be plain numbers (truth validation pins them) or pyomo
    # expressions (the aircraft's flight state supplies them); guesses come
    # from the *_g entries when the conditions are symbolic.
    T0, P0, MN, u0 = cond['T0'], cond['P0'], cond['MN'], cond['V0']
    T0g = cond.get('T0_g', T0 if isinstance(T0, (int, float)) else 288.15)
    P0g = cond.get('P0_g', P0 if isinstance(P0, (int, float)) else 101325.0)
    u0g = cond.get('u0_g', u0 if isinstance(u0, (int, float)) else 230.0)
    MNg = cond.get('MN_g', MN if isinstance(MN, (int, float)) else 0.75)
    # guess scale: mass flows and powers track ambient pressure across
    # operating points (house rule 7 -- an SLS point runs 3-4x the cruise
    # flows, and a decade-off guess is what non-convergence looks like).
    sc = P0g / pins.P0_Pa
    h0s_g = float(h_air(T0g))
    psi0s_g = float(psi_air(T0g))
    ram = (1 + 0.2 * MNg**2)

    Tt0 = V(P("Tt0"), T0g * ram, "freestream total temperature, K")
    ht0 = V(P("ht0"), h0s_g + u0g**2 / 2 + H_SHIFT, "shifted total h, J/kg")
    psi0 = V(P("psi0"), psi0s_g * ram**3.5, "psi at Tt0")
    Pt0 = V(P("Pt0"), P0g * ram**3.5, "freestream total pressure, Pa")
    h0sv = V(P("h0stat"), h0s_g + H_SHIFT, "shifted static h at T0, J/kg")
    psi0sv = V(P("psi0stat"), psi0s_g, "psi at T0")
    cons += [
        row_h_air(ht0, Tt0),
        row_h_air(h0sv, T0),
        row_psi_air(psi0sv, T0),
        ht0 == h0sv + u0**2 / 2.0,                          # SigEq (posy)
        row_psi_air(psi0, Tt0),
        Pt0 * psi0sv == P0 * psi0,
    ]
    Pt2 = V(P("Pt2"), P0g * ram**3.5, "station 2 total pressure, Pa")
    cons += [Pt2 == Pt0 * pins.ram_recovery]

    # spool speeds: design pins them; off-design solves them from power.
    if design:
        LPN, HPN = pins.LP_Nmech, pins.HP_Nmech
    else:
        LPN = V(P("LP_N"), pins.LP_Nmech, "LP spool speed, rpm")
        HPN = V(P("HP_N"), pins.HP_Nmech, "HP spool speed, rpm")

    # ---- compressors -----------------------------------------------------
    def compressor(key, Tt_in, ht_in, psi_in, Pt_in, Nmech, PR_pin,
                   eff_pin, T_guess, Pt_guess):
        """Thermodynamic rows now; the corrected-flow/map rows are added by
        ``comp_maps`` once the mass flows exist."""
        psi_g = float(psi_air(T_guess))
        h_g = H_SHIFT + float(h_air(T_guess))
        n = lambda s: P(f"{key}_{s}")

        if design:
            PR, eff = PR_pin, eff_pin
        else:
            PR = V(n("PR"), _gval(PR_pin, 1.8), f"{key} pressure ratio")
            eff = V(n("eff"), _gval(eff_pin, 0.89),
                    f"{key} adiabatic efficiency")

        # guard bounds, same story as the turbine/nozzle/burner blocks:
        # the restoration ray that survived all the earlier guards ran
        # through the COMPRESSOR block (lpc_dhs hit 1e16 in the aircraft
        # solve). Physical ranges, so they cannot pinch a real solution.
        Tts = V(n("Tts"), T_guess * 0.97, f"{key} ideal exit Tt, K",
                bounds=(150.0, 1400.0))
        Tt = V(n("Tt"), T_guess, f"{key} exit Tt, K",
               bounds=(150.0, 1400.0))
        psis = V(n("psis"), psi_g * 0.9, f"psi at {key} ideal exit")
        psi = V(n("psi"), psi_g, f"psi at {key} exit")
        Pt = V(n("Pt"), Pt_guess, f"{key} exit Pt, Pa",
               bounds=(5e2, 2e7))
        dhs = V(n("dhs"), 3e4, f"{key} ideal enthalpy rise, J/kg",
                bounds=(1e2, 1.5e6))
        dh = V(n("dh"), 3e4, f"{key} enthalpy rise, J/kg",
               bounds=(1e2, 1.5e6))
        ht = V(n("ht"), h_g, f"{key} exit shifted ht, J/kg",
               bounds=(4e5, 3.5e6))
        hts = V(n("hts"), H_SHIFT + float(h_air(T_guess * 0.97)),
                f"{key} ideal exit shifted ht, J/kg")
        cons.extend([
            Pt == Pt_in * PR,
            psis == psi_in * PR,          # isentrope: monomial, exact
            row_psi_air(psis, Tts),
            row_h_air(hts, Tts),
            hts == ht_in + dhs,                             # SigEq (posy)
            dh * eff == dhs,
            ht == ht_in + dh,
            row_h_air(ht, Tt),
            row_psi_air(psi, Tt),
        ])
        return dict(Tt=Tt, ht=ht, psi=psi, Pt=Pt, dh=dh, PR=PR, eff=eff,
                    Tt_in=Tt_in, Pt_in=Pt_in, Nmech=Nmech,
                    PR_pin=PR_pin, eff_pin=eff_pin, key=key)

    def comp_maps(c, W_in, shared_out):
        """Corrected-flow and map rows for compressor dict ``c``."""
        key = c['key']
        n = lambda s: P(f"{key}_{s}")
        M = {'fan': MAPS.FAN, 'lpc': MAPS.LPC, 'hpc': MAPS.HPC}[key]
        # Anchor the design scalars on the FIT evaluated at the map default
        # coordinates, not the exact table read: the off-design rows read
        # the fit, and anchoring on the table makes design and off-design
        # mutually inconsistent by the fit residual -- a contradiction the
        # solver cannot close below ~1e-4.
        x0c, y0c = M['x0'], M['y0']
        at = {k: float(map2d(M[f'terms_{k}'], M['NcMap_d'] / x0c,
                             M['RlineMap_d'] / y0c))
              for k in ('PR', 'eff')}
        at['Wc'] = float(sma_map_val(M['sma_Wc'], M['NcMap_d'] / x0c,
                                     M['RlineMap_d'] / y0c))
        Wc = V(n("Wc"), 300.0, f"{key} corrected flow, kg/s")
        Nc = V(n("Nc"), 4000.0, f"{key} corrected speed, rpm")
        cons.extend([
            Wc * c['Pt_in'] / P_STD == W_in * (c['Tt_in'] / T_STD)**0.5,
            Nc * (c['Tt_in'] / T_STD)**0.5 == c['Nmech'],
        ])
        if design:
            # RATIO FORM: the map scalars are pure ratios of design-point
            # quantities to map-default constants, so they are EXPRESSIONS,
            # not variables with defining rows.  Off-design rows then couple
            # directly to the design variables (Wc_od*at == Wc_des*WcM_od
            # after substitution) -- no intermediate scalar columns for the
            # SIA to drag along when the design point is FREE.
            sWc = Wc / (at['Wc'] * LB2KG)
            sNc = Nc / M['NcMap_d']
            # PR keeps pycycle's shifted convention s_PR = (PR-1)/(PRmap-1);
            # as an expression it is signomial-linear in the design PR.
            sPR = (c['PR_pin'] - 1.0) / (at['PR'] - 1.0)
            sEff = c['eff_pin'] / at['eff']
            shared_out.update({
                f"s_Wc_{key}": sWc, f"s_Nc_{key}": sNc,
                f"s_PR_{key}": sPR, f"s_eff_{key}": sEff})
        else:
            wNc, wR = M['window']['Nc'], M['window']['R']
            NcM = V(n("NcMap"), M['NcMap_d'], f"{key} map corrected speed",
                    bounds=tuple(wNc))
            Rl = V(n("R"), M['RlineMap_d'], f"{key} map R-line",
                   bounds=tuple(wR))
            # (window guards: the bounds= arguments above BECOME rows
            # through this module's V wrapper -- an earlier explicit
            # cons.extend here duplicated all four exactly, and the
            # duplicate pairs degenerate the active set whenever a
            # window binds. The stale comment claiming bounds are
            # dropped described the AIRCRAFT build path's variable
            # bounds, not this wrapper.)
            prm1 = V(n("prm1"), at['PR'] - 1.0, f"{key} map PR - 1")
            WcM = V(n("WcMv"), at['Wc'], f"{key} map corrected-flow value")
            cons.extend([
                NcM * shared[f"s_Nc_{key}"] == Nc,
                # Wc through the zero-cancellation SMA surface; PR and eff
                # stay signomial (not log-convex either way; their SMA fits
                # are 2.5-5% where the signomials hold 0.5-2%)
                row_sma_map(WcM, M['sma_Wc'], NcM / x0c, Rl / y0c),
                Wc == shared[f"s_Wc_{key}"] * LB2KG * WcM,
                prm1 + 1.0 == map2d(M['terms_PR'], NcM / x0c,
                                    Rl / y0c),             # [SP] SigEq
                c['PR'] == 1.0 + shared[f"s_PR_{key}"] * prm1,  # SigEq (posy)
                c['eff'] == shared[f"s_eff_{key}"]
                            * map2d(M['terms_eff'], NcM / x0c,
                                    Rl / y0c),             # [SP] SigEq
            ])

    fan = compressor('fan', Tt0, ht0, psi0, Pt2, LPN, pins.FPR,
                     pins.eff_fan, 290.0, P0g * 2.5)
    Pt_lpc_in = V(P("Pt_lpc_in"), P0g * 2.4, "LPC face Pt, Pa")
    cons += [Pt_lpc_in == fan['Pt'] * (1.0 - pins.dP_duct4)]
    lpc = compressor('lpc', fan['Tt'], fan['ht'], fan['psi'], Pt_lpc_in,
                     LPN, pins.LPC_PR, pins.eff_lpc, 350.0,
                     P0g * 2.5 * pins.LPC_PR)
    Pt_hpc_in = V(P("Pt_hpc_in"), P0g * 2.4 * pins.LPC_PR, "HPC face Pt, Pa")
    cons += [Pt_hpc_in == lpc['Pt'] * (1.0 - pins.dP_duct6)]
    hpc = compressor('hpc', lpc['Tt'], lpc['ht'], lpc['psi'], Pt_hpc_in,
                     HPN, pins.HPC_PR, pins.eff_hpc, 800.0,
                     P0g * 2.5 * pins.LPC_PR * pins.HPC_PR)
    Tt3, ht3, psi3, Pt3, dh_hpc = (hpc['Tt'], hpc['ht'], hpc['psi'],
                                   hpc['Pt'], hpc['dh'])
    ht25 = lpc['ht']

    # ---- flows -----------------------------------------------------------
    W = V(P("W"), 150.0 * sc, "inlet total mass flow, kg/s")
    Wcore = V(P("Wcore"), 25.0 * sc, "core mass flow, kg/s")
    Wbyp = V(P("Wbyp"), 125.0 * sc, "bypass mass flow, kg/s")
    if design:
        BPR = pins.BPR
    else:
        BPR = V(P("BPR"), _gval(pins.BPR, 5.5), "bypass ratio")
    cons += [W == Wcore * (1.0 + BPR), Wbyp == Wcore * BPR]
    fW_c1, fW_c2, fW_cust = (pins.cool1.frac_W, pins.cool2.frac_W,
                             pins.cust.frac_W)
    W3 = V(P("W3"), 20.0 * sc, "HPC exit flow, kg/s")
    W31 = V(P("W31"), 17.0 * sc, "burner face flow, kg/s")
    cons += [W3 == Wcore * (1.0 - fW_c1 - fW_c2 - fW_cust),
             W31 == W3 * (1.0 - pins.frac_cool3 - pins.frac_cool4)]
    Wc1, Wc2 = Wcore * fW_c1, Wcore * fW_c2
    Wc3, Wc4 = W3 * pins.frac_cool3, W3 * pins.frac_cool4

    shared_out = {}
    comp_maps(fan, W, shared_out)
    comp_maps(lpc, Wcore, shared_out)
    comp_maps(hpc, Wcore, shared_out)

    # far bounds bracket the vitiated-fit validity range with margin; the
    # restoration phases found an unbounded ray through (far, Wf, species)
    # on the climb-segment builds, the same guard-rail story as the nozzle
    # and turbine blocks.
    # far guards ARE the vitiated-surface fit window: the G fit's
    # boundary term carries far^23 and extrapolates violently outside
    # [0.012, 0.042] (measured: -100%+ G error by far 0.046 against
    # equilibrium truth), so the wider (0.008, 0.048) band admitted
    # corrupt enthalpies inside its own guards.
    far = V(P("far"), 0.025, "burner fuel-air ratio", bounds=(0.012, 0.042))
    Wf = V(P("Wf"), 0.5 * sc, "fuel flow, kg/s",
           bounds=(0.5 * sc / 300.0, 0.5 * sc * 300.0))
    W4 = V(P("W4"), 18.0 * sc, "burner exit flow, kg/s")
    cons += [Wf == far * W31, W4 == W31 + Wf]

    # HPC bleed states (pycycle BleedsAndPower: arithmetic interpolation).
    def hpc_bleed(key, b: Bleed):
        htb = V(P(f"ht_{key}"), H_SHIFT + 3e5, f"{key} bleed shifted ht")
        Ptb = V(P(f"Pt_{key}"), 5e5, f"{key} bleed Pt, Pa")
        Ttb = V(P(f"Tt_{key}"), 600.0, f"{key} bleed Tt, K")
        psib = V(P(f"psi_{key}"), float(psi_air(600.0)), f"psi {key} bleed")
        cons.extend([
            htb == ht25 + b.frac_work * dh_hpc,
            # Pt interpolation is ARITHMETIC in pycycle; mirrored exactly.
            Ptb == Pt_hpc_in * (1.0 - b.frac_P) + b.frac_P * Pt3,  # [SP] SigEq
            row_h_air(htb, Ttb),
            row_psi_air(psib, Ttb),
        ])
        return htb, Ptb, Ttb, psib

    ht_c1, Pt_c1, Tt_c1, psi_c1 = hpc_bleed("cool1", pins.cool1)
    ht_c2, Pt_c2, Tt_c2, psi_c2 = hpc_bleed("cool2", pins.cool2)

    # ---- burner: mass-action equilibrium block ---------------------------
    P4 = V(P("Pt4"), 2e6, "burner exit Pt, Pa")
    cons += [P4 == Pt3 * (1.0 - pins.dP_burner)]
    if cond['mode'] == 'T4':
        T4 = V(P("Tt4"), cond['T4'], "burner exit Tt, K")
        cons += [T4 == cond['T4']]
    else:
        # fit-validity guards on free T4 (thrust/PC modes): T4 feeds the
        # Kp posynomials, species enthalpy cubics, and psi_vit -- all
        # windowed fits that carry guards everywhere else. A restoration
        # ray through unguarded T4 is exactly the family those guards
        # exist for.
        T4 = V(P("Tt4"), _gval(pins.T4_K, 1550.0) * 0.95,
               "burner exit Tt, K", bounds=(1100.0, 2150.0))

    # Species are scaled by N_SCALE: trace radicals sit at 2.5e-10 kmol/kg
    # at cruise T4, BELOW the solver's 1e-9 positivity floor, and the
    # mass-action rows go infeasible against the floor. The scale factors
    # cancel identically in the mass-action rows PROVIDED the pressure
    # factor uses the scaled mole sum (see below).
    n_guess = _complete_combustion(0.025)
    # Lower bounds six decades under the stoichiometric guess: trace
    # radicals COLLAPSE at part-power T4 (H is 4e-7 scaled at 1360 K, three
    # decades under its 1900 K value), and a guess/1e3 floor pinched the
    # true solution -- complementarity 1e9, the bound-pinch signature.
    n = {sp: V(P(f"n_{sp}"), n_guess[sp] * N_SCALE,
               f"{sp} kmol/kg mix, x{N_SCALE:.0e}",
               bounds=(max(n_guess[sp] * N_SCALE / 1e6, 5e-9),
                       n_guess[sp] * N_SCALE * 1e3)) for sp in SPECIES}
    ntot = V(P("n_tot"), sum(n_guess.values()) * N_SCALE,
             f"total kmol/kg mix, x{N_SCALE:.0e}",
             bounds=(sum(n_guess.values()) * N_SCALE / 10.0,
                     sum(n_guess.values()) * N_SCALE * 10.0))
    cons += [ntot == sum(n.values())]                       # SigEq (posy)

    air_b = {e: 0.0 for e in _ELEM_ORDER}
    for sp, nj in TH.AIR_COMPOSITION_KMOL_KG.items():
        for e, cnt in TH.SPECIES_ELEM[sp].items():
            air_b[e] += cnt * nj
    fuel_b = {e: _FUEL['elements'].get(e, 0.0) / _FUEL['wt']
              for e in _ELEM_ORDER}
    for e in _ELEM_ORDER:
        if air_b[e] == 0.0 and fuel_b[e] == 0.0:
            continue
        lhs = sum(TH.SPECIES_ELEM[sp].get(e, 0.0) * n[sp] for sp in SPECIES
                  if TH.SPECIES_ELEM[sp].get(e, 0.0))
        cons += [(1.0 + far) * lhs
                 == N_SCALE * air_b[e] + far * (N_SCALE * fuel_b[e])]

    for sp, d in TH.KP.items():
        st = d['stoich']
        lhs = n[sp] * kp_inv(sp, T4)
        num, den = 1.0, 1.0
        dpow = 1.0 - sum(st.values())
        for base, c in st.items():
            if c > 0:
                num = num * n[base]**c
            else:
                den = den * n[base]**(-c)
        # pressure/mole-number factor. Written with the SCALED ntot on
        # purpose: the scaled product row needs pf_true^dpow * S^-dpow, and
        # pf_true / S is exactly P4/(ntot_scaled * Pref).
        pf = (P4 / (ntot * P_REF_PA))
        cons += [lhs * den * pf**dpow == num]

    h4m = V(P("h4_mix"), H_SHIFT + 4e5, "burner-exit shifted mix h, J/kg")
    cons += [
        h4m == H_SHIFT + sum(nj * (1000.0 / N_SCALE) * h_molar(sp, T4)
                             for sp, nj in n.items()),      # [SP] SigEq
        (1.0 + far) * h4m == ht3 + far * H_SHIFT,           # [SP] SigEq
    ]
    psi4 = V(P("psi4"), 6.0, "psi_vit at Tt4", bounds=(1e-2, 1e3))
    cons += [row_psi_vit(psi4, T4, far)]

    # ---- turbines --------------------------------------------------------
    def turbine(key, W_in, Tt_in, h_in, psi_in, Pt_in, far_in, Nmech,
                eff_pin, cool_at_inlet, cool_at_exit, PR_guess, T_out_guess):
        nm = lambda s: P(f"{key}_{s}")
        M = {'hpt': MAPS.HPT, 'lpt': MAPS.LPT}[key]
        x0t, y0t = M['x0'], M['y0']
        at = {k: float(sma_map_val(M[f'sma_{k}'], M['NpMap_d'] / x0t,
                                   M['PRmap_d'] / y0t))
              for k in ('Wp', 'eff')}
        # Bounds on the expansion blocks are the same NUMERICAL GUARDS as
        # the nozzle's: the slacked restoration phases found unbounded rays
        # through the cooling ideal-expansion variables (lpt_hci -> inf,
        # caught by instrumentation) exactly as they had through Fg_core.
        Pt_out = V(nm("Ptout"), 3e5, f"{key} exit Pt, Pa",
                   bounds=(1e3, 1e7))
        Ti = V(nm("Ti"), T_out_guess, f"{key} ideal exit Tt, K",
               bounds=(400.0, 2000.0))
        psii = V(nm("psii"), float(psi_vit(T_out_guess, 0.025)),
                 f"psi_vit at {key} ideal exit")
        hi = V(nm("hi"), H_SHIFT, f"{key} ideal exit shifted ht, J/kg",
               bounds=(1e5, 3e6))
        dh = V(nm("dh"), 4e5, f"{key} ideal enthalpy drop, J/kg",
               bounds=(1e2, 2e6))
        eff = (eff_pin if design
               else V(nm("eff"), _gval(eff_pin, 0.90),
                      f"{key} adiabatic efficiency"))
        cons.extend([
            psii * Pt_in == psi_in * Pt_out,
            row_psi_vit(psii, Ti, far_in),
            row_G(hi, far_in, Ti),
            hi + dh == h_in,                                # SigEq (posy)
        ])
        # frac_P=1 cooling enters at the TURBINE's inlet pressure (pycycle
        # BleedPressure), NOT its own supply Pt, and expands to Pt_out.
        Wci, htci, psici = cool_at_inlet
        Tci = V(nm("Tci"), 500.0, f"{key} inlet-cooling ideal exit Tt, K",
                bounds=(200.0, 1500.0))
        psci = V(nm("psci"), float(psi_air(500.0)), f"psi at that Tt")
        hci = V(nm("hci"), H_SHIFT, f"ideally expanded cooling ht, J/kg",
                bounds=(1e5, 3e6))
        dhc = V(nm("dhc"), 1e5, f"{key} cooling ideal drop, J/kg",
                bounds=(1e2, 2e6))
        cons.extend([
            psci * Pt_in == psici * Pt_out,
            row_psi_air(psci, Tci),
            row_h_air(hci, Tci),
            hci + dhc == htci,                              # SigEq (posy)
        ])
        Wce, htce = cool_at_exit
        Pwr = V(nm("P"), 1e7 * sc, f"{key} shaft power, W")
        cons.extend([Pwr == W_in * eff * dh + Wci * eff * dhc])
        W_out = V(nm("Wout"), 21.0 * sc, f"{key} exit flow, kg/s")
        ht_out = V(nm("htout"), H_SHIFT, f"{key} exit mixed ht, J/kg")
        far_out = V(nm("farout"), 0.021, f"fuel fraction at {key} exit",
                    bounds=(0.005, 0.05))
        Tt_out = V(nm("Ttout"), T_out_guess * 1.02, f"{key} exit Tt, K")
        psi_out = V(nm("psiout"), float(psi_vit(T_out_guess * 1.02, 0.02)),
                    f"psi_vit at {key} exit")
        cons.extend([
            W_out == W_in + Wci + Wce,
            W_out * ht_out == W_in * (h_in - eff * dh)
                            + Wci * (htci - eff * dhc)
                            + Wce * htce,                   # [SP] SigEq
            far_out * W_out == Wf * (1.0 + far_out),        # SigEq (posy)
            row_G(ht_out, far_out, Tt_out),
            row_psi_vit(psi_out, Tt_out, far_out),
        ])

        # map rows. Wp is scaled x1e-4 from SI (kg sqrt(K) / (s Pa)) to
        # keep it O(1); the scalar absorbs the choice.
        Wp = V(nm("Wp"), 2.0, f"{key} referred flow, x1e-4 SI")
        cons.extend([Wp * Pt_in == W_in * Tt_in**0.5 * 1e4])
        Np = V(nm("Np"), 350.0, f"{key} referred speed, rpm/sqrt(K)")
        cons.extend([Np * Tt_in**0.5 == Nmech])
        PRv = V(nm("PR"), PR_guess, f"{key} pressure ratio Pt_in/Pt_out")
        cons.extend([PRv * Pt_out == Pt_in])
        if design:
            # RATIO FORM (see comp_maps): scalars as expressions of the
            # design variables, no scalar columns or defining rows.
            sWp = Wp / at['Wp']
            sNp = Np / M['NpMap_d']
            sPR = (PRv - 1.0) / (M['PRmap_d'] - 1.0)
            sEfft = eff_pin / at['eff']
            shared_out.update({f"s_Wp_{key}": sWp, f"s_Np_{key}": sNp,
                               f"s_PR_{key}": sPR,
                               f"s_eff_{key}": sEfft})
        else:
            wNp, wPR = M['window']['Np'], M['window']['PR']
            NpM = V(nm("NpMap"), M['NpMap_d'], f"{key} map referred speed",
                    bounds=tuple(wNp))
            PRm = V(nm("PRmap"), M['PRmap_d'], f"{key} map PR",
                    bounds=tuple(wPR))
            # (window guards come from the bounds= rows above; see the
            # comp_maps note -- explicit duplicates removed.)
            WpM = V(nm("WpMv"), at['Wp'], f"{key} map referred-flow value")
            effM = V(nm("effMv"), at['eff'], f"{key} map efficiency value")
            cons.extend([
                NpM * shared[f"s_Np_{key}"] == Np,
                row_sma_map(WpM, M['sma_Wp'], NpM / x0t, PRm / y0t),
                Wp == shared[f"s_Wp_{key}"] * WpM,
                PRv == 1.0 + shared[f"s_PR_{key}"] * (PRm - 1.0),  # [SP] SigEq
                row_sma_map(effM, M['sma_eff'], NpM / x0t, PRm / y0t),
                eff == shared[f"s_eff_{key}"] * effM,
            ])
        return dict(Pt_out=Pt_out, W_out=W_out, ht_out=ht_out,
                    far_out=far_out, Tt_out=Tt_out, psi_out=psi_out,
                    Pwr=Pwr, dh=dh, eff=eff)

    hpt = turbine('hpt', W4, T4, h4m, psi4, P4, far, HPN, pins.eff_hpt,
                  (Wc3, ht3, psi3), (Wc4, ht3), 3.6, 1150.0)
    Pt_lpt_in = V(P("Pt_lpt_in"), 3e5, "LPT face Pt, Pa")
    cons += [Pt_lpt_in == hpt['Pt_out'] * (1.0 - pins.dP_duct11)]
    lpt = turbine('lpt', hpt['W_out'], hpt['Tt_out'], hpt['ht_out'],
                  hpt['psi_out'], Pt_lpt_in, hpt['far_out'], LPN,
                  pins.eff_lpt, (Wc1, ht_c1, psi_c1), (Wc2, ht_c2),
                  4.4, 810.0)

    # ---- shaft balances --------------------------------------------------
    P_hpc = V(P("P_hpc"), 1e7 * sc, "HPC shaft power, W")
    cons += [
        P_hpc == dh_hpc * (W3 + Wc1 * pins.cool1.frac_work
                           + Wc2 * pins.cool2.frac_work
                           + Wcore * fW_cust * pins.cust.frac_work),
        hpt['Pwr'] == P_hpc + pins.HPX_W,
    ]
    P_fan = V(P("P_fan"), 1.5e7 * sc, "fan shaft power, W")
    P_lpc = V(P("P_lpc"), 3e6 * sc, "LPC shaft power, W")
    cons += [P_fan == W * fan['dh'], P_lpc == Wcore * lpc['dh'],
             lpt['Pwr'] == P_fan + P_lpc]

    W5 = lpt['W_out']
    Pt5 = V(P("Pt5"), 6e4, "core nozzle Pt, Pa")
    cons += [Pt5 == lpt['Pt_out'] * (1.0 - pins.dP_duct13)]
    W15 = V(P("W15"), 120.0 * sc, "bypass nozzle flow, kg/s")
    Pt15 = V(P("Pt15"), 5e4, "bypass nozzle Pt, Pa")
    cons += [W15 == Wbyp * (1.0 - pins.frac_bypBld),
             Pt15 == fan['Pt'] * (1.0 - pins.dP_duct15)]

    R_vit_ = V(P("R_vit"), 288.0, "core-gas gas constant, J/(kg K)")
    # complete-combustion mole count; dissociation adds ~0.1% moles at
    # 1900 K, ignored for R only.
    cons += [R_vit_ * (1.0 + lpt['far_out'])
             == R_UNIV * 1000.0 * (_N_AIR + lpt['far_out'] * _N_FUEL_ADD)]

    # ---- nozzles ---------------------------------------------------------
    def nozzle(key, Wn, htn, psin, Ptn, Cv, choked, vit, farv, Rgas):
        """CV nozzle, pycycle branch structure: exit==throat; at Ps=P0 when
        unchoked, at M=1 when choked. Area rows exist in BOTH branches --
        the off-design closure matches them to the design values."""
        # The wide bounds here are NUMERICAL GUARDS, not modelling
        # statements: in the slacked restoration phase the trio (Fg, A, Ps)
        # of the choked-thrust row admits an unbounded ray, and the solver
        # rode design Fg_core to +inf (caught by instrumenting the phase
        # cache). Any physically meaningful value sits far inside them.
        Fg = V(P(f"Fg_{key}"), 3e4 * sc, f"{key} nozzle gross thrust, N",
               bounds=(1e1, 1e7))
        Ts = V(P(f"Ts_{key}"), 260.0, f"{key} exit static T, K",
               bounds=(150.0, 2000.0))
        psis_ = V(P(f"psis_{key}"), 0.7, f"psi at {key} exit static T")
        hs = V(P(f"hs_{key}"), H_SHIFT, f"{key} exit static shifted h")
        Vx = V(P(f"V_{key}"), 300.0, f"{key} exit velocity, m/s",
               bounds=(10.0, 1500.0))
        Ps = V(P(f"Ps_{key}"), P0, f"{key} exit static pressure, Pa",
               bounds=(1e3, 1e7))
        rho = V(P(f"rho_{key}"), 0.7, f"{key} exit density, kg/m^3",
                bounds=(1e-3, 20.0))
        A = V(P(f"A_{key}"), 0.3, f"{key} exit/throat area, m^2",
              bounds=(1e-3, 30.0))
        if vit:
            cons.extend([
                row_psi_vit(psis_, Ts, farv),
                row_G(hs, farv, Ts),
            ])
        else:
            cons.extend([
                row_psi_air(psis_, Ts),
                row_h_air(hs, Ts),
            ])
        cons.extend([
            hs + Vx**2 / 2.0 == htn,                        # SigEq (posy)
            psis_ * Ptn == psin * Ps,
            rho * Rgas * Ts == Ps,
            A * rho * Vx == Wn,
        ])
        if not choked:
            # BRANCH-CONSISTENCY GUARD: the unchoked closure is only
            # valid subsonic. Off-anchor (an optimizer crossing NPR ~
            # 1.85 with the flag fixed) the closure would otherwise go
            # silently wrong; this row makes it INFEASIBLE instead.
            cpu_ = V(P(f"cpu_{key}"), 1010.0,
                     f"{key} exit cp (branch guard), J/(kg K)")
            if vit:
                cons.extend([row_cp_vit(cpu_, farv, Ts)])
            else:
                cons.extend([row_cp_air(cpu_, Ts)])
            cons.extend([
                Ps == P0 * 1.0, Fg == Cv * Wn * Vx,
                # Vx^2 <= a^2 = (cp/cv) R Ts = cp R Ts / (cp - R)
                Vx**2 * (cpu_ - Rgas) <= cpu_ * Rgas * Ts,  # [SP] SigIneq
            ])
        else:
            cp_ = V(P(f"cp_{key}"), 1010.0, f"{key} throat cp, J/(kg K)")
            cv_ = V(P(f"cv_{key}"), 723.0, f"{key} throat cv, J/(kg K)")
            if vit:
                cons.extend([row_cp_vit(cp_, farv, Ts)])
            else:
                cons.extend([row_cp_air(cp_, Ts)])
            cons.extend([
                cv_ + Rgas == cp_,                          # SigEq (posy)
                Vx**2 * cv_ == cp_ * Rgas * Ts,   # sonic; monomial equality
                Fg == Cv * Wn * Vx + A * (Ps - P0),         # [SP] SigEq
                # BRANCH-CONSISTENCY GUARD: choked requires the throat
                # static at or above ambient; below it the (Ps - P0)
                # pressure-thrust term flips sign and the closure is the
                # wrong branch. Infeasible beats silently wrong.
                Ps >= P0 * 1.0,
            ])
        return Fg, A

    Fg_core, A_core = nozzle("core", W5, lpt['ht_out'], lpt['psi_out'], Pt5,
                             pins.Cv_core, cond['choked_core'], True,
                             lpt['far_out'], R_vit_)
    Fg_byp, A_byp = nozzle("byp", W15, fan['ht'], fan['psi'], Pt15,
                           pins.Cv_byp, cond['choked_byp'], False,
                           None, R_AIR)

    if not design:
        cons += [A_core == shared['A_core'], A_byp == shared['A_byp']]

    # ---- performance -----------------------------------------------------
    Fn = V(P("Fn"), pins.Fn_N * sc, "net thrust, N", bounds=(1e1, 1e7))
    cons += [Fn + W * u0 == Fg_core + Fg_byp]               # SigEq (posy)
    if cond.get('F') is not None:
        # mission mode: the aircraft states the thrust it needs; the rating
        # cap is one-sided BY ARGUMENT -- required thrust pushes T4 up, the
        # cap holds it down, so the row binds exactly when the rating does.
        cons += [Fn == cond['F']]
        if cond.get('T4_cap') is not None:
            cons += [T4 <= cond['T4_cap']]
    elif design:
        cons += [Fn == pins.Fn_N]
    elif cond['mode'] == 'PC':
        cons += [Fn == cond['PC'] * out_by_tag[cond['pc_of'] + "_"]['Fn']]
    TSFC = V(P("TSFC"), 1.7e-5, "TSFC, kg/(N s)")
    cons += [TSFC * Fn == Wf]

    out = dict(Fn=Fn, TSFC=TSFC, W=W, far=far, Wf=Wf, Fg_core=Fg_core,
               Fg_byp=Fg_byp, A_core=A_core, A_byp=A_byp, T4=T4)
    if design:
        shared_out.update({'A_core': A_core, 'A_byp': A_byp})
        out['shared'] = shared_out
    return out


def build(pins: CyclePins, od_points: tuple = (),
          warm: dict | None = None) -> Formulation:
    """Design point (everything pinned) plus optional off-design points
    sharing the design's map scalars and nozzle areas. ``warm`` overrides
    initial guesses by full variable name."""
    f = Formulation()
    g = f.group("cyc", prefix="")

    # NOTE on bounds: a blanket +-3-decade guard on every variable was
    # tried against the restoration-ray blowups and made things worse (the
    # design-only solve stopped converging); bounds stay TARGETED -- the
    # nozzle block and net thrust, where the unbounded rays were actually
    # observed, plus the map-coordinate windows.
    V = lambda n, gs, d, bounds=None: g.Variable(n, gs, "-", d,
                                                 bounds=bounds)
    cons = []

    cond_des = dict(T0=pins.T0_K, P0=pins.P0_Pa, MN=pins.MN,
                    V0=(pins.V0_m_s if pins.V0_m_s is not None
                        else _u0_from(pins.T0_K, pins.MN)),
                    mode='T4', T4=pins.T4_K,
                    choked_core=pins.choked_core, choked_byp=pins.choked_byp)
    out_by_tag = {}
    des = _point(V, cons, "", cond_des, pins, None, out_by_tag, warm)
    out_by_tag[""] = des

    for od in od_points:
        cond = dict(T0=od.T0_K, P0=od.P0_Pa, MN=od.MN, V0=od.V0_m_s,
                    mode=od.mode, T4=od.T4_K, PC=od.PC, pc_of=od.pc_of,
                    F=od.F_N, T4_cap=od.T4_cap_K,
                    choked_core=od.choked_core, choked_byp=od.choked_byp)
        out_by_tag[f"{od.name}_"] = _point(V, cons, f"{od.name}_", cond,
                                           pins, des['shared'], out_by_tag,
                                           warm)

    f.Objective(des['TSFC'] * 1e5)
    f.ConstraintList(cons)
    return f
