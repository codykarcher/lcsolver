"""SP-native on-design turbofan cycle: pycycle physics as signomial rows.

This is milestone 2 of HANDOFF_ENGINE.md: the two-spool separate-flow
turbofan of ``truth/hbtf.py`` -- pycycle's validated HBTF -- rewritten as
signomial-program rows. It is a NEW module; nothing in ``model.py`` is
touched, and the aircraft does not import it yet. Its stations are validated
one by one against ``truth/data/*.json`` by ``truth/validate_sp.py``.

Fidelity ingredients (all constants from ``sp_thermo.py``, generated from
the same janaf tables pycycle reads):

* Variable-cp enthalpies: h(T) cubics, signomial equalities.
* Isentropes as monomials: along a frozen-composition isentrope,
  P2/P1 == psi(T2)/psi(T1) with psi the fitted entropy exponential, so
  every pressure-temperature coupling is one monomial row.
* The burner is a mass-action equilibrium block: 11 species, 6 reaction
  rows (monomial equalities against fitted Kp), 5 element balances, a mole
  sum, and an absolute-enthalpy conservation row with the fuel stream at
  exactly zero enthalpy on the CEA scale (pycycle's convention, measured in
  the truth data). Dissociation at takeoff T4 is therefore in the model,
  not a calibration factor.
* Bleed and turbine-cooling bookkeeping copied line for line from
  pycycle's elements (arithmetic frac_P pressure interpolation, frac_work
  enthalpy, mass-weighted turbine remix, per-stream ideal expansions).

Conventions
-----------
Everything is dimensionless with SI magnitudes (K, Pa, J/kg, kg/s, N, W);
descriptions state the unit. Enthalpies are CEA-absolute SHIFTED by
``H_SHIFT`` so cold-side stations (negative on the CEA scale) stay
GP-positive: rows that conserve mass cancel the shift, and the two rows
where mass changes (burner fuel addition) carry the explicit ``far*H_SHIFT``
term. Nozzle choking is a per-build BRANCH (``choked_core`` /
``choked_byp``), fixed a priori exactly the way TASOPT's ichoke5/7 does it.

The cycle is written on a PER-POINT basis: ``add_cycle_point`` builds one
thermodynamic operating point (the design point here; off-design points
reuse the same rows with map-fit rows added in milestone 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from edi import Formulation

from . import sp_thermo as TH

R_UNIV = TH.R_UNIV            # J/(mol K)
H_SHIFT = 1.5e6               # J/kg; see module docstring
N_SCALE = 1.0e6               # species mole-number scaling; see burner block
P_REF_PA = TH.P_REF_BAR * 1e5

#: kmol/kg of air, and air's specific gas constant.
_N_AIR = sum(TH.AIR_COMPOSITION_KMOL_KG.values())
R_AIR = R_UNIV * 1000.0 * _N_AIR
#: kmol of gas added per kg of fuel by complete combustion of C12H23
#: (Delta n = y/4 per mole fuel).
_FUEL = TH.FUEL
_N_FUEL_ADD = (_FUEL['elements']['H'] / 4.0) / _FUEL['wt']


def h_air(T):
    """CEA-absolute air enthalpy, J/kg, as a pyomo expression. Coefficients
    are cp-first (cp fitted, integrated analytically), so dh/dT is accurate
    to the cp residual everywhere in range."""
    c, Tr = TH.AIR_H['c'], TH.AIR_H['T_ref_K']
    return sum(ck * (T / Tr)**k for k, ck in enumerate(c))


def cp_air(T):
    c, Tr = TH.AIR_H['c'], TH.AIR_H['T_ref_K']
    return sum(k * ck * (T / Tr)**(k - 1) / Tr
               for k, ck in enumerate(c) if k)


def psi_air(T):
    Tr = TH.AIR_PSI['T_ref_K']
    return sum(ck * (T / Tr)**a for ck, a in TH.AIR_PSI['terms'])


def h_vit(T, far):
    """CEA-absolute vitiated-gas enthalpy, J/kg. Multiply through by
    (1+far) at the call site -- the closed form here is
    (A + far B + far^2 C)/(1+far) and rows want the polynomial side."""
    t = T / TH.VIT_H['T_ref_K']
    A = sum(c * t**i for i, c in enumerate(TH.VIT_H['cA']))
    B = sum(c * t**i for i, c in enumerate(TH.VIT_H['cB']))
    C = sum(c * t**i for i, c in enumerate(TH.VIT_H['cC']))
    return A + far * B + far**2 * C          # == h_abs * (1 + far)


def cp_vit_times_1pf(T, far):
    """d/dT of (1+far)*h_vit, J/(kg K)."""
    Tr = TH.VIT_H['T_ref_K']
    t = T / Tr
    dA = sum(i * c * t**(i - 1) for i, c in enumerate(TH.VIT_H['cA']) if i)
    dB = sum(i * c * t**(i - 1) for i, c in enumerate(TH.VIT_H['cB']) if i)
    dC = sum(i * c * t**(i - 1) for i, c in enumerate(TH.VIT_H['cC']) if i)
    return (dA + far * dB + far**2 * dC) / Tr


def psi_vit(T, far):
    """far enters directly (see fits.py on the collinear (1+far) basis)."""
    Tr = TH.VIT_PSI['T_ref_K']
    return sum(ck * (T / Tr)**a * far**b
               for ck, a, b in TH.VIT_PSI['terms'])


def h_molar(sp, T):
    c, Tr = TH.SPECIES_H[sp]['c'], TH.SPECIES_H[sp]['T_ref_K']
    return sum(ck * (T / Tr)**k for k, ck in enumerate(c))


def kp_inv(sp, T):
    """1/Kp as a posynomial in T (all fits landed 'posy_inv')."""
    d = TH.KP[sp]
    assert d['kind'] == 'posy_inv', d['kind']
    return sum(ck * (T / d['T_ref_K'])**a for ck, a in d['terms'])


@dataclass(frozen=True)
class Bleed:
    frac_W: float
    frac_P: float = 0.0
    frac_work: float = 0.0


@dataclass(frozen=True)
class CyclePins:
    """Design-point inputs, mirroring truth/engines.py EngineSpec + the
    cycle parameters. All SI."""
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


SPECIES = list(TH.SPECIES_H)
_ELEM_ORDER = ['C', 'H', 'N', 'O', 'Ar']


def _complete_combustion(far):
    """Complete-combustion kmol/kg-mix -- initial guesses for the species
    block (house rule 7: guesses within a decade)."""
    air = TH.AIR_COMPOSITION_KMOL_KG
    w_air, w_fuel = 1.0 / (1 + far), far / (1 + far)
    bC = w_air * 2 * 0.0 + w_air * air['CO2'] + w_fuel * _FUEL['elements']['C'] / _FUEL['wt']
    bH = w_fuel * _FUEL['elements']['H'] / _FUEL['wt']
    nCO2 = bC
    nH2O = bH / 2.0
    nO2 = w_air * air['O2'] - (nCO2 - w_air * air['CO2']) - nH2O / 2.0
    return {'N2': w_air * air['N2'], 'O2': max(nO2, 1e-6),
            'Ar': w_air * air['Ar'], 'CO2': nCO2, 'H2O': nH2O,
            'CO': 3e-7, 'H2': 1e-7, 'OH': 3e-6, 'NO': 3e-5,
            'O': 3e-8, 'H': 1e-9}


def build(pins: CyclePins) -> Formulation:
    """The standalone on-design cycle, everything pinned, TSFC objective."""
    f = Formulation()
    g = f.group("cyc", prefix="")
    V = lambda n, gs, d: g.Variable(n, gs, "-", d)

    cons = []

    # ---- freestream -------------------------------------------------------
    T0, P0, MN = pins.T0_K, pins.P0_Pa, pins.MN
    # Static-point properties of the pinned freestream are plain numbers.
    h0s = float(h_air(T0))
    cp0 = float(cp_air(T0))
    gam0 = cp0 / (cp0 - R_AIR)
    u0 = (pins.V0_m_s if pins.V0_m_s is not None
          else MN * (gam0 * R_AIR * T0) ** 0.5)
    psi0s = float(psi_air(T0))

    Tt0 = V("Tt0", T0 * (1 + 0.2 * MN**2), "freestream total temperature, K")
    ht0 = V("ht0", h0s + u0**2 / 2 + H_SHIFT, "shifted total enthalpy, J/kg")
    psi0 = V("psi0", psi0s * (1 + 0.2 * MN**2)**3.5, "entropy exponential at Tt0")
    Pt0 = V("Pt0", P0 * (1 + 0.2 * MN**2)**3.5, "freestream total pressure, Pa")
    cons += [
        ht0 == H_SHIFT + h_air(Tt0),                       # [SP] SigEq
        ht0 == H_SHIFT + h0s + u0**2 / 2.0,                # [SP] SigEq
        psi0 == psi_air(Tt0),                              # [SP] SigEq
        Pt0 * psi0s == P0 * psi0,
    ]

    # ---- inlet ------------------------------------------------------------
    Pt2 = V("Pt2", P0 * 1.5, "station 2 total pressure, Pa")
    cons += [Pt2 == Pt0 * pins.ram_recovery]
    # ht2 == ht0, Tt2 == Tt0, psi2 == psi0: reuse the same variables.

    # ---- compressors ------------------------------------------------------
    def compressor(tag, Tt_in, ht_in, psi_in, Pt_in, PR, eff, T_guess,
                   Pt_guess):
        """PR at given adiabatic eff. Returns (Tt_out, ht_out, psi_out,
        Pt_out, dh). Guesses computed from T_guess through the fits --
        house rule 7, a variable ten decades off its magnitude is what
        non-convergence looks like, and psi spans 0.3..3e3 over the chain."""
        psi_g = float(psi_air(T_guess))
        h_g = H_SHIFT + float(h_air(T_guess))
        Tts = V(f"Tt{tag}s", T_guess * 0.97, f"{tag} ideal exit Tt, K")
        Tt = V(f"Tt{tag}", T_guess, f"{tag} exit Tt, K")
        psis = V(f"psi{tag}s", psi_g * 0.9, f"psi at Tt{tag}s")
        psi = V(f"psi{tag}", psi_g, f"psi at Tt{tag}")
        Pt = V(f"Pt{tag}", Pt_guess, f"{tag} exit Pt, Pa")
        dhs = V(f"dh{tag}s", 3e4, f"{tag} ideal enthalpy rise, J/kg")
        dh = V(f"dh{tag}", 3e4, f"{tag} enthalpy rise, J/kg")
        ht = V(f"ht{tag}", h_g, f"{tag} exit shifted ht, J/kg")
        cons.extend([
            Pt == Pt_in * PR,
            psis == psi_in * PR,          # isentrope: monomial, exact
            psis == psi_air(Tts),                          # [SP] SigEq
            H_SHIFT + h_air(Tts) == ht_in + dhs,           # [SP] SigEq
            dh == dhs / eff,
            ht == ht_in + dh,
            ht == H_SHIFT + h_air(Tt),                     # [SP] SigEq
            psi == psi_air(Tt),                            # [SP] SigEq
        ])
        return Tt, ht, psi, Pt, dh

    Tt21, ht21, psi21, Pt21, dh_fan = compressor(
        "21", Tt0, ht0, psi0, Pt2, pins.FPR, pins.eff_fan, 290.0,
        pins.P0_Pa * 2.5)

    # splitter: both legs carry the fan-exit state. duct4 core-side loss:
    Pt_lpc_in = V("Pt_lpc_in", 1e5, "LPC face Pt, Pa")
    cons += [Pt_lpc_in == Pt21 * (1.0 - pins.dP_duct4)]

    Tt25, ht25, psi25, Pt25, dh_lpc = compressor(
        "25", Tt21, ht21, psi21, Pt_lpc_in, pins.LPC_PR, pins.eff_lpc, 350.0,
        pins.P0_Pa * 2.5 * pins.LPC_PR)

    Pt_hpc_in = V("Pt_hpc_in", 2e5, "HPC face Pt, Pa")
    cons += [Pt_hpc_in == Pt25 * (1.0 - pins.dP_duct6)]

    Tt3, ht3, psi3, Pt3, dh_hpc = compressor(
        "3", Tt25, ht25, psi25, Pt_hpc_in, pins.HPC_PR, pins.eff_hpc, 800.0,
        pins.P0_Pa * 2.5 * pins.LPC_PR * pins.HPC_PR)

    # HPC bleed states (pycycle BleedsAndPower: arithmetic interpolation).
    def hpc_bleed(tag, b: Bleed):
        htb = V(f"ht_{tag}", H_SHIFT, f"{tag} bleed shifted ht, J/kg")
        Ptb = V(f"Pt_{tag}", 5e5, f"{tag} bleed Pt, Pa")
        Ttb = V(f"Tt_{tag}", 600.0, f"{tag} bleed Tt, K")
        psib = V(f"psi_{tag}", 30.0, f"psi at {tag} bleed Tt")
        cons.extend([
            htb == ht25 + b.frac_work * dh_hpc,
            # Pt interpolation is ARITHMETIC in pycycle; mirrored exactly.
            Ptb == Pt_hpc_in * (1.0 - b.frac_P) + b.frac_P * Pt3,  # [SP] SigEq
            htb == H_SHIFT + h_air(Ttb),                           # [SP] SigEq
            psib == psi_air(Ttb),                                  # [SP] SigEq
        ])
        return htb, Ptb, Ttb, psib

    ht_c1, Pt_c1, Tt_c1, psi_c1 = hpc_bleed("cool1", pins.cool1)
    ht_c2, Pt_c2, Tt_c2, psi_c2 = hpc_bleed("cool2", pins.cool2)
    # cust bleed leaves the engine; only its work matters (in power row).

    # ---- flows ------------------------------------------------------------
    W = V("W", 150.0, "inlet total mass flow, kg/s")
    Wcore = V("Wcore", 25.0, "core (splitter O1) mass flow, kg/s")
    Wbyp = V("Wbyp", 125.0, "bypass mass flow, kg/s")
    cons += [W == Wcore * (1.0 + pins.BPR),
             Wbyp == Wcore * pins.BPR]
    fW_c1, fW_c2, fW_cust = (pins.cool1.frac_W, pins.cool2.frac_W,
                             pins.cust.frac_W)
    W3 = V("W3", 20.0, "HPC exit flow, kg/s")
    W31 = V("W31", 17.0, "burner face flow (after bld3), kg/s")
    cons += [W3 == Wcore * (1.0 - fW_c1 - fW_c2 - fW_cust),
             W31 == W3 * (1.0 - pins.frac_cool3 - pins.frac_cool4)]
    Wc1 = Wcore * fW_c1
    Wc2 = Wcore * fW_c2
    Wc3 = W3 * pins.frac_cool3
    Wc4 = W3 * pins.frac_cool4

    far = V("far", 0.025, "burner fuel-air ratio")
    Wf = V("Wf", 0.5, "fuel flow, kg/s")
    W4 = V("W4", 18.0, "burner exit flow, kg/s")
    cons += [Wf == far * W31, W4 == W31 + Wf]

    # ---- burner: mass-action equilibrium block ---------------------------
    P4 = V("Pt4", 2e6, "burner exit Pt, Pa")
    cons += [P4 == Pt3 * (1.0 - pins.dP_burner)]
    T4 = V("Tt4", pins.T4_K, "burner exit Tt, K")
    cons += [T4 == pins.T4_K]

    # Species are scaled by N_SCALE: trace radicals sit at 2.5e-10 kmol/kg
    # at cruise T4, BELOW the solver's 1e-9 positivity floor, and the
    # mass-action rows go infeasible against the floor. The scale factors
    # cancel identically in the mass-action rows (S^(1-sum c) * S^(sum c-1)),
    # so only the element balances and the energy row carry N_SCALE.
    n_guess = _complete_combustion(0.025)
    n = {sp: V(f"n_{sp}", n_guess[sp] * N_SCALE,
               f"{sp} kmol per kg burner-exit mix, x{N_SCALE:.0e}")
         for sp in SPECIES}
    ntot = V("n_tot", sum(n_guess.values()) * N_SCALE,
             f"total kmol per kg mix, x{N_SCALE:.0e}")
    cons += [ntot == sum(n.values())]                       # SigEq (posy)

    # element balances: (1+far) * sum_j a_ej n_j == b_air_e + far * b_fuel_e
    air_b = {e: 0.0 for e in _ELEM_ORDER}
    for sp, nj in TH.AIR_COMPOSITION_KMOL_KG.items():
        for e, cnt in TH.SPECIES_ELEM[sp].items():
            air_b[e] += cnt * nj
    fuel_b = {e: _FUEL['elements'].get(e, 0.0) / _FUEL['wt']
              for e in _ELEM_ORDER}
    for e in _ELEM_ORDER:
        lhs = sum(TH.SPECIES_ELEM[sp].get(e, 0.0) * n[sp] for sp in SPECIES
                  if TH.SPECIES_ELEM[sp].get(e, 0.0))
        rhs = N_SCALE * air_b[e] + far * (N_SCALE * fuel_b[e])
        if air_b[e] == 0.0 and fuel_b[e] == 0.0:
            continue
        cons += [(1.0 + far) * lhs == rhs]                  # SigEq (posy)

    # mass action: for product p formed from bases with stoich c_b,
    #   x_p == Kp(T) * prod_b (x_b * Phat)^{c_b} / Phat,  Phat = P4/(ntot*Pref)
    # written with 1/Kp posynomial on the product side so each row is
    # monomial * posy(T) == monomial.
    for sp, d in TH.KP.items():
        st = d['stoich']
        lhs = n[sp] * kp_inv(sp, T4)
        num, den = 1.0, 1.0
        dpow = 1.0 - sum(st.values())   # exponent of (P4/(ntot*Pref*...))
        for base, c in st.items():
            if c > 0:
                num = num * n[base]**c
            else:
                den = den * n[base]**(-c)
        # pressure/mole-number factor. Written with the SCALED ntot on
        # purpose: the scaled product row needs pf_true^dpow * S^-dpow, and
        # pf_true / S is exactly P4/(ntot_scaled * Pref). Putting N_SCALE
        # back in here is the bug that left CO and O 1000x low (S^0.5 each)
        # and OH 31.6x low (S^0.25) while NO (dpow=0) sat at 1.001.
        pf = (P4 / (ntot * P_REF_PA))
        cons += [lhs * den * pf**dpow == num]

    # energy: absolute-enthalpy conservation, fuel at 0 (CEA scale):
    #   (1+far) * h4_abs == h3_abs   ->  shifted form carries far*H_SHIFT
    h4m = V("h4_mix", H_SHIFT + 2e5, "burner-exit shifted mix enthalpy, J/kg")
    cons += [
        h4m == H_SHIFT
        + sum(nj * (1000.0 / N_SCALE) * h_molar(sp, T4)
              for sp, nj in n.items()),                     # [SP] SigEq
        (1.0 + far) * h4m == ht3 + far * H_SHIFT,           # [SP] SigEq
    ]
    psi4 = V("psi4", 6.0, "psi_vit at Tt4")
    opf4 = V("opf4", 1.025, "1 + far at station 4")
    cons += [opf4 == 1.0 + far,
             psi4 == psi_vit(T4, far)]                     # [SP] SigEq

    # ---- HPT: solves its PR from the HP shaft balance --------------------
    # cool3 enters at Pt4 (frac_P=1), cool4 at Pt45 (frac_P=0).
    Pt45 = V("Pt45", 5e5, "HPT exit Pt, Pa")
    T41i = V("Tt41i", 1100.0, "HPT ideal exit Tt, K")
    psi41i = V("psi41i", 2.0, "psi_vit at Tt41i")
    h41i = V("ht41i", H_SHIFT, "HPT ideal exit shifted ht, J/kg")
    dh_hpt = V("dh_hpt", 4e5, "HPT ideal enthalpy drop, J/kg")
    cons += [
        psi41i * P4 == psi4 * Pt45,          # isentrope, monomial
        psi41i == psi_vit(T41i, far),                      # [SP] SigEq
        (1.0 + far) * (h41i - 0.0) == (1.0 + far) * H_SHIFT
            + h_vit(T41i, far),                             # [SP] SigEq
        h41i + dh_hpt == h4m,                               # SigEq (posy)
    ]
    # cool3's own ideal expansion Pt4 -> Pt45 (air):
    T_c3i = V("Tt_c3i", 550.0, "cool3 ideal expansion exit Tt, K")
    psi_c3i = V("psi_c3i", 20.0, "psi at Tt_c3i")
    h_c3i = V("ht_c3i", H_SHIFT, "cool3 ideally expanded shifted ht, J/kg")
    dh_c3 = V("dh_c3", 1e5, "cool3 ideal enthalpy drop, J/kg")
    cons += [
        psi_c3i * P4 == psi3 * Pt45,
        psi_c3i == psi_air(T_c3i),                          # [SP] SigEq
        h_c3i == H_SHIFT + h_air(T_c3i),                    # [SP] SigEq
        h_c3i + dh_c3 == ht3,                               # SigEq (posy)
    ]
    # power and exit mix (pycycle EnthalpyAndPower, mirrored):
    P_hpt = V("P_hpt", 1e7, "HPT shaft power, W")
    P_hpc = V("P_hpc", 1e7, "HPC shaft power, W")
    cons += [
        P_hpt == W4 * pins.eff_hpt * dh_hpt + Wc3 * pins.eff_hpt * dh_c3,
        # HPC power includes the work done on bleed streams up to their
        # extraction point (frac_work) and on cust which then leaves:
        P_hpc == dh_hpc * (W3
                           + Wc1 * pins.cool1.frac_work
                           + Wc2 * pins.cool2.frac_work
                           + Wcore * fW_cust * pins.cust.frac_work),
        P_hpt == P_hpc + pins.HPX_W,
    ]
    W45 = V("W45", 21.0, "HPT exit flow, kg/s")
    ht45 = V("ht45", H_SHIFT, "HPT exit (mixed) shifted ht, J/kg")
    far45 = V("far45", 0.021, "fuel fraction at 45")
    opf45 = V("opf45", 1.021, "1 + far45")
    Tt45 = V("Tt45", 1150.0, "HPT exit Tt, K")
    psi45 = V("psi45", 2.0, "psi_vit at Tt45")
    e_t = pins.eff_hpt
    cons += [
        W45 == W4 + Wc3 + Wc4,
        W45 * ht45 == W4 * (h4m - e_t * dh_hpt)
                    + Wc3 * (ht3 - e_t * dh_c3)
                    + Wc4 * ht3,                            # SigEq mixed [SP]
        far45 * W45 == Wf * (1.0 + far45),                  # SigEq (posy)
        opf45 == 1.0 + far45,
        opf45 * ht45 == opf45 * H_SHIFT + h_vit(Tt45, far45),  # [SP] SigEq
        psi45 == psi_vit(Tt45, far45),                      # [SP] SigEq
    ]

    # ---- LPT: solves its PR from the LP shaft balance --------------------
    Pt_lpt_in = V("Pt_lpt_in", 5e5, "LPT face Pt, Pa")
    cons += [Pt_lpt_in == Pt45 * (1.0 - pins.dP_duct11)]
    Pt49 = V("Pt49", 1e5, "LPT exit Pt, Pa")
    T49i = V("Tt49i", 800.0, "LPT ideal exit Tt, K")
    psi49i = V("psi49i", 0.5, "psi_vit at Tt49i")
    h49i = V("ht49i", H_SHIFT, "LPT ideal exit shifted ht, J/kg")
    dh_lpt = V("dh_lpt", 4e5, "LPT ideal enthalpy drop, J/kg")
    cons += [
        psi49i * Pt_lpt_in == psi45 * Pt49,
        psi49i == psi_vit(T49i, far45),                     # [SP] SigEq
        opf45 * h49i == opf45 * H_SHIFT + h_vit(T49i, far45),  # [SP] SigEq
        h49i + dh_lpt == ht45,                              # SigEq (posy)
    ]
    # cool1 (frac_P=1) expands Pt_c1 -> Pt49; cool2 (frac_P=0) does no work.
    T_c1i = V("Tt_c1i", 400.0, "cool1 ideal expansion exit Tt, K")
    psi_c1i = V("psi_c1i", 5.0, "psi at Tt_c1i")
    h_c1i = V("ht_c1i", H_SHIFT, "cool1 ideally expanded shifted ht, J/kg")
    dh_c1 = V("dh_c1", 1e5, "cool1 ideal enthalpy drop, J/kg")
    cons += [
        # frac_P=1: the bleed enters at the TURBINE's inlet pressure
        # (pycycle BleedPressure), not at its own HPC-supply Pt_c1 -- using
        # Pt_c1 overstated the cool1 expansion work and pushed LPT PR to
        # 0.975 of truth.
        psi_c1i * Pt_lpt_in == psi_c1 * Pt49,
        psi_c1i == psi_air(T_c1i),                          # [SP] SigEq
        h_c1i == H_SHIFT + h_air(T_c1i),                    # [SP] SigEq
        h_c1i + dh_c1 == ht_c1,                             # SigEq (posy)
    ]
    P_lpt = V("P_lpt", 2e7, "LPT shaft power, W")
    P_fan = V("P_fan", 1.5e7, "fan shaft power, W")
    P_lpc = V("P_lpc", 3e6, "LPC shaft power, W")
    e_l = pins.eff_lpt
    cons += [
        P_lpt == W45 * e_l * dh_lpt + Wc1 * e_l * dh_c1,
        P_fan == W * dh_fan,
        P_lpc == Wcore * dh_lpc,
        P_lpt == P_fan + P_lpc,
    ]
    W5 = V("W5", 21.5, "core nozzle flow, kg/s")
    ht49 = V("ht49", H_SHIFT, "LPT exit (mixed) shifted ht, J/kg")
    far49 = V("far49", 0.020, "fuel fraction at 49")
    opf49 = V("opf49", 1.020, "1 + far49")
    Tt49 = V("Tt49", 850.0, "LPT exit Tt, K")
    psi49 = V("psi49", 0.5, "psi_vit at Tt49")
    cons += [
        W5 == W45 + Wc1 + Wc2,
        W5 * ht49 == W45 * (ht45 - e_l * dh_lpt)
                   + Wc1 * (ht_c1 - e_l * dh_c1)
                   + Wc2 * ht_c2,                           # SigEq mixed [SP]
        far49 * W5 == Wf * (1.0 + far49),                   # SigEq (posy)
        opf49 == 1.0 + far49,
        opf49 * ht49 == opf49 * H_SHIFT + h_vit(Tt49, far49),  # [SP] SigEq
        psi49 == psi_vit(Tt49, far49),                      # [SP] SigEq
    ]
    Pt5 = V("Pt5", 5e4, "core nozzle Pt, Pa")
    cons += [Pt5 == Pt49 * (1.0 - pins.dP_duct13)]

    # ---- bypass duct ------------------------------------------------------
    W15 = V("W15", 120.0, "bypass nozzle flow, kg/s")
    Pt15 = V("Pt15", 5e4, "bypass nozzle Pt, Pa")
    cons += [W15 == Wbyp * (1.0 - pins.frac_bypBld),
             Pt15 == Pt21 * (1.0 - pins.dP_duct15)]

    # ---- nozzles ----------------------------------------------------------
    R_vit_ = V("R_vit", 288.0, "core-gas specific gas constant, J/(kg K)")
    # complete-combustion mole count: n_tot*(1+far) == n_air + far*(y/4)/wt_f;
    # dissociation adds ~0.1% moles at 1900 K, ignored for R only.
    cons += [R_vit_ * opf49 == R_UNIV * 1000.0
             * (_N_AIR + far49 * _N_FUEL_ADD)]  # SigEq (posy)

    def nozzle(tag, Wn, Ttn, htn, psin, Ptn, Cv, choked, vit, farv, opfv,
               Rgas):
        """CV nozzle, pycycle branch structure. Returns Fg expression."""
        Fg = V(f"Fg_{tag}", 3e4, f"{tag} nozzle gross thrust, N")
        if not choked:
            # exit at Ps = P0; Fg = Cv * W * V_exit.
            Ts = V(f"Ts_{tag}", 250.0, f"{tag} exit static T, K")
            psis_ = V(f"psis_{tag}", 0.5, f"psi at {tag} exit static T")
            hs = V(f"hs_{tag}", H_SHIFT, f"{tag} exit static shifted h, J/kg")
            Vx = V(f"V_{tag}", 300.0, f"{tag} exit velocity, m/s")
            if vit:
                cons.extend([
                    psis_ == psi_vit(Ts, farv),             # [SP] SigEq
                    opfv * hs == opfv * H_SHIFT + h_vit(Ts, farv),  # [SP] SigEq
                ])
            else:
                cons.extend([
                    psis_ == psi_air(Ts),                   # [SP] SigEq
                    hs == H_SHIFT + h_air(Ts),              # [SP] SigEq
                ])
            cons.extend([
                psis_ * Ptn == psin * P0,      # expand to ambient
                hs + Vx**2 / 2.0 == htn,                    # SigEq (posy)
                Fg == Cv * Wn * Vx,
            ])
        else:
            # throat at M=1; Fg = Cv*W*V_th + A_th*(Ps_th - P0).
            Ts = V(f"Ts_{tag}", 250.0, f"{tag} throat static T, K")
            psis_ = V(f"psis_{tag}", 0.5, f"psi at {tag} throat T")
            hs = V(f"hs_{tag}", H_SHIFT, f"{tag} throat static shifted h, J/kg")
            Vx = V(f"V_{tag}", 310.0, f"{tag} throat velocity (sonic), m/s")
            cp_ = V(f"cp_{tag}", 1010.0, f"{tag} throat cp, J/(kg K)")
            cv_ = V(f"cv_{tag}", 723.0, f"{tag} throat cv, J/(kg K)")
            Ps = V(f"Ps_{tag}", 5e4, f"{tag} throat static pressure, Pa")
            rho = V(f"rho_{tag}", 0.7, f"{tag} throat density, kg/m^3")
            A = V(f"A_{tag}", 0.3, f"{tag} throat area, m^2")
            if vit:
                cons.extend([
                    psis_ == psi_vit(Ts, farv),             # [SP] SigEq
                    opfv * hs == opfv * H_SHIFT + h_vit(Ts, farv),  # [SP] SigEq
                    opfv * cp_ == cp_vit_times_1pf(Ts, farv),  # [SP] SigEq
                ])
            else:
                cons.extend([
                    psis_ == psi_air(Ts),                   # [SP] SigEq
                    hs == H_SHIFT + h_air(Ts),              # [SP] SigEq
                    cp_ == cp_air(Ts),                      # [SP] SigEq
                ])
            cons.extend([
                cv_ + Rgas == cp_,                          # SigEq (posy)
                Vx**2 * cv_ == cp_ * Rgas * Ts,   # sonic; monomial equality
                hs + Vx**2 / 2.0 == htn,                    # SigEq (posy)
                psis_ * Ptn == psin * Ps,
                rho * Rgas * Ts == Ps,
                A * rho * Vx == Wn,
                Fg == Cv * Wn * Vx + A * (Ps - P0),         # [SP] SigEq
            ])
        return Fg

    Fg_core = nozzle("core", W5, Tt49, ht49, psi49, Pt5, pins.Cv_core,
                     pins.choked_core, True, far49, opf49, R_vit_)
    Fg_byp = nozzle("byp", W15, Tt21, ht21, psi21, Pt15, pins.Cv_byp,
                    pins.choked_byp, False, None, None, R_AIR)

    # ---- performance ------------------------------------------------------
    Fn = V("Fn", pins.Fn_N, "net thrust, N")
    TSFC = V("TSFC", 1.7e-5, "thrust specific fuel consumption, kg/(N s)")
    cons += [
        Fn + W * u0 == Fg_core + Fg_byp,                    # SigEq (posy)
        Fn == pins.Fn_N,
        TSFC * Fn == Wf,
    ]

    f.Objective(TSFC * 1e5)
    f.ConstraintList(cons)
    return f
