"""Forward evaluation of the SP cycle's DESIGN point, for warm starts.

Computes an (essentially) exactly-feasible value for every design-point
variable of ``sp_cycle._point`` from a ``CyclePins``, using the SAME fitted
property functions the rows use -- so the values satisfy the rows to fit
rounding by construction. Purpose: the SIA has a spurious attractor for
this system (freestream rows parked 4e-3 violated under 1e9-scale duals, a
condensation artifact), and which basin it lands in is decided by the warm
start. A self-consistent start removes the roulette.

Only the design point is evaluated here; off-design points warm-start from
the design state (or a neighbouring segment's) with flow-like variables
scaled by ambient pressure, which the accretive continuation in the
aircraft runner already does.

Everything is per unit inlet flow until the thrust match scales it, with a
1-D outer iteration on W because the HPX shaft extraction is absolute.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .. import sp_cycle as SC

H = SC.H_SHIFT


class _NoRoot(Exception):
    pass


def _inv(fun, target, lo, hi):
    """brentq with a self-expanding bracket -- the outer pressure searches
    probe states outside the physical window, and a fixed bracket throws."""
    g = lambda x: float(fun(x)) - target
    glo, ghi = g(lo), g(hi)
    n = 0
    while glo * ghi > 0 and n < 40:
        if abs(glo) < abs(ghi):
            lo = max(lo * 0.85, 60.0)
            glo = g(lo)
        else:
            hi = hi * 1.15
            ghi = g(hi)
        n += 1
    return brentq(g, lo, hi, xtol=1e-10)


def design_state(pins) -> dict:
    """name -> value for the design point of ``SC._point`` (tag "")."""
    out = {}
    T0, P0, u0 = pins.T0_K, pins.P0_Pa, (pins.V0_m_s if pins.V0_m_s
                                         else SC._u0_from(pins.T0_K,
                                                          pins.MN))
    FPR, PIlc, PIhc, BPR = pins.FPR, pins.LPC_PR, pins.HPC_PR, pins.BPR

    # freestream / inlet
    ht0 = H + float(SC.h_air(T0)) + u0**2 / 2
    Tt0 = _inv(lambda T: H + SC.h_air(T), ht0, T0, T0 + 120.0)
    psi0 = float(SC.psi_air(Tt0))
    Pt0 = P0 * psi0 / float(SC.psi_air(T0))
    Pt2 = Pt0 * pins.ram_recovery
    out.update(Tt0=Tt0, ht0=ht0, psi0=psi0, Pt0=Pt0, Pt2=Pt2)

    def compressor(key, Tt_in, ht_in, psi_in, Pt_in, PR, eff):
        psis = psi_in * PR
        Tts = _inv(SC.psi_air, psis, Tt_in - 5.0, Tt_in * 2.5)
        dhs = H + float(SC.h_air(Tts)) - ht_in
        dh = dhs / eff
        ht = ht_in + dh
        Tt = _inv(lambda T: H + SC.h_air(T), ht, Tts - 5.0, Tts * 1.4)
        psi = float(SC.psi_air(Tt))
        Pt = Pt_in * PR
        out.update({f"{key}_Tts": Tts, f"{key}_Tt": Tt,
                    f"{key}_psis": psis, f"{key}_psi": psi,
                    f"{key}_Pt": Pt, f"{key}_dhs": dhs, f"{key}_dh": dh,
                    f"{key}_ht": ht})
        return Tt, ht, psi, Pt, dh

    fanT, fanh, fanpsi, fanPt, dh_fan = compressor(
        "fan", Tt0, ht0, psi0, Pt2, FPR, pins.eff_fan)
    Pt_lpc_in = fanPt * (1.0 - pins.dP_duct4)
    lpcT, lpch, lpcpsi, lpcPt, dh_lpc = compressor(
        "lpc", fanT, fanh, fanpsi, Pt_lpc_in, PIlc, pins.eff_lpc)
    Pt_hpc_in = lpcPt * (1.0 - pins.dP_duct6)
    Tt3, ht3, psi3, Pt3, dh_hpc = compressor(
        "hpc", lpcT, lpch, lpcpsi, Pt_hpc_in, PIhc, pins.eff_hpc)
    out.update(Pt_lpc_in=Pt_lpc_in, Pt_hpc_in=Pt_hpc_in)

    # HPC bleed states
    for key, b in (("cool1", pins.cool1), ("cool2", pins.cool2)):
        htb = lpch + b.frac_work * dh_hpc
        Ptb = Pt_hpc_in * (1.0 - b.frac_P) + b.frac_P * Pt3
        Ttb = _inv(lambda T: H + SC.h_air(T), htb, lpcT, Tt3 + 5.0)
        out.update({f"ht_{key}": htb, f"Pt_{key}": Ptb, f"Tt_{key}": Ttb,
                    f"psi_{key}": float(SC.psi_air(Ttb))})

    # burner: far from absolute-enthalpy conservation with fuel at CEA-zero
    T4 = pins.T4_K
    P4 = Pt3 * (1.0 - pins.dP_burner)
    h3_abs = ht3 - H
    far = brentq(lambda fa: float(SC.h_vit(T4, fa)) - h3_abs,
                 0.005, 0.05, xtol=1e-12)
    h4m = H + float(SC.h_vit(T4, far)) / (1.0 + far)
    psi4 = float(SC.psi_vit(T4, far))
    out.update(Tt4=T4, Pt4=P4, far=far, h4_mix=h4m, psi4=psi4)
    # species from the offline Gibbs solver (identical chemistry source)
    try:
        from . import fits
        n = fits.equilibrium(T4, P4 / 1e5, far)
        for sp, v in n.items():
            out[f"n_{sp}"] = max(v, 1e-14) * SC.N_SCALE
        out["n_tot"] = sum(max(v, 1e-14) for v in n.values()) * SC.N_SCALE
    except Exception:
        pass

    # flow split per unit inlet flow
    wcore = 1.0 / (1.0 + BPR)
    wbyp = BPR * wcore
    fW1, fW2, fWc = (pins.cool1.frac_W, pins.cool2.frac_W,
                     pins.cust.frac_W)
    w3 = wcore * (1.0 - fW1 - fW2 - fWc)
    w31 = w3 * (1.0 - pins.frac_cool3 - pins.frac_cool4)
    wc1, wc2 = wcore * fW1, wcore * fW2
    wc3, wc4 = w3 * pins.frac_cool3, w3 * pins.frac_cool4
    wf = far * w31
    w4 = w31 + wf
    w15 = wbyp * (1.0 - pins.frac_bypBld)

    def turbine(key, w_in, h_in, psi_in, Pt_in, far_in, eff,
                wci, hci_in, psici, wce, hce, P_req_unit, W_total):
        """Solve Pt_out so shaft power balances P_req_unit*W + HPX-share."""
        def power_minus_req(Pt_out):
            psii = psi_in * Pt_out / Pt_in
            Ti = _inv(lambda T: SC.psi_vit(T, far_in), psii, 450.0, T4)
            hi = H + float(SC.h_vit(Ti, far_in)) / (1.0 + far_in)
            dh = h_in - hi
            psci = psici * Pt_out / Pt_in
            Tci = _inv(SC.psi_air, psci, 250.0, 1400.0)
            hci = H + float(SC.h_air(Tci))
            dhc = hci_in - hci
            P = w_in * eff * dh + wci * eff * dhc
            return P - P_req_unit, (Ti, hi, dh, Tci, hci, dhc, P)
        lo, hi_ = Pt_in * 0.05, Pt_in * 0.85
        glo = power_minus_req(lo)[0]
        ghi = power_minus_req(hi_)[0]
        if glo * ghi > 0:
            # no expansion in range can meet the requirement -- tell the
            # outer W search to move (small W makes HPX/W impossible).
            raise _NoRoot(f"{key}: g({lo:.0f})={glo:.0f} "
                          f"g({hi_:.0f})={ghi:.0f} req={P_req_unit:.0f}")
        Pt_out = brentq(lambda p: power_minus_req(p)[0], lo, hi_,
                        xtol=1e-6)
        _, (Ti, hi, dh, Tci, hci, dhc, P) = power_minus_req(Pt_out)
        w_out = w_in + wci + wce
        ht_out = (w_in * (h_in - eff * dh) + wci * (hci_in - eff * dhc)
                  + wce * hce) / w_out
        far_out = wf / (w_out - wf)
        Tt_out = _inv(lambda T: H + SC.h_vit(T, far_out) / (1 + far_out),
                      ht_out, 450.0, T4)
        psi_out = float(SC.psi_vit(Tt_out, far_out))
        out.update({f"{key}_Ptout": Pt_out, f"{key}_Ti": Ti,
                    f"{key}_psii": psi_in * Pt_out / Pt_in,
                    f"{key}_hi": hi, f"{key}_dh": dh,
                    f"{key}_Tci": Tci, f"{key}_hci": hci,
                    f"{key}_dhc": dhc,
                    f"{key}_psci": psici * Pt_out / Pt_in,
                    f"{key}_Wout": w_out * W_total,
                    f"{key}_htout": ht_out, f"{key}_farout": far_out,
                    f"{key}_Ttout": Tt_out, f"{key}_psiout": psi_out,
                    f"{key}_P": P * W_total})
        return Pt_out, w_out, ht_out, far_out, Tt_out, psi_out

    def nozzle(key, w_n, htn, psin, Ptn, Cv, choked, vit, farv, Rgas,
               W_total):
        if vit:
            h_of = lambda T: H + float(SC.h_vit(T, farv)) / (1 + farv)
            psi_of = lambda T: float(SC.psi_vit(T, farv))
            cp_of = lambda T: float(SC.cp_vit_times_1pf(T, farv)) / (1 + farv)
        else:
            h_of = lambda T: H + float(SC.h_air(T))
            psi_of = SC.psi_air
            cp_of = lambda T: float(SC.cp_air(T))
        if choked:
            def g(Ts):
                cp = cp_of(Ts)
                V2son = cp * Rgas * Ts / (cp - Rgas)
                return 2.0 * (htn - h_of(Ts)) - V2son
            Ts = brentq(g, 150.0, 1500.0, xtol=1e-9)
            Vx = np.sqrt(2.0 * (htn - h_of(Ts)))
            Ps = Ptn * float(psi_of(Ts)) / psin
            cp = cp_of(Ts)
            out.update({f"cp_{key}": cp, f"cv_{key}": cp - Rgas})
        else:
            Ps = P0
            Ts = _inv(lambda T: float(psi_of(T)),
                      psin * Ps / Ptn, 150.0, 1500.0)
            Vx = np.sqrt(max(2.0 * (htn - h_of(Ts)), 1.0))
        rho = Ps / (Rgas * Ts)
        A = w_n * W_total / (rho * Vx)
        Fg = Cv * w_n * W_total * Vx + (A * (Ps - P0) if choked else 0.0)
        out.update({f"Ts_{key}": Ts, f"psis_{key}": float(psi_of(Ts)),
                    f"hs_{key}": h_of(Ts), f"V_{key}": Vx,
                    f"Ps_{key}": Ps, f"rho_{key}": rho, f"A_{key}": A,
                    f"Fg_{key}": Fg})
        return Fg

    R_vit = (SC.R_UNIV * 1000.0 * (SC._N_AIR + 0.02 * SC._N_FUEL_ADD)
             / 1.02)

    def net_thrust(W):
        P_hpc_u = dh_hpc * (w3 + wc1 * pins.cool1.frac_work
                            + wc2 * pins.cool2.frac_work
                            + wcore * fWc * pins.cust.frac_work)
        hpt = turbine("hpt", w4, h4m, psi4, P4, far, pins.eff_hpt,
                      wc3, ht3, psi3, wc4, ht3,
                      P_hpc_u + pins.HPX_W / W, W)
        Pt45, w45, ht45, far45, Tt45, psi45 = hpt
        Pt_lpt_in = Pt45 * (1.0 - pins.dP_duct11)
        P_lp_u = dh_fan * 1.0 + dh_lpc * wcore
        lpt = turbine("lpt", w45, ht45, psi45, Pt_lpt_in, far45,
                      pins.eff_lpt, wc1, out["ht_cool1"], out["psi_cool1"],
                      wc2, out["ht_cool2"], P_lp_u, W)
        Pt49, w5, ht49, far49, Tt49, psi49 = lpt
        Pt5 = Pt49 * (1.0 - pins.dP_duct13)
        Pt15 = fanPt * (1.0 - pins.dP_duct15)
        Rv = (SC.R_UNIV * 1000.0
              * (SC._N_AIR + far49 * SC._N_FUEL_ADD) / (1.0 + far49))
        Fg_c = nozzle("core", w5, ht49, psi49, Pt5, pins.Cv_core,
                      pins.choked_core, True, far49, Rv, W)
        Fg_b = nozzle("byp", w15, fanh, fanpsi, Pt15, pins.Cv_byp,
                      pins.choked_byp, False, None, SC.R_AIR, W)
        out.update(Pt_lpt_in=Pt_lpt_in, Pt5=Pt5, Pt15=Pt15, R_vit=Rv,
                   P_hpc=P_hpc_u * W, P_fan=dh_fan * W,
                   P_lpc=dh_lpc * wcore * W)
        return Fg_c + Fg_b - W * u0

    def _fn(W_):
        try:
            return net_thrust(W_) - pins.Fn_N
        except _NoRoot as e:
            if not out.get('_noroot_logged'):
                out['_noroot_logged'] = True
                print(f"    [warm_start] no-root at W={W_:.0f}: {e}")
            return -pins.Fn_N       # thrust shortfall: push W up
    W = brentq(_fn, 10.0, 3000.0, xtol=1e-6)
    net_thrust(W)   # repopulate `out` at the solved W

    out.update(W=W, Wcore=W * wcore, Wbyp=W * wbyp, W3=W * w3,
               W31=W * w31, W4=W * w4, Wf=W * wf, W15=W * w15,
               Fn=pins.Fn_N, TSFC=W * wf / pins.Fn_N)

    # map block at the design defaults
    def _mapfit(M, k, x, y):
        return float(SC.map2d(M[f'terms_{k}'], x / M['x0'], y / M['y0']))
    for key, M, W_in, Tt_in, Pt_in, Nm in (
            ("fan", SC.MAPS.FAN, W, Tt0, Pt2, pins.LP_Nmech),
            ("lpc", SC.MAPS.LPC, W * wcore, fanT, Pt_lpc_in,
             pins.LP_Nmech),
            ("hpc", SC.MAPS.HPC, W * wcore, lpcT, Pt_hpc_in,
             pins.HP_Nmech)):
        Wc = W_in * (Tt_in / SC.T_STD) ** 0.5 / (Pt_in / SC.P_STD)
        Nc = Nm / (Tt_in / SC.T_STD) ** 0.5
        at_Wc = _mapfit(M, 'Wc', M['NcMap_d'], M['RlineMap_d'])
        out.update({f"{key}_Wc": Wc, f"{key}_Nc": Nc,
                    f"{key}_sWc": Wc / (at_Wc * SC.LB2KG),
                    f"{key}_sNc": Nc / M['NcMap_d']})
    for key, M, W_in, Tt_in, Pt_in, Pt_o, Nm in (
            ("hpt", SC.MAPS.HPT, W * w4, T4, P4, out["hpt_Ptout"],
             pins.HP_Nmech),
            ("lpt", SC.MAPS.LPT, out["hpt_Wout"], out["hpt_Ttout"],
             out["Pt_lpt_in"], out["lpt_Ptout"], pins.LP_Nmech)):
        Wp = W_in * Tt_in ** 0.5 / Pt_in * 1e4
        Np = Nm / Tt_in ** 0.5
        PRv = Pt_in / Pt_o
        at_Wp = _mapfit(M, 'Wp', M['NpMap_d'], M['PRmap_d'])
        out.update({f"{key}_Wp": Wp, f"{key}_Np": Np, f"{key}_PR": PRv,
                    f"{key}_sWp": Wp / at_Wp,
                    f"{key}_sNp": Np / M['NpMap_d'],
                    f"{key}_sPR": (PRv - 1.0) / (M['PRmap_d'] - 1.0)})
    return out
