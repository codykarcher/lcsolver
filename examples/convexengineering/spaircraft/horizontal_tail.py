"""Horizontal tail sizing, structure and drag.

Source model
------------
``horizontal_tail.py`` in https://github.com/convexengineering/SPaircraft.

A quirk worth knowing
---------------------
The wing links its planform taper to its box taper
(``wns['\\lambda'] == wb['taper']``) and so does the vertical tail
(``taper = surface['\\lambda_{vt}']``). The horizontal tail does **not**: in
``wingbox.py`` the ``horizontal_tail`` branch declares
``taper = Variable('taper', 0.3, '-')`` — a fixed constant — and nothing ever
ties it to ``\\lambda_{ht}``. So the box is sized for a 0.3-taper surface no
matter what planform taper the optimizer picks, and the ``L_{ht_{tri}}``
constraint that uses ``(1 - wb['taper'])`` is likewise pinned. Reproduced as
written, because the reference numbers depend on it.

The model does not include wing downwash on tail effectiveness; the source
says so explicitly.

What is sized elsewhere
-----------------------
``M_r``, ``L_{shear}``, ``b_{ht_{out}}`` and ``\\pi_{M-fac}`` belong to the
pi-tail bending case and close at aircraft level, as do the CG-relative
distances and the tail volume coefficient.
"""
from __future__ import annotations

from numpy import pi, tan

from .wingbox import add_wingbox

# The horizontal tail box taper, fixed in wingbox.py and deliberately not
# linked to the planform taper. See the module docstring.
BOX_TAPER = 0.3


def add_horizontal_tail(f, N, state, *, sweep_deg, prefix="HT_"):
    """Add the horizontal tail. Returns ``(vars, constraints)``."""
    ht = f.group("ht", prefix=prefix)
    V, C = ht.Variable, ht.Constant
    Vn = lambda n, g, u, d: ht.Variable(n, g, u, d, size=N)

    # ---- planform ---------------------------------------------------------
    ARht = V("AR_ht", 6.0, "-", "horizontal tail aspect ratio")
    Sh = V("S_ht", 30.0, "m^2", "horizontal tail area")
    bht = V("b_ht", 13.0, "m", "horizontal tail span")
    croot = V("c_root_ht", 3.5, "m", "horizontal tail root chord")
    ctip = V("c_tip_ht", 1.1, "m", "horizontal tail tip chord")
    chma = V("cbar_ht", 2.5, "m", "mean aerodynamic chord, HT")
    taper = V("lambda_ht", 0.3, "-", "horizontal tail taper ratio")
    tau = V("tau_ht", 0.12, "-", "horizontal tail thickness/chord ratio")
    p = V("p_ht", 1.6, "-", "substituted variable 1 + 2 taper")
    q = V("q_ht", 1.3, "-", "substituted variable 1 + taper")
    ymac = V("y_cbar_ht", 2.8, "m", "spanwise location of the mean aero chord")
    e = V("e_ht", 0.8, "-", "Oswald efficiency factor")
    fl = V("f_lambda_ht", 0.02, "-", "empirical efficiency function of taper")

    dxlead = V("dx_lead_ht", 16.0, "m", "CG to HT leading edge")
    dxtrail = V("dx_trail_ht", 20.0, "m", "CG to HT trailing edge")
    lht = V("l_ht", 17.0, "m", "horizontal tail moment arm")
    xcght = V("x_CG_ht", 34.0, "m", "horizontal tail CG location")
    Lmax = V("L_ht_max", 5e5, "N", "maximum load")
    Vh = V("V_ht", 0.8, "-", "horizontal tail volume coefficient")
    mrat = V("m_ratio", 0.5, "-", "wing to tail lift slope ratio")
    cattach = V("c_attach", 2.5, "m", "HT chord where it mounts to the VT")
    Wht = V("W_ht", 1e4, "N", "HT system weight")
    CLfCG = V("C_L_ht_fCG", 0.3, "-", "HT C_L at maximum forward CG")

    # ---- constants --------------------------------------------------------
    etaht = C("eta_ht", 1.0, "-", "tail efficiency")
    tanLh = C("tan_Lambda_ht", tan(sweep_deg * pi / 180), "-",
              "tangent of horizontal tail sweep")
    amax = C("alpha_ht_max", 2.5, "-", "max angle of attack, HT")
    CLhmax = C("C_L_ht_max", 2.0, "-", "max HT lift coefficient")
    Cht = C("C_ht", 1.0, "-", "HT weight margin and sensitivity factor")
    fht = C("f_ht", 0.3, "-", "rudder etc. fractional weight")

    out = dict(AR_ht=ARht, S_ht=Sh, b_ht=bht, c_root_ht=croot, c_tip_ht=ctip,
               cbar_ht=chma, lambda_ht=taper, tau_ht=tau, p_ht=p, q_ht=q,
               y_cbar_ht=ymac, e_ht=e, dx_lead_ht=dxlead, dx_trail_ht=dxtrail,
               l_ht=lht, x_CG_ht=xcght, L_ht_max=Lmax, V_ht=Vh,
               m_ratio=mrat, c_attach=cattach, W_ht=Wht, eta_ht=etaht,
               C_L_ht_max=CLhmax, C_L_ht_fCG=CLfCG)

    cons = [
        dxlead + ymac * tanLh + 0.25 * chma >= lht,                 # [SP]
        dxlead + croot <= dxtrail,
        p >= 1 + 2 * taper,
        2 * q >= 1 + p,
        ymac == (bht / 3) * q / p,
        (2. / 3) * (1 + taper + taper ** 2) * croot / q == chma,    # [SP] SigEq
        taper == ctip / croot,
        Sh == bht * (croot + ctip) / 2,                             # [SP] SigEq
        # Oswald efficiency (Nita & Scholz). The source marks this one
        # "slightly slack" with reltol=0.2.
        fl >= (0.0524 * taper ** 4 - 0.15 * taper ** 3 + 0.1659 * taper ** 2
               - 0.0706 * taper + 0.0119),
        e * (1 + fl * ARht) <= 1,
        ARht == bht ** 2 / Sh,
        # TODO in source: make less arbitrary.
        taper >= 0.2,
        taper <= 1,
    ]

    # ---- structure ---------------------------------------------------------
    box = ht.group("box", prefix=f"{ht.prefix}box_")
    wb, wbcons = add_wingbox("horizontal_tail", AR=ARht, b=bht, S=Sh, p=p,
                             q=q, tau=tau, Lmax=Lmax, group=box)
    cons += wbcons
    out["box"] = wb

    cons += [
        wb["L_ht_rect"] >= Lmax / 2. * ctip * bht / Sh,
        # Note BOX_TAPER, not the planform taper -- see the module docstring.
        wb["L_ht_tri"] >= Lmax / 4. * (1 - BOX_TAPER) * croot * bht / Sh,
        Wht >= Cht * (wb["W_struct"] + wb["W_struct"] * fht),
    ]

    # ---- per-segment aerodynamics -----------------------------------------
    Dht = Vn("D_ht", 3e3, "N", "horizontal tail drag")
    Lh = Vn("L_ht", 2e4, "N", "horizontal tail downforce")
    Rec = Vn("Re_c_h", 1.5e7, "-", "cruise Reynolds number, HT")
    CLah = Vn("C_L_alpha_ht", 4.0, "-", "lift curve slope, HT")
    CLah0 = Vn("C_L_alpha_ht_0", 6.28, "-", "isolated lift curve slope, HT")
    CLh = Vn("C_L_ht", 0.2, "-", "lift coefficient, HT")
    CDh = Vn("C_D_ht", 0.008, "-", "HT drag coefficient")
    CD0h = Vn("C_D_0_ht", 0.006, "-", "HT parasitic drag coefficient")
    alphah = Vn("alpha_ht", 0.05, "-", "horizontal tail angle of attack")
    out.update(D_ht=Dht, L_ht=Lh, C_L_ht=CLh, C_L_alpha_ht=CLah,
               C_L_alpha_ht_0=CLah0, alpha_ht=alphah, C_D_ht=CDh)

    rho, Vinf, M, mu = state.rho, state.V, state.M, state.mu
    cons += [
        Lh == 0.5 * rho * Vinf ** 2 * Sh * CLh,
        CLh == CLah * alphah,
        alphah <= amax,
        # Thin airfoil theory, used as an approximation.
        CLah0 == 2 * 3.14,
        Dht == 0.5 * rho * Vinf ** 2 * Sh * CDh,
        CDh >= CD0h + CLh ** 2 / (pi * e * ARht),
        Rec == rho * Vinf * chma / mu,
        # Martin's TASOPT tail drag fit.
        CD0h ** 6.48983 >= (
            5.28751e-20 * Rec ** 0.900672 * tau ** 0.912222 * M ** 8.64547
            + 1.67605e-28 * Rec ** 0.350958 * tau ** 6.29187 * M ** 10.2559
            + 7.09757e-25 * Rec ** 1.39489 * tau ** 1.96239 * M ** 0.567066
            + 3.73076e-14 * Rec ** -2.57406 * tau ** 3.12793 * M ** 0.448159
            + 1.44343e-12 * Rec ** -3.91046 * tau ** 4.66279 * M ** 7.68852),
    ]

    return ht, cons
