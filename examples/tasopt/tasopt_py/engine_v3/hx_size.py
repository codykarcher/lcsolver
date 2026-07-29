"""Cross-flow heat exchanger sizing and off-design -- ``hxsize!``/``hxoper!``.

The correlations live in :mod:`tasopt_py.engine_v3.heat_exchanger`; this is
the routine that assembles them into a sized exchanger, and its off-design
twin.

Two directions, one geometry
----------------------------
:func:`hx_size` is told an **effectiveness** and works out how much surface
area delivers it. :func:`hx_operate` is told the **area** and works out what
effectiveness results. They share the resistance network and differ only in
which end of the epsilon-NTU relation is the unknown -- so a bug in the
network shows up in both, which is why the tests round-trip one into the
other.

The inner loop
--------------
Both iterate, and for a reason that is easy to miss: the overall resistance
depends on the **wall temperature**, through the ``(Tw/T)^-0.5`` property
correction on the coolant Nusselt number. The wall temperature depends on
how the resistance splits. So resistance and wall temperature are mutually
defined and the reference sweeps them to convergence -- fifteen passes with
a relative tolerance, "expect fast convergence".

There is a second coupling inside the same loop: the Nusselt number across
the bank depends on the **row count**, which is what the loop is solving
for. Both are converged together.

What is *not* iterated
----------------------
The outlet temperatures are fixed before the loop from the effectiveness,
and the mean-temperature properties with them. So the property evaluation
never sees the converged wall temperature -- only the heat-transfer
correction does. That is deliberate in the reference and worth knowing if
you compare against a code that re-evaluates.

Verified against TASOPT.jl; see ``tests/test_hx_size.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..gas.transport import gas_Pr
from ..gas.mixture import gassum, gas_tset
from ..gas.properties import gasfun
from ..cryo.material_data import MATERIALS
from .heat_exchanger import (NTU_from_effectiveness, effectiveness_from_NTU,
                             colburn_j_pipe, nusselt_staggered,
                             pressure_drop_staggered, SAFETY_FACTOR,
                             TMIN_TUBE)

__all__ = ["HXGas", "HXTubular", "hx_size", "hx_operate", "tube_geometry",
           "gas_tset_single",
           "hx_optimize", "hx_objective"]

N_ITER = 15
TOL = 1.0e-10


def gas_tset_single(igas: int, hspec: float, tguess: float) -> float:
    """Temperature of a single species at enthalpy ``hspec``.

    Newton on ``h(T) - hspec``. The reference prints and returns ``nothing``
    on non-convergence, which propagates as a ``MethodError`` somewhere
    later; this raises where the failure is.
    """
    t = tguess
    for _ in range(15):
        st = gasfun(igas, t)
        dt = -(st.h - hspec) / st.h_t
        if abs(dt) < 1.0e-6:
            return t
        t += dt
    raise ValueError(
        f"gas_tset_single did not converge for igas={igas}, "
        f"hspec={hspec:.6g}: last step dT = {dt:.3g}")


@dataclass
class HXGas:
    """The two streams through an exchanger.

    ``fluid_p`` is the process (hot) side and ``fluid_c`` the coolant.
    ``alpha_p`` is the process-side mass-fraction vector; ``igas_c`` the
    coolant species index (40 is hydrogen).
    """
    fluid_p: str = "air"
    fluid_c: str = "h2"
    alpha_p: list = field(default_factory=lambda: [0.7532, 0.2315, 0.0006,
                                                   0.0020, 0.0127])
    igas_c: int = 40
    mdot_p: float = 0.0
    mdot_c: float = 0.0
    eps: float = 0.0
    Tp_in: float = 0.0
    Tc_in: float = 0.0
    Mp_in: float = 0.0
    Mc_in: float = 0.0
    pp_in: float = 0.0
    pc_in: float = 0.0
    # outputs
    Tp_out: float = 0.0
    Tc_out: float = 0.0
    Dh_p: float = 0.0
    Dh_c: float = 0.0
    Dp_p: float = 0.0
    Dp_c: float = 0.0
    Pl_p: float = 0.0
    Pl_c: float = 0.0


@dataclass
class HXTubular:
    """Geometry and material of a staggered-tube exchanger."""
    is_concentric: bool = False
    has_recirculation: bool = False
    has_shaft: bool = False
    N_t: float = 0.0            # tubes per row
    n_stages: float = 0.0       # independent coolant stages
    n_passes: float = 0.0       # coolant passes (rows per stage)
    A_cs: float = 0.0           # process free-stream area, m^2
    l: float = 0.0              # tube length, m
    t: float = 0.0              # wall thickness, m
    tD_o: float = 0.0           # tube outer diameter, m
    xt_D: float = 0.0           # transverse pitch ratio
    xl_D: float = 0.0           # longitudinal pitch ratio
    Rfp: float = 0.0            # process-side fouling, m^2 K/W
    Rfc: float = 0.0            # coolant-side fouling, m^2 K/W
    D_i: float = 0.0            # core inner diameter, m
    material: str = "Al-2219-T87"
    Dpdes: float = 0.0          # design pressure difference, Pa
    maxL: float = 0.0

    @property
    def k_wall(self) -> float:
        return MATERIALS[self.material]["thermal_conductivity"]

    @property
    def YTS(self) -> float:
        return MATERIALS[self.material]["YTS"]

    @property
    def tD_i(self) -> float:
        return self.tD_o - 2.0 * self.t


def tube_geometry(K: float, dp: float, YTS: float) -> tuple:
    """``(t, tD_o)`` from the hoop-stress balance -- ``tubesize!``.

    ``K = pi b n_stages / (4 xt_D A_cc)`` links the outer diameter to the
    coolant area the tubes must supply. Given ``t = C tD_o`` from hoop
    stress, the diameter follows from a quadratic whose positive root is
    taken.

    The thickness expression has a **pole at C = 0.5**, i.e. where the design
    pressure reaches the yield stress; the reference does not check for it.
    """
    C = SAFETY_FACTOR * dp / (2.0 * YTS)
    if C >= 0.5:
        raise ValueError(
            f"design pressure difference {dp:.4g} Pa against a yield stress "
            f"of {YTS:.4g} Pa gives C = {C:.4f}; the hoop-stress expression "
            "has a pole at C = 0.5 and no tube thickness satisfies it")
    t = max(C * K / (K - 2.0 * K * C) ** 2, TMIN_TUBE)
    tD_o = (4.0 * K * t + math.sqrt(8.0 * K * t + 1.0) + 1.0) / (2.0 * K)
    return t, tD_o


def _inlet_state(g: HXGas):
    """Process- and coolant-side inlet thermodynamic state."""
    sp = gassum(g.alpha_p, len(g.alpha_p), g.Tp_in)
    sc = gasfun(g.igas_c, g.Tc_in)
    return sp, sc


def _network(g: HXGas, x: HXTubular, Vc_m, nu_c_m, Pr_c_m, k_c_m, Tc_m,
             G, mu_p_m, Pr_p_m, k_p_m, Tp_m, xtm_D, Tw_c, Tw_p, N_L):
    """One pass of the resistance network. ``(RA, h_c, Cf, Tw_p, Tw_c)``.

    The five resistances in series, per unit *outer* area:
    coolant film (scaled by the diameter ratio), coolant fouling, wall
    conduction, process fouling, process film.
    """
    tD_o, tD_i, t = x.tD_o, x.tD_i, x.t
    Re_D_c = Vc_m * tD_i / nu_c_m
    jc, Cf = colburn_j_pipe(Re_D_c)
    Nu_cm = Re_D_c * jc * Pr_c_m ** (1.0 / 3.0)
    # Kays and London (1998) Eq. 4.1 -- gas coolants only.
    Nu_c = Nu_cm * (Tw_c / Tc_m) ** -0.5
    h_c = Nu_c * k_c_m / tD_i

    Re_D_p = G * tD_o / mu_p_m
    Nu_p = nusselt_staggered(Re_D_p, Pr_p_m, N_L, xtm_D, x.xl_D)
    h_p = Nu_p * k_p_m / tD_o

    RA = (1.0 / (h_c * (tD_i / tD_o)) + x.Rfp + tD_o / tD_i * x.Rfc
          + t / x.k_wall + 1.0 / h_p)
    Tw_p = Tc_m + ((Tp_m - Tc_m) / RA) * (
        1.0 / (h_c * (tD_i / tD_o)) + tD_o / tD_i * x.Rfc + x.Rfp
        + t / x.k_wall)
    Tw_c = Tc_m + ((Tp_m - Tc_m) / RA) * (1.0 / (h_c * (tD_i / tD_o)))
    return RA, h_c, Cf, Tw_p, Tw_c


def _common(g: HXGas, x: HXTubular, Q, C_p, C_c, cp_p_in, cp_c_in, Rp, Rc):
    """Everything both directions share once ``Q`` is known.

    Returns the mean-state bundle plus the free-stream and coolant areas.
    """
    sp_in, sc_in = _inlet_state(g)
    hp_out = sp_in.h - Q / g.mdot_p
    hc_out = sc_in.h + Q / g.mdot_c
    Tp_out = gas_tset(g.alpha_p, len(g.alpha_p), hp_out, g.Tp_in - Q / C_p)
    Tc_out = gas_tset_single(g.igas_c, hc_out, g.Tc_in + Q / C_c)

    gam_p = cp_p_in / (cp_p_in - Rp)
    rho_p_in = g.pp_in / (Rp * g.Tp_in)
    Vp_in = g.Mp_in * math.sqrt(gam_p * Rp * g.Tp_in)
    gam_c = cp_c_in / (cp_c_in - Rc)
    rho_c_in = g.pc_in / (Rc * g.Tc_in)
    Vc_in = g.Mc_in * math.sqrt(gam_c * Rc * g.Tc_in)

    Tp_m = (Tp_out + g.Tp_in) / 2.0
    Tc_m = (Tc_out + g.Tc_in) / 2.0
    _, Pr_p_m, _, _, mu_p_m, k_p_m = gas_Pr(g.fluid_p, Tp_m)
    rho_p_m = g.pp_in / (Rp * Tp_m)
    _, Pr_c_m, _, _, mu_c_m, k_c_m = gas_Pr(g.fluid_c, Tc_m)
    rho_c_m = g.pc_in / (Rc * Tc_m)
    Vc_m = rho_c_in * Vc_in / rho_c_m
    nu_c_m = mu_c_m / rho_c_m

    A_cs = g.mdot_p / (rho_p_in * Vp_in)
    A_cc = g.mdot_c / (rho_c_in * Vc_in)
    return dict(Tp_out=Tp_out, Tc_out=Tc_out, hp_out=hp_out, hc_out=hc_out,
                hp_in=sp_in.h, hc_in=sc_in.h, Tp_m=Tp_m, Tc_m=Tc_m,
                Pr_p_m=Pr_p_m, mu_p_m=mu_p_m, k_p_m=k_p_m, rho_p_m=rho_p_m,
                Pr_c_m=Pr_c_m, k_c_m=k_c_m, rho_c_m=rho_c_m, Vc_m=Vc_m,
                nu_c_m=nu_c_m, A_cs=A_cs, A_cc=A_cc)


def _pressure_drops(g: HXGas, x: HXTubular, m, Ah, Cf, Tw_p, Tw_c,
                    G, N_L, xtm_D):
    """Process- and coolant-side pressure drops and their power losses."""
    N_tubes_tot = x.N_t * x.n_passes * x.n_stages
    L = N_L * x.xl_D * x.tD_o
    NFV = m["A_cs"] * L - N_tubes_tot * math.pi * x.tD_o ** 2 * x.l / 4.0
    Dv = 4.0 * NFV / Ah
    Re_Dv = Dv * G / m["mu_p_m"]

    _, _, _, _, mu_p_w, _ = gas_Pr(g.fluid_p, Tw_p)
    mu_ratio = m["mu_p_m"] / mu_p_w

    if Re_Dv < 0.0:
        # Tubes fill more than the free volume: the bank is blocked.
        Dp_p = math.inf
    else:
        Dp_p = pressure_drop_staggered(Re_Dv, G, L, m["rho_p_m"], Dv,
                                       x.tD_o, xtm_D, x.xl_D, mu_ratio)
    Pl_p = Dp_p * g.mdot_p / m["rho_p_m"]

    tau_w = m["rho_c_m"] * m["Vc_m"] ** 2 / 2.0 * Cf
    A_s_c = math.pi * x.tD_i * x.l * x.n_passes
    A_cs_c = math.pi * x.tD_i ** 2 / 4.0
    Dp_c = tau_w * A_s_c / A_cs_c * (Tw_c / m["Tc_m"]) ** -0.1
    Pl_c = Dp_c * g.mdot_c / m["rho_c_m"]
    return Dp_p, Pl_p, Dp_c, Pl_c


def hx_size(g: HXGas, x: HXTubular) -> None:
    """Size an exchanger for the effectiveness in ``g.eps``. Mutates both.

    The effectiveness is **clipped** to 99% of what the flow arrangement can
    reach; ``g.eps`` is written back, so compare it against what you asked
    for.
    """
    sp_in, sc_in = _inlet_state(g)
    Rp, cp_p_in = sp_in.r, sp_in.cp
    Rc, cp_c_in = sc_in.r, sc_in.cp
    C_p = g.mdot_p * cp_p_in
    C_c = g.mdot_c * cp_c_in
    C_min = min(C_c, C_p)

    NTU, g.eps = NTU_from_effectiveness(g.eps, C_c, C_p)
    Q = g.eps * C_min * (g.Tp_in - g.Tc_in)

    m = _common(g, x, Q, C_p, C_c, cp_p_in, cp_c_in, Rp, Rc)
    x.A_cs = m["A_cs"]

    if x.is_concentric:
        b = math.pi * x.D_i          # inner circumference
    else:
        b = m["A_cs"] / x.l          # rectangular duct side

    K = math.pi * b * x.n_stages / (4.0 * x.xt_D * m["A_cc"])
    x.t, x.tD_o = tube_geometry(K, x.Dpdes, x.YTS)
    x.N_t = 4.0 * m["A_cc"] / (math.pi * x.tD_i ** 2 * x.n_stages)

    xtm_D = m["A_cs"] / (x.N_t * x.tD_o * x.l)
    A_min = m["A_cs"] - x.N_t * x.l * x.tD_o
    G = g.mdot_p / A_min

    Tw = (m["Tp_out"] + g.Tp_in + m["Tc_out"] + g.Tc_in) / 4.0
    Tw_p = Tw_c = Tw
    x.n_passes = n_prev = 4.0
    Ah = Cf = 0.0
    for _ in range(N_ITER):
        N_L = x.n_passes * x.n_stages
        RA, _, Cf, Tw_p, Tw_c = _network(
            g, x, m["Vc_m"], m["nu_c_m"], m["Pr_c_m"], m["k_c_m"], m["Tc_m"],
            G, m["mu_p_m"], m["Pr_p_m"], m["k_p_m"], m["Tp_m"], xtm_D,
            Tw_c, Tw_p, N_L)
        Ah = NTU * C_min * RA
        x.n_passes = Ah / (x.N_t * x.n_stages * math.pi * x.tD_o * x.l)
        if abs((n_prev - x.n_passes) / x.n_passes) < TOL:
            break
        n_prev = x.n_passes

    N_L = x.n_passes * x.n_stages
    Dp_p, Pl_p, Dp_c, Pl_c = _pressure_drops(
        g, x, m, Ah, Cf, Tw_p, Tw_c, G, N_L, xtm_D)

    g.Tp_out, g.Tc_out = m["Tp_out"], m["Tc_out"]
    g.Dh_p = m["hp_out"] - m["hp_in"]
    g.Dh_c = m["hc_out"] - m["hc_in"]
    g.Dp_p, g.Dp_c, g.Pl_p, g.Pl_c = Dp_p, Dp_c, Pl_p, Pl_c


def hx_operate(g: HXGas, x: HXTubular) -> float:
    """Run a sized exchanger off design. Mutates ``g``; returns the wall
    temperature.

    The geometry is fixed, so the **area** is known and the effectiveness is
    the unknown -- the reverse of :func:`hx_size`. But the resistance depends
    on the mean temperatures, the mean temperatures depend on the heat load,
    and the heat load depends on the resistance, so this is a fixed point in
    ``Q`` rather than a single evaluation.

    Two things differ from :func:`hx_size` beyond the direction, and neither
    is cosmetic:

    * The coolant velocity comes from the **tube area** here
      (``mdot / (N_ways rho A_tube)``) rather than from conservation of mass
      off the inlet velocity. Off design there is no inlet velocity to
      conserve from -- the geometry is what is known.
    * No power losses are computed. The reference sets the pressure drops
      but leaves ``Pl_p``/``Pl_c`` untouched, so they keep whatever they
      held; this leaves them alone too rather than inventing values.

    The starting guess is 0.95, which is above the achievable maximum for
    most capacity-rate ratios -- so the first pass overestimates the heat
    load and the loop walks down.
    """
    sp_in, sc_in = _inlet_state(g)
    Rp, cp_p_in = sp_in.r, sp_in.cp
    Rc, cp_c_in = sc_in.r, sc_in.cp

    rho_p_in = g.pp_in / (Rp * g.Tp_in)
    rho_c_in = g.pc_in / (Rc * g.Tc_in)
    Vp_in = g.mdot_p / (rho_p_in * x.A_cs)

    N_hyd_ways = x.N_t * x.n_stages
    A_cs_tube = math.pi * x.tD_i ** 2 / 4.0
    A_min = x.A_cs - x.N_t * x.l * x.tD_o
    G = rho_p_in * Vp_in * x.A_cs / A_min

    N_tubes_tot = x.N_t * x.n_passes * x.n_stages
    N_L = x.n_passes * x.n_stages
    L = N_L * x.xl_D * x.tD_o
    Ah = N_tubes_tot * math.pi * x.tD_o * x.l
    xtm_D = x.A_cs / (x.N_t * x.tD_o * x.l)

    C_p = g.mdot_p * cp_p_in
    C_c = g.mdot_c * cp_c_in
    C_min = min(C_c, C_p)

    eps = 0.95
    Q = Qprev = eps * C_min * (g.Tp_in - g.Tc_in)
    Tp_out = g.Tp_in - Q / C_p
    Tc_out = g.Tc_in + Q / C_c
    Tw_p = Tw_c = (Tp_out + g.Tp_in + Tc_out + g.Tc_in) / 4.0

    rho_p_m = mu_p_m = Vc_m = rho_c_m = Cf = 0.0
    hp_out = hc_out = 0.0

    for _ in range(20):
        Qmax = C_min * (g.Tp_in - g.Tc_in)
        Tp_m = (Tp_out + g.Tp_in) / 2.0
        Tc_m = (Tc_out + g.Tc_in) / 2.0
        _, Pr_p_m, _, _, mu_p_m, k_p_m = gas_Pr(g.fluid_p, Tp_m)
        rho_p_m = g.pp_in / (Rp * Tp_m)
        _, Pr_c_m, _, _, mu_c_m, k_c_m = gas_Pr(g.fluid_c, Tc_m)
        rho_c_m = g.pc_in / (Rc * Tc_m)
        nu_c_m = mu_c_m / rho_c_m
        Vc_m = g.mdot_c / (N_hyd_ways * rho_c_m * A_cs_tube)

        Re_D_p = G * x.tD_o / mu_p_m
        h_p = nusselt_staggered(Re_D_p, Pr_p_m, N_L, xtm_D,
                                x.xl_D) * k_p_m / x.tD_o
        Re_D_c = Vc_m * x.tD_i / nu_c_m
        jc, Cf = colburn_j_pipe(Re_D_c)
        Nu_c = (Re_D_c * jc * Pr_c_m ** (1.0 / 3.0)
                * (Tw_c / Tc_m) ** -0.5)
        h_c = Nu_c * k_c_m / x.tD_i

        RA = (1.0 / (h_c * (x.tD_i / x.tD_o)) + x.Rfp
              + x.tD_o / x.tD_i * x.Rfc + x.t / x.k_wall + 1.0 / h_p)
        NTU = Ah / (C_min * RA)
        eps = effectiveness_from_NTU(NTU, C_c, C_p)
        Q = eps * Qmax

        Tw_p = Tc_m + ((Tp_m - Tc_m) / RA) * (
            1.0 / (h_c * (x.tD_i / x.tD_o)) + x.tD_o / x.tD_i * x.Rfc
            + x.Rfp + x.t / x.k_wall)
        Tw_c = Tc_m + ((Tp_m - Tc_m) / RA) * (1.0 / (h_c * (x.tD_i / x.tD_o)))

        hp_out = sp_in.h - Q / g.mdot_p
        hc_out = sc_in.h + Q / g.mdot_c
        Tp_out = gas_tset(g.alpha_p, len(g.alpha_p), hp_out,
                          g.Tp_in - Q / C_p)
        Tc_out = gas_tset_single(g.igas_c, hc_out, g.Tc_in + Q / C_c)

        if abs((Q - Qprev) / Q) < TOL:
            break
        Qprev = Q

    Tc_m = (Tc_out + g.Tc_in) / 2.0
    Tp_m = (Tp_out + g.Tp_in) / 2.0
    _, _, _, _, mu_p_w, _ = gas_Pr(g.fluid_p, Tw_p)
    _, _, _, _, mu_p_m, _ = gas_Pr(g.fluid_p, Tp_m)

    NFV = x.A_cs * L - N_tubes_tot * math.pi * x.tD_o ** 2 * x.l / 4.0
    Dv = 4.0 * NFV / Ah
    Re_Dv = Dv * G / mu_p_m
    if Re_Dv < 0.0:
        Dp_p = math.inf
    else:
        Dp_p = pressure_drop_staggered(Re_Dv, G, L, rho_p_m, Dv, x.tD_o,
                                       xtm_D, x.xl_D, mu_p_m / mu_p_w)

    tau_w = rho_c_m * Vc_m ** 2 / 2.0 * Cf
    A_s_c = math.pi * x.tD_i * x.l * x.n_passes
    Dp_c = (tau_w * A_s_c / (math.pi * x.tD_i ** 2 / 4.0)
            * (Tw_c / Tc_m) ** -0.1)

    g.eps = eps
    g.Tp_out, g.Tc_out = Tp_out, Tc_out
    g.Dh_p = hp_out - sp_in.h
    g.Dh_c = hc_out - sc_in.h
    g.Dp_p, g.Dp_c = Dp_p, Dp_c
    return Tw_p


# --------------------------------------------------------------------------
# Design optimisation -- hxoptim!
# --------------------------------------------------------------------------

#: The four design variables, in the reference's order:
#: ``100*Mc_in``, ``n_stages``, ``xt_D``, ``l``.
OPTIM_LOWER = (1.0e-9, 1.0, 1.0)
OPTIM_UPPER = (30.0, 20.0, 6.0)

#: Constraint limits, all as ratios against unity.
MIN_PASSES, MAX_PASSES = 1.0, 20.0
MIN_TUBES, MAX_TUBES = 1.0, 200.0
#: Pressure drop may not exceed this fraction of the inlet pressure.
DP_THRESHOLD = 0.5


def _hx_constraints(g: HXGas, x: HXTubular) -> list:
    """The seven inequality constraints, in ``c <= 0`` form.

    All are written as ratios rather than differences, so they are scaled
    alike and a single tolerance means the same thing for each.

    Note how the reference gets these: the constraint closures **ignore
    their argument** and read the mutable structs, relying on NLopt having
    called the objective at the same point first. That is true of NLopt's
    COBYLA but is not a documented contract, and it is why this port
    evaluates the objective and the constraints from one cached solve
    instead.
    """
    return [MIN_PASSES / x.n_passes - 1.0,
            x.n_passes / MAX_PASSES - 1.0,
            MIN_TUBES / x.N_t - 1.0,
            x.N_t / MAX_TUBES - 1.0,
            g.Dp_p / (DP_THRESHOLD * g.pp_in) - 1.0,
            g.Dp_c / (DP_THRESHOLD * g.pc_in) - 1.0,
            (x.n_passes * x.n_stages * x.xl_D * x.tD_o) / x.maxL - 1.0]


def hx_objective(xv, g: HXGas, x: HXTubular) -> float:
    """Total pumping power, W -- the sum of both streams' pressure losses.

    Sets the four design variables, sizes, and returns ``Pl_p + Pl_c``. A
    failed solve returns a large value, so the optimiser walks away from it
    rather than stopping. The reference returns ``Inf``; a finite penalty
    behaves better in a derivative-free method that differences the
    objective, and the optimum is unaffected because feasible points are
    many orders below it.
    """
    g.Mc_in = xv[0] / 100.0
    x.n_stages, x.xt_D, x.l = xv[1], xv[2], xv[3]
    try:
        hx_size(g, x)
    except (ValueError, ZeroDivisionError, OverflowError):
        return 1.0e30
    if not math.isfinite(g.Pl_p + g.Pl_c):
        return 1.0e30
    return g.Pl_p + g.Pl_c


def hx_optimize(g: HXGas, x: HXTubular, initial_x, tol: float = 1.0e-4):
    """Choose ``Mc_in``, ``n_stages``, ``xt_D`` and ``l`` to minimise
    pumping power. Mutates both; returns the design vector.

    The lower bound on tube length is geometric: in a concentric duct a tube
    cannot be shorter than the annulus it spans, ``(D_o - D_i)/2``. In a
    rectangular one it comes from an aspect-ratio cap instead.

    **On reproducing the reference's answer.** The reference uses NLopt's
    COBYLA; this uses SciPy's, which descends from the same Powell code but
    is not the same implementation. On a problem whose objective is
    discontinuous -- a failed solve is a cliff, not a slope -- the two need
    not stop at the same point. The reference's own test acknowledges this
    and checks the *objective value* rather than the design vector, which is
    what ``tests/test_hx_size.py`` does too.
    """
    from scipy.optimize import minimize

    sp = gassum(g.alpha_p, len(g.alpha_p), g.Tp_in)
    gam = sp.cp / (sp.cp - sp.r)
    rho_p_in = g.pp_in / (sp.r * g.Tp_in)
    Vp_in = g.Mp_in * math.sqrt(gam * sp.r * g.Tp_in)
    A_cs = g.mdot_p / (rho_p_in * Vp_in)

    if x.is_concentric:
        D_o = math.sqrt(4.0 * (A_cs + math.pi * x.D_i ** 2 / 4.0) / math.pi)
        lmin = (D_o - x.D_i) / 2.0
    else:
        lmin = math.sqrt(A_cs / 10.0)      # AR_max = 10

    bounds = list(zip(OPTIM_LOWER, OPTIM_UPPER)) + [(lmin, None)]

    cache = {}

    def solve(xv):
        key = tuple(xv)
        if key not in cache:
            f = hx_objective(xv, g, x)
            cache.clear()
            cache[key] = (f, _hx_constraints(g, x) if f < 1.0e29
                          else [1.0e3] * 7)
        return cache[key]

    cons = [{"type": "ineq", "fun": (lambda xv, i=i: -solve(xv)[1][i])}
            for i in range(7)]

    res = minimize(lambda xv: solve(xv)[0], list(initial_x), method="COBYLA",
                   bounds=bounds, constraints=cons,
                   options={"rhobeg": 0.1, "tol": 1.0e-9, "maxiter": 5000,
                            "catol": tol})

    g.Mc_in = res.x[0] / 100.0
    x.n_stages, x.xt_D, x.l = res.x[1], res.x[2], res.x[3]
    return res.x
