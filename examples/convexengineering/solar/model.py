"""Solar-electric long-endurance aircraft (GP).

Source model
------------
``solar/solar.py`` in https://github.com/convexengineering/solar

Paper
-----
    M. Burton and W. Hoburg, "Solar and Gas Powered Long-Endurance Unmanned
    Aircraft Sizing via Geometric Programming", J. Aircraft 55(1), 2018.

The case rebuilt here is ``Npod=0``, GP (not SP), latitude 20, day 355
(winter solstice — the sizing case, shortest day). Objective is total
aircraft weight.

What makes it a solar aircraft
------------------------------
Everything else is a conventional airframe sizing problem; three constraints
carry the solar physics, and they are what close the design:

    ESirr        >= ESday + E/(etacharge etasolar Ssolar)
    E etadischarge >= Poper tnight + EStwi etasolar Ssolar
    Poper        == PSmin Ssolar etasolar

Read in order: the day's available irradiance must cover both the day's
flying and recharging the battery; the battery must then carry the night plus
twilight; and the operating power ties back to the minimum solar power the
wing area can collect. Battery energy and solar-cell area are the two
currencies, and both cost weight — that is the whole tradeoff.

Configuration differences from the standalone subsystems
--------------------------------------------------------
This is *not* the same wing as ``../wing/``. The solar aircraft reconfigures
its subsystems substantially, and porting it means honouring those overrides:

* **Materials are overridden.** ``cfrpud`` becomes rho=1.5 g/cm^3, E=200 GPa,
  sigma=1500 MPa (the library defaults are 1.6, 137, 1700); ``cfrpfabric``
  becomes rho=1.3, E=40 GPa, sigma=300 MPa, tau=80 MPa; ``foamhd`` becomes
  0.03 g/cm^3.
* **The wing has 20 nodes**, not 5.
* **No foam core** (``fillModel = None``) and the skin is
  ``WingSecondStruct`` — a flat areal density (0.35 kg/m^2 for the wing,
  0.4 for the tails) rather than the gauge-and-torsion skin model.
* **The tail boom is a box spar**, not a tube (``TailBoom.__bases__`` is
  reassigned), with a secondary areal weight of 0.15 kg/m^2.

Getting any of these wrong reproduces the failure mode already recorded in
DISCREPANCIES.md #8 — a structural constraint off by a constant, and the
optimizer buys span it has not paid for.

Status: WIP — does not converge
-------------------------------
The model is written and detected as a GP, but neither cvxopt nor IPOPT
converges on it. What has been established:

* **Not the wind fit.** Its exponents reach 169, which looked like the
  obvious suspect, but replacing it with a plain bound changes nothing.
* **Not the discretization.** Nwing of 5, 8, 12 and 20 all fail at the same
  transformed objective (6.1, i.e. ~446 lbf — close to the reference 436.4,
  so it gets near the answer and then stalls).
* **Not model size.** A stripped 13-variable core with only the energy
  balance, steady flight and the weight buildup fails the same way, under
  *both* solvers.
* **Not monomial equalities.** The K=1 fits emit equalities; a two-variable
  test model with one converges fine.

What it is: **under-bounded variables producing a degenerate optimum.** In
the stripped core ``V`` carries only a lower bound and does not enter the
objective, so every ``V >= 20`` is optimal — a ray of optima rather than a
point, which interior-point methods handle badly. The real model is supposed
to bound ``V`` through the drag/power coupling (faster flight costs power,
which costs battery, which costs weight), and the placeholders below break
that chain.

The placeholders are the actual defect, not incidental:

* ``cdw >= 0.0075 + CL^2/(pi AR e)`` — the source uses an XFOIL fit of the
  DAI1336a airfoil, which is not transcribed here. Without it, profile drag
  does not respond to Reynolds number or thickness.
* ``cda >= 0.005`` — should be the sum of tail and boom drag areas over wing
  area, which is what ties the empennage into the drag.
* ``Wemp``, ``Wmotor``, ``Wprop`` fixed at nominal values instead of being
  sized by the empennage, motor and propeller models.

Fixing these means composing the real empennage, motor and propeller
submodels rather than stubbing them, and transcribing the DAI1336a polar
fit. That is the next step; the energy balance and weight buildup here are
believed correct and are the parts worth keeping.
"""
from __future__ import annotations

import numpy as np
from pyomo.environ import units

from edi import Formulation
from environment import ENVIRONMENT, fit_constraints

G = 9.81

# --- materials AS OVERRIDDEN BY solar.py (not the gplibrary defaults) ------
CFRPUD = dict(rho=1.5, E=200e9, tmin=0.1, sigma=1500e6)   # g/cm^3, Pa, mm, Pa
CFRPFABRIC = dict(rho=1.3, E=40e9, tmin=0.1, sigma=300e6, tau=80e6)
FOAMHD = dict(rho=0.03)


def planform_constants(N: int, lam: float):
    eta = np.linspace(0.0, 1.0, N)
    cbar = np.array([2.0 / (1 + lam) * (1 + (lam - 1) * e) for e in eta])
    cbave = (cbar[:-1] + cbar[1:]) / 2.0
    deta = np.diff(eta)
    lami = cbar[1:] / cbar[:-1]
    maci = 2.0 / 3 * cbar[:-1] * (1 + lami + lami**2) / (1 + lami)
    num = sum((cbar[i] + cbar[i + 1]) / 2 * maci[i] * deta[i] for i in range(len(deta)))
    den = sum((cbar[i] + cbar[i + 1]) / 2 * deta[i] for i in range(len(deta)))
    return eta, cbar, cbave, deta, num / den / cbar[0]


def _box_spar(f, tag, N, cave, tau_expr, b, g_u, wlim=0.15):
    """BoxSpar over N-1 segments. Returns (vars, constraints)."""
    Nseg = N - 1
    _, _, _, deta, _ = planform_constants(N, 0.5)
    I  = f.Variable(name=f"{tag}_I",  guess=1e-6, units="m^4", size=Nseg, description="spar inertia")
    Sy = f.Variable(name=f"{tag}_Sy", guess=1e-5, units="m^3", size=Nseg, description="section modulus")
    dm = f.Variable(name=f"{tag}_dm", guess=0.2,  units="kg",  size=Nseg, description="segment mass")
    w  = f.Variable(name=f"{tag}_w",  guess=0.1,  units="in",  size=Nseg, description="spar width")
    t  = f.Variable(name=f"{tag}_t",  guess=0.02, units="in",  size=Nseg, description="cap thickness")
    ts = f.Variable(name=f"{tag}_tshear", guess=0.02, units="in", size=Nseg, description="shear web thickness")
    tc = f.Variable(name=f"{tag}_tcore",  guess=0.02, units="in", size=Nseg, description="core thickness")
    hin = f.Variable(name=f"{tag}_hin", guess=0.5, units="in", size=Nseg, description="height between caps")
    W  = f.Variable(name=f"{tag}_Wspar", guess=5.0, units="lbf", description="spar weight")

    rho_ud = CFRPUD["rho"] * units.g / units.cm**3
    rho_fb = CFRPFABRIC["rho"] * units.g / units.cm**3
    rho_fm = FOAMHD["rho"] * units.g / units.cm**3
    mfac_s, tcoret = 0.97, 0.02
    cons = []
    for i in range(Nseg):
        cav = cave[i] if not isinstance(cave, (int, float)) else cave
        cons += [
            I[i] / mfac_s <= w[i] * t[i] * hin[i]**2,
            dm[i] >= (rho_ud * 4 * w[i] * t[i]
                      + 4 * ts[i] * rho_fb * (hin[i] + w[i])
                      + 2 * rho_fm * tc[i] * (w[i] + hin[i])) * b / 2 * deta[i],
            w[i] <= wlim * cav,
            cav * tau_expr >= hin[i] + 4 * t[i] + 2 * tc[i],
            t[i] >= CFRPUD["tmin"] * units.mm,
            Sy[i] * (hin[i] / 2 + 2 * t[i] + tc[i]) <= I[i],
            ts[i] >= CFRPFABRIC["tmin"] * units.mm,
            tc[i] >= tcoret * cav * tau_expr,
        ]
    cons.append(W >= 2 * sum(dm[i] for i in range(Nseg)) * g_u)
    return dict(I=I, Sy=Sy, w=w, t=t, W=W, hin=hin), cons


def _beam(f, tag, N, deta, b, I, E, sigma, Sy, load_expr, kappa=0.2,
          tiny=1e-2):
    """Spanwise beam: shear -> moment -> angle -> deflection. Same chain as
    ../wing/, which is verified against its reference."""
    Sh = f.Variable(name=f"{tag}_S",  guess=100.0, units="N",   size=N, description="shear")
    M  = f.Variable(name=f"{tag}_M",  guess=100.0, units="N*m", size=N, description="moment")
    th = f.Variable(name=f"{tag}_th", guess=0.05,  units="-",   size=N, description="angle")
    wd = f.Variable(name=f"{tag}_w",  guess=0.1,   units="m",   size=N, description="deflection")
    q  = f.Variable(name=f"{tag}_q",  guess=50.0,  units="N/m", size=N, description="distributed load")
    cons = []
    for i in range(N - 1):
        cons += [
            Sh[i] >= Sh[i + 1] + 0.5 * deta[i] * (b / 2) * (q[i] + q[i + 1]),
            M[i] >= M[i + 1] + 0.5 * deta[i] * (b / 2) * (Sh[i] + Sh[i + 1]),
            th[i + 1] >= th[i] + 0.5 * deta[i] * (b / 2) * (M[i + 1] + M[i]) / E / I[i],
            wd[i + 1] >= wd[i] + 0.5 * deta[i] * (b / 2) * (th[i + 1] + th[i]),
        ]
    cons += [Sh[N - 1] >= tiny * units.N, M[N - 1] >= tiny * units.N * units.m,
             th[0] >= tiny, wd[0] >= tiny * units.m,
             wd[N - 1] / (b / 2) <= kappa]
    for i in range(N - 1):
        cons.append(sigma >= M[i] / Sy[i])
    for i in range(N):
        cons.append(q[i] >= load_expr(i))
    return dict(S=Sh, M=M, th=th, w=wd, q=q), cons


def build(latitude: int = 20, Nwing: int = 20, Ntail: int = 5,
          Nboom: int = 5) -> Formulation:
    """Solar aircraft, Npod=0, GP, at the given latitude."""
    env = ENVIRONMENT[latitude]
    f = Formulation()
    pi = np.pi
    g_u = G * units.m / units.s**2

    eta_w, cbar_w, cbave_w, deta_w, cbarmac_w = planform_constants(Nwing, 0.5)
    eta_t, cbar_t, cbave_t, deta_t, cbarmac_t = planform_constants(Ntail, 0.8)

    # ---------------- aircraft-level constants ----------------------------
    Wpay = f.Constant(name="Wpay", value=11.0, units="lbf", description="payload weight")
    Wavn = f.Constant(name="Wavn", value=22.0, units="lbf", description="avionics weight")
    Nprop = 4.0

    # ---------------- wing planform ---------------------------------------
    Sw   = f.Variable(name="Sw",   guess=200.0, units="ft^2", description="wing area")
    ARw  = f.Variable(name="ARw",  guess=25.0,  units="-",    description="wing aspect ratio")
    bw   = f.Variable(name="bw",   guess=70.0,  units="ft",   description="wing span")
    crootw = f.Variable(name="crootw", guess=4.0, units="ft", description="wing root chord")
    cmacw  = f.Variable(name="cmacw",  guess=3.0, units="ft", description="wing MAC")
    cavew  = f.Variable(name="cavew",  guess=3.0, units="ft", size=Nwing - 1, description="wing mid chord")
    tauw   = f.Variable(name="tauw",   guess=0.12, units="-", description="wing thickness ratio")
    Wwingtot = f.Variable(name="Wwing", guess=60.0, units="lbf", description="wing group weight")
    Wwstruct = f.Variable(name="Wwstruct", guess=40.0, units="lbf", description="wing structural weight")
    Wwskin = f.Variable(name="Wwskin", guess=15.0, units="lbf", description="wing skin weight")

    # ---------------- power system ----------------------------------------
    Ssolar = f.Variable(name="Ssolar", guess=150.0, units="ft^2", description="solar cell area")
    Wsolar = f.Variable(name="Wsolar", guess=10.0,  units="lbf",  description="solar cell weight")
    Ebatt  = f.Variable(name="E",      guess=3.0e5, units="kJ",   description="battery energy")
    Wbatt  = f.Variable(name="Wbatt",  guess=150.0, units="lbf",  description="battery weight")
    Volbatt = f.Variable(name="Volbatt", guess=0.1, units="m^3",  description="battery volume")

    # ---------------- overall ----------------------------------------------
    Wtotal = f.Variable(name="Wtotal", guess=430.0, units="lbf", description="aircraft weight")
    Wcent  = f.Variable(name="Wcent",  guess=100.0, units="lbf", description="center weight")
    Wland  = f.Variable(name="Wland",  guess=9.0,   units="lbf", description="landing gear weight")
    Wemp   = f.Variable(name="Wemp",   guess=10.0,  units="lbf", description="empennage weight")
    Wmotor = f.Variable(name="Wmotor", guess=2.0,   units="lbf", description="motor weight")
    Wprop  = f.Variable(name="Wprop",  guess=1.0,   units="lbf", description="propeller weight")

    f.Objective(Wtotal)

    cons = []

    # ---------------- wing structure ---------------------------------------
    cons += [
        bw**2 == Sw * ARw,
        crootw == Sw / bw * cbar_w[0],
        cmacw == crootw * cbarmac_w,
    ]
    for i in range(Nwing - 1):
        cons.append(cavew[i] == cbave_w[i] * Sw / bw)
    # WingSecondStruct: flat areal density, not a gauge model
    cons.append(Wwskin >= 0.35 * units.kg / units.m**2 * Sw * g_u)

    spar, spar_cons = _box_spar(f, "wing", Nwing, cavew, tauw, bw, g_u)
    cons += spar_cons
    # mfac = 1.0 for the solar wing, and there is no foam core
    cons.append(Wwstruct >= Wwskin + spar["W"])

    # ---------------- power system -----------------------------------------
    cons += [
        Ssolar <= Sw,                                    # mfsolar = 1
        Wsolar >= 0.3 * units.kg / units.m**2 * Ssolar * g_u,
        # minSOC/hbatt/etaRTE/etapack
        Wbatt >= Ebatt * 1.03 / (350 * units.W * units.hr / units.kg)
                 / 0.95 / 0.85 * g_u,
        Volbatt >= Ebatt / (800 * units.W * units.hr / units.liter),
        Volbatt <= cmacw**2 * 0.5 * tauw * bw,           # Npod = 0
    ]

    # ---------------- flight state -----------------------------------------
    V     = f.Variable(name="V",     guess=30.0,  units="m/s",    description="true airspeed")
    Vwind = f.Variable(name="Vwind", guess=25.0,  units="m/s",    description="wind speed")
    rho   = f.Variable(name="rho",   guess=0.4,   units="kg/m^3", description="air density")
    Vne   = f.Variable(name="Vne",   guess=45.0,  units="m/s",    description="never-exceed speed")
    qne   = f.Variable(name="qne",   guess=400.0, units="kg/s^2/m", description="never-exceed dynamic pressure")
    PSmin = f.Variable(name="PSmin", guess=15.0,  units="W/m^2",  description="minimum solar power")
    ESday = f.Variable(name="ESday", guess=200.0, units="W*hr/m^2", description="daytime solar energy")
    EStwi = f.Variable(name="EStwi", guess=1.0,   units="W*hr/m^2", description="twilight battery energy")

    ESvar = 1.0 * units.W * units.hr / units.m**2
    PSvar = 1.0 * units.W / units.m**2
    cons += [
        V >= Vwind,                                       # mfac = 1
        Vne == 1.4 * V,
        qne == 0.5 * rho * Vne**2,
    ]
    cons += fit_constraints(env["wind"], Vwind / (100.0 * units.m / units.s),
                            [rho / (1.0 * units.kg / units.m**3), 0.9],
                            mfac=1.0 + env["wind"]["rms_err"])
    cons += fit_constraints(env["ESday"], ESday / ESvar, [PSmin / PSvar],
                            mfac=1.0 + env["ESday"]["rms_err"])
    cons += fit_constraints(env["EStwi"], EStwi / ESvar, [PSmin / PSvar],
                            mfac=1.0 + env["EStwi"]["rms_err"])

    # ---------------- energy balance (the solar physics) -------------------
    Poper = f.Variable(name="Poper", guess=400.0, units="W", description="operating power")
    ESirr = env["esirr"] * units.W * units.hr / units.m**2
    tnight = env["tnight"] * units.hr
    etasolar, etacharge, etadischarge = 0.2, 0.98, 0.98
    cons += [
        ESirr >= ESday + Ebatt / etacharge / etasolar / Ssolar,
        Ebatt * etadischarge >= Poper * tnight + EStwi * etasolar * Ssolar,
        Poper == PSmin * Ssolar * etasolar,
    ]

    # ---------------- aerodynamics and steady level flight -----------------
    CL  = f.Variable(name="CL",  guess=1.0,   units="-", description="lift coefficient")
    CD  = f.Variable(name="CD",  guess=0.05,  units="-", description="drag coefficient")
    cda = f.Variable(name="cda", guess=0.01,  units="-", description="non-wing drag coefficient")
    cdw = f.Variable(name="cdw", guess=0.03,  units="-", description="wing drag coefficient")
    Thrust = f.Variable(name="T", guess=20.0, units="lbf", description="thrust")
    Pelec  = f.Variable(name="Pelec", guess=200.0, units="W", description="electrical power")

    cons += [
        Wtotal <= 0.5 * rho * V**2 * CL * Sw,
        Thrust >= 0.5 * rho * V**2 * CD * Sw,
        CD / 1.05 >= cda + cdw,
        # Poper/mpower >= Pavn + Ppay + Pelec*Nprop
        Poper / 1.05 >= 200 * units.W + 100 * units.W + Pelec * Nprop,
        # wing profile drag: induced plus a nominal profile term. The source
        # uses an XFOIL fit of the DAI1336a airfoil; that fit is not
        # transcribed here (see STATUS.md) so a fixed cdp is used instead.
        cdw >= 0.0075 + CL**2 / pi / ARw / 0.95,
        cda >= 0.005,
    ]

    # ---------------- wing loading (manoeuvre + gust) ----------------------
    E_ud = CFRPUD["E"] * units.Pa
    sig_ud = CFRPUD["sigma"] * units.Pa
    Nload = 2.0 * 1.5     # Nmax=2, Nsafety=1.5 for the solar wing
    _, wing_load_cons = _beam(
        f, "wingg", Nwing, deta_w, bw, spar["I"], E_ud, sig_ud, spar["Sy"],
        lambda i: Nload * Wcent / bw * cbar_w[i])
    cons += wing_load_cons

    # ---------------- weight buildup ---------------------------------------
    cons += [
        Wland >= 0.02 * Wtotal,
        Wwingtot >= Wwstruct + Wbatt + Wsolar,
        Wcent >= Wpay + Wavn + Wemp + Wmotor * Nprop,
        Wemp >= 8.0 * units.lbf,      # placeholder, see STATUS.md
        Wmotor >= 1.5 * units.lbf,
        Wprop >= 0.5 * units.lbf,
        Wtotal / 1.05 >= (Wpay + Wavn + Wland + Wsolar + Wwstruct + Wbatt
                          + Wemp + Nprop * (Wmotor + Wprop)),
        tauw <= 0.144,
    ]

    f.ConstraintList(cons)
    return f


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from harness import solve_edi, feasibility

    f = build(latitude=20)
    sol, obj, note = solve_edi(f)
    nv, worst, where = feasibility(f)
    print(f"Wtotal = {obj:.5g} lbf     (gpkit npod0_gp_lat20 = 436.4348)")
    print(f"feasibility: {nv} violated, worst rel {worst:.2e}"
          + (f" at {where}" if where else ""))
    if note:
        print("note:", note)
    for k in ("Sw", "ARw", "bw", "V", "Vwind", "rho", "E", "Wbatt",
              "Ssolar", "Wsolar", "CL", "PSmin"):
        if k in sol:
            print(f"   {k:8s} {sol[k]:14.6g}")
