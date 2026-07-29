"""Jungle Hawk Owl — gas-powered long-endurance UAV (GP).

Source model
------------
``gassolar/gas/gas.py`` in https://github.com/convexengineering/gassolar
(the same aircraft as the ``jho`` repository).

Paper
-----
    M. Burton and W. Hoburg, "Solar and Gas Powered Long-Endurance Unmanned
    Aircraft Sizing via Geometric Programming", J. Aircraft 55(1), 2018.

The case rebuilt here is the default mission: latitude 38, 90th-percentile
winds, day 355, a ten-segment climb to 15000 ft followed by a six-day loiter.
Objective is maximum take-off weight.

The gas counterpart to ``../solar/``
------------------------------------
Same airframe library, opposite energy story. The solar aircraft carries its
energy as *battery mass that never leaves*, so weight is constant and the
sizing fight is between cell area and battery mass. This one *burns* its
energy, so weight falls through the mission and the sizing fight is between
fuel fraction and the structure needed to carry it. Fuel is about 59% of MTOW
at the optimum (64.1 of 108.5 lbf).

That difference shows up as the Breguet range equation, which is the one
genuinely awkward piece to make GP-compatible:

    z_bre    >= P_total t BSFC g / sqrt(W_end W_start)
    f_fo W_fuel / W_end >= te_exp_minus1(z_bre, 3)

``te_exp_minus1(z, 3)`` is the Taylor expansion of ``exp(z) - 1`` to third
order, ``z + z^2/2 + z^3/6``. Written that way it is a posynomial in z, where
``exp(z) - 1`` is not — the expansion is what keeps the fuel-burn relation
inside a GP. The geometric mean ``sqrt(W_end W_start)`` plays the same role
for the weight the aircraft flies at during the segment.

Wing configuration
------------------
Unlike ``../solar/`` this uses the gplibrary **defaults**: ``CapSpar`` (not
BoxSpar), ``WingSkin`` (gauge and torsion, not areal density) and a
``WingCore`` foam fill. ``tau`` is fixed at 0.115. The cap spar differs from
the box spar in section:

    box:  I <= w t hin^2          cap:  I <= 2 w t (hin/2)^2

Status: verified to ~2%
-----------------------
Solves as a GP and matches the gpkit reference:

| quantity | rebuild | reference | delta |
|---|---|---|---|
| MTOW (lbf) | 110.78 | 108.55 | +2.1% |
| fuel (lbf) | 64.30 | 64.06 | +0.4% |
| fuselage W (lbf) | 3.80 | 3.79 | +0.4% |
| W_cent (lbf) | 98.91 | 97.54 | +1.4% |
| wing AR | 17.75 | 18.24 | -2.7% |
| wing S (ft^2) | 12.81 | 13.38 | -4.3% |
| wing W (lbf) | 10.32 | 9.60 | +7.5% |
| engine W (lbf) | 12.81 | 11.69 | +9.5% |

Getting from an initial -28% to +2% took four fixes, two of them **bugs in
the source that have to be reproduced** to match the reference numbers.

**1. Posynomial on the greater side.** ``sum(t_i) >= 6 days`` is not
GP-representable and was the single constraint of 397 that pushed the model
out of GP into SP; maidas' ``check_problem_form`` located it directly. The
source constrains each segment instead, ``t_i >= t/N`` — the same device the
wind turbine uses for equal-power spanwise bins.

**2. The ``W`` dict collision**, as in ``../solar/``. Caught immediately here
only because the merge used ``dict(**sp)``, which raises on a duplicate key
where ``.update()`` silently overwrites. Prefer the former.

**3. SOURCE BUG — the fuselage is charged for drag twice.**
``AircraftPerf`` loops over the area-drag components appending a term for
each of ``"Cf"``, ``"Cd"``, ``"C_d"`` that a component's flight model happens
to define. ``TailBoomAero`` defines only ``Cf`` and ``TailAero`` only ``Cd``,
but ``FuselageAero`` defines **both** — and since ``Cd/mfac >= Cf*k``, the
fuselage is charged about ``(1+k) = 2.15x`` its own drag. It is load-bearing
for the reference numbers: CDA is 0.010314 with the double count and 0.006441
without, and without it MTOW lands 28% low.

**4. SOURCE BUG — the wing load factors end up swapped.** ``gas.py`` reads

    loading[0].substitutions[loading[0].Nmax] = 5
    loading[1].substitutions[loading[0].Nmax] = 2

The second line keys *loading[1]'s* substitution with **loading[0]'s**
varkey, so both entries target the manoeuvre case; the later wins and sets it
to 2, while the gust case silently keeps its default of 5. The solved
reference confirms it — ``SparLoading.N = 2``, ``GustL.N = 5`` — and the gust
moments dominate throughout (2420 vs 1503 N*m at the root). The evident
intent was the reverse.

``reference.json`` records the gpkit
solution; note the endurance requirement ``Loiter.t = 6`` days must be
substituted or the model is unbounded — gpkit reports
``Mission.Loiter.t has no lower bound``.

Two bugs already found and fixed here, both familiar:

**The same ``W`` dict collision as ../solar/**, this time caught immediately
because the merge used ``dict(**sp)``, which raises on a duplicate key, where
``.update()`` silently overwrites. Worth preferring the former.

**A posynomial on the greater side.** ``sum(t_i) >= 6 days`` is not
GP-representable, and it was the single constraint of 397 that pushed the
model out of GP into SP. maidas' ``check_problem_form`` located it directly.
The source avoids it by constraining each segment, ``t_i >= t/N`` — the same
device the wind turbine uses with equal-power spanwise bins.
"""
from __future__ import annotations

import math

import numpy as np
from pyomo.environ import units
from pyomo.core.base.var import IndexedVar

from edi import Formulation
from environment import (CLIMB, LOITER, POWER_LAW, BSFC_FIT,
                         JHO_POLAR, NACA0008)

G = 9.81
G_U = G * units.m / units.s**2

# gplibrary defaults (NOT the solar overrides)
CFRPUD = dict(rho=1.6, E=137e9, tmin=0.1, sigma=1700e6)   # g/cm^3, Pa, mm, Pa
CFRPFABRIC = dict(rho=1.6, E=150e9, tmin=0.3048, tau=570e6, sigma=400e6)
FOAMHD = dict(rho=0.036)


def planform_constants(N: int, lam: float):
    eta = np.linspace(0.0, 1.0, N)
    cbar = np.array([2.0 / (1 + lam) * (1 + (lam - 1) * e) for e in eta])
    cbave = (cbar[:-1] + cbar[1:]) / 2.0
    deta = np.diff(eta)
    lami = cbar[1:] / cbar[:-1]
    maci = 2.0 / 3 * cbar[:-1] * (1 + lami + lami**2) / (1 + lami)
    num = sum((cbar[i] + cbar[i + 1]) / 2 * maci[i] * deta[i]
              for i in range(len(deta)))
    den = sum((cbar[i] + cbar[i + 1]) / 2 * deta[i] for i in range(len(deta)))
    return eta, cbar, cbave, deta, num / den / cbar[0]


def fit_constraints(fit, ivar, dvars, mfac=1.0):
    """EDI constraints for a gpfit fit. See ../solar/environment.py."""
    monos = []
    for k in range(fit["K"]):
        m = fit["c"][k]
        for i, u in enumerate(dvars):
            m = m * u ** fit["e"][k][i]
        monos.append(m)
    lhs = ivar / mfac
    if fit["ftype"] == "SMA":
        lhs = lhs ** fit["a1"]
        rhs = sum(monos)
        return [lhs == rhs] if fit["K"] == 1 else [lhs >= rhs]
    if fit["ftype"] == "MA":
        if fit["K"] == 1:
            return [lhs == monos[0]]
        return [lhs >= m for m in monos]
    raise NotImplementedError(fit["ftype"])


def te_exp_minus1(z, order=3):
    """Taylor expansion of exp(z) - 1, which is a posynomial in z.

    ``exp(z) - 1`` is not GP-representable; its truncation
    ``z + z^2/2 + z^3/6`` is. Since every term is positive the expansion is a
    lower bound on the true value for z > 0, so using it in
    ``W_fuel/W_end >= te_exp_minus1(z)`` under-predicts fuel burn slightly —
    the standard, and conservative in the optimizer's favour, so the source
    keeps the order at 3.
    """
    return sum(z**i / math.factorial(i) for i in range(1, order + 1))


def _cap_spar(f, tag, N, cave, tau, b):
    """CapSpar over N-1 segments (the gplibrary default spar)."""
    Nseg = N - 1
    deta = np.diff(np.linspace(0.0, 1.0, N))
    # The spar is a region of its surface, so it nests inside that group.
    # Its weight is `W` in its own namespace -- `wing.spar.W` beside
    # `wing.W` -- rather than a `Wspar` renamed to dodge a collision.
    g = f.group(tag, prefix=f"{tag}_").group("spar")
    I   = g.Variable("I",   1e-6, "m^4", f"{tag} spar inertia",        size=Nseg)
    Sy  = g.Variable("Sy",  1e-5, "m^3", f"{tag} section modulus",     size=Nseg)
    dm  = g.Variable("dm",  0.1,  "kg",  f"{tag} segment mass",        size=Nseg)
    w   = g.Variable("w",   0.3,  "in",  f"{tag} cap width",           size=Nseg)
    t   = g.Variable("t",   0.02, "in",  f"{tag} cap thickness",       size=Nseg)
    ts  = g.Variable("ts",  0.02, "in",  f"{tag} shear web thickness", size=Nseg)
    hin = g.Variable("hin", 0.4,  "in",  f"{tag} height between caps", size=Nseg)
    W   = g.Variable("W",   3.0,  "lbf", f"{tag} spar weight")

    rho_ud = CFRPUD["rho"] * units.g / units.cm**3
    rho_fb = CFRPFABRIC["rho"] * units.g / units.cm**3
    rho_fm = FOAMHD["rho"] * units.g / units.cm**3
    cons = []
    # `cave` is either a vector of panel chords or a single chord shared by
    # every panel; both combine with the vectors below.
    cav = cave[:Nseg] if isinstance(cave, IndexedVar) else cave
    cons += [
        # cap spar section, NOT the box form used in ../solar/
        I / 0.97 <= 2 * w * t * (hin / 2)**2,
        dm >= (rho_ud * (2 * w * t)
               + 2 * ts * rho_fb * (hin + 2 * t)
               + rho_fm * w * hin) * b / 2 * deta,
        w <= 0.15 * cav,
        cav * tau >= hin + 2 * t,
        Sy * (hin / 2 + t) <= I,
        ts >= CFRPFABRIC["tmin"] * units.mm,
    ]
    cons.append(W >= 2 * f.sum(dm) * G_U)
    return g, cons


def _beam(f, tag, N, b, I, Sy, load_expr, kappa=0.2, tiny=1e-2):
    deta = np.diff(np.linspace(0.0, 1.0, N))
    g = f.group(tag, prefix=f"{tag}_")
    Sh = g.Variable("S",  50.0, "N",   f"{tag} shear",      size=N)
    M  = g.Variable("M",  50.0, "N*m", f"{tag} moment",     size=N)
    th = g.Variable("th", 0.05, "-",   f"{tag} angle",      size=N)
    wd = g.Variable("w",  0.1,  "m",   f"{tag} deflection", size=N)
    q  = g.Variable("q",  20.0, "N/m", f"{tag} load",       size=N)
    E_ud, sig_ud = CFRPUD["E"] * units.Pa, CFRPUD["sigma"] * units.Pa
    cons = []
    # Beam recursion outboard along the span: shear and moment accumulate
    # inboard, slope and deflection outboard.
    cons += [
        Sh[:-1] >= Sh[1:] + 0.5 * deta * (b / 2) * (q[:-1] + q[1:]),
        M[:-1] >= M[1:] + 0.5 * deta * (b / 2) * (Sh[:-1] + Sh[1:]),
        th[1:] >= th[:-1] + 0.5 * deta * (b / 2) * (M[1:] + M[:-1]) / E_ud / I,
        wd[1:] >= wd[:-1] + 0.5 * deta * (b / 2) * (th[1:] + th[:-1]),
    ]
    cons += [Sh[N - 1] >= tiny * units.N, M[N - 1] >= tiny * units.N * units.m,
             th[0] >= tiny, wd[0] >= tiny * units.m,
             wd[N - 1] / (b / 2) <= kappa]
    cons.append(sig_ud >= M[:N - 1] / Sy)
    for i in range(N):
        cons.append(q[i] >= load_expr(i))
    return g, cons


def build(Nwing: int = 5, Ntail: int = 5, t_loiter_days: float = 6.0,
          Nclimb: int = 10, Nloiter: int = 5) -> Formulation:
    """Jungle Hawk Owl at the default mission."""
    f = Formulation()
    V_, C_ = f.Variable, f.Constant
    pi = np.pi
    _, cbar, cbave, deta, cbarmac = planform_constants(Nwing, 0.5)
    _, cbart, cbavet, detat, cbarmact = planform_constants(Ntail, 0.8)

    # ================= wing ==============================================
    Sw    = V_(name="Sw",    guess=13.0, units="ft^2", description="wing area")
    ARw   = V_(name="ARw",   guess=18.0, units="-",    description="wing aspect ratio")
    bw    = V_(name="bw",    guess=15.0, units="ft",   description="wing span")
    crootw = V_(name="crootw", guess=1.2, units="ft",  description="wing root chord")
    cmacw = V_(name="cmacw", guess=0.9,  units="ft",   description="wing MAC")
    cavew = V_(name="cavew", guess=0.9,  units="ft", size=Nwing - 1, description="wing mid chord")
    Wwing = V_(name="Wwing", guess=9.6,  units="lbf",  description="wing weight")
    Wskin = V_(name="Wskin", guess=3.0,  units="lbf",  description="wing skin weight")
    Wcore = V_(name="Wcore", guess=2.0,  units="lbf",  description="wing core weight")
    tskin = V_(name="tskin", guess=0.012, units="in",  description="wing skin thickness")
    tauw = C_(name="tauw", value=0.115, units="-", description="wing thickness ratio")

    rho_fab = CFRPFABRIC["rho"] * units.g / units.cm**3
    rho_fm = FOAMHD["rho"] * units.g / units.cm**3
    cons = [
        bw**2 == Sw * ARw,
        crootw == Sw / bw * cbar[0],
        cmacw == crootw * cbarmac,
        # WingSkin: minimum gauge plus a torsion requirement
        Wskin >= rho_fab * Sw * 2 * tskin * G_U,
        tskin >= CFRPFABRIC["tmin"] * units.mm,
        CFRPFABRIC["tau"] * units.Pa
            >= 1 / (0.01114 / units.mm) / crootw**2 / tskin * 0.121 * Sw
               * (1.225 * units.kg / units.m**3) * (45 * units.m / units.s)**2,
        # WingCore foam fill
        Wcore >= 2 * sum(G_U * rho_fm * 0.0753449 * cavew[i]**2 * bw / 2 * deta[i]
                         for i in range(Nwing - 1)),
    ]
    cons.append(cavew == cbave * Sw / bw)
    spar, c = _cap_spar(f, "wing", Nwing, cavew, tauw, bw); cons += c
    cons.append(Wwing / 1.2 >= Wskin + Wcore + spar.W)

    # ================= empennage =========================================
    def tail(tag, tau_val, AR_val):
        # The surface and its spar share one namespace, so the spar's own
        # weight is `Wspar` and the surface's is `W` -- distinct names rather
        # than two dictionaries merged under a renaming rule. The collision
        # that silently broke ../solar/ (see DISCREPANCIES.md) cannot arise:
        # declaring the same name twice in a group is an error.
        g = f.group(tag, prefix=f"{tag}_")
        S = g.Variable("S", 1.0, "ft^2", f"{tag} area")
        b = g.Variable("b", 2.2, "ft", f"{tag} span")
        croot = g.Variable("croot", 0.5, "ft", f"{tag} root chord")
        cave = g.Variable("cave", 0.45, "ft", f"{tag} mid chord", size=Ntail - 1)
        W = g.Variable("W", 0.5, "lbf", f"{tag} weight")
        Wsk = g.Variable("Wskin", 0.3, "lbf", f"{tag} skin weight")
        cc = [b**2 == S * AR_val, croot == S / b * cbart[0],
              Wsk >= rho_fab * S * 2 * (CFRPFABRIC["tmin"] * units.mm) * G_U]
        cc.append(cave == cbavet * S / b)
        sp, c2 = _cap_spar(f, tag, Ntail, cave, tau_val, b)
        cc += c2
        cc.append(W / 1.1 >= Wsk + sp.W)
        return g, cc

    htail, c = tail("htail", 0.08, 5.0); cons += c
    vtail, c = tail("vtail", 0.08, 4.0); cons += c

    lboom = V_(name="lboom", guess=4.0, units="ft",   description="tail boom length")
    Sboom = V_(name="Sboom", guess=1.5, units="ft^2", description="tail boom wetted area")
    dboom = V_(name="dboom", guess=1.0, units="in",   description="tail boom diameter")
    tboom = V_(name="tboom", guess=0.012, units="in", description="tail boom wall thickness")
    Wboom = V_(name="Wboom", guess=1.0, units="lbf",  description="tail boom weight")
    Iboom = V_(name="Iboom", guess=1e-8, units="m^4", description="tail boom inertia")
    Syboom = V_(name="Syboom", guess=1e-7, units="m^3", description="tail boom section modulus")
    cons += [
        Iboom <= pi * tboom * dboom**3 / 8.0,
        Syboom <= 2 * Iboom / dboom,
        Wboom >= pi * rho_fab * dboom * tboom * lboom * G_U,
        tboom >= CFRPFABRIC["tmin"] * units.mm,
        Sboom == lboom * pi * dboom,
        tauw * crootw >= dboom,
    ]
    Wemp = V_(name="Wemp", guess=2.5, units="lbf", description="empennage weight")
    cons += [
        Wemp >= htail.W + vtail.W + Wboom,
        0.45 <= htail.S * lboom / Sw**2 * bw,     # Vh
        0.04 <= vtail.S * lboom / Sw / bw,        # Vv
    ]

    # ---- boom bending from both tail loads -------------------------------
    # Without this the boom has no lower bound on diameter: a thinner boom is
    # lighter and lower drag, so d -> 0, the boom becomes weightless, and the
    # volume-coefficient constraints are then satisfied by growing the moment
    # arm instead of the tail area -- the empennage collapses to ~0 lbf. The
    # source applies TailBoomBending twice, once per tail.
    qne_j = V_(name="qne", guess=980.0, units="kg/s^2/m",
               description="never-exceed dynamic pressure")
    cons.append(qne_j == 0.5 * (1.225 * units.kg / units.m**3)
                * (40 * units.m / units.s)**2)
    E_ud, sig_ud = CFRPUD["E"] * units.Pa, CFRPUD["sigma"] * units.Pa
    for tag, surf, clmax in (("hbend", htail, 1.39), ("vbend", vtail, 1.39)):
        Fb = V_(name=f"{tag}_F", guess=50.0, units="N",
                description=f"{tag} tail force")
        thb = V_(name=f"{tag}_th", guess=0.05, units="-",
                 description=f"{tag} boom deflection angle")
        Mrb = V_(name=f"{tag}_Mr", guess=20.0, units="N*m",
                 description=f"{tag} boom root moment")
        cons += [
            Fb >= qne_j * surf["S"] * clmax,
            # cantilever with a tip load: M_root = F l, theta = F l^2/(2 EI)
            Mrb >= Fb * lboom,
            sig_ud >= Mrb / Syboom,
            thb >= Fb * lboom**2 / (2 * E_ud * Iboom),
            thb * clmax * 1.5 <= 0.1,          # kappa = 0.1
        ]

    # ================= fuselage ==========================================
    Rfuse = V_(name="Rfuse", guess=0.4, units="ft",   description="fuselage radius")
    lfuse = V_(name="lfuse", guess=3.0, units="ft",   description="fuselage length")
    Sfuse = V_(name="Sfuse", guess=6.0, units="ft^2", description="fuselage wetted area")
    Wfuse = V_(name="Wfuse", guess=3.8, units="lbf",  description="fuselage weight")
    Volfuse = V_(name="Volfuse", guess=1.4, units="ft^3", description="fuselage volume")
    kfuse = V_(name="kfuse", guess=1.1, units="-", description="fuselage form factor")
    frfuse = V_(name="frfuse", guess=6.0, units="-", description="fuselage fineness ratio")
    tfuse = V_(name="tfuse", guess=0.024, units="in", description="fuselage skin thickness")
    P = 1.6075
    cons += [
        frfuse == lfuse / Rfuse / 2.0,
        kfuse >= 1.0 + 60.0 / frfuse**3 + frfuse / 400.0,
        3.0 * (Sfuse / pi)**P >= 2.0 * (lfuse * Rfuse * 2.0)**P + (2.0 * Rfuse)**(2 * P),
        Volfuse <= 4.0 * pi / 3.0 * (lfuse / 2.0) * Rfuse**2,
        Wfuse / 2.0 >= Sfuse * rho_fab * tfuse * G_U,
        tfuse >= 2 * CFRPFABRIC["tmin"] * units.mm,
    ]

    # ================= engine ============================================
    Pslmax = V_(name="Pslmax", guess=3.1, units="hp",  description="max sea-level shaft power")
    Weng   = V_(name="Weng",   guess=5.2, units="lbf", description="engine weight")
    Weng_i = V_(name="Weng_installed", guess=11.7, units="lbf", description="installed engine weight")
    cons += fit_constraints(POWER_LAW, Weng / (10.0 * units.lbf),
                            [Pslmax / (10.0 * units.hp)],
                            mfac=1.0 + POWER_LAW["rms_err"])
    cons.append(Weng_i >= 2.572 * Weng**0.922 * units.lbf**0.078)

    # ================= weights ===========================================
    MTOW  = V_(name="MTOW",  guess=108.0, units="lbf", description="max take-off weight")
    Wzfw  = V_(name="Wzfw",  guess=44.0,  units="lbf", description="zero fuel weight")
    Wfuel = V_(name="Wfuel", guess=64.0,  units="lbf", description="total fuel weight")
    Wcent = V_(name="Wcent", guess=97.0,  units="lbf", description="center weight")
    Wpay = C_(name="Wpay", value=10.0, units="lbf", description="payload weight")
    Wavn = C_(name="Wavn", value=8.0,  units="lbf", description="avionics weight")

    f.Objective(MTOW)
    cons += [
        Wzfw >= Wfuse + Wwing + Weng_i + Wemp + Wpay + Wavn,
        MTOW >= Wzfw + Wfuel,
        Wcent >= Wfuel + Wpay + Wavn + Wfuse + Weng_i,
        Volfuse >= Wfuel / (6.01 * units.lbf / units.gallon),
    ]

    # ================= mission ===========================================
    # Segment weights run from MTOW down to Wzfw. Each segment burns fuel by
    # the Breguet relation; the aircraft flies at the geometric mean weight.
    segs = [("climb", i, CLIMB["rho"][i], CLIMB["mu"][i], CLIMB["Vwind_ms"][i])
            for i in range(Nclimb)]
    segs += [("loiter", i, LOITER["rho"], LOITER["mu"], LOITER["Vwind_ms"])
             for i in range(Nloiter)]
    n = len(segs)

    Wstart = V_(name="Wstart", guess=100.0, units="lbf", size=n, description="segment start weight")
    Wend   = V_(name="Wend",   guess=95.0,  units="lbf", size=n, description="segment end weight")
    Wfs    = V_(name="Wfs",    guess=4.0,   units="lbf", size=n, description="segment fuel weight")
    Vseg   = V_(name="V",      guess=35.0,  units="m/s", size=n, description="true airspeed")
    CLseg  = V_(name="CL",     guess=1.0,   units="-",   size=n, description="lift coefficient")
    CDseg  = V_(name="CD",     guess=0.04,  units="-",   size=n, description="drag coefficient")
    CDAseg = V_(name="CDA",    guess=0.01,  units="-",   size=n, description="area drag coefficient")
    cdw    = V_(name="cdw",    guess=0.02,  units="-",   size=n, description="wing drag coefficient")
    cdp    = V_(name="cdp",    guess=0.01,  units="-",   size=n, description="wing profile drag coefficient")
    Rew    = V_(name="Rew",    guess=3e5,   units="-",   size=n, description="wing Reynolds number")
    Reh    = V_(name="Reh",    guess=1e5,   units="-",   size=n, description="htail Reynolds number")
    Rev    = V_(name="Rev",    guess=1e5,   units="-",   size=n, description="vtail Reynolds number")
    Reb    = V_(name="Reb",    guess=5e5,   units="-",   size=n, description="boom Reynolds number")
    Ref    = V_(name="Ref",    guess=5e5,   units="-",   size=n, description="fuselage Reynolds number")
    Cfb    = V_(name="Cfb",    guess=0.005, units="-",   size=n, description="boom skin friction coefficient")
    Cff    = V_(name="Cff",    guess=0.005, units="-",   size=n, description="fuselage skin friction coefficient")
    cdf    = V_(name="cdf",    guess=0.006, units="-",   size=n, description="fuselage drag coefficient")
    cdh    = V_(name="cdh",    guess=0.008, units="-",   size=n, description="htail drag coefficient")
    cdv    = V_(name="cdv",    guess=0.008, units="-",   size=n, description="vtail drag coefficient")
    Tseg   = V_(name="T",      guess=20.0,  units="N",   size=n, description="thrust")
    Pshaft = V_(name="Pshaft", guess=1.0,   units="hp",  size=n, description="shaft power")
    Ptot   = V_(name="Ptotal", guess=1.1,   units="hp",  size=n, description="total power")
    Pmax   = V_(name="Pshaftmax", guess=3.0, units="hp", size=n, description="max shaft power at altitude")
    bsfc   = V_(name="BSFC",   guess=0.4,   units="kg/kW/hr", size=n, description="brake specific fuel consumption")
    zbre   = V_(name="zbre",   guess=0.1,   units="-",   size=n, description="Breguet coefficient")
    tseg   = V_(name="tseg",   guess=0.5,   units="day", size=n, description="segment time")

    BSFC_MIN = 0.3162 * units.kg / units.kW / units.hr
    etaprop = 0.8
    cons += [Wstart[0] == MTOW, Wend[n - 1] >= Wzfw]
    cons.append(Wend[:-1] >= Wstart[1:])

    for i, (kind, k, rho_i, mu_i, vw_i) in enumerate(segs):
        rho_u = rho_i * units.kg / units.m**3
        mu_u = mu_i * units.N * units.s / units.m**2
        alt_ft = CLIMB["altitude_ft"][k] if kind == "climb" else LOITER["altitude_ft"]
        Leng = -0.035 * (alt_ft / 1000.0) + 1.0     # shaft power lapse
        cons += [
            Vseg[i] >= vw_i * units.m / units.s,
            # steady level flight at the geometric-mean weight
            (Wend[i] * Wstart[i])**0.5 <= 0.5 * rho_u * Vseg[i]**2 * CLseg[i] * Sw,
            Tseg[i] >= 0.5 * rho_u * Vseg[i]**2 * CDseg[i] * Sw,
            Pshaft[i] >= Tseg[i] * Vseg[i] / etaprop,
            CLseg[i] <= 1.39,
            # drag buildup, from the real component fits rather than fixed
            # coefficients: profile drag has to respond to Reynolds number or
            # the wing sizing comes out ~35% small.
            Rew[i] == rho_u * Vseg[i] * cmacw / mu_u,
            Reh[i] == rho_u * Vseg[i] * htail.S / htail.b / mu_u,
            Rev[i] == rho_u * Vseg[i] * vtail.S / vtail.b / mu_u,
            Reb[i] == rho_u * Vseg[i] * lboom / mu_u,
            Ref[i] == rho_u * Vseg[i] * lfuse / mu_u,
            Cfb[i] >= 0.455 / Reb[i]**0.3,
            Cff[i] >= 0.455 / Ref[i]**0.3,
            cdf[i] >= Cff[i] * kfuse,
            cdw[i] >= cdp[i] + CLseg[i]**2 / pi / ARw / 0.9,
            # NOTE the fuselage appears TWICE, via both Cf and Cd. That is
            # what the source does -- AircraftPerf loops over the area-drag
            # components appending a term for each of "Cf", "Cd", "C_d" that
            # the component's flight model happens to define, and
            # FuselageAero defines both. Since Cd/mfac >= Cf*k, the fuselage
            # is charged ~(1+k) = 2.15x its own drag. See DISCREPANCIES.md;
            # reproduced here because it is load-bearing for the reference
            # numbers -- CDA is 0.010314 with it and 0.006441 without.
            CDAseg[i] >= (cdf[i] * Sfuse / Sw + Cff[i] * Sfuse / Sw
                          + cdh[i] * htail.S / Sw
                          + cdv[i] * vtail.S / Sw + Cfb[i] * Sboom / Sw),
            CDseg[i] >= CDAseg[i] + cdw[i],
            # engine: power lapse with altitude, and BSFC penalty at part power
            Pmax[i] == Pslmax * Leng,
            Pmax[i] >= Ptot[i],
            Ptot[i] >= Pshaft[i] + 40 * units.W / 0.8,
            # Breguet
            zbre[i] >= Ptot[i] * tseg[i] * bsfc[i] * G_U
                       / (Wend[i] * Wstart[i])**0.5,
            0.98 * Wfs[i] / Wend[i] >= te_exp_minus1(zbre[i], 3),
            Wstart[i] >= Wend[i] + Wfs[i],
        ]
        cons += fit_constraints(BSFC_FIT, bsfc[i] / BSFC_MIN,
                                [Ptot[i] / Pmax[i]],
                                mfac=1.0 + BSFC_FIT["rms_err"])
        cons += fit_constraints(JHO_POLAR, cdp[i], [CLseg[i], Rew[i]],
                                mfac=1.0 + JHO_POLAR["rms_err"])
        cons += fit_constraints(NACA0008, cdh[i], [Reh[i], 0.08],
                                mfac=1.0 + NACA0008["rms_err"])
        cons += fit_constraints(NACA0008, cdv[i], [Rev[i], 0.08],
                                mfac=1.0 + NACA0008["rms_err"])

    # Loiter endurance requirement. The source constrains each segment
    # individually, ``be.t >= t/N``, rather than bounding the sum -- and that
    # is not a stylistic choice. ``sum(t_i) >= T`` puts a *posynomial on the
    # greater side*, which is not GP-representable, and it is the single
    # constraint (of 397) that pushed this model out of GP and into SP.
    # Dividing the requirement equally is the same device the wind turbine
    # uses with its equal-power spanwise bins.
    cons.append(tseg[Nclimb:] >= t_loiter_days / Nloiter * units.day)
    cons.append(Wfuel >= f.sum(Wfs))

    # ================= wing loading ======================================
    # Load factors: manoeuvre Nmax = 2, gust Nmax = 5.
    #
    # That is NOT what gas.py appears to intend, and the reason is a bug in
    # the source worth knowing about. Lines 194-195 read
    #
    #     loading[0].substitutions[loading[0].Nmax] = 5
    #     loading[1].substitutions[loading[0].Nmax] = 2
    #
    # The second keys loading[1]'s substitution with **loading[0]'s** varkey,
    # so both entries target the manoeuvre case; the later one wins and sets
    # it to 2, while the gust case keeps its default of 5. The solved
    # reference confirms it: SparLoading.N = 2, GustL.N = 5, and the gust
    # moments dominate throughout (2420 vs 1503 N*m at the root).
    #
    # Reproduced as the source behaves, since that is what the reference
    # numbers come from. See DISCREPANCIES.md.
    _, c = _beam(f, "wingg", Nwing, bw, spar.I, spar.Sy,
                 lambda i: 2.0 * Wcent / bw * cbar[i])
    cons += c

    ARCTAN = dict(ftype="MA", K=1, d=1, a1=1.0,
                  c=[0.9460414492363466], e=[[0.9960249757710423]],
                  rms_err=0.039722989129247634)
    etaw = np.linspace(0.0, 1.0, Nwing)
    cosm1 = np.hstack([1e-10, 1 - np.cos(etaw[1:] * pi / 2)])
    agust = V_(name="agust", guess=0.05, units="-", size=Nwing,
               description="gust angle of attack")
    vgust = 10.0 * units.m / units.s
    Vref = Vseg[Nclimb]                      # loiter speed sizes the gust case
    CLref = CLseg[Nclimb]
    for i in range(Nwing):
        cons += fit_constraints(ARCTAN, agust[i], [cosm1[i] * vgust / Vref],
                                mfac=1.0 + ARCTAN["rms_err"])
    _, c = _beam(f, "winggust", Nwing, bw, spar.I, spar.Sy,
                 lambda i: 5.0 * Wcent / bw * cbar[i]
                 * (1 + 2 * pi * agust[i] / CLref * (1 + Wwing / Wcent)))
    cons += c

    f.ConstraintList(cons)
    return f


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from harness import solve_edi, feasibility

    fm = build()
    sol, obj, note = solve_edi(fm)
    nv, worst, where = feasibility(fm)
    print(f"MTOW = {obj:.5g} lbf     (gpkit reference = 108.5455)")
    print(f"feasibility: {nv} violated, worst rel {worst:.2e}"
          + (f" at {where}" if where else ""))
    if note:
        print("note:", note)
    for k in ("Sw", "ARw", "bw", "Wzfw", "Wfuel", "Wwing", "Weng_installed",
              "Wfuse", "Wemp", "Wcent"):
        if k in sol:
            print(f"   {k:15s} {sol[k]:12.5g}")
