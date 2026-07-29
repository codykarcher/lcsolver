"""Solar-electric long-endurance aircraft (GP).

Source model
------------
``solar/solar.py`` in https://github.com/convexengineering/solar

Paper
-----
    M. Burton and W. Hoburg, "Solar and Gas Powered Long-Endurance Unmanned
    Aircraft Sizing via Geometric Programming", J. Aircraft 55(1), 2018.

The case rebuilt here is ``Npod=0``, GP (not SP), latitude 20, day 355 —
winter solstice, the sizing case, because it is the shortest day and the
longest night. Objective is total aircraft weight.

What makes it a solar aircraft
------------------------------
Everything else is conventional airframe sizing; three constraints carry the
solar physics and they are what close the design:

    ESirr          >= ESday + E/(etacharge etasolar Ssolar)
    E etadischarge >= Poper tnight + EStwi etasolar Ssolar
    Poper          == PSmin Ssolar etasolar

The day's irradiance must cover both the day's flying and recharging the
battery; the battery must then carry the night plus twilight; and operating
power ties back to the solar power the wing area can collect. Battery energy
and cell area are the two currencies and both cost weight.

Why the whole airframe has to be here
-------------------------------------
An early version of this file stubbed the empennage, motor and propeller and
used a fixed profile-drag term. It would not converge, and the reason is
worth recording because it is not obvious: **flight speed became unbounded.**

``V`` enters the objective only through a chain —

    V -> Re -> cdw -> CD -> thrust -> propeller -> motor -> Pelec
      -> Poper -> battery energy -> battery weight -> Wtotal

Break any link and flying faster becomes free, every ``V`` above the stall
bound is optimal, and the GP has a ray of optima rather than a point. Interior
point methods stall on that. So the drag fit, the propulsion chain and the
tail drag areas are all load-bearing for *convergence*, not just accuracy.

Status: verified to ~1%
-----------------------
Solves on ``convex_backend="ipopt"`` and matches the gpkit reference:

| quantity | rebuild | reference | delta |
|---|---|---|---|
| Wtotal (lbf), lat 20 | 434.25 | 436.43 | -0.5% |
| Wtotal (lbf), lat 10 | 295.38 | 297.56 | -0.7% |
| wing AR | 38.17 | 38.10 | +0.2% |
| wing S (ft^2) | 400.1 | 394.2 | +1.5% |
| wing b (ft) | 123.6 | 122.6 | +0.8% |
| battery E (kJ) | 1.0995e5 | 1.0915e5 | +0.7% |
| battery W (lbf) | 245.5 | 243.7 | +0.7% |
| Poper (W) | 2116.9 | 2101.3 | +0.7% |
| V (m/s) | 22.71 | 22.91 | -0.9% |
| PSmin (W/m^2) | 284.8 | 286.9 | -0.7% |
| empennage W (lbf) | 14.99 | 15.83 | -5.3% |

Four bugs were found getting here, and each masked the next — worth reading
in order, because three of the four produced a *perfectly self-consistent*
model that simply had the wrong answer.

**1. The backend.** cvxopt stalled with ``status='unknown'`` and no amount of
model work moved it; the same model converges immediately on
``convex_backend="ipopt"``. Everything below was invisible until this was
fixed, and two real defects were misattributed to the model in the meantime.

**2. A dict-key collision.** ``_lifting_surface`` merged the spar dict into
the surface dict and both carry a ``"W"``, so ``wing.W`` silently became
the *spar* weight. The total-weight constraint then used the spar weight
while the real surface-weight variable kept only a lower bound — free to run
to 1e36 while the model stayed feasible and the solver stayed happy.
Wtotal 198 -> 330.

**3. A missing load case.** Only the manoeuvre load was applied; the source
has manoeuvre *and* gust, and gust is the sizing case. Without it the
structure is too cheap and the optimizer answers with span and thinness —
AR 48.8 against 38.1, and ``tau`` pinned at the bottom of its range instead
of the top. Wtotal 330 -> 465, AR to within 0.1%.

**4. Conflating two boom widths.** In the solar build the tail boom is a
*box* spar (``TailBoom.__bases__`` is reassigned), so its section has two
distinct widths: ``cave``, the box chord that ``wlim`` scales, and ``d == w``,
the spar cap width — and the wetted area is ``S = l*pi*w[0]``, built from the
*cap*, not the chord. Using one variable for both made ``Sboom`` 50x too
large and the non-wing drag ``cda`` 64% high. Wtotal 465 -> 432.

Both latitudes agree to under 1%, which is an independent check on the
embedded environment fits: latitude 10 and 20 use entirely different wind,
ESday and EStwi coefficients.

The remaining gap is concentrated in the empennage (-5.3%). The ``Climb``
mission segment is not modelled, which is why the propeller's max static
thrust ``T_m`` comes out 0.37x the reference — climb sizes it, cruise does
not, and ``T_m`` feeds the propeller and motor weights.

Configuration differences from the standalone subsystems
--------------------------------------------------------
This is not the same wing as ``../wing/``. Solar reconfigures its subsystems:

* **Materials are overridden**: ``cfrpud`` becomes rho=1.5 g/cm^3, E=200 GPa,
  sigma=1500 MPa (library defaults are 1.6, 137, 1700); ``cfrpfabric``
  becomes rho=1.3, E=40 GPa, sigma=300 MPa, tau=80 MPa; ``foamhd`` 0.03.
* **20 wing nodes**, not 5.
* **No foam core**, and the skin is ``WingSecondStruct`` — a flat areal
  density (0.35 kg/m^2 wing, 0.4 tails) rather than a gauge model.
* **The tail boom is a box spar**, not a tube, with 0.15 kg/m^2 secondary
  weight.

Getting one of these wrong reproduces DISCREPANCIES.md #8 — a structural
constraint off by a constant and the optimizer buying free span.
"""
from __future__ import annotations

import numpy as np
from pyomo.environ import units

from pyomo.core.base.var import IndexedVar

from edi import Formulation
from airfoils import DAI1336A, NACA0008
from environment import ENVIRONMENT, fit_constraints

G = 9.81
G_U = G * units.m / units.s**2

# --- materials AS OVERRIDDEN BY solar.py (not the gplibrary defaults) ------
CFRPUD = dict(rho=1.5, E=200e9, tmin=0.1, sigma=1500e6)   # g/cm^3, Pa, mm, Pa
CFRPFABRIC = dict(rho=1.3, E=40e9, tmin=0.1, sigma=300e6, tau=80e6)
FOAMHD = dict(rho=0.03)


def planform_constants(N: int, lam: float):
    """Normalized chord distribution, as gplibrary's linked variables give it."""
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


# ---------------------------------------------------------------------------
# Reusable subsystem builders
# ---------------------------------------------------------------------------

def _box_spar(f, tag, N, cave, tau_expr, b, wlim=0.15):
    """BoxSpar over N-1 segments. cave may be an indexed var or a scalar."""
    Nseg = N - 1
    deta = np.diff(np.linspace(0.0, 1.0, N))
    # The spar is a region of its surface, so it nests inside that group:
    # `wing.spar.W` beside `wing.W`, each weight in its own namespace.
    g = f.group(tag, prefix=f"{tag}_").group("spar")
    I   = g.Variable("I",   1e-6, "m^4", f"{tag} spar inertia",        size=Nseg)
    Sy  = g.Variable("Sy",  1e-5, "m^3", f"{tag} section modulus",     size=Nseg)
    dm  = g.Variable("dm",  0.2,  "kg",  f"{tag} segment mass",        size=Nseg)
    w   = g.Variable("w",   0.5,  "in",  f"{tag} spar width",          size=Nseg)
    t   = g.Variable("t",   0.02, "in",  f"{tag} cap thickness",       size=Nseg)
    ts  = g.Variable("ts",  0.02, "in",  f"{tag} shear web thickness", size=Nseg)
    tc  = g.Variable("tc",  0.02, "in",  f"{tag} core thickness",      size=Nseg)
    hin = g.Variable("hin", 0.5,  "in",  f"{tag} height between caps", size=Nseg)
    W   = g.Variable("W",   5.0,  "lbf", f"{tag} spar weight")

    rho_ud = CFRPUD["rho"] * units.g / units.cm**3
    rho_fb = CFRPFABRIC["rho"] * units.g / units.cm**3
    rho_fm = FOAMHD["rho"] * units.g / units.cm**3
    cons = []
    # `cave` is either a vector of panel chords or one chord shared by every
    # panel; both combine with the vectors below.
    cav = cave[:Nseg] if isinstance(cave, IndexedVar) else cave
    cons += [
        I / 0.97 <= w * t * hin**2,
        dm >= (rho_ud * 4 * w * t
               + 4 * ts * rho_fb * (hin + w)
               + 2 * rho_fm * tc * (w + hin)) * b / 2 * deta,
        w <= wlim * cav,
        cav * tau_expr >= hin + 4 * t + 2 * tc,
        t >= CFRPUD["tmin"] * units.mm,
        Sy * (hin / 2 + 2 * t + tc) <= I,
        ts >= CFRPFABRIC["tmin"] * units.mm,
        tc >= 0.02 * cav * tau_expr,
    ]
    cons.append(W >= 2 * f.sum(dm) * G_U)
    return g, cons


def _beam(f, tag, N, b, I, Sy, load_expr, Nsafety_kappa=0.2, tiny=1e-2):
    """Spanwise beam: shear -> moment -> angle -> deflection.

    Identical chain to ``../wing/``, which is verified against its reference.
    """
    deta = np.diff(np.linspace(0.0, 1.0, N))
    g = f.group(tag, prefix=f"{tag}_")
    Sh = g.Variable("S",  100.0, "N",   f"{tag} shear",      size=N)
    M  = g.Variable("M",  100.0, "N*m", f"{tag} moment",     size=N)
    th = g.Variable("th", 0.05,  "-",   f"{tag} angle",      size=N)
    wd = g.Variable("w",  0.1,   "m",   f"{tag} deflection", size=N)
    q  = g.Variable("q",  50.0,  "N/m", f"{tag} load",       size=N)
    E_ud, sig_ud = CFRPUD["E"] * units.Pa, CFRPUD["sigma"] * units.Pa
    cons = []
    # Beam recursion outboard along the span.
    cons += [
        Sh[:-1] >= Sh[1:] + 0.5 * deta * (b / 2) * (q[:-1] + q[1:]),
        M[:-1] >= M[1:] + 0.5 * deta * (b / 2) * (Sh[:-1] + Sh[1:]),
        th[1:] >= th[:-1] + 0.5 * deta * (b / 2) * (M[1:] + M[:-1]) / E_ud / I,
        wd[1:] >= wd[:-1] + 0.5 * deta * (b / 2) * (th[1:] + th[:-1]),
    ]
    cons += [Sh[N - 1] >= tiny * units.N, M[N - 1] >= tiny * units.N * units.m,
             th[0] >= tiny, wd[0] >= tiny * units.m,
             wd[N - 1] / (b / 2) <= Nsafety_kappa]
    cons.append(sig_ud >= M[:N - 1] / Sy)
    for i in range(N):
        cons.append(q[i] >= load_expr(i))
    return g, cons


def _lifting_surface(f, tag, N, lam, rhoA, mfac, wlim=0.15):
    """Planform + areal-density skin + box spar. Used for wing and both tails."""
    _, cbar, cbave, deta, cbarmac = planform_constants(N, lam)
    g = f.group(tag, prefix=f"{tag}_")
    S     = g.Variable("S",     50.0, "ft^2", f"{tag} area")
    AR    = g.Variable("AR",    15.0, "-",    f"{tag} aspect ratio")
    b     = g.Variable("b",     25.0, "ft",   f"{tag} span")
    croot = g.Variable("croot", 2.5,  "ft",   f"{tag} root chord")
    cmac  = g.Variable("cmac",  2.0,  "ft",   f"{tag} MAC")
    cave  = g.Variable("cave",  2.0,  "ft",   f"{tag} mid chord", size=N - 1)
    tau   = g.Variable("tau",   0.12, "-",    f"{tag} thickness ratio")
    Wsk   = g.Variable("Wskin", 5.0,  "lbf",  f"{tag} skin weight")
    W     = g.Variable("W",     15.0, "lbf",  f"{tag} weight")

    cons = [b**2 == S * AR,
            croot == S / b * cbar[0],
            cmac == croot * cbarmac,
            Wsk >= rhoA * units.kg / units.m**2 * S * G_U]
    cons.append(cave == cbave * S / b)

    spar, spar_cons = _box_spar(f, tag, N, cave, tau, b, wlim=wlim)
    cons += spar_cons
    cons.append(W / mfac >= Wsk + spar.W)   # no foam core in the solar build

    # The surface and its spar share one namespace, so the spar's weight is
    # `Wspar` and the surface's is `W`. There is nothing to merge and nothing
    # to rename: declaring a name twice in a group is an error, where the two
    # dictionaries this replaces could clobber one another silently -- the
    # total-weight constraint would then use the spar weight while the real
    # surface weight kept only a lower bound, free to run to 1e36 without ever
    # looking infeasible.
    #
    # `cbar` and `deta` are the planform's fixed fractions, not model
    # quantities, so they are returned alongside rather than declared.
    return g, cbar, deta, cons


def build(latitude: int = 20, Nwing: int = 20, Ntail: int = 5,
          Nboom: int = 5) -> Formulation:
    """Solar aircraft, Npod=0, GP, at the given latitude."""
    env = ENVIRONMENT[latitude]
    f = Formulation()
    V_ = f.Variable
    C_ = f.Constant
    pi = np.pi
    Nprop = 4.0

    # ================= airframe =========================================
    # never-exceed dynamic pressure is needed by both the flight state and
    # the tail load case, so it is declared before the airframe
    qne_ph = V_(name="qne", guess=400.0, units="kg/s^2/m", description="never-exceed dynamic pressure")

    wing, wing_cbar, wing_deta, cons = _lifting_surface(
        f, "wing", Nwing, 0.5, rhoA=0.35, mfac=1.0)
    htail, htail_cbar, _, c = _lifting_surface(
        f, "htail", Ntail, 0.8, rhoA=0.4, mfac=1.1); cons += c
    vtail, vtail_cbar, _, c = _lifting_surface(
        f, "vtail", Ntail, 0.8, rhoA=0.4, mfac=1.1); cons += c
    cons += [htail.AR == 4.0, vtail.AR == 4.0]

    # ---- tail boom: box spar with a secondary areal weight ---------------
    lboom = V_(name="lboom", guess=15.0, units="ft",   description="tail boom length")
    Sboom = V_(name="Sboom", guess=1.0,  units="ft^2", description="tail boom wetted area")
    Wboom = V_(name="Wboom", guess=5.0,  units="lbf",  description="tail boom weight")
    # The boom is a *box* spar in the solar build (TailBoom.__bases__ is
    # reassigned to BoxSpar), so its section is described by two different
    # widths and conflating them is easy to get wrong:
    #   cave  -- the box "chord", a free vector, what wlim scales
    #   d == w -- the spar cap width, and what the wetted area is built from
    # The wetted area is S = l*pi*w[0], i.e. it tracks the slender cap, not
    # the chord. Using the chord here made Sboom 50x too large and the
    # non-wing drag cda 64% high.
    caveb = V_(name="caveb", guess=3.0, units="in", size=Nboom - 1,
               description="tail boom box chord")
    boom, c = _box_spar(f, "boom", Nboom, caveb, 1.0, 2 * lboom, wlim=1.0)
    cons += c
    cons += [Sboom == lboom * pi * boom.w[0],
             Wboom >= boom.W + 0.15 * units.kg / units.m**2 * Sboom * G_U]
    cons.append(caveb[:-1] >= caveb[1:])          # boom tapers inboard-out

    Wemp = V_(name="Wemp", guess=15.0, units="lbf", description="empennage weight")
    cons.append(Wemp / 1.0 >= htail.W + vtail.W + Wboom)

    # ---- boom bending under the tail loads -------------------------------
    # Without this the boom is unbounded *below*: a thinner boom is lighter
    # AND lower drag, so nothing stops dboom -> 0. The horizontal tail's
    # max download is what sizes it. Normalized cantilever beam with unit tip
    # shear, rescaled by the actual tip force and length (TailBoomBending).
    Fbend = V_(name="Fbend", guess=200.0, units="N", description="tail bending force")
    Mbar  = V_(name="Mbar",  guess=0.5,  units="-", size=Nboom, description="normalized moment")
    thbar = V_(name="thbar", guess=0.05, units="-", size=Nboom, description="normalized angle")
    dbar  = V_(name="dbar",  guess=0.05, units="-", size=Nboom, description="normalized deflection")
    EIbar = V_(name="EIbar", guess=1.0,  units="-", size=Nboom - 1, description="normalized EI")
    Mr    = V_(name="Mr",    guess=100.0, units="N*m", size=Nboom - 1, description="section root moment")
    detab = 1.0 / (Nboom - 1)
    E_ud, sig_ud = CFRPUD["E"] * units.Pa, CFRPUD["sigma"] * units.Pa
    cons.append(Fbend >= qne_ph * htail.S * 1.39)   # qne * S * CLmax
    cons += [
        Mbar[:-1] >= Mbar[1:] + 0.5 * detab * 2.0,               # unit tip shear
        thbar[1:] >= thbar[:-1] + 0.5 * detab * (Mbar[1:] + Mbar[:-1]) / EIbar,
        dbar[1:] >= dbar[:-1] + 0.5 * detab * (thbar[1:] + thbar[:-1]),
    ]
    cons += [Mbar[Nboom - 1] >= 1e-10, thbar[0] >= 1e-10, dbar[0] >= 1e-10,
             dbar[Nboom - 1] * 1.39 * 1.5 <= 0.1]
    cons += [
        EIbar <= E_ud * boom.I / Fbend / lboom**2 / 2,
        Mr >= Mbar[:Nboom - 1] * Fbend * lboom,
        sig_ud >= Mr / boom.Sy,
    ]

    # ================= power system =====================================
    Ssolar  = V_(name="Ssolar",  guess=150.0, units="ft^2", description="solar cell area")
    Wsolar  = V_(name="Wsolar",  guess=15.0,  units="lbf",  description="solar cell weight")
    Ebatt   = V_(name="E",       guess=6.5e4, units="kJ",   description="battery energy")
    Wbatt   = V_(name="Wbatt",   guess=150.0, units="lbf",  description="battery weight")
    Volbatt = V_(name="Volbatt", guess=0.05,  units="m^3",  description="battery volume")
    cons += [
        Ssolar <= wing.S,
        Wsolar >= 0.3 * units.kg / units.m**2 * Ssolar * G_U,
        Wbatt >= Ebatt * 1.03 / (350 * units.W * units.hr / units.kg) / 0.95 / 0.85 * G_U,
        Volbatt >= Ebatt / (800 * units.W * units.hr / units.liter),
        Volbatt <= wing.cmac**2 * 0.5 * wing.tau * wing.b,   # Npod = 0
    ]

    # ================= flight state =====================================
    V     = V_(name="V",     guess=30.0,  units="m/s",      description="true airspeed")
    Vwind = V_(name="Vwind", guess=25.0,  units="m/s",      description="wind speed")
    rho   = V_(name="rho",   guess=0.4,   units="kg/m^3",   description="air density")
    Vne   = V_(name="Vne",   guess=45.0,  units="m/s",      description="never-exceed speed")
    PSmin = V_(name="PSmin", guess=25.0,  units="W/m^2",    description="minimum solar power")
    ESday = V_(name="ESday", guess=300.0, units="W*hr/m^2", description="daytime solar energy")
    EStwi = V_(name="EStwi", guess=2.0,   units="W*hr/m^2", description="twilight battery energy")
    mu    = C_(name="mu",    value=1.42e-5, units="N*s/m^2", description="air viscosity")

    ESvar = 1.0 * units.W * units.hr / units.m**2
    PSvar = 1.0 * units.W / units.m**2
    cons += [V >= Vwind, Vne == 1.4 * V, qne_ph == 0.5 * rho * Vne**2]
    cons += fit_constraints(env["wind"], Vwind / (100.0 * units.m / units.s),
                            [rho / (1.0 * units.kg / units.m**3), 0.9],
                            mfac=1.0 + env["wind"]["rms_err"])
    cons += fit_constraints(env["ESday"], ESday / ESvar, [PSmin / PSvar],
                            mfac=1.0 + env["ESday"]["rms_err"])
    cons += fit_constraints(env["EStwi"], EStwi / ESvar, [PSmin / PSvar],
                            mfac=1.0 + env["EStwi"]["rms_err"])

    # ================= aerodynamics =====================================
    CL  = V_(name="CL",  guess=1.2,  units="-", description="lift coefficient")
    CD  = V_(name="CD",  guess=0.05, units="-", description="drag coefficient")
    cda = V_(name="cda", guess=0.01, units="-", description="non-wing drag coefficient")
    cdw = V_(name="cdw", guess=0.03, units="-", description="wing drag coefficient")
    cdp = V_(name="cdp", guess=0.02, units="-", description="wing profile drag coefficient")
    Rew = V_(name="Rew", guess=3e5,  units="-", description="wing Reynolds number")
    cdht = V_(name="cdht", guess=0.01, units="-", description="horizontal tail drag coefficient")
    cdvt = V_(name="cdvt", guess=0.01, units="-", description="vertical tail drag coefficient")
    Reht = V_(name="Reht", guess=2e5, units="-", description="horizontal tail Reynolds number")
    Revt = V_(name="Revt", guess=2e5, units="-", description="vertical tail Reynolds number")
    Cftb = V_(name="Cftb", guess=0.005, units="-", description="tail boom skin friction coefficient")
    Retb = V_(name="Retb", guess=1e6, units="-", description="tail boom Reynolds number")

    cons += [
        Rew == rho * V * wing.cmac / mu,
        # DAI1336a profile drag: (cdp/mfac)^a >= sum_k c_k CL^e0 Re^e1 tau^e2.
        # This is the link that makes flying faster cost power.
        cdw >= cdp + CL**2 / pi / wing.AR / 0.95,
        CL <= 1.5,
        Reht == rho * V * htail.S / htail.b / mu,
        Revt == rho * V * vtail.S / vtail.b / mu,
        Retb == rho * V * lboom / mu,
        Cftb >= 0.455 / Retb**0.3,
        cda >= (cdht * htail.S / wing.S + cdvt * vtail.S / wing.S
                + Cftb * Sboom / wing.S),
        CD / 1.05 >= cda + cdw,
    ]
    cons += fit_constraints(DAI1336A, cdp, [CL, Rew, wing.tau],
                            mfac=1.0 + DAI1336A["rms_err"])
    for cd, Re, t in ((cdht, Reht, htail.tau), (cdvt, Revt, vtail.tau)):
        cons += fit_constraints(NACA0008, cd, [Re, t],
                                mfac=1.0 + NACA0008["rms_err"])

    # ================= propulsion =======================================
    Thrust = V_(name="T",      guess=15.0,   units="lbf", description="thrust")
    Tprop  = V_(name="Tprop",  guess=4.0,    units="lbf", description="thrust per propeller")
    Rprop  = V_(name="Rprop",  guess=1.5,    units="ft",  description="propeller radius")
    Wprop  = V_(name="Wprop",  guess=1.0,    units="lbf", description="propeller weight")
    Tm     = V_(name="Tm",     guess=10.0,   units="lbf", description="propeller max static thrust")
    etaprop = V_(name="etaprop", guess=0.8,  units="-",   description="propeller efficiency")
    etai   = V_(name="etai",   guess=0.95,   units="-",   description="inviscid efficiency")
    Tc     = V_(name="Tc",     guess=0.05,   units="-",   description="thrust coefficient")
    z2     = V_(name="z2",     guess=1.05,   units="-",   description="efficiency helper")
    lam    = V_(name="lam",    guess=0.3,    units="-",   description="advance ratio")
    CT     = V_(name="CT",     guess=0.005,  units="-",   description="prop thrust coefficient")
    CP     = V_(name="CP",     guess=0.003,  units="-",   description="prop power coefficient")
    omega  = V_(name="omega",  guess=2000.0, units="rpm", description="rotation rate")
    Qprop  = V_(name="Q",      guess=20.0,   units="N*m", description="torque")
    Pshaft = V_(name="Pshaft", guess=300.0,  units="W",   description="shaft power")
    Pelec  = V_(name="Pelec",  guess=350.0,  units="W",   description="electrical power")
    etam   = V_(name="etam",   guess=0.9,    units="-",   description="motor efficiency")
    imot   = V_(name="i",      guess=5.0,    units="amp", description="motor current")
    vmot   = V_(name="v",      guess=100.0,  units="V",   description="motor voltage")
    Kv     = V_(name="Kv",     guess=200.0,  units="rpm/V", description="motor voltage constant")
    Qmax   = V_(name="Qmax",   guess=30.0,   units="N*m", description="motor max torque")
    Wmotor = V_(name="Wmotor", guess=3.0,    units="lbf", description="motor weight")

    z1 = 2.0 - 1.0 / 0.7            # etaadd = 0.7
    Vtip = omega * Rprop / units.rad
    cons += [
        Tprop == Thrust / Nprop,
        etaprop <= 0.85 * etai,
        Tc >= Tprop / (0.5 * rho * V**2 * pi * Rprop**2),
        z2 >= Tc + 1.0,
        etai * (z1 + z2**0.5 / 0.7) <= 2.0,
        lam >= V / Vtip,
        CT >= Tc * lam**2,
        CP <= Qprop * omega / units.rad / (0.5 * rho * Vtip**3 * pi * Rprop**2),
        etaprop >= CT * lam / CP,
        omega <= 10000 * units.rpm,
        Pshaft == Qprop * omega / units.rad,
        (0.5 * 295 * units.m / units.s)**2 >= Vtip**2 + V**2,
        Tm >= Tprop,
        Wprop >= 4e-4 / units.ft**2 * Tm * Rprop**2,
        # motor
        Pelec == vmot * imot,
        etam == Pshaft / Pelec,
        Qmax >= Qprop,
        vmot <= 300 * units.V,
        imot >= Qprop * Kv / units.rad + 4.5 * units.amp,
        vmot >= omega / Kv + imot * 0.033 * units.ohm,
        Wmotor >= 0.8 * units.kg / (units.N * units.m) * Qmax * G_U,
        Kv >= 1 * units.rpm / units.V,
        Kv <= 1000 * units.rpm / units.V,
    ]

    # ================= energy balance ===================================
    Poper = V_(name="Poper", guess=500.0, units="W", description="operating power")
    ESirr = env["esirr"] * units.W * units.hr / units.m**2
    tnight = env["tnight"] * units.hr
    cons += [
        ESirr >= ESday + Ebatt / 0.98 / 0.2 / Ssolar,
        Ebatt * 0.98 >= Poper * tnight + EStwi * 0.2 * Ssolar,
        Poper == PSmin * Ssolar * 0.2,
        Poper / 1.05 >= 200 * units.W + 100 * units.W + Pelec * Nprop,
    ]

    # ================= steady level flight ==============================
    Wtotal = V_(name="Wtotal", guess=430.0, units="lbf", description="aircraft weight")
    Wcent  = V_(name="Wcent",  guess=90.0,  units="lbf", description="center weight")
    Wland  = V_(name="Wland",  guess=9.0,   units="lbf", description="landing gear weight")
    Wwing  = V_(name="Wwinggroup", guess=250.0, units="lbf", description="wing group weight")
    Wpay = C_(name="Wpay", value=11.0, units="lbf", description="payload weight")
    Wavn = C_(name="Wavn", value=22.0, units="lbf", description="avionics weight")

    f.Objective(Wtotal)
    cons += [
        Wtotal <= 0.5 * rho * V**2 * CL * wing.S,
        Thrust >= 0.5 * rho * V**2 * CD * wing.S,
    ]

    # ================= tail sizing and geometry =========================
    cons += [
        0.45 <= htail.S * lboom / wing.S / wing.cmac,   # Vh
        0.02 <= vtail.S * lboom / wing.S / wing.b,      # Vv
        boom.w[0] <= wing.tau * wing.croot,
        vtail.tau >= 0.09,
        htail.tau >= 0.06,
        wing.tau <= 0.144,
        wing.tau >= DAI1336A["bounds"]["tau"][0],
        wing.tau <= DAI1336A["bounds"]["tau"][1],
    ]

    # ================= wing loading =====================================
    # Two load cases, as the source has. The gust case is usually the sizing
    # one: leaving it out makes the structure too cheap, and the optimizer
    # answers with too much span and too thin a section (AR 48.8 vs 38.1 and
    # tau pinned at the bottom of its range rather than the top).
    #
    # Manoeuvre: N-g on the centre weight, distributed by chord.
    _, c = _beam(f, "wingg", Nwing, wing.b, wing.spar.I, wing.spar.Sy,
                 lambda i: 2.0 * 1.5 * Wcent / wing.b * wing_cbar[i])
    cons += c

    # Gust: adds the incremental lift from the gust angle of attack. Ww is the
    # wing *group* weight (structure + battery + solar), which relieves the
    # root bending, so it appears as (1 + Ww/W).
    ARCTAN_FIT = dict(ftype="MA", K=1, d=1, a1=1.0,
                      c=[0.9460414492363466], e=[[0.9960249757710423]],
                      rms_err=0.039722989129247634)
    eta_w = np.linspace(0.0, 1.0, Nwing)
    cosm1 = np.hstack([1e-10, 1 - np.cos(eta_w[1:] * pi / 2)])
    agust = V_(name="agust", guess=0.05, units="-", size=Nwing,
               description="gust angle of attack")
    vgust = 5.0 * units.m / units.s          # solar sets winggust.vgust = 5
    for i in range(Nwing):
        cons += fit_constraints(ARCTAN_FIT, agust[i], [cosm1[i] * vgust / V],
                                mfac=1.0 + ARCTAN_FIT["rms_err"])
    _, c = _beam(f, "winggust", Nwing, wing.b, wing.spar.I, wing.spar.Sy,
                 lambda i: 2.0 * 1.5 * Wcent / wing.b * wing_cbar[i]
                 * (1 + 2 * pi * agust[i] / CL * (1 + Wwing / Wcent)))
    cons += c

    # ================= tail spar loading ================================
    # Each tail carries its own max-download case, W = qne*S*CLmax, through
    # the same beam chain as the wing. Without it the tail spars are sized
    # only by minimum gauge and the empennage comes out light.
    for tag, surf, cbar_t in (("htailg", htail, htail_cbar),
                              ("vtailg", vtail, vtail_cbar)):
        Wt_ = V_(name=f"{tag}_W", guess=50.0, units="lbf",
                 description=f"{tag} load")
        cons.append(Wt_ == qne_ph * surf.S * 1.39)      # CLmax = 1.39
        _, c = _beam(f, tag, Ntail, surf.b, surf.spar.I,
                     surf.spar.Sy, lambda i, W=Wt_, sf=surf, cb=cbar_t:
                     W / sf.b * cb[i])
        cons += c

    # ================= weight buildup ===================================
    cons += [
        Wland >= 0.02 * Wtotal,
        Wwing >= wing.W + Wbatt + Wsolar,
        Wcent >= Wpay + Wavn + Wemp + Wmotor * Nprop,
        Wtotal / 1.05 >= (Wpay + Wavn + Wland + Wsolar + wing.W + Wbatt
                          + Wemp + Nprop * (Wmotor + Wprop)),
    ]

    f.ConstraintList(cons)
    return f


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from harness import solve_edi, feasibility

    fm = build(latitude=20)
    sol, obj, note = solve_edi(fm)
    nv, worst, where = feasibility(fm)
    print(f"Wtotal = {obj:.5g} lbf     (gpkit npod0_gp_lat20 = 436.4348)")
    print(f"feasibility: {nv} violated, worst rel {worst:.2e}"
          + (f" at {where}" if where else ""))
    if note:
        print("note:", note)
    for k in ("wing_S", "wing_AR", "wing_b", "V", "Vwind", "rho", "CL",
              "E", "Wbatt", "Ssolar", "Wsolar", "PSmin", "Poper", "Wemp"):
        if k in sol:
            print(f"   {k:10s} {sol[k]:14.6g}")
