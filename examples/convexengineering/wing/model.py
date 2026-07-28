"""Wing box: planform, structure, aerodynamics and beam loading (GP).

STATUS: NOT YET VERIFIED. Solves and lands close, but does not reproduce the
reference. See "Open discrepancy" at the end of this docstring before
building anything on top of it.


Source model
------------
``gpkitmodels/GP/aircraft/wing/`` in
https://github.com/convexengineering/gplibrary — ``wing.py`` (Planform, Wing,
WingAero), ``boxspar.py``, ``wing_skin.py``, ``wing_core.py``,
``sparloading.py``, ``gustloading.py``.

The case rebuilt here is ``wing_test()`` from ``wing_test.py``: a wing of
fixed weight and thickness ratio, minimizing profile drag coefficient,
carrying both a manoeuvre load and a gust load.

This is the keystone subsystem — the solar, gassolar and jho aircraft all
build on it.

Discretization
--------------
The wing is cut into ``N`` spanwise nodes and ``N-1`` segments. Node
quantities (shear ``S``, moment ``M``, deflection angle ``th``, deflection
``w``, load ``q``) have length N; segment quantities (chord ``cave``, spar
inertia ``I``, section modulus ``Sy``, masses) have length N-1.

The beam is integrated from tip to root as a chain of inequality
constraints — shear accumulates the distributed load, moment accumulates
shear, angle accumulates moment/EI, deflection accumulates angle. Each is a
posynomial in the next, so the whole beam solve is GP-representable:

    S[i]   >= S[i+1]  + 0.5 deta (b/2) (q[i] + q[i+1])
    M[i]   >= M[i+1]  + 0.5 deta (b/2) (S[i] + S[i+1])
    th[i+1]>= th[i]   + 0.5 deta (b/2) (M[i+1] + M[i]) / (E I)
    w[i+1] >= w[i]    + 0.5 deta (b/2) (th[i+1] + th[i])

The tip values are pinned to small positive constants (``Stip``, ``Mtip``,
``throot``, ``wroot``) rather than zero, because a GP cannot represent an
exact zero. The originals loosen these to 1e-2 when solving with cvxopt
rather than MOSEK; that is reproduced here.

Fixed planform
--------------
``lam`` (taper) and ``eta`` (node stations) are *constants* in the source,
so the whole normalized chord distribution ``cbar``, its segment averages
``cbave``, the segment widths ``deta`` and the normalized MAC ``cbarmac``
are compile-time constants rather than variables. They are computed here by
the same formulas as gpkit's linked-variable callbacks.

Fitted drag polar
-----------------
Profile drag comes from an XFOIL-derived posynomial fit of the JHO airfoil,
of "SMA" (softmax-affine) form:

    (cdp/mfac)^a  >=  sum_k c_k * CL^e_k0 * Re^e_k1

and the gust angle from a 1-term "MA" fit of an arctangent. Both fit
datasets are transcribed from the CSVs in the source tree.

Open discrepancy
----------------
This rebuild solves but lands about 1.4% below the reference objective
(Cd 0.007095 vs 0.007195 at the 1e-2 tip relaxation) with a noticeably
higher aspect ratio (22.7 vs 20.2), i.e. it is under-constrained somewhere
and buys induced drag with span it should not have.

What has been ruled out:

* the beam chain — printing the reference's constraints shows them
  character-for-character equivalent to the ones built here;
* ``WingCore`` — initially guessed as ``0.5*tau*cave^2``; it is actually
  ``Abar*cave^2`` with a fixed normalized section area ``Abar = 0.0753449``.
  Corrected, which moved the objective from 0.007056 to 0.007095;
* the tip relaxation value — ``wing_test`` uses 1e-1 where ``box_spar`` uses
  1e-2, and the two give materially different answers (0.007682 vs
  0.007195). The reference JSON here was recorded at 1e-2.

What is unexplained: in the reference solution the manoeuvre load ``q`` sits
*well above* the lower bound its own constraint states — at the root,
1605.8 N/m against ``N*W/b*cbar[0] = 422.9 N/m``. The gust case's ``q``
matches its bound exactly. Since ``q`` appears only on the loosening side of
the shear chain, the optimizer should drive it to that bound, so either the
substitution ``loading.substitutions["W"] = 100`` is not reaching the
variable this port assumes, or ``cbar`` (a gpkit *linked* variable, evaluated
at solve time rather than a fixed array) resolves to something other than the
taper-derived values used here.

Resolving that is the next step. Until then this model should not be treated
as a verified reference, and the solar/gassolar/jho aircraft that would build
on it are blocked behind it.
"""
from __future__ import annotations

import numpy as np
from pyomo.environ import units

from edi import Formulation

G = 9.81  # m/s^2, gpkitmodels.g

# --- material properties (gplibrary GP/materials) --------------------------
CFRPUD = dict(rho=1.6, tmin=0.1, sigma=1.5e9, E=190e9)      # g/cm^3, mm, Pa, Pa
CFRPFABRIC = dict(rho=1.6, tmin=0.3048, tau=570e6)          # g/cm^3, mm, Pa
FOAMHD = dict(rho=0.036)                                    # g/cm^3

# --- XFOIL drag polar fit, jho_fitdata.csv (SMA, K=4, d=2) -----------------
JHO_FIT = dict(
    ftype="SMA", K=4, d=2, a1=3.0904265782626554,
    c=[1.0237722536072613e-07, 0.0016819467416519377,
       2.2831557182654378e-06, 175843713.87100798],
    e=[[18.8561449416522, -0.1833195640493263],
       [2.963071814236627, -0.667476893160359],
       [-1.653325906904553, -0.3496409845113647],
       [-0.17866429593832706, -2.713007500787293]],
    rms_err=0.006577663039320368,
)

# --- arctan fit for gust angle, arctan_fit.csv (MA, K=1, d=1) --------------
ARCTAN_FIT = dict(ftype="MA", K=1, d=1, a1=1.0,
                  c=[0.9460414492363466], e=[[0.9960249757710423]],
                  rms_err=0.039722989129247634)


# ---------------------------------------------------------------------------
# Planform constants (gpkit computes these as linked variables)
# ---------------------------------------------------------------------------

def planform_constants(N: int, lam: float = 0.5):
    """Return (eta, cbar, cbave, deta, cbarmac) exactly as gplibrary does."""
    eta = np.linspace(0.0, 1.0, N)
    cbar = np.array([2.0 / (1 + lam) * (1 + (lam - 1) * e) for e in eta])
    cbave = (cbar[:-1] + cbar[1:]) / 2.0
    deta = np.diff(eta)

    # normalized MAC — area-weighted mean of the segment MACs
    lami = cbar[1:] / cbar[:-1]
    maci = 2.0 / 3 * cbar[:-1] * (1 + lami + lami**2) / (1 + lami)
    num = sum((cbar[i] + cbar[i + 1]) / 2 * maci[i] * deta[i]
              for i in range(len(deta)))
    den = sum((cbar[i] + cbar[i + 1]) / 2 * deta[i] for i in range(len(deta)))
    cbarmac = num / den / cbar[0]
    return eta, cbar, cbave, deta, cbarmac


def fit_constraints(fit, ivar, dvars, mfac=1.0):
    """Build the EDI constraint(s) for a gpfit fit.

    Mirrors ``gpfit.fit_constraintset.FitCS``:

        ISMA : 1        >= sum_k c_k prod_i u_i^e_ki / (w/mfac)^alpha_k
        SMA  : (w/mfac)^a >= sum_k c_k prod_i u_i^e_ki
        MA   : (w/mfac)   >= c_k prod_i u_i^e_ki      (one per k)

    With K == 1 the source emits an *equality* instead of an inequality,
    which matters: it pins the variable rather than bounding it.
    """
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
    raise NotImplementedError(f"fit type {fit['ftype']!r}")


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def build(N: int = 5, cvxopt_tip_relax: bool = True) -> Formulation:
    """Wing sized by manoeuvre and gust loads, minimizing profile drag."""
    f = Formulation()
    eta, cbar, cbave, deta, cbarmac = planform_constants(N)
    Nseg = N - 1
    pi = np.pi

    # ---- flight state ----------------------------------------------------
    V   = f.Constant(name="V",   value=50.0,   units="m/s",     description="airspeed")
    rho = f.Constant(name="rho", value=1.255,  units="kg/m^3",  description="air density")
    mu  = f.Constant(name="mu",  value=1.5e-5, units="N*s/m^2", description="air viscosity")
    qne = f.Constant(name="qne", value=1.2 * 1.255 * 50.0**2, units="kg/s^2/m",
                     description="never-exceed dynamic pressure")

    # ---- fixed wing inputs (substituted in wing_test) --------------------
    Wwing = f.Constant(name="W",   value=50.0,  units="lbf", description="wing weight")
    tau   = f.Constant(name="tau", value=0.115, units="-",   description="airfoil thickness ratio")
    Wload = f.Constant(name="Wload", value=100.0, units="lbf", description="loading weight")

    mfac_w = f.Constant(name="mfac_w", value=1.2, units="-", description="wing weight margin")

    # ---- planform --------------------------------------------------------
    S     = f.Variable(name="S",     guess=190.0, units="ft^2", description="wing area")
    AR    = f.Variable(name="AR",    guess=20.0,  units="-",    description="aspect ratio")
    b     = f.Variable(name="b",     guess=60.0,  units="ft",   description="span")
    croot = f.Variable(name="croot", guess=4.0,   units="ft",   description="root chord")
    cmac  = f.Variable(name="cmac",  guess=3.0,   units="ft",   description="mean aerodynamic chord")
    cave  = f.Variable(name="cave",  guess=3.0,   units="ft", size=Nseg,
                       description="mid-section chord")

    # ---- skin ------------------------------------------------------------
    Wskin = f.Variable(name="Wskin", guess=10.0,  units="lbf", description="skin weight")
    tskin = f.Variable(name="tskin", guess=0.012, units="in",  description="skin thickness")

    # ---- box spar (per segment) -------------------------------------------
    Wspar  = f.Variable(name="Wspar",  guess=10.0,  units="lbf", description="spar weight")
    hin    = f.Variable(name="hin",    guess=1.0,   units="in",  size=Nseg, description="height between caps")
    Ispar  = f.Variable(name="I",      guess=1e-6,  units="m^4", size=Nseg, description="spar x moment of inertia")
    Sy     = f.Variable(name="Sy",     guess=1e-5,  units="m^3", size=Nseg, description="section modulus")
    dm     = f.Variable(name="dm",     guess=0.5,   units="kg",  size=Nseg, description="segment spar mass")
    wspar  = f.Variable(name="w",      guess=0.5,   units="in",  size=Nseg, description="spar width")
    tcap   = f.Variable(name="t",      guess=0.02,  units="in",  size=Nseg, description="spar cap thickness")
    tshear = f.Variable(name="tshear", guess=0.02,  units="in",  size=Nseg, description="shear web thickness")
    tcore  = f.Variable(name="tcore",  guess=0.02,  units="in",  size=Nseg, description="core thickness")

    # ---- core / foam ------------------------------------------------------
    Wcore = f.Variable(name="Wcore", guess=5.0, units="lbf", description="core weight")

    # ---- aerodynamics -----------------------------------------------------
    Cd  = f.Variable(name="Cd",  guess=0.007, units="-", description="wing drag coefficient")
    CL  = f.Variable(name="CL",  guess=0.3,   units="-", description="lift coefficient")
    Re  = f.Variable(name="Re",  guess=4e5,   units="-", description="Reynolds number")
    cdp = f.Variable(name="cdp", guess=0.006, units="-", description="profile drag coefficient")

    # ---- loading (manoeuvre and gust share the beam structure) -----------
    def loading_vars(tag):
        return dict(
            q  = f.Variable(name=f"{tag}_q",  guess=50.0, units="N/m", size=N,   description="distributed load"),
            Sh = f.Variable(name=f"{tag}_S",  guess=200.0, units="N",  size=N,   description="shear"),
            M  = f.Variable(name=f"{tag}_M",  guess=200.0, units="N*m", size=N,  description="moment"),
            th = f.Variable(name=f"{tag}_th", guess=0.05, units="-",   size=N,   description="deflection angle"),
            w  = f.Variable(name=f"{tag}_w",  guess=0.5,  units="m",   size=N,   description="deflection"),
        )

    man = loading_vars("SparLoading")
    gus = loading_vars("GustL")
    agust = f.Variable(name="agust", guess=0.05, units="-", size=N, description="gust angle")

    f.Objective(Cd)

    # tip/root pins: a GP cannot express an exact zero. cvxopt needs these
    # looser than MOSEK does, which is what the source does too.
    tiny = 1e-2 if cvxopt_tip_relax else 1e-10
    Stip   = tiny * units.N
    Mtip   = tiny * units.N * units.m
    throot = tiny
    wroot  = tiny * units.m

    Nmax, Nsafety, kappa = 5.0, 1.0, 0.2
    Nload = Nmax * Nsafety
    E     = CFRPUD["E"] * units.Pa
    sigma = CFRPUD["sigma"] * units.Pa

    cons = []

    # ---- planform geometry ------------------------------------------------
    cons += [
        b**2 == S * AR,
        croot == S / b * cbar[0],
        cmac == croot * cbarmac,
    ]
    for i in range(Nseg):
        cons.append(cave[i] == cbave[i] * S / b)

    # ---- skin -------------------------------------------------------------
    rho_fab = CFRPFABRIC["rho"] * units.g / units.cm**3
    cons += [
        Wskin >= rho_fab * S * 2 * tskin * (G * units.m / units.s**2),
        tskin >= CFRPFABRIC["tmin"] * units.mm,
        # torsion: skin must carry the never-exceed pitching moment
        CFRPFABRIC["tau"] * units.Pa
            >= 1 / (0.01114 / units.mm) / croot**2 / tskin * 0.121 * S
               * (1.225 * units.kg / units.m**3) * (45 * units.m / units.s)**2,
    ]

    # ---- box spar ---------------------------------------------------------
    rho_ud   = CFRPUD["rho"] * units.g / units.cm**3
    rho_core = FOAMHD["rho"] * units.g / units.cm**3
    wlim, mfac_s, tcoret = 0.15, 0.97, 0.02
    for i in range(Nseg):
        cons += [
            Ispar[i] / mfac_s <= wspar[i] * tcap[i] * hin[i]**2,
            dm[i] >= (rho_ud * 4 * wspar[i] * tcap[i]
                      + 4 * tshear[i] * rho_fab * (hin[i] + wspar[i])
                      + 2 * rho_core * tcore[i] * (wspar[i] + hin[i])) * b / 2 * deta[i],
            wspar[i] <= wlim * cave[i],
            cave[i] * tau >= hin[i] + 4 * tcap[i] + 2 * tcore[i],
            tcap[i] >= CFRPUD["tmin"] * units.mm,
            Sy[i] * (hin[i] / 2 + 2 * tcap[i] + tcore[i]) <= Ispar[i],
            tshear[i] >= CFRPFABRIC["tmin"] * units.mm,
            tcore[i] >= tcoret * cave[i] * tau,
        ]
    cons.append(Wspar >= 2 * sum(dm[i] for i in range(Nseg)) * (G * units.m / units.s**2))

    # ---- core -------------------------------------------------------------
    # WingCore: foam fills the section. Abar is the *normalized* cross section
    # area of the airfoil (a fixed 0.0753449 in the source), so the section
    # area is Abar*cave^2 — it is not a function of tau.
    ABAR = 0.0753449
    cons.append(Wcore >= 2 * sum(
        (G * units.m / units.s**2) * rho_core * ABAR * cave[i]**2 * b / 2 * deta[i]
        for i in range(Nseg)))

    # ---- wing weight buildup ---------------------------------------------
    cons.append(Wwing / mfac_w >= Wskin + Wspar + Wcore)

    # ---- aerodynamics -----------------------------------------------------
    cons += [
        Cd >= cdp + CL**2 / pi / AR / 0.9,
        Re == rho * V * cmac / mu,
        CL <= 1.3,
    ]
    cons += fit_constraints(JHO_FIT, cdp, [CL, Re], mfac=1.0 + JHO_FIT["rms_err"])

    # ---- beam loading, shared by both load cases -------------------------
    def beam(v, qexpr):
        c = []
        for i in range(N - 1):
            c += [
                v["Sh"][i] >= v["Sh"][i + 1] + 0.5 * deta[i] * (b / 2) * (v["q"][i] + v["q"][i + 1]),
                v["M"][i] >= v["M"][i + 1] + 0.5 * deta[i] * (b / 2) * (v["Sh"][i] + v["Sh"][i + 1]),
                v["th"][i + 1] >= v["th"][i] + 0.5 * deta[i] * (b / 2) * (v["M"][i + 1] + v["M"][i]) / E / Ispar[i],
                v["w"][i + 1] >= v["w"][i] + 0.5 * deta[i] * (b / 2) * (v["th"][i + 1] + v["th"][i]),
            ]
        c += [v["Sh"][N - 1] >= Stip, v["M"][N - 1] >= Mtip,
              v["th"][0] >= throot, v["w"][0] >= wroot,
              v["w"][N - 1] / (b / 2) <= kappa]
        for i in range(Nseg):
            c += [sigma >= v["M"][i] / Sy[i]]
        for i in range(N):
            c.append(v["q"][i] >= qexpr(i))
        return c

    # manoeuvre: uniform N-g load distributed by chord
    cons += beam(man, lambda i: Nload * Wload / b * cbar[i])

    # gust: adds the incremental lift from the gust angle of attack
    cosm1 = np.hstack([1e-10, 1 - np.cos(eta[1:] * pi / 2)])
    vgust = 10.0 * units.m / units.s
    for i in range(N):
        cons += fit_constraints(ARCTAN_FIT, agust[i],
                                [cosm1[i] * vgust / V],
                                mfac=1.0 + ARCTAN_FIT["rms_err"])
    cons += beam(gus, lambda i: Wload * Nload / b * cbar[i]
                 * (1 + 2 * pi * agust[i] / CL * (1 + Wwing / Wload)))

    f.ConstraintList(cons)
    return f


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from harness import solve_edi, feasibility

    f = build()
    sol, obj, note = solve_edi(f)
    nv, worst, where = feasibility(f)
    print(f"objective Cd = {obj:.7g}    (gpkit wing_test = 0.007195170)")
    print(f"feasibility: {nv} violated, worst rel {worst:.2e}"
          + (f" at {where}" if where else ""))
    if note:
        print("note:", note)
    for k in ("S", "AR", "b", "CL", "Re", "cdp", "Wskin", "Wspar", "Wcore"):
        if k in sol:
            print(f"   {k:8s} {sol[k]:14.6g}")
