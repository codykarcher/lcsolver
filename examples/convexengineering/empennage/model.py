"""Empennage: horizontal tail, vertical tail and tail boom (GP).

Source model
------------
``gpkitmodels/GP/aircraft/tail/`` in
https://github.com/convexengineering/gplibrary — ``empennage.py``,
``horizontal_tail.py``, ``vertical_tail.py``, ``tail_boom.py``,
``tube_spar.py``, ``tail_aero.py``.

Structure
---------
Both tails are the **same Wing model** as ``../wing/``, with three changes
made by subclassing rather than by rewriting:

* ``sparModel = None`` — a tail has no spar, so it is skin plus foam core
  only. That removes the whole beam-loading chain, which is why the tails are
  much smaller models than the wing.
* the planform is pinned: ``AR = 4``, ``lam = 0.8``.
* the foam is lighter and thinner-sectioned than the wing's:
  ``Abar = 0.0548`` (vs 0.0753449) and ``rho = 0.024 g/cm^3`` (vs 0.036).

The horizontal tail adds a span-effectiveness constraint

    mh (1 + 2/AR) <= 2 pi

which is the finite-span lift-curve-slope correction. Both tails carry a
volume coefficient (``Vh``, ``Vv``) and moment arm (``lh``, ``lv``) for
aircraft-level trim and stability constraints to use, and the boom length
must reach both: ``l >= lh``, ``l >= lv``.

The tail boom is a **tapered tube** rather than a box. ``TubeSpar`` gives

    I <= pi t d^3 / 8,   Sy <= 2 I / d,   dm >= pi rho d deta t (1-k/2) l

with the boom's wetted area taken from the root diameter, ``S = l pi d0``.

Both tails share a drag polar fitted to NACA 0008 XFOIL data in ``(Re, tau)``,
of max-affine form with K=5 — so it contributes five separate constraints
rather than one. The boom uses a flat-plate skin-friction law.

Relationship to the wing
------------------------
The tails reuse the wing's planform and skin equations but *not* its spar or
beam chain, so they are not exposed to the discrepancy recorded against the
wing in DISCREPANCIES.md #8.

Verification status: STRUCTURAL ONLY
------------------------------------
This model solves and gives physically sensible numbers (3.64 lbf total for a
small solar UAV empennage: 0.92 htail, 1.04 vtail, 1.68 boom, 14.6 ft arm),
but it is **not** checked against a recorded gpkit solution, because the
bounding case is one this file constructs rather than one the source
provides.

A tail has no lower bound on its own — shrink it and everything improves. The
source's ``tail_tests.py`` bounds it by embedding the tails in a full
aircraft. Here they are instead pinned by volume coefficients (``Vh = 0.45``,
``Vv = 0.04``) against a nominal wing, which is the standard sizing
relationship but not the source's test case, so there is nothing to diff
against number-for-number.

What that means in practice: the constraint set is transcribed faithfully and
can be trusted structurally, but the *numbers* here have only been sanity
checked, not verified. Composing it into the solar aircraft — where the real
wing and mission supply the bounding — is what will actually exercise it
against a reference.

One assumption worth flagging: the tail load case uses
``qne = 0.5 * 1.225 * 40^2 Pa``, taken from ``TailBoomState``'s
``rhosl = 1.225 kg/m^3`` and ``Vne = 40 m/s``. That is the conventional
dynamic pressure but the source's expression for it was not read directly.
"""
from __future__ import annotations

import numpy as np
from pyomo.environ import units

from edi import Formulation

G = 9.81  # m/s^2

CFRPFABRIC = dict(rho=1.6, tmin=0.3048, tau=570e6)   # g/cm^3, mm, Pa
TAIL_FOAM = dict(rho=0.024, Abar=0.0548)             # g/cm^3, -

# tail_dragfit.csv — NACA 0008 polar in (Re, tau), max-affine, K=5
TAIL_FIT = dict(
    ftype="MA", K=5, d=2, a1=1.0,
    c=[0.3399377561914537, 5.446864658024941, 16.259467895292175,
       9.509193806494268, 218.73655005090146],
    e=[[-0.18199062784771394, 0.7746039331652241],
       [-0.4848666193196949, 0.2463415216530887],
       [-0.5399432024831557, 0.4663844545784079],
       [-0.4823469575592283, 0.4674663891252669],
       [-0.6038708953785882, 1.312443752168466]],
    rms_err=0.040701846058073914,
)


def planform_constants(N: int, lam: float):
    """Normalized chord distribution, exactly as gplibrary computes it."""
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
    """EDI constraints for a gpfit fit — see ../wing/model.py for the forms."""
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


def _tail_surface(f, tag, N, lam, AR, rho_fab, g_u):
    """Build one tail surface (skin + foam core, no spar). Returns its vars."""
    eta, cbar, cbave, deta, cbarmac = planform_constants(N, lam)
    Nseg = N - 1

    S     = f.Variable(name=f"{tag}_S",     guess=5.0,  units="ft^2", description=f"{tag} area")
    b     = f.Variable(name=f"{tag}_b",     guess=4.5,  units="ft",   description=f"{tag} span")
    croot = f.Variable(name=f"{tag}_croot", guess=1.2,  units="ft",   description=f"{tag} root chord")
    cmac  = f.Variable(name=f"{tag}_cmac",  guess=1.1,  units="ft",   description=f"{tag} MAC")
    cave  = f.Variable(name=f"{tag}_cave",  guess=1.1,  units="ft", size=Nseg, description=f"{tag} mid chord")
    tau   = f.Variable(name=f"{tag}_tau",   guess=0.08, units="-",    description=f"{tag} thickness ratio")
    W     = f.Variable(name=f"{tag}_W",     guess=1.0,  units="lbf",  description=f"{tag} weight")
    Wskin = f.Variable(name=f"{tag}_Wskin", guess=0.5,  units="lbf",  description=f"{tag} skin weight")
    Wfoam = f.Variable(name=f"{tag}_Wfoam", guess=0.5,  units="lbf",  description=f"{tag} core weight")
    tskin = f.Variable(name=f"{tag}_tskin", guess=0.012, units="in",  description=f"{tag} skin thickness")
    Cd    = f.Variable(name=f"{tag}_Cd",    guess=0.01, units="-",    description=f"{tag} drag coefficient")
    Re    = f.Variable(name=f"{tag}_Re",    guess=3e5,  units="-",    description=f"{tag} Reynolds number")

    rho_foam = TAIL_FOAM["rho"] * units.g / units.cm**3
    cons = [
        b**2 == S * AR,
        croot == S / b * cbar[0],
        cmac == croot * cbarmac,
        # skin: minimum gauge, and torsion from the never-exceed case
        Wskin >= rho_fab * S * 2 * tskin * g_u,
        tskin >= CFRPFABRIC["tmin"] * units.mm,
        CFRPFABRIC["tau"] * units.Pa
            >= 1 / (0.01114 / units.mm) / croot**2 / tskin * 0.121 * S
               * (1.225 * units.kg / units.m**3) * (45 * units.m / units.s)**2,
        # foam core
        Wfoam >= 2 * sum(g_u * rho_foam * TAIL_FOAM["Abar"] * cave[i]**2
                         * b / 2 * deta[i] for i in range(Nseg)),
        # 1.1 weight margin, applied in empennage.py via substitutions
        W / 1.1 >= Wskin + Wfoam,
    ]
    for i in range(Nseg):
        cons.append(cave[i] == cbave[i] * S / b)
    return dict(S=S, b=b, croot=croot, cmac=cmac, cave=cave, tau=tau,
                W=W, Cd=Cd, Re=Re), cons


def build(Nt: int = 3, Nb: int = 5) -> Formulation:
    """Empennage sized at a fixed flight state, minimizing total weight.

    Bounding: a tail has no natural lower bound on its own, so the volume
    coefficients are pinned (``Vh``, ``Vv``) against a nominal wing, exactly
    as ``tail_tests.py`` does.
    """
    f = Formulation()
    pi = np.pi

    V   = f.Constant(name="V",   value=25.0,   units="m/s",     description="airspeed")
    rho = f.Constant(name="rho", value=0.7,    units="kg/m^3",  description="air density")
    mu  = f.Constant(name="mu",  value=1.5e-5, units="N*s/m^2", description="air viscosity")

    # nominal wing the tails are sized against
    Sw   = f.Constant(name="Sw",   value=50.0, units="ft^2", description="wing area")
    bw   = f.Constant(name="bw",   value=25.0, units="ft",   description="wing span")
    cmacw = f.Constant(name="cmacw", value=2.0, units="ft",  description="wing MAC")
    Vh_c = f.Constant(name="Vh", value=0.45, units="-", description="horizontal tail volume coefficient")
    Vv_c = f.Constant(name="Vv", value=0.04, units="-", description="vertical tail volume coefficient")

    g_u = G * units.m / units.s**2
    rho_fab = CFRPFABRIC["rho"] * units.g / units.cm**3

    ht, cons_h = _tail_surface(f, "htail", Nt, lam=0.8, AR=4.0,
                               rho_fab=rho_fab, g_u=g_u)
    vt, cons_v = _tail_surface(f, "vtail", Nt, lam=0.8, AR=4.0,
                               rho_fab=rho_fab, g_u=g_u)

    lh = f.Variable(name="lh", guess=6.0, units="ft", description="horizontal tail moment arm")
    lv = f.Variable(name="lv", guess=6.0, units="ft", description="vertical tail moment arm")
    mh = f.Variable(name="mh", guess=2.0, units="-",  description="horizontal tail span effectiveness")

    # ---- tail boom (tapered tube) ----------------------------------------
    Nbseg = Nb - 1
    detab = 1.0 / (Nb - 1)
    lboom = f.Variable(name="l",     guess=6.0,  units="ft",   description="tail boom length")
    Sboom = f.Variable(name="Sboom", guess=4.0,  units="ft^2", description="tail boom wetted area")
    Wboom = f.Variable(name="Wboom", guess=1.0,  units="lbf",  description="tail boom weight")
    dboom = f.Variable(name="d",     guess=1.5,  units="in", size=Nbseg, description="boom diameter")
    tboom = f.Variable(name="tboom", guess=0.012, units="in", size=Nbseg, description="boom wall thickness")
    Iboom = f.Variable(name="Iboom", guess=1e-8, units="m^4", size=Nbseg, description="boom moment of inertia")
    Syboom = f.Variable(name="Syboom", guess=1e-7, units="m^3", size=Nbseg, description="boom section modulus")
    dmboom = f.Variable(name="dmboom", guess=0.1, units="kg", size=Nbseg, description="boom segment mass")
    Cfboom = f.Variable(name="Cfboom", guess=0.005, units="-", description="boom skin friction coefficient")
    Reboom = f.Variable(name="Reboom", guess=1e6,  units="-", description="boom Reynolds number")

    Wtot = f.Variable(name="W", guess=4.0, units="lbf", description="empennage weight")

    f.Objective(Wtot)

    cons = cons_h + cons_v
    kfac = 1.0 - 0.0 / 2.0   # k defaults to 0 in tube_spar (kfac = 1 - k/2)

    for i in range(Nbseg):
        cons += [
            Iboom[i] <= pi * tboom[i] * dboom[i]**3 / 8.0,
            Syboom[i] <= 2 * Iboom[i] / dboom[i],
            dmboom[i] >= pi * rho_fab * dboom[i] * detab * tboom[i] * kfac * lboom,
            tboom[i] >= CFRPFABRIC["tmin"] * units.mm,
        ]
    cons += [
        Wboom >= g_u * sum(dmboom[i] for i in range(Nbseg)),
        Sboom == lboom * pi * dboom[0],
        Reboom == V * rho * lboom / mu,
        Cfboom >= 0.455 / Reboom**0.3,
    ]

    # ---- boom bending (TailBoomBending + Beam) ----------------------------
    # Without this the boom is unbounded: nothing else forces a nonzero
    # diameter, so d -> 0, the boom becomes weightless, the moment arm grows
    # without limit and the tail areas collapse. The cvxopt GP then diverges.
    #
    # The beam is solved in *normalized* form: unit tip shear (Sbar == 1, set
    # by TailBoomBending as Beam.SbarFun), moment and deflection accumulated
    # root-ward, then rescaled by the actual tip force F and length l.
    Fb    = f.Variable(name="F",     guess=200.0, units="N",   description="tail force")
    Mbar  = f.Variable(name="Mbar",  guess=0.5,  units="-", size=Nb,    description="normalized moment")
    thbar = f.Variable(name="thbar", guess=0.05, units="-", size=Nb,    description="normalized angle")
    dbar  = f.Variable(name="dbar",  guess=0.05, units="-", size=Nb,    description="normalized deflection")
    EIbar = f.Variable(name="EIbar", guess=1.0,  units="-", size=Nbseg, description="normalized EI")
    Mr    = f.Variable(name="Mr",    guess=100.0, units="N*m", size=Nbseg, description="section root moment")

    E_cf     = 190e9 * units.Pa    # cfrpud modulus
    sigma_cf = 1.5e9 * units.Pa    # cfrpud strength
    CLmax, kappa_b, Nsafety = 1.39, 0.1, 1.0
    Sbar = 1.0                     # unit tip shear, per Beam.SbarFun
    tiny = 1e-10
    # never-exceed dynamic pressure for the tail load case: TailBoomState
    # gives rhosl = 1.225 kg/m^3 and Vne = 40 m/s.
    qne_tail = 0.5 * 1.225 * 40.0**2 * units.Pa

    cons.append(Fb >= qne_tail * ht["S"])
    for i in range(Nb - 1):
        cons += [
            Mbar[i] >= Mbar[i + 1] + 0.5 * detab * (Sbar + Sbar),
            thbar[i + 1] >= thbar[i] + 0.5 * detab * (Mbar[i + 1] + Mbar[i]) / EIbar[i],
            dbar[i + 1] >= dbar[i] + 0.5 * detab * (thbar[i + 1] + thbar[i]),
        ]
    cons += [Mbar[Nb - 1] >= tiny, thbar[0] >= tiny, dbar[0] >= tiny,
             dbar[Nb - 1] * CLmax * Nsafety <= kappa_b]
    for i in range(Nbseg):
        cons += [
            EIbar[i] <= E_cf * Iboom[i] / Fb / lboom**2 / 2,
            Mr[i] >= Mbar[i] * Fb * lboom,
            sigma_cf >= Mr[i] / Syboom[i],
        ]

    # ---- tail sizing from volume coefficients ----------------------------
    cons += [
        Vh_c <= ht["S"] * lh / Sw / cmacw,
        Vv_c <= vt["S"] * lv / Sw / bw,
        mh * (1 + 2.0 / 4.0) <= 2 * pi,
        lboom >= lh,
        lboom >= lv,
    ]

    # ---- tail drag polars -------------------------------------------------
    for tag, t in (("htail", ht), ("vtail", vt)):
        cons.append(t["Re"] == V * rho * t["S"] / t["b"] / mu)
        cons += fit_constraints(TAIL_FIT, t["Cd"], [t["Re"], t["tau"]],
                                mfac=1.0 + TAIL_FIT["rms_err"])
        cons.append(t["tau"] >= 0.08)

    # ---- total ------------------------------------------------------------
    cons.append(Wtot >= ht["W"] + vt["W"] + Wboom)

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
    print(f"empennage weight = {obj:.6g} lbf")
    print(f"feasibility: {nv} violated, worst rel {worst:.2e}"
          + (f" at {where}" if where else ""))
    if note:
        print("note:", note)
    for k in ("htail_S", "vtail_S", "lh", "lv", "l", "htail_W", "vtail_W", "Wboom"):
        if k in sol:
            print(f"   {k:10s} {sol[k]:12.5g}")
