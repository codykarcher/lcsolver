"""Empennage: horizontal tail, vertical tail and tail boom (GP).

Source model
------------
``gpkitmodels/GP/aircraft/tail/`` in
https://github.com/convexengineering/gplibrary — ``empennage.py``,
``horizontal_tail.py``, ``vertical_tail.py``, ``tail_boom.py``,
``tube_spar.py``, ``tail_aero.py``.

The case rebuilt here is ``test_emp`` from ``tail_tests.py``: an empennage of
*fixed* weight (10 lbf) and boom length (5 ft), sized against a nominal wing,
minimizing total tail drag

    Cd_htail + Cd_vtail + Cf_tailboom

What drives this case
---------------------
Worth understanding before reading the numbers, because it is the opposite of
the usual sizing logic. Drag *falls* with tail area here — a bigger tail flies
at higher Reynolds number, and the NACA 0008 polar drops with Re — so the
objective pushes area up. What stops it is the fixed 10 lbf weight budget:

    W_emp / mfac >= W_htail + W_vtail + W_boom

Both tails therefore grow until the budget is spent, and since they are
symmetric in both the objective and the budget they come out **identical**
(S = 12.167 ft^2 each), even though their volume coefficients differ. The
volume-coefficient constraints (Vh = 0.4, Vv = 0.04) require only 5.0 and
8.0 ft^2 respectively, so neither is binding.

Structure
---------
Both tails are the same ``Wing`` model as ``../wing/`` with three changes:

* ``sparModel = None`` — a tail has no spar, so it is skin plus foam core
  only, which is why there is no beam chain on the surfaces themselves;
* the planform is pinned: ``AR = 4``, ``lam = 0.8``, ``tau = 0.08``;
* the foam is lighter and thinner-sectioned than the wing's:
  ``Abar = 0.0548`` (vs 0.0753449) and ``rho = 0.024 g/cm^3`` (vs 0.036).

The tail boom is a **tapered tube** (``TubeSpar``), not a box:

    I <= pi t d^3 / 8,   Sy <= 2I/d,   dm >= pi rho d deta t (1-k/2) l

with ``k = 0.8`` the taper index, so ``kfac = 0.6``. Its wetted area comes
from the root diameter, ``S = l pi d0``, and ``b = 2 l``.

The boom is sized by **bending, not by its own drag**: each tail's maximum
download is applied through a normalized cantilever with unit tip shear, then
rescaled by the actual tip force and length. Without those two load cases the
boom diameter has no lower bound — thinner is both lighter and lower drag —
and the whole empennage collapses toward zero.

Verification
------------
Matches the gpkit ``test_emp`` solution on all 20 compared variables to
better than 1.1e-3, and the objective to 7e-6 (0.01113479 against
0.011134710).

The reference is solved with ``use_leqs=False`` and with the beam tip values
relaxed to 1e-3, which is what ``tail_tests.py`` does under cvxopt; both are
reproduced here.
"""
from __future__ import annotations

import numpy as np
from pyomo.environ import units

from lcsolver import Formulation

G = 9.81
G_U = G * units.m / units.s**2

# gplibrary GP/materials defaults
CFRPFABRIC = dict(rho=1.6, E=150e9, tmin=0.3048, tau=570e6, sigma=400e6)
TAIL_FOAM = dict(rho=0.024, Abar=0.0548)   # overridden by HorizontalTail/VerticalTail

# tail_dragfit.csv — NACA 0008 polar in (Re, tau), max-affine with K=5
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
    """LCsolver constraints for a gpfit fit. K>1 max-affine gives one per term."""
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


def build(Ntail: int = 3, Nboom: int = 2, tip_relax: float = 1e-3) -> Formulation:
    """Empennage at the ``test_emp`` operating point."""
    f = Formulation()
    V_, C_ = f.Variable, f.Constant
    pi = np.pi
    _, cbar, cbave, deta, cbarmac = planform_constants(Ntail, 0.8)
    Nseg = Ntail - 1

    # ---- fixed inputs from test_emp --------------------------------------
    Sw   = C_(name="Sw",   value=50.0, units="ft^2", description="nominal wing area")
    bw   = C_(name="bw",   value=20.0, units="ft",   description="nominal wing span")
    cmac = C_(name="cmac", value=15.0, units="in",   description="nominal wing MAC")
    Wemp = C_(name="Wemp", value=10.0, units="lbf",  description="empennage weight")
    lboom = C_(name="l",   value=5.0,  units="ft",   description="tail boom length")
    Vh   = C_(name="Vh",   value=0.4,  units="-",    description="horizontal tail volume coefficient")
    Vv   = C_(name="Vv",   value=0.04, units="-",    description="vertical tail volume coefficient")
    tau  = C_(name="tau",  value=0.08, units="-",    description="tail thickness ratio")

    # ---- flight state (gplibrary FlightState) ----------------------------
    V   = C_(name="V",   value=50.0,   units="m/s",     description="airspeed")
    rho = C_(name="rho", value=1.255,  units="kg/m^3",  description="air density")
    mu  = C_(name="mu",  value=1.5e-5, units="N*s/m^2", description="air viscosity")
    qne = C_(name="qne", value=1.2 * 1.255 * 50.0**2, units="kg/s^2/m",
             description="never-exceed dynamic pressure")

    rho_fab = CFRPFABRIC["rho"] * units.g / units.cm**3
    rho_foam = TAIL_FOAM["rho"] * units.g / units.cm**3

    cons = []
    surfaces = {}
    for tag in ("htail", "vtail"):
        S     = V_(name=f"{tag}_S",     guess=12.0, units="ft^2", description=f"{tag} area")
        b     = V_(name=f"{tag}_b",     guess=7.0,  units="ft",   description=f"{tag} span")
        croot = V_(name=f"{tag}_croot", guess=1.9,  units="ft",   description=f"{tag} root chord")
        cmac_t = V_(name=f"{tag}_cmac", guess=1.75, units="ft",   description=f"{tag} MAC")
        cave  = V_(name=f"{tag}_cave",  guess=1.8,  units="ft", size=Nseg, description=f"{tag} mid chord")
        W     = V_(name=f"{tag}_W",     guess=4.6,  units="lbf",  description=f"{tag} weight")
        Wsk   = V_(name=f"{tag}_Wskin", guess=2.4,  units="lbf",  description=f"{tag} skin weight")
        Wcr   = V_(name=f"{tag}_Wcore", guess=1.7,  units="lbf",  description=f"{tag} core weight")
        tsk   = V_(name=f"{tag}_tskin", guess=0.012, units="in",  description=f"{tag} skin thickness")
        Cd    = V_(name=f"{tag}_Cd",    guess=0.004, units="-",   description=f"{tag} drag coefficient")
        Re    = V_(name=f"{tag}_Re",    guess=2.2e6, units="-",   description=f"{tag} Reynolds number")

        cons += [
            b**2 == S * 4.0,                      # AR = 4
            croot == S / b * cbar[0],
            cmac_t == croot * cbarmac,
            # WingSkin: minimum gauge plus a torsional requirement
            Wsk >= rho_fab * S * 2 * tsk * G_U,
            tsk >= CFRPFABRIC["tmin"] * units.mm,
            CFRPFABRIC["tau"] * units.Pa
                >= 1 / (0.01114 / units.mm) / croot**2 / tsk * 0.121 * S
                   * (1.225 * units.kg / units.m**3) * (45 * units.m / units.s)**2,
            # WingCore, with the tails' lighter foam and thinner section
            Wcr >= 2 * sum(G_U * rho_foam * TAIL_FOAM["Abar"] * cave[i]**2
                           * b / 2 * deta[i] for i in range(Nseg)),
            W / 1.1 >= Wsk + Wcr,                 # mfac = 1.1 per tail
            # NACA 0008 polar; Re is built on the mean chord S/b
            Re == V * rho * S / b / mu,
        ]
        cons.append(cave == cbave * S / b)
        cons += fit_constraints(TAIL_FIT, Cd, [Re, tau],
                                mfac=1.0 + TAIL_FIT["rms_err"])
        surfaces[tag] = dict(S=S, b=b, croot=croot, cmac=cmac_t, W=W, Cd=Cd, Re=Re)

    ht, vt = surfaces["htail"], surfaces["vtail"]

    # ---- tail boom: tapered tube -----------------------------------------
    Nbseg = Nboom - 1
    detab = 1.0 / (Nboom - 1)
    Sboom = V_(name="Sboom", guess=13.0, units="ft^2", description="tail boom wetted area")
    Wboom = V_(name="Wboom", guess=0.8,  units="lbf",  description="tail boom weight")
    dboom = V_(name="d",     guess=10.0, units="in", size=Nbseg, description="boom diameter")
    tboom = V_(name="t",     guess=0.012, units="in", size=Nbseg, description="boom wall thickness")
    Iboom = V_(name="I",     guess=2.1e-6, units="m^4", size=Nbseg, description="boom moment of inertia")
    Syboom = V_(name="Sy",   guess=1.6e-5, units="m^3", size=Nbseg, description="boom section modulus")
    dmboom = V_(name="dm",   guess=0.36, units="kg", size=Nbseg, description="boom segment mass")
    Cftb  = V_(name="Cftb",  guess=0.004, units="-",  description="boom skin friction coefficient")
    Retb  = V_(name="Retb",  guess=1.3e6, units="-",  description="boom Reynolds number")

    KFAC = 1.0 - 0.8 / 2.0          # k = 0.8 taper index
    cons += [
        Iboom <= pi * tboom * dboom**3 / 8.0,
        Syboom <= 2 * Iboom / dboom,
        dmboom >= pi * rho_fab * dboom * detab * tboom * KFAC * lboom,
        tboom >= CFRPFABRIC["tmin"] * units.mm,
    ]
    cons += [
        Wboom >= G_U * sum(dmboom[i] for i in range(Nbseg)),
        Sboom == lboom * pi * dboom[0],
        Retb == V * rho * lboom / mu,
        Cftb >= 0.455 / Retb**0.3,
    ]

    # ---- objective: total tail drag --------------------------------------
    f.Objective(ht["Cd"] + vt["Cd"] + Cftb)

    # ---- the binding constraint: a fixed weight budget --------------------
    cons.append(Wemp / 1.0 >= ht["W"] + vt["W"] + Wboom)

    # ---- volume coefficients (not binding at this operating point) -------
    cons += [
        Vh <= ht["S"] * lboom / Sw / cmac,
        Vv <= vt["S"] * lboom / Sw / bw,
    ]

    # ---- boom bending, one case per tail ---------------------------------
    # Normalized cantilever with unit tip shear (Beam.SbarFun = [1]*N), then
    # rescaled by the actual tip force F = qne*S*CLmax and the length.
    E_fab = CFRPFABRIC["E"] * units.Pa
    sig_fab = CFRPFABRIC["sigma"] * units.Pa
    CLMAX, KAPPA, NSAFETY = 1.39, 0.1, 1.0
    for tag, surf in (("hbend", ht), ("vbend", vt)):
        Fb    = V_(name=f"{tag}_F",   guess=1000.0, units="N", description=f"{tag} tail force")
        Mbar  = V_(name=f"{tag}_Mbar", guess=1.0, units="-", size=Nboom, description=f"{tag} normalized moment")
        thbar = V_(name=f"{tag}_thbar", guess=0.03, units="-", size=Nboom, description=f"{tag} normalized angle")
        dbar  = V_(name=f"{tag}_dbar", guess=0.03, units="-", size=Nboom, description=f"{tag} normalized deflection")
        EIbar = V_(name=f"{tag}_EIbar", guess=100.0, units="-", size=Nbseg, description=f"{tag} normalized EI")
        Mr    = V_(name=f"{tag}_Mr",  guess=1000.0, units="N*m", size=Nbseg, description=f"{tag} section root moment")
        cons.append(Fb >= qne * surf["S"])
        # Beam recursion along the boom: each station against the next.
        cons += [
            Mbar[:-1] >= Mbar[1:] + 0.5 * detab * (1.0 + 1.0),
            thbar[1:] >= thbar[:-1] + 0.5 * detab * (Mbar[1:] + Mbar[:-1]) / EIbar,
            dbar[1:] >= dbar[:-1] + 0.5 * detab * (thbar[1:] + thbar[:-1]),
        ]
        cons += [Mbar[Nboom - 1] >= tip_relax,
                 thbar[0] >= tip_relax, dbar[0] >= tip_relax,
                 dbar[Nboom - 1] * CLMAX * NSAFETY <= KAPPA]
        cons += [
            EIbar <= E_fab * Iboom / Fb / lboom**2 / 2,
            Mr >= Mbar[:Nbseg] * Fb * lboom,
            sig_fab >= Mr / Syboom,
        ]

    f.ConstraintList(cons)
    return f


ALIASES = {
    "Empennage.HorizontalTail.Planform.S": "htail_S",
    "Empennage.HorizontalTail.Planform.b": "htail_b",
    "Empennage.HorizontalTail.Planform.croot": "htail_croot",
    "Empennage.HorizontalTail.Planform.cmac": "htail_cmac",
    "Empennage.HorizontalTail.W": "htail_W",
    "Empennage.HorizontalTail.WingSkin.W": "htail_Wskin",
    "Empennage.HorizontalTail.WingSkin.t": "htail_tskin",
    "Empennage.HorizontalTail.WingCore.W": "htail_Wcore",
    "Empennage.VerticalTail.Planform.S": "vtail_S",
    "Empennage.VerticalTail.Planform.b": "vtail_b",
    "Empennage.VerticalTail.Planform.croot": "vtail_croot",
    "Empennage.VerticalTail.Planform.cmac": "vtail_cmac",
    "Empennage.VerticalTail.W": "vtail_W",
    "Empennage.VerticalTail.WingSkin.W": "vtail_Wskin",
    "Empennage.VerticalTail.WingSkin.t": "vtail_tskin",
    "Empennage.VerticalTail.WingCore.W": "vtail_Wcore",
    "Empennage.TailBoom.W": "Wboom",
    "Empennage.TailBoom.S": "Sboom",
    "TailAero.Cd": "htail_Cd",
    "TailAero.Re": "htail_Re",
}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from harness import solve_edi, compare, load_reference, feasibility

    fm = build()
    sol, obj, note = solve_edi(fm)
    ref = load_reference(Path(__file__).with_name("reference.json"))
    ref = {k: v for k, v in ref.items() if not isinstance(v, list)}
    rep = compare("Empennage (test_emp)", sol, ref, rtol=5e-3,
                  only=sorted(ALIASES), aliases=ALIASES)
    nv, worst, where = feasibility(fm)
    if note:
        rep.notes.append(note)
    rep.notes.append(f"objective Cdh+Cdv+Cftb = {obj:.7g}   (gpkit = 0.011134710)")
    rep.notes.append(f"feasibility: {nv} violated, worst rel {worst:.2e}"
                     + (f" at {where}" if where else ""))
    print(rep)
