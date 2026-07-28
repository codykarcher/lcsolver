"""A signomial-programming form of TASOPT's surface profile drag.

Why this exists
---------------
York et al. showed that TASOPT can be recast almost entirely as a signomial
program. The claim is only worth as much as the agreement between the SP form
and the exact physics, so this module puts the two side by side: an EDI
signomial model of ``surfcd``, checked against the verified Fortran-matching
port in ``tasopt_py.aero.drag`` over a sweep of designs.

What has to change to make it GP/SP-compatible
----------------------------------------------
``surfcd`` is close to monomial already -- it is products and powers of taper
ratios, Reynolds numbers and chords. Three things obstruct it:

1. **The spanwise integral factors ``lsfac`` and ``lfac``** are ratios of
   differences, e.g. ``(1 - lambdas^(2+aRexp)) / (1 - lambdas^2)``. Both
   numerator and denominator are signomials, and each is singular where the
   taper ratios coincide. Written directly they are a difference-of-convex
   quotient, which is not signomial-representable as a *constraint*; here
   each is introduced as a variable defined by a signomial equality, which is
   what an SP allows.

2. **The sweep/unsweep blend** ``(fSuns + (1-fSuns) cos^2 L) cos L`` carries a
   ``1 - fSuns``. With ``cos L`` fixed by the sweep, this is affine in
   ``fSuns`` and becomes a signomial equality.

3. **The area sum** ``Ssurf`` and the three-term bracket in ``CDsurf`` are
   posynomials appearing on the *greater* side of their relations, so they are
   signomial rather than GP.

Everything else -- the Reynolds scaling ``(Reco/Reref)^aRexp``, the taper
powers, the chord/span products -- is monomial and passes through untouched.

The exact model is the reference, not the other way round: ``verify()``
solves the SP with the geometry fixed and compares ``CDsurf`` against
``tasopt_py.aero.drag.surfcd``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from pyomo.environ import units

from edi import Formulation


@dataclass(frozen=True)
class Design:
    """One surface. Mirrors the argument list of ``surfcd``."""
    S: float = 105.0
    b: float = 35.0
    bs: float = 12.0
    bo: float = 3.6
    lambdat: float = 0.25
    lambdas: float = 0.65
    sweep: float = 26.0
    co: float = 5.6
    cdf: float = 0.0050
    cdp: float = 0.0025
    Reco: float = 2.0e7
    Reref: float = 1.0e7
    aRexp: float = -0.15
    kSuns: float = 0.5
    fCDcen: float = 1.0


def build(d: Design) -> Formulation:
    """Build the SP form of ``surfcd`` for one design.

    Geometry is fixed by the design; the free variables are the intermediate
    quantities that the signomial equalities define. Minimizing ``CDsurf``
    with those equalities is a feasibility problem with a unique answer, so
    the optimum reproduces the exact routine rather than trading against it.
    """
    f = Formulation()
    V = lambda n, g, u, de: f.Variable(name=n, guess=g, units=u, description=de)
    C = lambda n, v, u, de: f.Constant(name=n, value=v, units=u, description=de)

    cosL = math.cos(d.sweep * math.pi / 180.0)
    etao, etas = d.bo / d.b, d.bs / d.b

    # ---- fixed geometry ---------------------------------------------------
    S = C("S", d.S, "m^2", "reference area")
    b = C("b", d.b, "m", "span")
    co = C("co", d.co, "m", "root chord")
    lambdas = C("lambda_s", d.lambdas, "-", "inner panel taper ratio")
    lambdat = C("lambda_t", d.lambdat, "-", "outer panel taper ratio")
    cdf = C("cdf", d.cdf, "-", "section friction drag coefficient")
    cdp = C("cdp", d.cdp, "-", "section pressure drag coefficient")
    Reco = C("Reco", d.Reco, "-", "Reynolds number at the root chord")
    Reref = C("Reref", d.Reref, "-", "reference Reynolds number")
    kSuns = C("kSuns", d.kSuns, "-", "shock-unsweep area constant")
    fCDcen = C("fCDcen", d.fCDcen, "-", "centre-section drag factor")

    # ---- free intermediates ------------------------------------------------
    Ssurf = V("Ssurf", 100.0, "m^2", "exposed surface area")
    fSuns = V("fSuns", 0.3, "-", "fraction subject to root shock unsweep")
    sweepfac = V("sweepfac", 0.9, "-", "sweep/unsweep blend on pressure drag")
    cdo = V("cdo", 0.006, "-", "section cd at the root chord")
    lsfac = V("lsfac", 1.0, "-", "inner panel spanwise integral factor")
    lfac = V("lfac", 1.0, "-", "outer panel spanwise integral factor")
    bracket = V("bracket", 1.0, "-", "area-weighted spanwise bracket")
    CDsurf = V("CDsurf", 0.007, "-", "overall profile CD")

    f.Objective(CDsurf)

    aR = d.aRexp
    cons = [
        # Exposed area: a posynomial equality, hence signomial.
        Ssurf == co * 0.5 * b * ((1.0 + lambdas) * (etas - etao)
                                 + (lambdas + lambdat) * (1.0 - etas)),
        # Unsweep fraction -- monomial in the fixed geometry.
        fSuns * 0.5 * Ssurf == kSuns * co ** 2,
        # Sweep blend: affine in fSuns, so a signomial equality.
        sweepfac == (fSuns + (1.0 - fSuns) * cosL ** 2) * cosL,
        # Section cd with Reynolds scaling. (Reco/Reref)^aRexp is monomial.
        cdo == (cdf + cdp * sweepfac) * (Reco / Reref) ** aR,
    ]

    # The two spanwise integral factors. Each is a quotient of signomials, so
    # it is introduced as a variable and pinned by a signomial equality --
    # cross-multiplied, so no division by a signomial appears.
    if abs(d.lambdas - 1.0) < 0.02:
        cons += [lsfac == 1.0 - aR * (1.0 - lambdas) / (1.0 + lambdas)]
    else:
        cons += [lsfac * (1.0 - lambdas ** 2)
                 == 2.0 / (2.0 + aR) * (1.0 - lambdas ** (2 + aR))]

    if abs(d.lambdat - d.lambdas) < 0.02:
        cons += [lfac == 1.0 - aR * (lambdas - lambdat) / (lambdas + lambdat)]
    else:
        cons += [lfac * (lambdas ** 2 - lambdat ** 2)
                 == 2.0 / (2.0 + aR) * (lambdas ** (2 + aR)
                                        - lambdat ** (2 + aR))]

    cons += [
        bracket == (2.0 * etao * fCDcen
                    + (1.0 + lambdas) * (etas - etao) * lsfac
                    + (lambdas + lambdat) * (1.0 - etas) * lfac),
        CDsurf == (co * 0.5 * b / S) * cdo * bracket,
    ]

    f.ConstraintList(cons)
    return f


def verify(designs=None, solver: str = "ipopt-convex") -> list:
    """Solve the SP for each design and diff against the exact routine."""
    import pyomo.environ as pyo
    from edi.solvers.solver import solve
    from tasopt_py.aero.drag import surfcd

    designs = designs or DESIGNS
    out = []
    for name, d in designs:
        fm = build(d)
        solve(fm, solver=solver)
        got = float(pyo.value(fm.find_component("CDsurf")))
        exact = surfcd(S=d.S, b=d.b, bs=d.bs, bo=d.bo, lambdat=d.lambdat,
                       lambdas=d.lambdas, sweep=d.sweep, co=d.co, cdf=d.cdf,
                       cdp=d.cdp, Reco=d.Reco, Reref=d.Reref, aRexp=d.aRexp,
                       kSuns=d.kSuns, fCDcen=d.fCDcen).CDsurf
        out.append((name, got, exact, abs(got - exact) / exact))
    return out


DESIGNS = [
    ("baseline", Design()),
    ("unswept", Design(sweep=0.0)),
    ("high sweep", Design(sweep=35.0, lambdat=0.15, lambdas=0.50, Reco=3.5e7)),
    ("near-untapered inner", Design(lambdas=0.995, lambdat=0.30)),
    ("coincident tapers", Design(lambdas=0.60, lambdat=0.59)),
    ("positive Re exponent", Design(lambdas=0.99, lambdat=0.985,
                                    aRexp=0.20, fCDcen=0.6)),
]


if __name__ == "__main__":
    print(f"{'design':24} {'SP':>12} {'TASOPT':>12} {'rel':>10}")
    worst = 0.0
    for name, got, exact, rel in verify():
        worst = max(worst, rel)
        print(f"{name:24} {got:12.7g} {exact:12.7g} {rel:10.2e}")
    print(f"worst relative difference: {worst:.2e}")
