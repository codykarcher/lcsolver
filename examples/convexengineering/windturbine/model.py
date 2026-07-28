"""Wind turbine rotor aerodynamics as a geometric program.

Paper
-----
    W. Hoburg and P. Abbeel, "Fast Wind Turbine Design via Geometric
    Programming", 53rd AIAA/ASME/ASCE/AHS/ASC Structures, Structural
    Dynamics and Materials Conference, 2012.

There is **no published source code** for this model — unlike every other
model in this directory, the paper is the only reference. Verification is
therefore against the two results the paper states exactly:

* the **Betz limit**: with no viscous drag and no tip loss, Cp -> 16/27
  as the tip speed ratio grows (paper section IV.B);
* **Figure 1**: Cp(lambda) curves for c_l/c_d in {5, 10, 20, 40}, with and
  without the Prandtl tip loss correction, at 10 spanwise stations.

Formulation
-----------
Blade-element/vortex theory following Drela's QPROP, cast as a GP. Per
spanwise station the flow field is described by nondimensional factors

    a = v_a/V   axial induction        b = W_a/V = 1 - a
    s = v_t/V   swirl induction        xi = lambda*y + 2s

and the paper's GP-compatible constraint set is

    (14)  s*b   >= (dCp/dy) / (8 y^2 F G)  +  eps*a*b
    (15)  2ab   >= s*lambda*y + s*xi
    (16)  xi^2  >= lambda^2 y^2 + 4ab
    (17)  1     >= a + b
    (25)  1/y   >= 1 + 2 f s / (B a)          Prandtl helper
    (26)  1     >= F^1.56 + 1.655 F^5.51 / f^2.76
    (27)  G     <= 1

Every one of these is (posynomial <= monomial), so the whole model is a GP.

The spanwise discretization is the paper's key trick (section IV.C). Writing
Cp as a *sum* of per-station contributions would bound Cp from below by a
posynomial, and maximizing that is not GP-representable. Instead the bins are
chosen to have **equal power** dCp = Cp/N and variable width, so maximizing
Cp reduces to maximizing the single scalar dCp:

    (20)  1       >= y_N + dy_N/2
    (21)  y_{i+1} >= y_i + dy_i/2 + dy_{i+1}/2

Objective: minimize 1/dCp  (equivalently maximize Cp = N*dCp).

Guessed inputs
--------------
* ``B`` (number of blades) is **a guess of 3**. The paper does not state it
  for Figure 1. It only affects the tip-loss curves; with ``tip_loss=False``
  the model is independent of B, so the solid curves of Figure 1 are
  verifiable without this assumption.
"""
from __future__ import annotations

import numpy as np

from edi import Formulation

BETZ = 16.0 / 27.0


def build(N: int = 10, lam: float = 3.0, eps: float = 0.05,
          B: int = 3, tip_loss: bool = True) -> Formulation:
    """Rotor GP at one operating point.

    Parameters
    ----------
    N : spanwise stations (equal-power bins)
    lam : tip speed ratio, lambda = Omega R / V
    eps : local drag-to-lift ratio c_d/c_l
    B : number of blades  (GUESS — see module docstring)
    tip_loss : include the Prandtl tip loss correction (26)
    """
    f = Formulation()

    lam_c = f.Constant(name="lam", value=lam, units="-", description="tip speed ratio")
    eps_c = f.Constant(name="eps", value=max(eps, 1e-9), units="-", description="drag to lift ratio")

    # ---- per-station flow field -----------------------------------------
    a  = f.Variable(name="a",  guess=0.33, units="-", size=N, description="axial induction v_a/V")
    b  = f.Variable(name="b",  guess=0.67, units="-", size=N, description="axial velocity W_a/V")
    s  = f.Variable(name="s",  guess=0.05, units="-", size=N, description="swirl induction v_t/V")
    xi = f.Variable(name="xi", guess=2.0,  units="-", size=N, description="lambda*y + 2s")
    y  = f.Variable(name="y",  guess=0.5,  units="-", size=N, description="spanwise station r/R")
    dy = f.Variable(name="dy", guess=1.0 / N, units="-", size=N, description="bin width")
    G  = f.Variable(name="G",  guess=1.0,  units="-", size=N, description="near-axis correction")

    dCp = f.Variable(name="dCp", guess=BETZ / N, units="-",
                     description="power coefficient per equal-power bin")

    if tip_loss:
        F = f.Variable(name="F", guess=0.9, units="-", size=N, description="Prandtl tip loss factor")
        ff = f.Variable(name="f", guess=3.0, units="-", size=N, description="Prandtl exponent argument")

    # maximize Cp = N*dCp
    f.Objective(1.0 / dCp)

    cons = []
    for i in range(N):
        Fi = F[i] if tip_loss else 1.0
        cons += [
            # (14) differential power, with dCp/dy -> dCp/dy_i.
            #
            # PAPER TYPO (corrected here): equation (14) as printed omits the
            # tip speed ratio from the denominator. Deriving it from the
            # paper's own (13),
            #     dCp/dy = 8 s lambda y^2 F G (b - eps(lambda y + s))
            # and substituting the induced-velocity relation (6),
            # ab = s(lambda y + s), gives
            #     dCp/dy = 8 lambda y^2 F G (s b - eps a b)
            # so the lambda belongs with the 8 y^2 F G. Confirmed exactly by
            # the paper's own (18): with F=G=1 and eps->0 the corrected form
            # reduces symbolically to
            #     dCp <= 16 a b^2 y dy / (1 + sqrt(1 + 4ab/(lambda^2 y^2)))
            # which is (18) verbatim, and yields the Betz limit 16/27 as
            # lambda grows. Without the lambda, Cp falls off as 1/lambda and
            # never approaches Betz.
            s[i] * b[i] >= dCp / (8.0 * lam_c * dy[i] * y[i] ** 2 * Fi * G[i])
                           + eps_c * a[i] * b[i],
            # (15)+(16) encode the induced-velocity relation ab = s(lambda y + s)
            2.0 * a[i] * b[i] >= s[i] * lam_c * y[i] + s[i] * xi[i],
            xi[i] ** 2 >= lam_c**2 * y[i] ** 2 + 4.0 * a[i] * b[i],
            # (17) b = 1 - a
            1.0 >= a[i] + b[i],
            # (27) near-axis factor must be bounded from *above*: (14) loosens
            # as G grows, so a lower bound would be unconservative.
            G[i] <= 1.0,
        ]
        if tip_loss:
            cons += [
                # (25) is (23) rewritten with ab = s(lambda y + s) substituted,
                # which removes the sum (lambda*y + s) from a denominator and
                # makes the constraint GP-compatible.
                1.0 / y[i] >= 1.0 + 2.0 * ff[i] * s[i] / (B * a[i]),
                # (26) implicit posynomial fit to F = (2/pi) arccos(exp(-f))
                1.0 >= F[i] ** 1.56 + 1.655 * F[i] ** 5.51 / ff[i] ** 2.76,
            ]

    # ---- equal-power bins must tile the span without overlap -------------
    cons.append(1.0 >= y[N - 1] + dy[N - 1] / 2.0)             # (20)
    for i in range(N - 1):
        cons.append(y[i + 1] >= y[i] + dy[i] / 2.0 + dy[i + 1] / 2.0)  # (21)

    f.ConstraintList(cons)
    return f


def cp_at(N=10, lam=3.0, eps=0.05, B=3, tip_loss=True) -> float:
    """Solve one operating point and return the total power coefficient."""
    import pyomo.environ as pyo
    from edi.solvers.solver import solve

    f = build(N=N, lam=lam, eps=eps, B=B, tip_loss=tip_loss)
    solve(f)
    return float(N * pyo.value(f.dCp))


if __name__ == "__main__":
    print(f"Betz limit = 16/27 = {BETZ:.4f}\n")
    print("Cp with no viscous drag and no tip loss, approaching the Betz limit:")
    for lam in (2.0, 4.0, 8.0, 16.0, 32.0):
        cp = cp_at(N=10, lam=lam, eps=0.0, tip_loss=False)
        print(f"   lambda={lam:5.1f}   Cp={cp:.4f}   Cp/Betz={cp / BETZ:.4f}")
