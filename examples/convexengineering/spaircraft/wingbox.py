"""Wing box structural model, shared by the wing, horizontal and vertical tail.

Source model
------------
``wingbox.py`` in https://github.com/convexengineering/SPaircraft, which is in
turn the box-beam model of

    W. Hoburg and P. Abbeel, "Geometric Programming for Aircraft Design
    Optimization", AIAA Journal 52(11), 2014.

One box, three surfaces
-----------------------
The same beam model sizes all three surfaces, but each gets a different set of
constraints, and the differences are not cosmetic:

* **wing** — carries ``N_lift = 3``, applies its own ``tau <= tau_max``, and
  takes the root moment ``M_r`` from outside (the wing's root bending comes
  from the aircraft-level load case, not from this model).
* **vertical tail** — halves the structural weight (``W_struct >= 0.5*(...)``)
  because the VT is a half-span surface while the beam model is written for a
  full span, and correspondingly doubles ``b``, ``S`` and ``L_max`` on the way
  in. ``N_lift = 1``, and it computes its own root moment ``M_r >= L_max*AR*p/24``.
* **horizontal tail** — ``N_lift = 1``, and adds pi-tail sizing: the maximum
  lift is split into triangular and rectangular parts so that root bending can
  be evaluated at an outboard pin joint, with ``\\pi_{M-fac}`` scaling the spar
  cap weight.

Note the shear-web constraint differs in form between the three: the wing uses
``12 >= AR*L_max*q^2/(...)`` with no load factor, the VT multiplies by
``N_lift``, and the HT uses ``2*AR*L_shear*N_lift`` against the pin-joint shear
rather than the root load. The source carries ``#TODO`` markers on the two HT
lines; they are reproduced as written.
"""
from __future__ import annotations


def add_wingbox(surfacetype, *, AR, b, S, p, q, tau, Lmax, group,
                Mr=None, taper=None, tau_max=None):
    """Add a wing box to the group's formulation, and return its variables.

    Parameters are the *linked* quantities owned by the surface model: aspect
    ratio, span, area, the taper substitutions ``p`` and ``q``, thickness
    ratio, and maximum load. ``Mr`` is supplied for the wing (whose root
    moment comes from the aircraft load case) and created here otherwise.

    ``group`` is the surface's own group; the box nests inside it, so a cap
    thickness reads ``wing.box.t_cap``.
    """
    V, C = group.Variable, group.Constant

    # ---- box variables ----------------------------------------------------
    Icap = V("I_cap", 1e-5, "-", "non-dim spar cap area moment of inertia")
    nu = V("nu", 0.8, "-", "dummy variable (t^2+t+1)/(t+1)^2")
    Wcap = V("W_cap", 1e4, "N", "weight of spar caps")
    Wweb = V("W_web", 1e3, "N", "weight of shear web")
    Wstruct = V("W_struct", 2e4, "N", "structural weight")
    tcap = V("t_cap", 0.003, "-", "non-dim spar cap thickness")
    tweb = V("t_web", 0.001, "-", "non-dim shear web thickness")
    if Mr is None:
        Mr = V("M_r", 1e6, "N", "root moment per root chord")

    # ---- box constants ----------------------------------------------------
    g = C("g", 9.81, "m/s^2", "gravitational acceleration")
    rh = C("r_h", 0.75, "-", "fractional wing thickness at spar web")
    rhocap = C("rho_cap", 2700.0, "kg/m^3", "density of spar cap material")
    rhoweb = C("rho_web", 2700.0, "kg/m^3", "density of shear web material")
    sigmax = C("sigma_max", 250e6, "Pa", "allowable tensile stress")
    sigmaxshear = C("sigma_max_shear", 167e6, "Pa", "allowable shear stress")
    wwb = C("r_w_c", 0.5, "-", "wingbox width-to-chord ratio")

    out = dict(I_cap=Icap, nu=nu, W_cap=Wcap, W_web=Wweb, W_struct=Wstruct,
               t_cap=tcap, t_web=tweb, M_r=Mr)

    cons = [
        AR == b ** 2 / S,
        # Root stiffness (Hoburg 2014). Assumes r_h = 0.75, so the rms box
        # height is ~0.92 t_max -- hence the 0.92 factors.
        0.92 * wwb * tau * tcap ** 2 + Icap <= 0.92 ** 2 / 2 * wwb * tau ** 2 * tcap,
        # Posynomial approximation of nu = (1+lam+lam^2)/(1+lam)^2
        nu ** 3.94 >= 0.86 * p ** (-2.38) + 0.14 * p ** 0.56,
        Wweb >= 8 * rhoweb * g * rh * tau * tweb * S ** 1.5 * nu / (3 * AR ** 0.5),
    ]

    if surfacetype == "wing":
        Nlift = C("N_lift", 3.0, "-", "wing loading multiplier")
        cons += [
            tau <= tau_max,
            Wstruct >= Wweb + Wcap,
            # Shear web sizing; assumes all shear carried by the web, r_h=0.75.
            12 >= AR * Lmax * q ** 2 / (tau * S * tweb * sigmaxshear),
            Wcap >= 8 * rhocap * g * wwb * tcap * S ** 1.5 * nu / (3 * AR ** 0.5),
            # Stress limit; assumes bending carried by caps (I_cap >> I_web).
            8 >= Mr * AR * q ** 2 * tau / (S * Icap * sigmax),
        ]
    elif surfacetype == "vertical_tail":
        Nlift = C("N_lift", 1.0, "-", "wing loading multiplier")
        cons += [
            tau <= 0.14,
            # Half of the beam model's weight: the VT is a half-span surface.
            Wstruct >= 0.5 * (Wweb + Wcap),
            # Root moment; assumes lift per unit span follows local chord.
            Mr >= Lmax * AR * p / 24,
            12 >= AR * Lmax * Nlift * q ** 2 / (tau * S * tweb * sigmaxshear),
            Wcap >= 8 * rhocap * g * wwb * tcap * S ** 1.5 * nu / (3 * AR ** 0.5),
            8 >= Nlift * Mr * AR * q ** 2 * tau / (S * Icap * sigmax),
        ]
    elif surfacetype == "horizontal_tail":
        Nlift = C("N_lift", 1.0, "-", "wing loading multiplier")
        # Pi-tail sizing: split max lift into triangular and rectangular parts
        # so root bending can be taken at the outboard pin joint.
        bhtout = V("b_ht_out", 3.0, "m", "HT outboard half-span")
        Lhtri = V("L_ht_tri", 1e4, "N", "triangular HT load")
        Lhrect = V("L_ht_rect", 1e4, "N", "rectangular HT load")
        Lhtriout = V("L_ht_tri_out", 1e4, "N", "triangular HT load outboard")
        Lhrectout = V("L_ht_rect_out", 1e4, "N", "rectangular HT load outboard")
        Lshear = V("L_shear", 1e4, "N", "maximum shear load at pin joint")
        piMfac = V("pi_M_fac", 30.0, "-", "pi-tail bending structural factor")
        out.update(b_ht_out=bhtout, L_ht_tri=Lhtri, L_ht_rect=Lhrect,
                   L_ht_tri_out=Lhtriout, L_ht_rect_out=Lhrectout,
                   L_shear=Lshear, pi_M_fac=piMfac)
        cons += [
            tau <= 0.14,
            Wstruct >= Wweb + Wcap,
            Lhtriout >= Lhtri * bhtout ** 2 / (0.5 * b) ** 2,
            Lhrectout >= Lhrect * bhtout / (0.5 * b),
            12 >= 2 * AR * Lshear * Nlift * q ** 2 / (tau * S * tweb * sigmaxshear),
            Wcap >= piMfac * 8 * rhocap * g * wwb * tcap * S ** 1.5 * nu / (3 * AR ** 0.5),
            8 >= Nlift * Mr * AR * q ** 2 * tau / (S * Icap * sigmax),
        ]
    else:
        raise ValueError(f"unknown surfacetype {surfacetype!r}")

    out["N_lift"] = Nlift
    return group, cons
