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

#: Box materials. TASOPT and SPaircraft both assume aluminium throughout;
#: a 787-class aircraft is carbon composite and should not be charged
#: aluminium's density.
ALUMINIUM = {"rho": 2700.0, "sigma": 250e6, "tau": 167e6}
COMPOSITE = {"rho": 1600.0, "sigma": 450e6, "tau": 300e6}


def add_wingbox(surfacetype, *, AR, b, S, p, q, tau, Lmax, group,
                Mr=None, taper=None, tau_max=None, material=None,
                tau_limits=True, cosL=None, weight_credit=1.0):
    """Add a wing box to the group's formulation, and return its variables.

    Parameters are the *linked* quantities owned by the surface model: aspect
    ratio, span, area, the taper substitutions ``p`` and ``q``, thickness
    ratio, and maximum load. ``Mr`` is supplied for the wing (whose root
    moment comes from the aircraft load case) and created here otherwise.

    ``group`` is the surface's own group; the box nests inside it, so a cap
    thickness reads ``wing.box.t_cap``.
    """
    material = material or ALUMINIUM
    # ``tau_limits=False`` drops the thickness band on the TAIL surfaces, to
    # measure what the band is actually holding back.
    #
    # Both ends are fit-validity limits, not physics. The tail polars carry
    # tau^6.29 and tau^133.8 terms fitted over t/c 0.08-0.15; outside that
    # range they are extrapolation. The lower end also used to be a band-aid
    # over a collapsing pi-tail load path, but that is cured at the source now
    # (pi_tail_supports="fixed"), so the floor has only the fit left to justify
    # it. The wing keeps its own tau <= tau_max either way: that one is a real
    # aerodynamic limit (TASOPT's 0.14733), not a curve fit.
    _band = []
    if tau_limits:
        _band = [tau <= 0.14, tau >= 0.08]

    # ---- sweep pricing, ported from TASOPT surfw.f ------------------------
    # A swept box is heavier, and SPaircraft's box does not know it. TASOPT
    # sizes the root station as
    #
    #     tbwebo = So*0.5      /(co**2 * tauweb * rh * hboxo * cosL**2)
    #     con    = Mo*6.0*hboxo/(co**3 * sigcap * wbox        * cosL**4)
    #
    # -- shear web thickness as 1/cos^2 L, cap sizing as 1/cos^4 L. Without
    # them sweep is a free variable that buys transonic relief and pays
    # nothing, and on the TAILS it is worse than free: tail sweep appears in
    # no drag polar and no weight, only in the moment arm, so sweeping the
    # tail buys arm -- and therefore a smaller tail -- at zero cost. That is
    # the same defect as the one-sided `mac`, `l_ht` and `l_vt` rows: a
    # quantity the optimizer can inflate for nothing.
    #
    # cosL is a monomial whether it is a constant or a design variable, so
    # both rows stay GP.
    _c2 = cosL ** 2 if cosL is not None else 1.0
    _c4 = cosL ** 4 if cosL is not None else 1.0
    # ...and the OTHER half of the same transformation, surfw.f:155-161:
    #
    #     Vinn = co**2*b * (etas-etao) * (...)/6.0 * cosL
    #     Vout = co**2*b * (1.0 -etas) * (...)/6.0 * cosL
    #
    # The box VOLUME carries a factor of cosL, because the structural box runs
    # along the swept span while S and AR here are streamwise. Taking only the
    # thickness terms above and not this one over-charges sweep by a full power
    # of cosL in each row: measured on the E175 that drove the wing to 4.2
    # degrees and cruise to M 0.55, which is not an airliner. Net, as TASOPT
    # has it: web weight ~ 1/cosL, cap weight ~ 1/cosL^3.
    _cv = cosL if cosL is not None else 1.0
    # CRANKED-PLANFORM CREDIT, applied to the cap and web directly so the
    # COMPONENTS match a reference rather than only their sum.
    #
    # This box integrates a single taper line in closed form. A real transport
    # wing is cranked -- an inner panel tapering far less (TASOPT's 737: 0.70
    # root-to-break against 0.25 break-to-tip) -- which carries root bending
    # more efficiently than one taper can, so the closed form over-predicts by
    # a roughly constant factor. Measured against TASOPT's own station-based
    # calculation at matched inputs: W_cap 1.241, W_web 1.423, W_wing 1.249.
    #
    # WING ONLY. The caller passes it; the tails do not, and must not -- this
    # model already comes out LIGHTER than TASOPT on both (W_ht 0.671, W_vt
    # 0.750), so a credit there would widen a gap rather than close one.
    #
    # A correction, not a discovery. The honest version is the station-based
    # cranked box in wingbox_tasopt.py; when that converges this goes.
    _wc = weight_credit
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
    # Material. 2700 kg/m3 and 250 MPa are aluminium, which is right for a
    # 737 and wrong for a 787 -- that aircraft's wing and empennage boxes are
    # carbon composite: lighter and stronger, roughly 1600 kg/m3 at 450 MPa
    # allowable with the same shear ratio. Applying aluminium to it overstates
    # its structural weight throughout.
    rhocap = C("rho_cap", material["rho"], "kg/m^3", "spar cap density")
    rhoweb = C("rho_web", material["rho"], "kg/m^3", "shear web density")
    sigmax = C("sigma_max", material["sigma"], "Pa", "allowable tensile stress")
    sigmaxshear = C("sigma_max_shear", material["tau"], "Pa",
                    "allowable shear stress")
    wwb = C("r_w_c", 0.5, "-", "wingbox width-to-chord ratio")

    out = dict(r_w_c=wwb,
               I_cap=Icap, nu=nu, W_cap=Wcap, W_web=Wweb, W_struct=Wstruct,
               t_cap=tcap, t_web=tweb, M_r=Mr)

    cons = [
        AR == b ** 2 / S,
        # Root stiffness (Hoburg 2014). Assumes r_h = 0.75, so the rms box
        # height is ~0.92 t_max -- hence the 0.92 factors.
        0.92 * wwb * tau * tcap ** 2 + Icap <= 0.92 ** 2 / 2 * wwb * tau ** 2 * tcap,
        # Posynomial approximation of nu = (1+lam+lam^2)/(1+lam)^2
        nu ** 3.94 >= 0.86 * p ** (-2.38) + 0.14 * p ** 0.56,
        _wc * Wweb >= _cv * 8 * rhoweb * g * rh * tau * tweb * S ** 1.5 * nu / (3 * AR ** 0.5),
    ]

    if surfacetype == "wing":
        Nlift = C("N_lift", 3.0, "-", "wing loading multiplier")
        cons += [
            tau <= tau_max,
            Wstruct >= Wweb + Wcap,
            # Shear web sizing; assumes all shear carried by the web, r_h=0.75.
            12 * _c2 >= AR * Lmax * q ** 2 / (tau * S * tweb * sigmaxshear),
            _wc * Wcap >= _cv * 8 * rhocap * g * wwb * tcap * S ** 1.5 * nu / (3 * AR ** 0.5),
            # Stress limit; assumes bending carried by caps (I_cap >> I_web).
            8 * _c4 >= Mr * AR * q ** 2 * tau / (S * Icap * sigmax),
        ]
    elif surfacetype == "vertical_tail":
        Nlift = C("N_lift", 1.0, "-", "wing loading multiplier")
        cons += _band + [
            # Half of the beam model's weight: the VT is a half-span surface.
            Wstruct >= 0.5 * (Wweb + Wcap),
            # Root moment; assumes lift per unit span follows local chord.
            Mr >= Lmax * AR * p / 24,
            12 * _c2 >= AR * Lmax * Nlift * q ** 2 / (tau * S * tweb * sigmaxshear),
            _wc * Wcap >= _cv * 8 * rhocap * g * wwb * tcap * S ** 1.5 * nu / (3 * AR ** 0.5),
            8 * _c4 >= Nlift * Mr * AR * q ** 2 * tau / (S * Icap * sigmax),
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
        cons += _band + [
            # A LOWER bound on thickness, which this model did not have.
            #
            # The reference solution puts the horizontal tail at
            # tau = 3.45e-10 -- a surface with no thickness at all. The box
            # then has no structural depth, the cap inertia goes as tau so
            # t_cap follows it to zero, and a 36.9 m2 tail comes out weighing
            # 59 lbf next to a 5.9 m2 fin at 260 lbf.
            #
            # The wing and the fin do not do this because their shear webs
            # carry tau in the DENOMINATOR, so thin costs web weight and both
            # sit at their upper bounds. The pi-tail horizontal reacts shear
            # at an outboard pin joint instead, where the load is far smaller,
            # so the web never pushes back and thickness is free to vanish.
            #
            # NOTE this is a band-aid on a load path, not the cure. The
            # cure is not to let M_r collapse -- use pi_tail_supports="fixed",
            # whose centreline moment carries no subtraction. The bound stays
            # because it is ALSO the edge of where the airfoil drag fits were
            # trained (t/c 0.08-0.15): outside that range the tau^6.3 and
            # tau^133.8 terms in the tail polars are extrapolation, not
            # physics. Both bounds are FIT VALIDITY limits, which is why the
            # upper one stays too even though drag does penalise thickness.
            Wstruct >= Wweb + Wcap,
            Lhtriout >= Lhtri * bhtout ** 2 / (0.5 * b) ** 2,
            Lhrectout >= Lhrect * bhtout / (0.5 * b),
            12 * _c2 >= 2 * AR * Lshear * Nlift * q ** 2 / (tau * S * tweb * sigmaxshear),
            _wc * Wcap >= _cv * piMfac * 8 * rhocap * g * wwb * tcap * S ** 1.5 * nu / (3 * AR ** 0.5),
            8 * _c4 >= Nlift * Mr * AR * q ** 2 * tau / (S * Icap * sigmax),
        ]
    elif surfacetype == "horizontal_tail_conventional":
        # A CANTILEVER horizontal tail: one surface, root-mounted on the
        # fuselage or the fin, carrying its own root moment.
        #
        # SPaircraft has no such model, because optimalD8 is all it builds and
        # a D8's horizontal spans BETWEEN two fins -- a beam on two supports,
        # which is what the "horizontal_tail" branch above sizes. Applying
        # that to a 737 or a 787 is the wrong support condition entirely: it
        # reacts the root bending at an outboard pin joint that a
        # single-finned aircraft does not have.
        #
        # The form here is the vertical tail's, which is ALREADY a cantilever
        # in this same file -- own root moment ``M_r >= L_max*AR*p/24``, shear
        # against the root load rather than a pin joint. Two differences: the
        # VT halves its structural weight because it is a half-span surface
        # and the beam model is written for a full span, which a horizontal
        # tail is not; and the horizontal keeps its own ``tau <= 0.14``.
        Nlift = C("N_lift", 1.0, "-", "wing loading multiplier")
        cons += _band + [
            Wstruct >= Wweb + Wcap,
            # p/24, the wing's coefficient -- because a CONVENTIONAL horizontal
            # tail has the wing's topology: full span, two cantilevers off a
            # centre box, fed its true b, S and L_max. TASOPT agrees on the
            # topology explicitly, calling `tailpo(Sh, ARh, ...)` for the HT
            # with true area and aspect ratio against
            # `tailpo(2.0*Sv/nvtail, 2.0*ARv, ...)` for the fin -- the doubling
            # is the fin's alone.
            #
            # This row said /12 for a while, and the reasoning was sound at the
            # time: match the wing, which aircraft.py then wrote as
            # `M_r c_root >= L_eff b^2/(12 S)(c_root + 2 c_tip)`. The wing has
            # since been corrected to /(24 S) -- the /12 form is the moment
            # about the centreline of BOTH panels, and the box carries one --
            # and this row did not follow, leaving the HT on twice its root
            # moment. Measured before the fix: M_r 69,952 against
            # L_max*AR*p/24 = 34,976, a clean factor of two.
            Mr >= Lmax * AR * p / 24,
            12 * _c2 >= AR * Lmax * Nlift * q ** 2 / (tau * S * tweb * sigmaxshear),
            _wc * Wcap >= _cv * 8 * rhocap * g * wwb * tcap * S ** 1.5 * nu / (3 * AR ** 0.5),
            8 * _c4 >= Nlift * Mr * AR * q ** 2 * tau / (S * Icap * sigmax),
        ]
    else:
        raise ValueError(f"unknown surfacetype {surfacetype!r}")

    out["N_lift"] = Nlift
    return group, cons
