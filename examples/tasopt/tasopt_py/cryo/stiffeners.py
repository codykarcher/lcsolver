"""Stiffener rings for a cryogenic pressure vessel -- ``tankWmech.jl``.

A tank slung inside a fuselage is a thin shell carrying its own contents'
weight through a small number of discrete supports. Between supports the
shell would ovalise, so it is held round by stiffener rings, and those rings
have to be sized for the bending moment the support reactions put into them.

All of this follows Barron, *Cryogenic Systems* (1985), chapter 7. The moment
distribution around a ring supported at angle ``theta`` from the bottom is
Barron's Eqs. (7.4)-(7.5); the outer vessel, which has two support rings and
can also collapse under external pressure, is Eqs. (7.13)-(7.15) and (7.11).

The quantity being maximised is ``k = 2 pi M / (W R)`` -- moment normalised by
the total supported weight and the ring radius -- because that makes the
answer a pure function of the support angles.

Nothing here exists in TASOPT 2.16, which has no tank of any kind.

Things in the source worth knowing
----------------------------------
* **The reference doubts its own equation.** ``stiffeners_bendingM`` carries
  a ``TODO`` saying Eq. (7.5) "does not match the curves in Fig. 7.3" and
  suspecting an error in Barron. That is reproduced as written, and pinned
  by a test, because changing it would silently move every inner-tank
  stiffener weight away from the reference's.
* **The two maxima are searched differently.** The inner ring evaluates ``k``
  at 23 angles -- the support angle, 21 points spanning 1.1 to 1.2 radians,
  and pi -- because the maximum is known to live near 1.15 rad. The outer
  ring sweeps 361 points across the half-circle. So the inner search is
  *sharper but narrower*: it would miss a maximum that moved outside that
  window, which for supports far from the usual position it can. Reproduced,
  with a test showing where the assumption holds and where it stops.
* ``stiffener_weight`` reuses the name ``W`` for two different things -- the
  supported weight coming in, and then the I-beam flange width, which
  overwrites it. Harmless because the weight is consumed first, but it means
  the returned expression reads as if it involved the load when it does not.

Verified against TASOPT.jl; see ``tests/test_stiffeners.py``.
"""
from __future__ import annotations

import math

__all__ = ["stiffeners_bending_moment", "stiffeners_bending_moment_outer",
           "stiffener_weight", "find_K1_head", "GEE", "PREF",
           "IBEAM_FLANGE_WIDTH", "IBEAM_WEB_THICKNESS",
           "IBEAM_FLANGE_THICKNESS"]

GEE = 9.81
#: ``constants.jl``'s ``pref``. Note it is 101320, not the 101325 that
#: :mod:`tasopt_py.cryo.fuel_thermo` uses for ``p_atm`` -- the reference
#: carries both and they differ by 5 Pa.
PREF = 101320.0

#: The stiffener section is assumed to be a 100 x 100 I-beam, fixed.
IBEAM_FLANGE_WIDTH = 100.0e-3       # m
IBEAM_WEB_THICKNESS = 7.1e-3        # m
IBEAM_FLANGE_THICKNESS = 8.8e-3     # m

#: Barron table 7.6 (p. 367): equivalent-radius parameter for
#: hemi-ellipsoidal heads, ``R = K1 * D``.
_HEAD_AR = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0)
_HEAD_K1 = (0.50, 0.57, 0.65, 0.73, 0.81, 0.90, 0.99, 1.08, 1.18, 1.27, 1.36)


def _linspace(a: float, b: float, n: int) -> list:
    if n == 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + step * i for i in range(n)]


def stiffeners_bending_moment(theta: float) -> tuple:
    """Peak ring moment for the **inner** vessel, Barron Eqs. (7.4)-(7.5).

    ``theta`` is the angular position of the supports measured from the
    bottom of the tank, in radians. Returns ``(phi_max, k_max)`` where
    ``k = 2 pi M / (W R)``.

    The search set is the reference's: the support angle itself, 21 points
    from 1.1 to 1.2 rad, and pi. That is a deliberate shortcut -- "take
    advantage of known form of maximum" -- and it is narrow. See the module
    docstring.
    """
    phis = [theta] + _linspace(1.1, 1.2, 21) + [math.pi]
    k = []
    for phi in phis:
        if 0.0 <= phi < theta:
            v = (0.5 * math.cos(phi) + phi * math.sin(phi)
                 - (math.pi - theta) * math.sin(theta) + math.cos(theta)
                 + math.cos(phi) * math.sin(theta) ** 2)
        elif theta <= phi <= math.pi:
            # Barron Eq. (7.5). The reference notes this does not match the
            # curves in Barron's Fig. 7.3 and suspects an error in the book;
            # reproduced as written.
            v = (0.5 * math.cos(phi) - (math.pi - phi) * math.sin(phi)
                 + theta + math.cos(theta)
                 + math.cos(phi) * math.sin(theta) ** 2)
        else:
            v = 0.0
        k.append(v)

    kmax, imax = _absmax(k)
    return phis[imax], kmax


def stiffeners_bending_moment_outer(theta1: float, theta2: float) -> tuple:
    """Peak ring moment for the **outer** vessel, Barron Eqs. (7.13)-(7.15).

    ``theta1`` and ``theta2`` are the angular positions of the bottom and top
    support rings, from the bottom of the tank. Swept over 361 points across
    the half circle, unlike the inner ring's 23.
    """
    phis = _linspace(0.0, math.pi, 361)
    s1, s2 = math.sin(theta1), math.sin(theta2)
    c1, c2 = math.cos(theta1), math.cos(theta2)
    base = s2 ** 2 - s1 ** 2
    k = []
    for phi in phis:
        if 0.0 <= phi <= theta1:
            v = (math.cos(phi) * base + (c2 - c1)
                 - (math.pi - theta2) * s2 + (math.pi - theta1) * s1)
        elif theta1 <= phi <= theta2:
            v = (math.cos(phi) * base + (c2 - c1)
                 - (math.pi - theta2) * s2 + math.pi * math.sin(phi)
                 - theta1 * s1)
        elif theta2 <= phi <= math.pi:
            v = (math.cos(phi) * base + (c2 - c1)
                 + (theta2 * s2 - theta1 * s1))
        else:
            v = 0.0
        k.append(v)

    kmax, imax = _absmax(k)
    return phis[imax], kmax


def _absmax(values) -> tuple:
    """``findmax(abs.(k))`` -- the largest magnitude, and where it is.

    Julia's ``findmax`` returns the **first** maximum on ties, and so does
    this; the tie matters because the inner search puts the support angle and
    a uniform sweep in the same list.
    """
    best, at = abs(values[0]), 0
    for i, v in enumerate(values[1:], start=1):
        if abs(v) > best:
            best, at = abs(v), i
    return best, at


def stiffener_weight(tanktype: str, W: float, Rtank: float, perim: float,
                     s_a: float, rho_stiff: float, theta1: float,
                     theta2: float = 0.0, Nstiff: float = 2.0,
                     l_cyl: float = 0.0, E: float = 0.0) -> float:
    """Weight of one stiffener ring, N.

    ``W`` is the supported weight, ``Rtank`` the ring radius, ``perim`` the
    ring perimeter, ``s_a`` the allowable stress and ``rho_stiff`` the
    stiffener density. For ``tanktype = "outer"`` the ring must also resist
    collapse under external pressure, which needs ``Nstiff``, ``l_cyl`` and
    the modulus ``E``.

    The section is a fixed 100 x 100 I-beam; only its height is solved for.
    """
    if tanktype == "inner":
        _, kmax = stiffeners_bending_moment(theta1)
        # A pressurised inner vessel cannot collapse inward.
        Icollapse = 0.0
    elif tanktype == "outer":
        _, kmax = stiffeners_bending_moment_outer(theta1, theta2)
        # Barron Eq. (7.11): critical pressure taken as 4 atmospheres.
        pc = 4.0 * PREF
        Do = 2.0 * Rtank
        L = l_cyl / (Nstiff - 1.0)          # span between supports
        Icollapse = pc * Do ** 3 * L / (24.0 * E)
    else:
        raise ValueError(f"tanktype must be 'inner' or 'outer', "
                         f"not {tanktype!r}")

    Mmax = kmax * W * Rtank / (2.0 * math.pi)
    Z = Mmax / s_a                          # required section modulus

    # The reference reassigns `W` here, from the supported weight to the
    # I-beam flange width. Kept as separate names.
    bw = IBEAM_FLANGE_WIDTH
    t_w = IBEAM_WEB_THICKNESS
    t_f = IBEAM_FLANGE_THICKNESS

    # I > b H^2 t_f / 2 + t_f^3 b / 6, and the requirement is
    # I = Icollapse + Z (H + t_f)/2. Solve the quadratic for the height H.
    a = t_f * bw / 2.0
    b = -Z / 2.0
    c = -Icollapse - Z * t_f / 2.0 + t_f ** 3 * bw / 6.0

    H = (-b + math.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)
    S = 2.0 * bw * t_f + (H - t_f) * t_w    # cross-sectional area

    return GEE * rho_stiff * S * perim


def find_K1_head(AR: float) -> float:
    """Equivalent-radius parameter for a hemi-ellipsoidal head, Barron 7.6.

    ``R = K1 * D`` with ``D`` the major diameter. Linearly interpolated in
    aspect ratio.

    Outside 1.0-3.0 the reference prints a message and returns 1.0; this
    raises instead. Returning a silent default that is neither an
    interpolation nor an error puts a wrong head radius into the pressure
    sizing with nothing to show for it, and the message goes to stdout in the
    middle of an optimisation where nobody reads it.
    """
    if not (_HEAD_AR[0] <= AR <= _HEAD_AR[-1]):
        raise ValueError(
            f"head aspect ratio {AR} is outside Barron table 7.6's "
            f"{_HEAD_AR[0]}-{_HEAD_AR[-1]}. TASOPT.jl prints "
            "'ARtank of heads not supported' and returns K1 = 1.0, which is "
            "neither an interpolation nor a failure.")

    i = 0
    while AR > _HEAD_AR[i]:
        i += 1
    if math.isclose(AR, _HEAD_AR[i], rel_tol=1e-9, abs_tol=1e-12):
        return _HEAD_K1[i]
    return (_HEAD_K1[i - 1]
            + (_HEAD_K1[i] - _HEAD_K1[i - 1])
            / (_HEAD_AR[i] - _HEAD_AR[i - 1]) * (AR - _HEAD_AR[i - 1]))
