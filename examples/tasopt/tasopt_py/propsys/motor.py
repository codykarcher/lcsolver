"""Permanent-magnet motor losses -- ``PMSM.jl``.

The loss physics of a permanent-magnet synchronous machine, which is what
sets both its efficiency and how hard it is to cool. Nothing in TASOPT 2.16
corresponds to it -- the Fortran has no electrical machine of any kind.

Four loss mechanisms, and they scale differently
------------------------------------------------
=================  ============================  =========================
loss               scales as                     what it is
=================  ============================  =========================
ohmic              ``I^2 R``                     current through windings
hysteresis         ``m f B^alpha``               re-magnetising the steel
eddy               ``m f^2 B^2``                 currents induced in it
windage            ``Cf rho Omega^3 R^4 l``      air dragged in the gap
=================  ============================  =========================

The exponents are what matter for design. Ohmic loss falls with speed at
fixed power (less torque, less current), while the core and windage losses
all *rise* with it -- windage as the **cube** of speed. That is the whole
reason a motor has a best speed rather than simply wanting to be fast, and
the tests exercise it.

Hysteresis carries a Steinmetz exponent ``alpha`` of about 1.8 rather than a
round 2, which is why the flux density is raised to a material constant here
and squared in the eddy term.

The windage coefficient is implicit
-----------------------------------
Vrancik (1968) gives the skin friction in the annular gap as

    1/sqrt(Cf) = 2.04 + 1.768 ln(Re sqrt(Cf))

which has to be solved for ``Cf``. The reference uses ``find_zero(res,
1e-2)``; this brackets instead, for the same reason as the tank's buckling
solve -- see :func:`_solve_friction`.

Verified against TASOPT.jl; see ``tests/test_motor.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["MU_0", "airgap_flux", "remanent_flux", "ohmic_loss",
           "hysteresis_loss", "eddy_loss", "core_loss", "windage_loss",
           "slot_resistance", "cross_sectional_area", "STEELS",
           "MotorDesign", "SizedMotor", "size_motor"]

#: Vacuum permeability, N/A^2. NIST.
MU_0 = 1.25663706127e-6

#: Electrical steels, from v3's material database.
#: ``(k_hysteresis, k_eddy, steinmetz_alpha, density)``.
STEELS = {
    "M19": (0.02351412, 7.0963515e-5, 1.793, 7750.0),
}


def airgap_flux(M: float, thickness: float, airgap: float) -> float:
    """Flux density in the airgap, T.

    A magnetic-circuit result assuming no MMF drop across the steel, so it
    is the magnet's own field divided between magnet and gap by their
    thicknesses. The consequence is a hard ceiling: however thick the magnet,
    ``B_gap`` cannot exceed ``mu0 M``.
    """
    return MU_0 * M * thickness / (thickness + airgap)


def remanent_flux(Br: float, alpha: float, T: float,
                  Tbase: float = 20.0) -> float:
    """Magnet remanent flux at temperature ``T``, T.

    Linear in temperature, with ``alpha`` a percentage per kelvin -- which is
    why the expression divides by 100. Neodymium loses roughly 0.1%/K, so a
    motor run 100 K hot has lost a tenth of its field.
    """
    return Br * (1.0 - alpha * (T - (273.15 + Tbase)) / 100.0)


def ohmic_loss(I: float, phase_resistance: float,
               energized_phases: int = 2) -> float:
    """Resistive loss, W. Two phases are energised at a time by default."""
    return I ** 2 * (energized_phases * phase_resistance)


def hysteresis_loss(mass: float, f: float, B: float, k_h: float,
                    alpha: float) -> float:
    """Hysteresis loss in a steel component, W.

    Linear in frequency and raised to a Steinmetz exponent in flux density
    -- about 1.8, not 2.
    """
    return mass * k_h * f * B ** alpha


def eddy_loss(mass: float, f: float, B: float, k_e: float) -> float:
    """Eddy-current loss in a steel component, W. Quadratic in both."""
    return mass * k_e * f ** 2 * B ** 2


def core_loss(mass: float, f: float, B: float, steel: str = "M19") -> float:
    """Hysteresis plus eddy loss for one component, W."""
    k_h, k_e, alpha, _ = STEELS[steel]
    return (hysteresis_loss(mass, f, B, k_h, alpha)
            + eddy_loss(mass, f, B, k_e))


def _solve_friction(Re: float, lo: float = 1.0e-8,
                    hi: float = 1.0) -> float:
    """Vrancik's skin-friction coefficient, bracketed.

    ``1/sqrt(Cf) = 2.04 + 1.768 ln(Re sqrt(Cf))``. The reference solves it
    with ``find_zero(res, 1e-2)`` from a guess. Bracketing is used here for
    the same reason as the tank's buckling solve: the residual has a
    logarithm, so it is undefined below ``Cf = 0`` and an unbracketed solver
    started from a guess can step there.
    """
    def res(Cf):
        return 1.0 / math.sqrt(Cf) - 2.04 - 1.768 * math.log(
            Re * math.sqrt(Cf))

    flo, fhi = res(lo), res(hi)
    if flo * fhi > 0.0:
        raise ValueError(
            f"Vrancik's friction residual does not change sign on "
            f"[{lo}, {hi}] at Re = {Re:.4g}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = res(mid)
        if fm == 0.0 or (hi - lo) < 1.0e-16 * max(1.0, mid):
            return mid
        if flo * fm < 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def windage_loss(Omega: float, radius_gap: float, airgap: float,
                 length: float, rho: float, nu: float) -> float:
    """Air-friction loss in the rotor gap, W -- Vrancik (1968).

    ``Omega`` is in rad/s. The **cube** of speed and the **fourth power** of
    radius, which is why a fast large-diameter rotor is dominated by it.
    """
    Re = Omega * radius_gap * airgap / nu
    if Re <= 0.0:
        raise ValueError(
            f"windage needs a positive gap Reynolds number, got {Re}; "
            "Vrancik's correlation takes its logarithm")
    Cf = _solve_friction(Re)
    return Cf * math.pi * rho * Omega ** 3 * radius_gap ** 4 * length


def slot_resistance(resistivity: float, area: float, length: float) -> float:
    """Resistance of one winding slot, ohm."""
    return resistivity * length / area


def cross_sectional_area(Ro: float, Ri: float) -> float:
    """Annulus area, m^2."""
    return math.pi * (Ro ** 2 - Ri ** 2)


# --------------------------------------------------------------------------
# Geometric sizing -- size_PMSM!
# --------------------------------------------------------------------------

@dataclass
class MotorDesign:
    """The design choices a PMSM is sized from.

    These are limits and technology levels rather than dimensions: the
    geometry falls out of them. ``U_max`` and ``J_max`` in particular are
    the two that set the machine's size -- tip speed bounds the radius, and
    current density bounds the slot area.
    """
    U_max: float = 200.0            # rotor tip speed limit, m/s
    J_max: float = 1.0e7            # winding current density limit, A/m^2
    B_sat: float = 1.8              # steel saturation flux density, T
    rB_sat: float = 0.98            # fraction of saturation actually used
    N_pole_pairs: int = 8
    N_slots: int = 6
    N_slots_per_phase: int = 2
    N_energized_slots: int = 2
    phases: int = 3
    N_inverters: int = 1
    airgap_thickness: float = 2.0e-3
    magnet_thickness: float = 20.0e-3
    magnet_M: float = 1.0e6         # magnetisation, A/m
    teeth_thickness: float = 10.0e-3
    winding_kpf: float = 0.35       # slot packing factor
    #: Winding temperature, K. The reference ties this to the cooling air,
    #: defaulting to 363.15 K -- not ambient, and worth 28% on the phase
    #: resistance.
    winding_T: float = 363.15
    #: Densities, kg/m^3.
    rho_steel: float = 7750.0
    rho_magnet: float = 7501.0
    rho_conductor: float = 8960.0
    rho_insulator: float = 2150.0
    rho_shaft: float = 7850.0
    #: Shaft material limits.
    tau_max_shaft: float = 4.5963636363636357e8    # Pa, AISI-4340
    YTS_shaft: float = 8.62e8                     # Pa
    l_extra_shaft: float = 1.2      # shaft length as a multiple of stack


@dataclass(frozen=True)
class SizedMotor:
    radius_gap: float     # m, rotor outer radius at the airgap
    length: float         # m, active stack length
    mass: float           # kg, total
    masses: dict          # kg, by component
    A_slot: float         # m^2, one slot
    B_gap: float          # T
    I: float              # A, peak slot current
    phase_resistance: float   # ohm
    Omega: float          # rad/s


def size_motor(design: MotorDesign, shaft_speed: float,
               design_power: float) -> SizedMotor:
    """Size a PMSM from a shaft speed (rpm) and a design shaft power (W).

    The chain is short and almost entirely limit-driven:

    1. **Tip speed sets the radius.** ``radius_gap = U_max / Omega`` -- so a
       faster machine is a *smaller* one, which is the whole argument for
       high-speed motors.
    2. **Flux continuity sets the back-iron thickness.** The gap flux has to
       return through the rotor and stator yokes without saturating them.
    3. **Saturation splits the annulus** between teeth and slots, in closed
       form: ``A_teeth = A (1 - B_gap / sqrt(B_teeth^2 - B_windings^2))``.
    4. **Current density sets the slot current**, and hence the torque per
       unit length, and hence the stack length.

    Three failure modes are checked, all of which the reference reports as
    errors too: a rotor bore that closes up, a shaft too thin for the
    torque, and a shaft that bursts under its own rotation.
    """
    Omega = shaft_speed * (2.0 * math.pi / 60.0)
    if Omega <= 0.0:
        raise ValueError(f"shaft speed must be positive, got {shaft_speed}")

    radius_gap = design.U_max / Omega
    B_steel = design.rB_sat * design.B_sat
    B_gap = airgap_flux(design.magnet_M, design.magnet_thickness,
                        design.airgap_thickness)

    # Back iron: enough to carry the gap flux around one pole pitch.
    yoke_t = (B_gap / B_steel * math.pi * radius_gap
              / (2.0 * design.N_pole_pairs))

    rotor_Ro = radius_gap - design.magnet_thickness
    rotor_Ri = rotor_Ro - yoke_t
    if rotor_Ri <= 0.0:
        raise ValueError(
            f"rotor bore closes up: outer radius {rotor_Ro:.4f} m minus a "
            f"{yoke_t:.4f} m yoke leaves {rotor_Ri:.4f} m. The tip-speed "
            "limit has made the machine too small for its own flux.")

    teeth_Ri = radius_gap + design.airgap_thickness
    stator_Ri = teeth_Ri + design.teeth_thickness
    stator_Ro = stator_Ri + yoke_t

    B_windings = (MU_0 * design.J_max * design.winding_kpf
                  * design.teeth_thickness)
    if B_steel ** 2 <= B_windings ** 2:
        raise ValueError(
            "the winding's own field exceeds the teeth's saturation flux; "
            "the tooth-area expression takes the square root of a negative")

    A_annulus = cross_sectional_area(teeth_Ri + design.teeth_thickness,
                                     teeth_Ri)
    A_teeth = A_annulus - B_gap * A_annulus / math.sqrt(
        B_steel ** 2 - B_windings ** 2)
    A_slots = A_annulus - A_teeth
    A_slot = A_slots / design.N_slots

    lam = A_slots / (A_slots + A_teeth)
    l_end_turns = (math.pi / (2.0 * design.N_pole_pairs) * teeth_Ri
                   * (lam / math.sqrt(1.0 - lam ** 2)))

    torque = design_power / Omega
    slot_current = design.J_max * design.winding_kpf * A_slot
    force_length = slot_current * design.N_energized_slots * B_gap
    length = torque / (force_length * radius_gap)

    # Shaft: hollow, sized on torsion, then checked on burst.
    shaft_Ro = rotor_Ri
    margin = torque / design.tau_max_shaft * 2.0 * shaft_Ro / math.pi
    if shaft_Ro ** 4 <= margin:
        raise ValueError(
            f"shaft too thin: a solid shaft of radius {shaft_Ro:.4f} m "
            f"cannot carry {torque:.1f} N m at "
            f"{design.tau_max_shaft:.2e} Pa")
    shaft_Ri = (shaft_Ro ** 4 - margin) ** 0.25
    if Omega ** 2 * shaft_Ri ** 2 * design.rho_shaft > design.YTS_shaft:
        raise ValueError(
            "rotational stress exceeds the shaft yield strength")

    masses = {
        "rotor": cross_sectional_area(rotor_Ro, rotor_Ri) * length
        * design.rho_steel,
        "stator": cross_sectional_area(stator_Ro, stator_Ri) * length
        * design.rho_steel,
        "magnet": cross_sectional_area(radius_gap, rotor_Ro) * length
        * design.rho_magnet,
        "teeth": A_teeth * length * design.rho_steel,
    }
    winding_volume = A_slots * (length + 2.0 * l_end_turns)
    masses["windings"] = (
        design.winding_kpf * winding_volume * design.rho_conductor
        + (1.0 - design.winding_kpf) * winding_volume * design.rho_insulator)
    masses["shaft"] = (cross_sectional_area(shaft_Ro, shaft_Ri)
                       * (length * design.l_extra_shaft) * design.rho_shaft)

    # Winding resistance: the end turns are carried twice, once each end,
    # and the copper sits at the cooling-air temperature, not ambient.
    rho_con = 1.68e-8 * (1.0 + 0.00404 * (design.winding_T - 293.15))
    R_phase = design.N_slots_per_phase * slot_resistance(
        rho_con, design.winding_kpf * A_slot, length + 2.0 * l_end_turns)

    return SizedMotor(radius_gap=radius_gap, length=length,
                      mass=sum(masses.values()), masses=masses,
                      A_slot=A_slot, B_gap=B_gap, I=slot_current,
                      phase_resistance=R_phase, Omega=Omega)
