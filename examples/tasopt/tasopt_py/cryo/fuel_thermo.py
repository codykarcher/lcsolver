"""Saturated properties of cryogenic fuels -- ``fuel_thermo.jl``.

**This has no counterpart in TASOPT 2.16.** The Fortran stores fuel as a
density and a temperature (``rhofuel``, ``Tfuel`` in the ``.tas`` file) and
never asks what phase it is in, because kerosene is a liquid at every
condition an airliner sees. A cryogen is not: liquid hydrogen boils at 20 K
at one atmosphere, and everything about carrying it -- tank pressure, wall
temperature, boil-off, how much of the fuel is vapour -- turns on where the
saturation line is.

So this is the first thing the tank needs: given a tank pressure, the
saturation temperature and the density, enthalpy and internal energy of each
phase on it, plus their pressure derivatives.

Where the numbers come from
---------------------------
TASOPT.jl fits NIST saturation data at 0.1 atm increments from 0.1 to 10 atm
with polynomials in pressure-in-atmospheres, up to sixth order. Those
coefficients are generated into :mod:`~tasopt_py.cryo.fuel_thermo_fits` by
``tools/gen_fuel_thermo.py`` rather than transcribed.

**The derivatives are differentiated here, not copied.** The Julia writes
``ρ_p`` and ``u_p`` out by hand, term by term with the powers brought down
manually. Differentiating the value polynomial instead means a slip in those
expressions cannot propagate into this port -- and comparing the two is a
check on the reference, which ``tests/test_fuel_thermo.py`` performs.

Range
-----
The fits are valid over the pressure range they were made on, 0.1 to 10 atm.
Outside it a sixth-order polynomial diverges fast and silently, so
:func:`gas_properties` and :func:`liquid_properties` refuse rather than
extrapolate. TASOPT.jl does not check; it is a fit, not a physical model, and
a sixth-order fit evaluated at 40 atm is not a number anybody should use.
That is a deliberate departure and it is tested.

Verified against TASOPT.jl; see ``tests/test_fuel_thermo.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

from .fuel_thermo_fits import FITS

__all__ = ["gas_properties", "liquid_properties", "SaturatedPhase",
           "P_ATM", "P_MIN_ATM", "P_MAX_ATM", "SPECIES"]

#: Standard atmosphere, Pa -- TASOPT.jl's ``p_atm``.
P_ATM = 101325.0

#: The pressure range the NIST fits were made over, in atmospheres.
P_MIN_ATM, P_MAX_ATM = 0.1, 10.0

#: Fuels with a fit. ``LH2`` is accepted as a synonym for ``H2``, as in the
#: Julia, so a caller can name the liquid.
SPECIES = ("H2", "CH4")
_ALIAS = {"LH2": "H2", "LCH4": "CH4"}


@dataclass(frozen=True)
class SaturatedPhase:
    """One phase on the saturation line, at a given pressure."""
    Tsat: float      # saturation temperature, K
    rho: float       # density, kg/m^3
    rho_p: float     # d(rho)/dp, kg/(m^3 Pa)
    h: float         # specific enthalpy, J/kg
    u: float         # specific internal energy, J/kg
    u_p: float       # d(u)/dp, J/(kg Pa)


def _polyval(c, x: float) -> float:
    """Horner, highest power first."""
    out = 0.0
    for a in c:
        out = out * x + a
    return out


def _polyder(c) -> list:
    """Coefficients of the derivative, highest power first."""
    n = len(c) - 1
    return [a * (n - i) for i, a in enumerate(c[:-1])]


def _normalise(species: str) -> str:
    key = species.upper()
    key = _ALIAS.get(key, key)
    if key not in SPECIES:
        raise ValueError(
            f"no saturation fit for {species!r}; have {list(SPECIES)}. "
            "TASOPT.jl silently returns undefined values for an unknown "
            "species -- its `if` chain has no `else` -- so this refuses "
            "instead.")
    return key


def _properties(kind: str, species: str, p: float) -> SaturatedPhase:
    key = _normalise(species)
    x = p / P_ATM                       # pressure in atmospheres
    if not (P_MIN_ATM <= x <= P_MAX_ATM):
        raise ValueError(
            f"pressure {p:.0f} Pa is {x:.3f} atm, outside the {P_MIN_ATM}-"
            f"{P_MAX_ATM} atm range these NIST fits were made over. A "
            "sixth-order polynomial diverges quickly outside its fit range; "
            "extrapolating one is not a physical model.")

    f = FITS[(kind, key)]
    # h and u are fitted in kJ/kg and converted after evaluation, as the
    # Julia does; rho_p and u_p are per atmosphere and converted to per Pa.
    return SaturatedPhase(
        Tsat=_polyval(f["Tsat"], x),
        rho=_polyval(f["rho"], x),
        rho_p=_polyval(_polyder(f["rho"]), x) / P_ATM,
        h=_polyval(f["h"], x) * 1.0e3,
        u=_polyval(f["u"], x) * 1.0e3,
        u_p=_polyval(_polyder(f["u"]), x) * 1.0e3 / P_ATM,
    )


def gas_properties(species: str, p: float) -> SaturatedPhase:
    """Saturated **vapour** properties at pressure ``p`` [Pa]."""
    return _properties("gas", species, p)


def liquid_properties(species: str, p: float) -> SaturatedPhase:
    """Saturated **liquid** properties at pressure ``p`` [Pa]."""
    return _properties("liquid", species, p)
