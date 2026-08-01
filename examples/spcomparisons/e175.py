"""The Embraer 175 validation case.

The second cell with a real aeroplane behind it. Run at the same technology
level as the 737-800 case: the E175 entered service in 2004 with GE CF34-8E
engines (BPR ~5.0, OPR ~28.5) on an aluminium airframe, which is the same
generation as the 737NG rather than a modern geared fan. ``cfm56_era`` carries
allowables from TASOPT's 737 deck and an OPR/BPR ceiling of 32 / 5.5; the
CF34-8E sits just inside both, so the ceilings should not be what sizes this
engine. The class already selects the ``CFM56`` deck, whose design pressure
ratio (1.685 x 1.935 x 9.369 = 30.6) is a fair stand-in for the CF34.

Seat count
----------
This class is sized for 88 passengers, the E175's single-class certified
capacity, NOT the 76 of a typical dual-class layout. Every reference figure
below is for the airframe as built, so sizing to 76 compared a smaller
aeroplane against a larger one: fuselage 17% short and MTOW 20% light, versus
the 737's 2% and 0.1% where the class was already near its real maximum. At 88
those become 0.914 and 0.910. What remains is a real gap; the rest was the
question being asked wrong.

Fewer reference figures than the 737
------------------------------------
Wing area, span and length for the E170/175 are widely published and are used
here. Tail areas and landing gear stations are NOT -- I do not have figures I
can attribute, so they are absent rather than guessed. That makes this a
weaker validation than the 737-800 case, and saying so is the point: an
invented reference is worse than a missing one, because it looks like
evidence.

Span
----
The model does NOT run to its span limit here (26.25 m against a 28.7 m gate
box), so unlike the 737 this is a free result. It lands within 1% of the real
BARE span of 26.00 m, which is the right comparison because there is no
winglet model. An earlier draft compared against the 28.65 m winglet span and
made a correct answer look 8% short.
"""
from __future__ import annotations

from validation import Case, Ref, run

#: The real Embraer 175. MTOW and OEW are the figures already carried in
#: classes.py for this class; geometry is from published E170/175 data.
REFERENCE = {
    "W_total_max": Ref(85_517, "lbf", "E175 MTOW, as carried in classes.py"),
    "Wing_S":      Ref(72.72, "m^2", "E170/175 wing reference area"),
    # BARE span. The model has no winglet representation, so comparing to the
    # 28.65 m winglet span would charge it for a device it does not model --
    # and the 737 case uses bare span, so winglets here would be inconsistent.
    "Wing_b":      Ref(26.00, "m", "span WITHOUT winglets (28.65 m with)"),
    "Fuse_l_fuse": Ref(31.68, "m", "E175 overall length"),
}

KNOWN_GAPS = {
    "Wing_b": (
        "Not resolvable with this model. The real E175 is quoted at 26.00 m "
        "bare and 28.65 m with winglets; the model has no winglet "
        "representation, so a wing reaching equivalent induced drag without "
        "them should be physically LONGER than 26.00. The model lands at "
        "27.6-28.4 m, i.e. between the two published figures, which is the "
        "expected place for a no-winglet wing to sit. Neither number is the "
        "right target, so this quantity is weak evidence in either direction "
        "until winglets are modelled. It is NOT sitting on the 28.7 m gate "
        "limit, so it is at least a free result."),
}

CASE = Case(
    key="e175",
    label="Embraer 175 validation",
    locked={
        "TECH": "cfm56_era",
        "GEAR_BOX_FRAC": "1.00",     # main gear on the rear spar
        "V_HT_FLOOR": "1.00",        # Raymer horizontal tail volume floor
    },
    reference=REFERENCE,
    known_gaps=KNOWN_GAPS,
    lock_mach=True,
    notes="Regional jet; CF34-8E class engine on an aluminium airframe.",
)


if __name__ == "__main__":
    run(CASE, both_mach=True)
