"""The structural choices that distinguish one SPaircraft configuration.

Where do the engines hang, what shape is the fuselage cross-section, and how
is the horizontal tail mounted -- those three questions are the whole
difference between the D8.2 and a 737, once the numbers in ``subs/`` are set
aside. The source answers them with module-level flags read from inside the
model (``if rearengine: ... if tube: ... if piHT: ...``). Here each answer is
a named function, and a deck calls the ones its aeroplane has.

That is the same physics with the branch removed. Nothing dispatches on a
configuration name; there is no table mapping ``'optimal737'`` to a set of
booleans. A deck that mounts its engines on the wing calls
:func:`wing_engines`, and the reason its wing root moment gets engine load
relief is that wing engines relieve the wing root -- not that a flag was set
four files away.

The pairings are not arbitrary. A wing-engined aeroplane is a tube with a
conventional tail because that is what the source's ``if conventional:
wingengine = True; tube = True`` says, and because the combination is what the
substitution decks were tuned against. Mixing them is possible and is exactly
what ``D8_eng_wing`` does -- but that configuration does not converge in
gpkit, so there is nothing to check a mixture against.
"""
from __future__ import annotations

from numpy import pi, tan

__all__ = ["rear_engines", "wing_engines",
           "double_bubble_floor", "tube_floor",
           "pi_tail", "conventional_tail"]


# ---------------------------------------------------------------------------
# Engine location
# ---------------------------------------------------------------------------

def rear_engines(p, v, K):
    """Engines on the aft fuselage, ahead of the tail.

    Three consequences, all of them about where the mass is:

    * the wing root carries no engine, so its moment gets relief from wing
      weight and wing fuel only;
    * the aft fuselage does, so the engine weight enters both horizontal
      bending load cases and the tail's yaw inertia;
    * the engine sits on the tail cone, which is what brackets ``x_eng``.

    ``y_eng`` is *not* set here -- how far outboard the engine sits depends on
    whether it is buried at the centreline ingesting boundary layer or podded
    beside the fuselage, and that is the deck's to say.
    """
    wing, vt, ht, fu = p.wing, p.vt, p.ht, p.fu
    return [
        # Wing root moment, with wing weight and fuel load relief.
        wing.box.M_r * wing.c_root >= (
            (wing.L_max - wing.box.N_lift * (wing.W_wing
                                             + K.f_wingfuel * v.W_ftotal))
            * (wing.b ** 2 / (12 * wing.S)
               * (wing.c_root + 2 * wing.c_tip))),                  # [SP]

        # Horizontal tail aero + landing load constants. The engines are aft,
        # so they bend the fuselage along with the tail.
        fu.A_1h_Land >= (fu.N_land * (fu.W_tail + K.n_eng * v.Wengsys
                                      + fu.W_apu))
                        / (fu.h_fuse * fu.sigma_bend),
        fu.A_1h_MLF >= (fu.N_lift * (fu.W_tail + K.n_eng * v.Wengsys
                                     + fu.W_apu)
                        + fu.r_M_h * ht.L_ht_max)
                       / (fu.h_fuse * fu.sigma_M_h),

        # Moments of inertia. No engine term on the wing; the engines are on
        # the tail, at l_vt.
        v.Izwing >= ((wing.W_fuel_wing + wing.W_wing) / (wing.S * K.g)
                     * wing.c_root * wing.b ** 3.
                     * (1. / 12. - (1. - wing.lambda_) / 16.)),
        v.Iztail >= ((fu.W_apu + vt.W_vt + K.n_eng * v.Wengsys)
                     * vt.l_vt ** 2. / K.g
                     + ht.W_ht * ht.l_ht ** 2. / K.g),
        # x_wing and l_vt stand in for CG-relative distances so I_z stays
        # scalar: x_CG moves through the flight, and a per-segment inertia
        # would size the VT against a moving target.
        v.Izfuse >= ((fu.W_fuse + fu.W_payload_max) / fu.l_fuse
                     * (fu.x_wing ** 3. + vt.l_vt ** 3.) / (3. * K.g)),

        # Engine weight centroid, on the tail cone.
        v.xeng <= fu.x_shell2 + 1.00 * fu.l_cone,
        v.xeng >= fu.x_shell2 + 0.75 * fu.l_cone,
    ]


def wing_engines(p, v, K, *, sweep_w):
    """Engines podded under the wing.

    The mirror image of :func:`rear_engines`. The engine now hangs off the
    wing at ``y_eng``, which

    * **relieves** the wing root moment -- the engine weight acts downward
      against the lift the root has to carry, and it enters with a minus sign,
      which is what makes this constraint signomial;
    * takes its weight *out* of the fuselage bending cases and the tail
      inertia, and puts it into the wing's yaw inertia at ``y_eng**2``;
    * has to clear the ground when the aeroplane rolls, which is a constraint
      a rear-engined aeroplane simply does not have.

    ``x_eng`` follows the quarter-chord aft as the wing sweeps, so it is built
    from the wing station rather than the tail cone.
    """
    wing, vt, ht, lg, fu = p.wing, p.vt, p.ht, p.lg, p.fu
    return [
        # Wing root moment, with wing, fuel AND engine weight load relief.
        wing.box.M_r * wing.c_root >= (
            (wing.L_max - wing.box.N_lift * (wing.W_wing
                                             + K.f_wingfuel * v.W_ftotal))
            * (wing.b ** 2 / (12 * wing.S)
               * (wing.c_root + 2 * wing.c_tip))
            - wing.box.N_lift * v.Wengsys * v.y_eng),               # [SP]

        # Horizontal tail aero + landing load constants, tail and APU only.
        fu.A_1h_Land >= (fu.N_land * (fu.W_tail + fu.W_apu))
                        / (fu.h_fuse * fu.sigma_bend),
        fu.A_1h_MLF >= (fu.N_lift * (fu.W_tail + fu.W_apu)
                        + fu.r_M_h * ht.L_ht_max)
                       / (fu.h_fuse * fu.sigma_M_h),

        # Moments of inertia. The engines are the wing's, at y_eng.
        v.Izwing >= (K.n_eng * v.Wengsys * v.y_eng ** 2. / K.g
                     + (wing.W_fuel_wing + wing.W_wing) / (wing.S * K.g)
                     * wing.c_root * wing.b ** 3.
                     * (1. / 12. - (1. - wing.lambda_) / 16.)),     # [SP]
        v.Iztail >= ((fu.W_apu + vt.W_vt) * vt.l_vt ** 2. / K.g
                     + ht.W_ht * ht.l_ht ** 2. / K.g),
        v.Izfuse >= ((fu.W_fuse + fu.W_payload_max) / fu.l_fuse
                     * (fu.x_wing ** 3. + vt.l_vt ** 3.) / (3. * K.g)),

        # Engine ground clearance in a roll: the wing has dihedral, so an
        # engine outboard of the main gear gains height at tan(gamma).
        lg.d_nacelle + lg.h_nacelle
            <= lg.l_m + (v.y_eng - lg.y_m) * lg.tan_gamma,          # [SP]

        # Engine weight centroid, swept aft with the wing.
        v.xeng >= (fu.x_wing + tan(sweep_w * pi / 180) * v.y_eng
                   - 0.5 * v.lnace),
    ]


# ---------------------------------------------------------------------------
# Fuselage cross-section
# ---------------------------------------------------------------------------

def double_bubble_floor(p):
    """Two intersecting circles joined by a tension web.

    The floor spans between the two bubbles and is supported by the web at
    mid-span, so it is a two-span continuous beam rather than a simple one:
    shear 5/16 and moment 9/256 of the simply-supported values, against 1/2
    and 1/4 for a tube. ``dR_fuse`` -- the straight section between the two
    half-circles -- is tied to the radius rather than left free.
    """
    fu = p.fu
    return [
        fu.S_floor == (5. / 16.) * fu.P_floor,
        fu.M_floor == 9. / 256. * fu.P_floor * fu.w_floor,
        fu.dR_fuse == fu.R_fuse * 0.43 / 1.75,
    ]


def tube_floor(p):
    """A single circular barrel.

    The floor is a simply supported beam across the full width: shear
    ``P/2``, moment ``P*w/4``. Nothing constrains ``dR_fuse`` here, because a
    tube has no straight section -- the deck pins it near zero instead, along
    with ``theta_db``, which is what collapses ``fuselage.py``'s double-bubble
    geometry onto a circle.
    """
    fu = p.fu
    return [
        fu.S_floor == 1. / 2. * fu.P_floor,
        fu.M_floor == 1. / 4. * fu.P_floor * fu.w_floor,
    ]


# ---------------------------------------------------------------------------
# Horizontal tail mounting
# ---------------------------------------------------------------------------

def pi_tail(f, p, v, K, *, supports="pinned"):
    """Horizontal tail spanning the tops of two vertical tails.

    There is no root in the structural sense: the horizontal is carried at
    ``+/- w_fuse`` and overhangs outboard of each vertical. Two stations can
    size it -- the centreline and the attachment -- and which one governs
    depends on whether the joints carry moment.

    ``supports='pinned'`` reproduces the source. ``supports='fixed'`` is the
    more physical alternative; see the comments below and DISCREPANCIES.md.
    """
    ht, fu = p.ht, p.fu
    hb = ht.box
    Mrout = f.Variable(name="M_r_out", guess=1e5, units="N",
                       description="HT moment at the VT attachment")
    cons = [
        hb["b_ht_out"] == 0.5 * ht.b_ht - fu.w_fuse,                # [SP] SigEq
        Mrout * ht.c_attach >= (hb["L_ht_rect_out"] * (0.5 * hb["b_ht_out"])
                                + hb["L_ht_tri_out"] * (1. / 3. * hb["b_ht_out"])),
        hb["L_shear"] >= hb["L_ht_rect_out"] + hb["L_ht_tri_out"],
        ht.c_tip_ht + (1. - ht.lambda_ht) * 2. * hb["b_ht_out"] / ht.b_ht
            * ht.c_root_ht == ht.c_attach,                          # [SP] SigEq
    ]

    if supports == "pinned":
        # SOURCE BEHAVIOUR. The verticals are treated as pin joints carrying
        # no moment, so the inboard span is simply supported and the
        # centreline moment is the applied moment MINUS the support reaction.
        # That subtraction is what makes M_r degenerate: the two terms can
        # very nearly cancel, the constraint stops binding, and M_r collapses
        # onto the 1e-30 box floor along with I_cap and t_cap. Reproduced
        # because the gpkit reference depends on it -- see DISCREPANCIES.md.
        cons += [
            ht.b_ht / 4. * hb["L_ht_rect"] + ht.b_ht / 3. * hb["L_ht_tri"]
                == hb["b_ht_out"] * ht.L_ht_max / 2.,               # [SP] SigEq
            hb["M_r"] * ht.c_root_ht >= (hb["L_ht_rect"] * (ht.b_ht / 4.)
                                         + hb["L_ht_tri"] * (ht.b_ht / 6.)
                                         - fu.w_fuse * ht.L_ht_max / 2.),
            hb["pi_M_fac"] >= ((0.5 * (Mrout * ht.c_attach
                                       + hb["M_r"] * ht.c_root_ht)
                                * fu.w_fuse
                                / (0.5 * Mrout * ht.c_attach * hb["b_ht_out"])
                                + 1.0) * hb["b_ht_out"] / (0.5 * ht.b_ht)),
        ]
    elif supports == "fixed":
        # FIXED SUPPORTS. A pi-tail horizontal joins two verticals rigidly, so
        # the inboard span is a beam BUILT IN at both ends, not pin-jointed.
        # For span L under load W the standard results are
        #
        #     hogging at each support   W*L/12
        #     sagging at midspan        W*L/24
        #
        # against W*L/8 at midspan and zero at the supports if pinned. Two
        # consequences, and they are the point of the change:
        #
        # 1. The sizing station moves to the ATTACHMENT, where the fixed-end
        #    moment adds to the overhang moment. A root moment never sizes a
        #    pi-tail horizontal -- there is no root, only two supports.
        # 2. Every moment is now a SUM of positive terms. Nothing can cancel,
        #    so M_r cannot collapse, and the constraint is posynomial rather
        #    than signomial -- strictly easier for the solver as well as more
        #    physical.
        #
        # The verticals sit at +/- w_fuse, so the built-in span is 2*w_fuse.
        Lin = f.Variable(name="L_ht_in", guess=1e5, units="N",
                         description="HT load inboard of the VT attachments")
        Mfe = f.Variable(name="M_fe", guess=1e4, units="N",
                         description="fixed-end moment per attachment chord")
        cons += [
            # Load inboard of the attachments. The section is untapered over
            # this span, so its share of the load is its share of the span.
            Lin >= ht.L_ht_max * (2. * fu.w_fuse) / ht.b_ht,
            # Fixed-end (hogging) moment at each support, W*L/12.
            Mfe * ht.c_attach >= Lin * (2. * fu.w_fuse) / 12.,
            # The attachment carries the overhang AND the fixed-end moment;
            # both hog the beam over the support, so they add.
            Mrout * ht.c_attach >= (hb["L_ht_rect_out"] * (0.5 * hb["b_ht_out"])
                                    + hb["L_ht_tri_out"] * (1. / 3. * hb["b_ht_out"])
                                    + Mfe * ht.c_attach),
            # Sagging at the centreline, W*L/24 -- half the fixed-end value
            # and a third of what a pinned span would carry.
            hb["M_r"] * ht.c_root_ht >= Lin * (2. * fu.w_fuse) / 24.,
            # Load split, unchanged in form but now with no cancellation.
            ht.b_ht / 4. * hb["L_ht_rect"] + ht.b_ht / 3. * hb["L_ht_tri"]
                == hb["b_ht_out"] * ht.L_ht_max / 2.,               # [SP] SigEq
            # The cap must carry the larger of the two stations.
            hb["pi_M_fac"] >= 1.0,
            hb["pi_M_fac"] >= Mrout * ht.c_attach / (hb["M_r"] * ht.c_root_ht),
        ]
    else:
        raise ValueError(f"supports must be 'pinned' or 'fixed', got "
                         f"{supports!r}")
    return cons


def conventional_tail(p, v, K, *, sweep_vt, sweep_ht):
    """Horizontal tail mounted on the fuselage, at the base of one fin.

    This is the simple case the pi-tail is not. There *is* a root, at the
    centreline; the attachment chord is the root chord; the outboard half-span
    is half the span; and the cap carries one station, so the pi-tail's
    structural amplification factor is exactly one. Every moment is a sum of
    positive terms, so nothing here is signomial and nothing can collapse the
    way ``pi_M_fac`` does on the D8.

    The trailing edge is limited by the fuselage running out rather than by
    the fin's sweep, which is the other half of the difference: a conventional
    tail can extend to the end of the tail cone, and the source states that as
    ``l_fuse >= x_CG + dx_trail_ht``. ``sweep_vt`` and ``sweep_ht`` are taken
    for signature symmetry with the pi-tail and are not used -- on this layout
    the fin geometry does not enter.

    The source also states ``M_{r_{out}} == M_r`` here. That variable exists
    only so the pi-tail can distinguish its two sizing stations; nothing else
    in this port reads it, and on a conventional tail the two stations are the
    same station. It is left out rather than created and immediately pinned.
    """
    ht, fu = p.ht, p.fu
    hb = ht.box
    return [
        # HT root moment: rectangular plus triangular load over the half-span.
        hb["M_r"] * ht.c_attach >= (1. / 3. * hb["L_ht_tri_out"] * hb["b_ht_out"]
                                    + 1. / 2. * hb["L_ht_rect_out"] * hb["b_ht_out"]),
        # The attachment IS the root.
        ht.c_attach == ht.c_root_ht,
        hb["b_ht_out"] == 0.5 * ht.b_ht,
        hb["L_shear"] >= hb["L_ht_rect_out"] + hb["L_ht_tri_out"],
        # One sizing station, so no amplification.
        hb["pi_M_fac"] == 1.0,
        # Trailing edge limited by the fuselage, not by the fin.
        fu.l_fuse >= v.xCG + ht.dx_trail_ht,
    ]
