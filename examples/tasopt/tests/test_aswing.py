"""The ASWING export (aswout.f, BOUTPUT) against the real TASOPT 737 run.

``tests/data/737.asw`` is the deck the shipped program writes for the 737,
made by flipping ``Laswwrite`` to ``T`` in a copy of ``737.tas``. This is a
**byte-for-byte diff against it** -- all 320 lines, same MD5.

That is a stronger check than it looks, because the file's shape is not
mechanical. Which variables share a table depends on comparing station
positions to a tolerance; which column gets which scale factor depends on a
``log10(2*max)`` rule; and the deck is full of the source's own
mistranslations, five of which are listed in the module docstring of
:mod:`tasopt_py.aswing`. Reproducing the file exactly means reproducing all
of that.

The individual tests below pin the pieces, so a failure says which one moved
rather than only that the file changed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tasopt_py.aswing import aswout, boutput
from tasopt_py.model import beam_indices as B
from tasopt_py.model import indices as I
from tasopt_py.run import run_case

DATA = Path(__file__).parent / "data"
TAS = Path("/Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.tas")
REF = DATA / "737.asw"

pytestmark = pytest.mark.skipif(not TAS.exists() or not REF.exists(),
                                reason="737 case or reference deck absent")


@pytest.fixture(scope="module")
def built():
    r = run_case(TAS, off_design=False)
    m = r.case.missions[0]
    deck = aswout(r.case.pari, r.case.parg,
                  m.para.column(I.IPCRUISE1), r.case.configname,
                  r.fuselage_bl)
    return r, deck, boutput(deck)


def test_the_deck_matches_the_fortran_byte_for_byte(built):
    _, _, text = built
    got = text.split("\n")
    if got and got[-1] == "":
        got.pop()
    want = REF.read_text().split("\n")
    if want and want[-1] == "":
        want.pop()
    assert len(got) == len(want) == 320
    for i in range(len(want)):
        assert got[i] == want[i], (
            f"line {i + 1}\n  port: {got[i]!r}\n  ref : {want[i]!r}")


def test_the_whole_file_is_identical(built):
    _, _, text = built
    assert text == REF.read_text()


# --- the deck's structure -------------------------------------------------

def test_four_beams_in_the_right_order(built):
    _, deck, _ = built
    assert [b.name for b in deck.beams] == [
        "Fuselage", "Wing", "Horizontal Tail", "Vertical Tail"]
    assert [b.kbnum for b in deck.beams] == [1, 2, 3, 4]


def test_the_vertical_tail_belongs_to_the_horizontal_tails_lifting_system(
        built):
    """`IBEAM(IS) = IBEAM(IShtail)`, which is why the deck writes
    `Beam      4      3` rather than `Beam      4`."""
    _, deck, text = built
    vt = deck.beams[3]
    assert vt.kbnum == 4 and vt.ibeam == 3
    assert "Beam      4      3" in text
    assert "Beam      3\n" in text


def test_the_fuselage_is_written_as_five_tables(built):
    """The beam has three griddings -- 31 geometry stations, 8 structural,
    21 aerodynamic -- and BOUTPUT writes a separate table for each. But the
    six-column cap then splits the first two again, each leaving one
    variable stranded in a table of its own: ``radius`` off the geometry
    group and ``Dmg`` off the structural one. Five tables, not three, and
    which variable gets stranded depends only on its index order."""
    _, deck, _ = built
    fuse = deck.beams[0]
    assert fuse.nb[B.JXA] == 31
    assert fuse.nb[B.JECC] == 8
    assert fuse.nb[B.JCDF] == 21

    lines = REF.read_text().splitlines()
    start = lines.index("Beam      1")
    end = start + lines[start:].index("End")
    heads = [ln for ln in lines[start:end] if ln.startswith("       t")]
    assert len(heads) == 5
    widths = [(len(h) - 16) // 11 for h in heads]
    assert widths == [6, 6, 1, 1, 2]
    assert heads[2].split() == ["t", "Dmg"]
    assert heads[3].split() == ["t", "radius"]


def test_no_table_is_wider_than_six_variables(built):
    """`IF(KSAV.GE.6) GO TO 21` -- the source caps a block at six columns
    so the file stays editable."""
    for ln in REF.read_text().splitlines():
        if ln.startswith("       t"):
            assert (len(ln) - 16) // 11 <= 6


def test_the_shell_inertia_never_reaches_the_deck(built):
    """mgcc1 and mgnn1 are computed for the fuselage and their LQBDEF lines
    are commented out, so the shell's own rotational inertia is dropped."""
    _, deck, _ = built
    fuse = deck.beams[0]
    assert B.JMCC1 in fuse.nb and B.JMCC1 not in fuse.defined
    assert B.JMNN1 in fuse.nb and B.JMNN1 not in fuse.defined
    assert fuse.q(3, B.JMCC1) > 0.0        # it was computed


# --- the mistranslations, each pinned -------------------------------------

def test_the_fuselage_drag_uses_the_last_stations_edge_velocity(built):
    """§46. `ue` is updated outside the interval test, so it ends up at the
    last BL interval whatever the station is. Every station therefore gets
    the same ue, and Cdf is off by the cube of a velocity ratio."""
    r, deck, _ = built
    fuse = deck.beams[0]
    bl = r.fuselage_bl
    # The ue the loop actually leaves behind, at the trailing edge.
    ue_used = bl.ue[bl.iblte - 1]
    # Reconstruct Cdiss at a mid-body station and check Cdf is 2*Cdiss*ue^3
    # with *that* ue rather than the local one.
    x = fuse.t(11, B.JCDF)
    local = None
    for ibl in range(1, bl.iblte):
        if bl.x[ibl - 1] <= x <= bl.x[ibl]:
            local = bl.ue[ibl - 1]
    assert local is not None
    assert abs(local - ue_used) > 0.01        # they really are different
    Cdf = fuse.q(11, B.JCDF)
    Cdiss = Cdf / (2.0 * ue_used ** 3)
    assert Cdf == pytest.approx(2.0 * Cdiss * ue_used ** 3)
    assert Cdf != pytest.approx(2.0 * Cdiss * local ** 3, rel=1e-3)


def test_the_wing_tip_gets_the_break_sections_moment_and_drag(built):
    """§47. `cm = cms*(1-frac) + cms*frac` in the outer panel, and the same
    for cdf and cdp, so all three are pinned at the break value; cmt, cdft
    and cdpt are computed just above and never used.

    It costs nothing on any shipped case, and the test says why: the three
    inputs are equal anyway. `cdf` and `cdp` are hard-wired to 0.006 and
    0.003 at all three stations, and the 737's section moment is uniform
    across the span. So this is a latent bug -- visible in the source,
    invisible in the deck -- and it would start to matter the moment
    someone gave the tip its own airfoil.
    """
    r, deck, _ = built
    wing = deck.beams[1]
    for j in (B.JCM, B.JCDF, B.JCDP):
        assert len({wing.q(i, j) for i in range(7, 12)}) == 1

    para = r.case.missions[0].para
    assert (para[I.IACMPO, I.IPCRUISE1] == para[I.IACMPS, I.IPCRUISE1]
            == para[I.IACMPT, I.IPCRUISE1])
    assert wing.q(11, B.JCM) == pytest.approx(
        para[I.IACMPS, I.IPCRUISE1])
    assert wing.q(11, B.JCDF) == 0.006 and wing.q(11, B.JCDP) == 0.003


def test_the_wing_shell_variables_carry_over_instead_of_interpolating(
        built):
    """§48. `Csh = Csho*(1-frac) + Csh*frac` -- the right-hand side is the
    loop variable, not the break value Cshs. The result is that the inner
    panel never reaches Cshs, and the outer panel starts from Cshs anyway,
    so the column jumps at the break."""
    _, deck, _ = built
    wing = deck.beams[1]
    inner_end = wing.q(6, B.JCSH)      # last station of the inner panel
    outer_start = wing.q(7, B.JCSH)    # first of the outer, = Cshs
    assert inner_end != pytest.approx(outer_start, rel=1e-6)
    # The centre-section value is what it is dragged toward instead.
    assert wing.q(1, B.JCSH) == pytest.approx(wing.q(2, B.JCSH))


def test_the_wing_fuel_inertia_ends_at_the_fuel_mass(built):
    """§49. `mgnnf = mgnnofuel*(1-frac) + mgsfuel*frac` -- the outboard end
    of the interpolation is a mass where an inertia was meant, so the column
    steps discontinuously across the doubled break station."""
    _, deck, _ = built
    wing = deck.beams[1]
    at_break_inner = wing.q(6, B.JMNN2)
    at_break_outer = wing.q(7, B.JMNN2)
    assert at_break_inner / at_break_outer == pytest.approx(3.5, abs=0.2)


def test_the_reference_points_are_all_at_the_wing_box(built):
    """Moment, acceleration and velocity references, all three columns."""
    r, deck, _ = built
    for L in range(3):
        assert deck.xyzref[0][L] == r.case.parg[I.IGXWBOX]
        assert deck.xyzref[1][L] == 0.0
        assert deck.xyzref[2][L] == 0.0


# --- the pieces that hold the beams together ------------------------------

def test_eight_point_weights_and_two_engines(built):
    _, deck, _ = built
    weights = [p for p in deck.pylons if p.kptype == 1]
    engines = [p for p in deck.pylons if p.kptype > 10]
    assert len(weights) == 8      # fixed, APU, hyd/elec, nose gear,
    assert len(engines) == 2      # 2 main gear, 2 engine masses
    assert deck.engtyp == [0, 0]


def test_the_engines_are_on_the_wing_beam_and_push_forward(built):
    """iengloc = 1 on the 737, so the engines hang off beam 2 at the wing
    break, and their thrust vector is -x."""
    r, deck, _ = built
    engines = [p for p in deck.pylons if p.kptype > 10]
    for p in engines:
        assert p.kbeam == 2
        assert (p.q[4], p.q[5], p.q[6]) == (-1.0, 0.0, 0.0)
        assert p.q[7] == pytest.approx(0.5)      # 1/neng
        assert abs(p.q[0]) == pytest.approx(0.5 * r.case.parg[I.IGBS])
    assert engines[0].q[0] == -engines[1].q[0]


def test_the_wing_is_grounded_rather_than_jointed(built):
    """The two fuselage-to-wing joints are commented out in the source; the
    wing is held by ground attachments at its two root stations instead."""
    r, deck, _ = built
    assert [g.beam for g in deck.grounds] == [1, 2, 2]
    assert deck.grounds[0].t == r.case.parg[I.IGXWBOX]
    assert deck.grounds[1].t == pytest.approx(0.5 * r.case.parg[I.IGBO])
    assert deck.grounds[1].t == -deck.grounds[2].t
    assert all(set(j.beams) != {1, 2} for j in deck.joints)


def test_the_tail_joints_are_the_conventional_pair(built):
    """Not a T-tail or a Pi-tail on the 737: the fin joins the fuselage at
    the vertical box and the stabiliser joins it at the horizontal box."""
    _, deck, _ = built
    assert [j.beams for j in deck.joints] == [(1, 4), (1, 3)]


# --- the parts nothing shipped exercises ----------------------------------

def test_a_sensor_block_cannot_be_produced(built):
    """BOUTPUT can write Strut and Sensor blocks; aswout never builds one,
    so the port says so rather than emitting an untested format."""
    from tasopt_py.aswing import Pylon, boutput as write

    _, deck, _ = built
    deck.pylons.append(Pylon(kptype=101, kbeam=1))
    try:
        with pytest.raises(NotImplementedError, match="pylon type 4"):
            write(deck)
    finally:
        deck.pylons.pop()


def test_the_beam_indices_are_generated_and_zero_based(built):
    """Unlike index.inc's, ASWING's VARS is declared (0:JBTOT)."""
    assert B.JSA == 0 and B.JXA == 1
    assert B.JBTOT == 102 and B.JBFUSE == 35
    assert len(B.VARS) == B.JBTOT + 1 == len(B.KBREAK)
    assert all(len(v) == 11 for v in B.VARS)
    assert B.VARS[B.JCSH].strip() == "Cshell"
