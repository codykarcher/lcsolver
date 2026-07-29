"""Reading .tas files (getparm.f, getval.f) against the real TASOPT 737 run.

The check is total and exact. ``tests/data/wsize_in.txt`` is the complete
parameter state the shipped program handed to ``wsize`` when it sized the
737 -- every one of the 254 ``parg`` entries, the flags, the mission
parameters and both 17-point matrices. This test reads ``runs/737/737.tas``
with the port and asserts the result is that state, entry for entry, with no
tolerance: 27 ``pari`` + 254 ``parg`` + 17 ``parm`` + 51x17 ``para`` +
269x17 ``pare``, unset entries included.

Together with ``tests/test_wsize.py``, which shows what that state sizes to,
this closes the loop: the port can be pointed at a ``.tas`` file and produce
the aircraft TASOPT produces.

Why exactness is reasonable here
--------------------------------
Nothing is computed. The reader parses decimal literals and multiplies them by
unit-conversion factors from the same line, so every value is one or two
IEEE operations away from the text -- the same one or two the Fortran does.
The only derived quantities are the cabin pressure (``atmos`` at the cabin
altitude), the offtake fractions, and the nozzle-area interpolations, and
those match too.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tasopt_py.model import indices as I
from tasopt_py.tasfile import (BIGNUM, TasFormatError, _rkey, _Reader,
                               read_tas)

DATA = Path(__file__).parent / "data"
TAS = Path("/Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.tas")

pytestmark = pytest.mark.skipif(
    not TAS.exists() or not (DATA / "wsize_in.txt").exists(),
    reason="737.tas or the wsize dump not present")


@pytest.fixture(scope="module")
def case():
    return read_tas(TAS)


@pytest.fixture(scope="module")
def reference():
    from test_wsize import _load_in
    return _load_in()


def test_reproduces_the_state_handed_to_wsize(case, reference):
    ref, _ = reference
    m = case.missions[0]
    n = 0
    for k in range(1, I.IITOTAL + 1):
        assert case.pari[k] == ref.pari[k], f"pari[{k}]"
        n += 1
    for k in range(1, I.IGTOTAL + 1):
        assert case.parg[k] == ref.parg[k], f"parg[{k}]"
        n += 1
    for k in range(1, I.IMTOTAL + 1):
        assert m.parm[k] == ref.parm[k], f"parm[{k}]"
        n += 1
    for k in range(1, I.IATOTAL + 1):
        for ip in range(1, I.IPTOTAL + 1):
            assert m.para[k, ip] == ref.para[k, ip], f"para[{k},{ip}]"
            n += 1
    for k in range(1, I.IETOTAL + 1):
        for ip in range(1, I.IPTOTAL + 1):
            assert m.pare[k, ip] == ref.pare[k, ip], f"pare[{k},{ip}]"
            n += 1
    assert n == (I.IITOTAL + I.IGTOTAL + I.IMTOTAL
                 + (I.IATOTAL + I.IETOTAL) * I.IPTOTAL)


def test_run_settings_match_the_dump(case, reference):
    _, kw = reference
    s = case.settings
    assert s.iterwmax == kw["iterwmax"]
    assert (s.wrlx1, s.wrlx2, s.wrlx3) == (kw["wrlx1"], kw["wrlx2"],
                                           kw["wrlx3"])
    assert s.Litprint is True and s.Lopt is False
    assert s.iterfmax == 15


def test_the_737_has_two_missions(case):
    """The weighting line is ``1.0  0.0  ! 0.0  0.0  !...`` -- two values
    before the comment, so two missions, the second weighted zero. That is
    why the run calls woper exactly once."""
    assert case.nmission == 2
    assert case.missions[0].parm[I.IMWOPT] == 1.0
    assert case.missions[1].parm[I.IMWOPT] == 0.0
    # The flags and geometry are shared; only parm/para/pare are per-mission.
    assert case.missions[1].parg is case.parg
    assert case.missions[1].para is not case.missions[0].para


def test_airfoil_path_resolves_against_the_file(case):
    assert case.airfoil_file.endswith("air/C.air")
    assert Path(case.airfoil_file).exists()


def test_unset_parameters_are_huge_not_zero(case):
    """tasopt.f fills every array with 2**1023 first, so a value the file does
    not set is obviously unset rather than silently zero."""
    assert case.parg[I.IGWMTO] == BIGNUM
    assert case.missions[0].para[I.IACD, I.IPCRUISE1] == BIGNUM


def test_tail_sweep_in_the_file_is_dead_input(case):
    """getparm.f reads sweeph and then overwrites it with sweep, under a
    ``###`` comment."""
    assert case.parg[I.IGSWEEPH] == case.parg[I.IGSWEEP]


def test_nfweb_is_hard_wired(case):
    """Its read is commented out in getparm.f."""
    assert case.parg[I.IGNFWEB] == 1.0


def test_the_sequence_lines_are_commented_out_in_this_case(case):
    """``AR ! 8 9 10 11 12`` -- the values are behind a ``!``, so no sweep."""
    assert case.ispars == "" and case.jspars == ""


def test_optimisation_perturbations_are_read(case):
    """Lopt is false here, so these are read and not used -- but a keyword
    with its value commented out must come back as zero, not as missing."""
    d = case.dvarso
    assert d["CL"] == 0.001 and d["AR"] == 0.05
    assert d["lambdas"] == 0.0          # 'lambdas ! 0.001'
    assert d["FPR"] == 0.0 and d["OPR"] == 0.0
    assert set(d) == set(I.CPARO)


# --- the getval.f primitives ---------------------------------------------

def _r(text):
    return _Reader(text)


def test_unit_conversions_on_the_value_line():
    r = _r("77.0 * 0.0254 ! Rfuse\n3600.0 / 2.0 ! something\n5.5 ! plain\n")
    assert r.real() == pytest.approx(77.0 * 0.0254)
    assert r.real() == pytest.approx(1800.0)
    assert r.real() == 5.5


def test_comment_and_blank_lines_are_skipped():
    r = _r("# a comment\n\n% another\n!  a third\n   \n42 ! the value\n")
    assert r.integer() == 42


def test_a_line_that_is_only_a_comment_after_data_is_skipped():
    r = _r("   ! nothing but a comment\n7\n")
    assert r.integer() == 7


def test_logicals():
    r = _r("T\nF\n.TRUE.\n.false.\n")
    assert [r.logical() for _ in range(4)] == [True, False, True, False]


def test_end_of_file_raises_rather_than_stopping():
    r = _r("1.0\n")
    r.real()
    with pytest.raises(TasFormatError, match="end-of-file"):
        r.real()


def test_backspace_rewinds_so_the_line_is_read_again():
    """``backspace(lu)`` in the optimisation loop puts back the first line
    that is not a keyword, so the next read sees it."""
    r = _r("1.0\n2.0\n")
    assert r.real() == 1.0
    r.back()
    assert r.real() == 1.0
    assert r.real() == 2.0


def test_rkey_reports_absent_present_and_valued():
    assert _rkey("Mach 0.8 0.7", "AR", 9)[1] == -2        # keyword absent
    assert _rkey("AR ", "AR", 9)[1] == 0                  # present, no values
    vals, n = _rkey("AR  8 9 10 11 12", "AR", 9)
    assert n == 5 and vals == [8.0, 9.0, 10.0, 11.0, 12.0]


def test_rkey_applies_a_multiplier_to_every_value():
    vals, n = _rkey("Range 3000.0 2000.0 * 1852.0", "Range", 9)
    assert n == 2
    assert vals == pytest.approx([3000.0 * 1852.0, 2000.0 * 1852.0])


def test_rkey_caps_the_number_of_values_read():
    vals, n = _rkey("1.0 2.0 3.0 4.0", " ", 2)
    assert n == 2 and vals == [1.0, 2.0]


def test_rkey_stops_at_a_comment_character():
    vals, n = _rkey("1.0 2.0 ! 3.0 4.0", " ", 9)
    assert n == 2 and vals == [1.0, 2.0]


def test_a_null_list_field_is_refused_rather_than_guessed():
    """Fortran leaves the target unchanged for ``1.0,,2.0``. Nothing shipped
    does it, and quietly picking an interpretation would be worse."""
    with pytest.raises(TasFormatError, match="null list field"):
        _rkey("1.0,,2.0", " ", 9)


# --- the whole way through -------------------------------------------------

def test_sizes_the_737_straight_from_the_tas_file(case):
    """Two iterations is enough to show the wiring: the first row of the
    convergence table has to match the shipped program's."""
    from tasopt_py.aero.airfoil import airtable
    from tasopt_py.sizing.wsize import wsize

    s = case.settings
    r = wsize(*case.design, iterwmax=2, wrlx1=s.wrlx1, wrlx2=s.wrlx2,
              wrlx3=s.wrlx3, initwgt=0, initeng=0,
              table=airtable(case.airfoil_file))
    want = [float(x) for x in
            (DATA / "wsize_history.txt").read_text().split("\n")[1].split()]
    assert int(r.history[0][0]) == 1
    # One unit in the last printed place, per column, as in test_wsize.
    ulp = [1e-10] + [1e-4] * 5 + [1e-3] * 3 + [1e-5]
    for got, w, u in zip(r.history[0][1:], want[1:], ulp):
        assert got == pytest.approx(w, abs=1.5 * u)


# --- the i/j parameter sweeps ----------------------------------------------
# These mutate, so they get their own case rather than the module-scoped one.

@pytest.fixture
def fresh():
    return read_tas(TAS)


def test_sweeping_a_scalar_parameter(fresh):
    from tasopt_py.tasfile import apply_sweep
    apply_sweep(fresh, "AR", 12.0)
    assert fresh.parg[I.IGAR] == 12.0
    apply_sweep(fresh, "bmax", 40.0)
    assert fresh.parg[I.IGBMAX] == 40.0


def test_sweeping_mach_touches_only_the_cruise_block(fresh):
    from tasopt_py.tasfile import apply_sweep
    before = fresh.missions[0].para[I.IAMACH, I.IPCLIMB1]
    apply_sweep(fresh, "Mach", 0.72)
    m = fresh.missions[0]
    for ip in range(I.IPCLIMBN, I.IPDESCENT1 + 1):
        assert m.para[I.IAMACH, ip] == 0.72
    assert m.para[I.IAMACH, I.IPCLIMB1] == before


def test_sweeping_opr_uses_one_lpc_ratio_for_every_point(fresh):
    """The HPC ratio is worked out once, from the start-of-cruise LPC ratio of
    the first mission, and written everywhere -- the per-point form is
    commented out beside it. So a point whose LPC ratio differs does not end
    up at the OPR that was asked for."""
    from tasopt_py.tasfile import apply_sweep
    m = fresh.missions[0]
    m.pare[I.IEPILC, I.IPSTATIC] = m.pare[I.IEPILC, I.IPCRUISE1] * 2.0
    apply_sweep(fresh, "OPR", 35.0)
    got_cruise = (m.pare[I.IEPIHC, I.IPCRUISE1]
                  * m.pare[I.IEPILC, I.IPCRUISE1])
    got_static = (m.pare[I.IEPIHC, I.IPSTATIC]
                  * m.pare[I.IEPILC, I.IPSTATIC])
    assert got_cruise == pytest.approx(35.0)
    assert got_static == pytest.approx(70.0)     # not 35, by construction


def test_sweeping_range_moves_only_the_design_mission(fresh):
    """The loop that would have changed the other missions is commented out."""
    from tasopt_py.tasfile import apply_sweep
    other = fresh.missions[1].parm[I.IMRANGE]
    apply_sweep(fresh, "Range", 4.0e6)
    assert fresh.parg[I.IGRANGE] == 4.0e6
    assert fresh.missions[0].parm[I.IMRANGE] == 4.0e6
    assert fresh.missions[1].parm[I.IMRANGE] == other


def test_an_unsweepable_keyword_is_refused(fresh):
    from tasopt_py.tasfile import TasFormatError, apply_sweep
    with pytest.raises(TasFormatError, match="cannot sweep"):
        apply_sweep(fresh, "Wpay", 1.0)
