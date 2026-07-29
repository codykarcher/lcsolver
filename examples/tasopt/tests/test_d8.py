"""A second aircraft, end to end -- the strut-braced, BLI, Pi-tailed D8.

Every other end-to-end check in this suite is against the 737. That is one
aircraft, and it exercises one path through the code: cantilever wing, two
wing-mounted engines, one fin, no boundary-layer ingestion. ``runs/D8/sd81``
is none of those. It has

* ``iwplan = 2``   a strut-braced wing (a fifth ASWING beam),
* ``nvtail = 2``   a Pi-tail,
* ``iengloc = 2``  three tail-mounted engines,
* ``fBLIf = 0.4``  the fan ingests 40% of the fuselage wake.

The last is the important one for the *numbers*: on the 737 ``fBLIf`` is zero,
so ``Phiinl`` and ``Kinl`` are identically zero and the whole BLI credit path
is dead. Nothing before this file had ever run it.

What it found
-------------
The port could not size this aircraft at all until §51 was fixed. ``mission``
had an ``ip > IPDESCENT1`` guard on the descent engine-state seeding that the
source does not have, so the first descent point cold-started where the
reference program seeds it from the last cruise point. The 737 survives that;
the D8's cold start diverges into a state where the LPT is asked to extract
more enthalpy than the flow contains.

That is the same shape of bug as the four in ``test_mission.py``: a path the
737 does not take, invisible until a second aircraft ran.

Agreement
---------
Sizing matches the reference program's 134921.7659 lbf in the same 18
iterations, and **2346 of the 2347 lines of the report are identical**. The
one that is not is ``Kinl`` at climb 1, off by 1.8e-5 relative -- not a
rounding boundary, a real difference, recorded as §52.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tasopt_py.model import indices as I
from tasopt_py.output import report
from tasopt_py.run import run_case

DATA = Path(__file__).parent / "data"
TAS = Path("/Users/codykarcher/Desktop/Tasopt2.16/runs/D8/sd81.tas")
REF = DATA / "sd81.out"

pytestmark = pytest.mark.skipif(not TAS.exists() or not REF.exists(),
                                reason="D8 case or reference report absent")

#: What the reference program sizes this aircraft to.
REF_WTO_LBF = 134921.7659
REF_ITERATIONS = 18


@pytest.fixture(scope="module")
def flown():
    return run_case(TAS)


def test_it_sizes_to_the_same_aircraft(flown):
    assert flown.sized.converged
    assert flown.sized.iterations == REF_ITERATIONS
    assert flown.WTO_lbf == pytest.approx(REF_WTO_LBF, abs=5e-4)


def test_the_configuration_is_the_one_the_737_is_not(flown):
    """If any of these stops being true the test has stopped covering what
    it was written to cover."""
    case = flown.case
    assert case.pari[I.IIWPLAN] == 2          # strut-braced
    assert case.parg[I.IGNVTAIL] == 2.0       # Pi-tail
    assert case.pari[I.IIENGLOC] == 2         # tail-mounted engines
    assert case.parg[I.IGNENG] == 3.0
    assert case.parg[I.IGFBLIF] == 0.4        # the fan ingests the wake
    assert case.parg[I.IGFBLIW] == 0.0


def test_the_bli_credit_is_actually_nonzero(flown):
    """The whole point of this case: on the 737 both of these are zero at
    every point, so the ingestion path was dead code until now."""
    from tasopt_py.engine.tfcalc import _ingestion

    case, m = flown.case, flown.case.missions[0]
    Phi, K = _ingestion(case.parg, m.para.column(I.IPCRUISE1),
                        m.pare.column(I.IPCRUISE1), case.parg[I.IGNENG])
    assert Phi > 1.0e3 and K > 1.0e3


def test_the_report_matches_except_for_one_known_line(flown):
    got = report(flown.case, flown).split("\n")
    if got and got[-1] == "":
        got.pop()
    want = REF.read_text().split("\n")
    if want and want[-1] == "":
        want.pop()
    assert len(got) == len(want) == 2347

    differed = [i + 1 for i in range(len(want)) if got[i] != want[i]]
    assert len(differed) == 1, [
        (n, got[n - 1], want[n - 1]) for n in differed[:5]]
    # §52: Kinl at climb 1, 24.928 against 24.927.
    assert got[differed[0] - 1].strip().startswith("B1: Kinl")


def test_the_one_difference_is_not_a_rounding_boundary(flown):
    """Said explicitly, because the .oute case *is* one and the two should
    not be confused. Kinl prints at five significant figures, so the boundary
    is 24.9275; the port is 4.4e-4 above it, which is 1.8e-5 relative -- far
    too big to be a tie-break."""
    from tasopt_py.engine.tfcalc import _ingestion

    case, m = flown.case, flown.case.missions[0]
    _, K = _ingestion(case.parg, m.para.column(I.IPCLIMB1),
                      m.pare.column(I.IPCLIMB1), case.parg[I.IGNENG])
    kW = K / 1000.0
    assert kW == pytest.approx(24.9279, abs=1e-4)
    assert abs(kW - 24.9275) / 24.9275 > 1e-5


def test_the_report_honours_the_case_files_own_bl_flag(flown):
    """sd81 sets Lfblwrite = F where the 737 sets T. Taking that from a
    default rather than the case made the D8's report 54 lines too long."""
    assert flown.case.settings.Lfblwrite is False
    text = report(flown.case, flown)
    assert "Fuselage BL+Wake development" not in text
    # ...but the solve still ran; nothing else would have KAfTE.
    assert flown.fuselage_bl is not None
    assert flown.case.missions[0].para[I.IAKAFTE, I.IPCRUISE1] > 0.0
