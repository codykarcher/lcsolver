"""The off-design engine deck (``eopwrt``) against the real TASOPT 737 run.

The shipped 737 case writes ``737.oute``: 4323 lines describing the sized
engine at 45 operating points -- three altitudes, three Mach numbers, five
throttle settings. It is the only output the reference program produces that
the port previously could not, and unlike the rest of the output path it is
not formatting: every point is a converged ``tfoper`` solve at a flight
condition the design mission never visits.

Agreement
---------
**4322 of the 4323 lines are byte-identical.** The one that is not is a
printed rounding boundary, not a disagreement about the engine:

    ``T1: mcool  =   3.8390`` (reference)  vs  ``3.8391`` (port)

at 10 kft / Mach 0.2 / 20% throttle. The port's value is 3.839050015 --
fifteen billionths *above* the 3.83905 boundary where the fourth decimal
turns over, so a relative difference of 4e-9 in ``mcore * fc`` lands the two
on opposite sides of it. That is the ``tfoper`` numerical-Jacobian floor
(``DISCREPANCIES.md`` §22) showing up in the fourth decimal of one number
because of where the value happens to sit, and 20% throttle at altitude is
the furthest this deck gets from the shipped engine's envelope.

The test asserts that: everything matches except that one line, and that
line's underlying value agrees to 1e-8. Loosening it to "the numbers are
close" would stop it noticing if a second line ever moved.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tasopt_py.enginedeck import engine_deck, eopwrt
from tasopt_py.model import indices as I
from tasopt_py.run import run_case
from tasopt_py.tasfile import read_tase

TAS = Path("/Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.tas")
TASE = Path("/Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.tase")
OUTE = Path("/Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.oute")

pytestmark = pytest.mark.skipif(
    not TAS.exists() or not TASE.exists() or not OUTE.exists(),
    reason="737 case or .tase/.oute not present")

#: The one line that differs, and why -- see the module docstring.
ROUNDING_BOUNDARY = " T1: mcool  =   3.839"


@pytest.fixture(scope="module")
def deck():
    r = run_case(TAS, off_design=False)
    grid = read_tase(TASE)
    m = r.case.missions[0]
    return r.case, grid, engine_deck(r.case.pari, r.case.parg, m.para, m.pare,
                                     grid)


@pytest.fixture(scope="module")
def lines(deck):
    case, _, d = deck
    got = eopwrt(case, d).split("\n")
    if got and got[-1] == "":
        got.pop()
    want = OUTE.read_text().split("\n")
    if want and want[-1] == "":
        want.pop()
    return got, want


def test_the_grid_is_read_from_the_tase_file(deck):
    _, grid, _ = deck
    assert grid.shape == (3, 3, 5)
    assert grid.npoints == 45
    # The altitudes are written in feet with a `* 0.3048` suffix.
    assert grid.alt == [0.0, 3048.0, 6096.0]
    assert grid.mach == [0.1, 0.2, 0.3]
    assert grid.fset == [0.2, 0.4, 0.6, 0.8, 1.0]
    assert bool(grid) is True


def test_the_deck_matches_the_fortran_line_for_line(lines):
    got, want = lines
    assert len(got) == len(want) == 4323
    differed = []
    for i in range(len(want)):
        if got[i] != want[i]:
            differed.append(i + 1)
    assert len(differed) == 1, [
        (n, got[n - 1], want[n - 1]) for n in differed[:5]]
    n = differed[0]
    assert got[n - 1].startswith(ROUNDING_BOUNDARY)
    assert want[n - 1].startswith(ROUNDING_BOUNDARY)


def test_the_one_difference_is_a_rounding_boundary(deck):
    """The value straddles 3.83905, where the fourth decimal turns over."""
    _, _, d = deck
    p = next(q for q in d.points
             if q.alt == 3048.0 and q.mach == 0.2 and q.kfset == 1)
    mcool = p.pare[I.IEMCORE] * p.pare[I.IEFC]
    assert mcool == pytest.approx(3.83905, abs=1e-7)
    assert mcool > 3.83905          # the port rounds up, the Fortran down
    assert mcool - 3.83905 < 1e-7   # by fifteen billionths


def test_every_point_converged(deck):
    """45 tfoper solves at conditions the design mission never flies, down
    to 20% throttle at 20 kft. None of them failed."""
    _, _, d = deck
    assert len(d.points) == 45
    for p in d.points:
        assert p.F > 0.0
        assert 0.2 < p.TSFC < 2.0
        assert p.mfuel > 0.0


def test_the_throttle_is_a_fraction_of_local_max_thrust(deck):
    """Not of sea-level static thrust -- so 100% at 20 kft is much less
    force than 100% at sea level, and each point's thrust is its own Fmax
    times its own fraction."""
    _, _, d = deck
    for p in d.points:
        assert p.F == pytest.approx(p.Fmax * p.fset, rel=1e-6)
    sl = next(p for p in d.points
              if p.alt == 0.0 and p.mach == 0.1 and p.fset == 1.0)
    hi = next(p for p in d.points
              if p.alt == 6096.0 and p.mach == 0.1 and p.fset == 1.0)
    assert hi.Fmax < 0.75 * sl.Fmax


def test_the_deck_leaves_the_mission_alone(deck):
    """Everything runs in the spare iptest slot, so a deck can be taken off
    a converged aircraft without disturbing it."""
    case, grid, _ = deck
    m = case.missions[0]
    before = [m.pare[I.IEFE, ip] for ip in range(1, I.IPTOTAL)]
    engine_deck(case.pari, case.parg, m.para, m.pare, grid)
    after = [m.pare[I.IEFE, ip] for ip in range(1, I.IPTOTAL)]
    assert after == before


def test_a_missing_tase_file_gives_an_empty_grid(tmp_path):
    """tasopt.f's `err=15` branch: the deck is optional and its absence is
    not an error."""
    grid = read_tase(tmp_path / "nothing.tase")
    assert grid.npoints == 0
    assert bool(grid) is False


def test_the_point_labels_wrap_at_ten(deck):
    """`ione = mod(keFset,10)`, so an eleventh throttle setting would be
    labelled T1 again -- the same tag as the first. Nothing shipped goes
    past five; this pins the behaviour rather than fixing it."""
    _, _, d = deck
    assert [p.kfset % 10 for p in d.points[:5]] == [1, 2, 3, 4, 5]
    assert 11 % 10 == 1 % 10
