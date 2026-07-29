"""The ``.out`` report (output.f) against the real TASOPT 737 run.

This is a **byte-for-byte diff against the reference program's own
``runs/737/737.out``**, not a comparison against numbers somebody chose to
check. Every value in the report is compared at its printed precision, in its
printed column, including the padding of the fixed-length name fields and the
zero-length records Fortran writes for ``write(lu,*)``.

The report is not a plain prefix of the reference: TASOPT writes each
mission's summary, then that mission's engine dump, then the next mission's
summary. So the comparison is done in two blocks -- everything up to the first
``Aero, Engine parameters...``, and then the second mission's summary against
the corresponding block further down.

Coverage
--------
:func:`tasopt_py.output.report` writes the summary sections -- header,
``Airframe parameters``, ``Fuselage BL+Wake development``, and per mission
``Cruise performance``, ``Takeoff performance`` and ``Mission profile
summary``. It does not write ``Aero, Engine parameters``, which is ``airwrt``
and ``engwrt`` dumping 124 lines for each of 17 mission points -- some 2100
of ``737.out``'s 4565 lines.

Five lines per mission summary depend on ``noise.f``, which is not ported:

* the three rows of the ``Noise...`` table -- the sideline, cutback and
  flyover dB values and the positions they are measured at;
* the ``TO:`` and ``CB:`` rows of the mission profile, because ``noise.f``
  re-runs ``cdsum`` and ``tfcalc`` at those two points, setting their
  flight-path angle, burner temperature and fuel flow.

Those ten lines are identified by content, skipped and counted. The other
345 must match exactly.

The report is regenerated with::

    python -m tasopt_py runs/737/737.tas --out /tmp/port.out
    diff /tmp/port.out runs/737/737.out
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tasopt_py.output import _efmt, _ffmt, _gfmt, report
from tasopt_py.run import run_case

TAS = Path("/Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.tas")
OUT = Path("/Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.out")

pytestmark = pytest.mark.skipif(
    not TAS.exists() or not OUT.exists(),
    reason="737.tas or 737.out not present")

#: Where the reference switches to the section this port does not write.
UNCOVERED = " Aero, Engine parameters..."
RULE2 = " " + "-" * 59


@pytest.fixture(scope="module")
def lines():
    r = run_case(TAS)
    got = report(r.case, r).split("\n")
    if got and got[-1] == "":
        got.pop()
    want = OUT.read_text().split("\n")
    if want and want[-1] == "":
        want.pop()
    return got, want


def _noise_dependent(want, i):
    """True if reference line ``i`` is one ``noise.f`` fills in."""
    line = want[i]
    if line.startswith(" TO:") or line.startswith(" CB:"):
        return True
    # The three rows under the noise table's column header.
    for k in range(1, 4):
        if i - k >= 0 and want[i - k].startswith("    x [m]   z [m]"):
            return True
    return False


def _compare(got, want, gi, wi, n):
    """Compare ``n`` lines, skipping the ones noise.f fills. Returns
    ``(compared, skipped)``."""
    compared = skipped = 0
    for k in range(n):
        if _noise_dependent(want, wi + k):
            skipped += 1
            continue
        assert got[gi + k] == want[wi + k], (
            f"port line {gi + k + 1} / ref line {wi + k + 1}\n"
            f"  port: {got[gi + k]!r}\n  ref : {want[wi + k]!r}")
        compared += 1
    return compared, skipped


def test_report_matches_the_reference_byte_for_byte(lines):
    got, want = lines
    # Block 1: everything down to the rule line above the engine dump.
    end = want.index(UNCOVERED) - 1
    assert end == 297, end
    c1, s1 = _compare(got, want, 0, 0, end)

    # Block 2: the second mission's summary, which the reference resumes
    # after 2100 lines of engine dump.
    start = want.index(" Fleet mission   2") - 1
    assert want[start] == " " + "=" * 61
    n = len(got) - end
    c2, s2 = _compare(got, want, end, start, n)

    # 355 report lines: 345 matched exactly, 10 skipped (five per mission).
    assert (c1 + c2, s1 + s2) == (345, 10)


def test_the_uncovered_section_is_where_we_think_it_is():
    """If a future change writes ``airwrt``/``engwrt``, this is the line the
    comparison has to be extended past."""
    want = OUT.read_text().split("\n")
    assert want[298] == UNCOVERED
    assert want.count(UNCOVERED) == 2            # once per fleet mission


# --- Fortran's edit descriptors -------------------------------------------

def test_f_editing_keeps_the_point_at_zero_decimals():
    """Fortran ``F9.0`` writes ``3510566.``; C and Python write ``3510566``."""
    assert _ffmt(3510566.4, 9, 0) == " 3510566."
    assert _ffmt(3000.0, 8, 0) == "   3000."
    assert _ffmt(1.5, 8, 3) == "   1.500"


def test_e_editing_normalises_the_mantissa_below_one():
    """Fortran writes ``0.207E+04`` where C writes ``2.067E+03``."""
    assert _efmt(2067.0, 11, 3) == "  0.207E+04"
    assert _efmt(0.0, 11, 3) == "  0.000E+00"
    assert _efmt(-2067.0, 11, 3) == " -0.207E+04"
    # A mantissa that rounds up to 1.000 carries into the exponent.
    assert _efmt(9999.9, 11, 3) == "  0.100E+05"


def test_g_editing_switches_between_f_and_e():
    """``G11.3``: F(7, 3-N) plus four blanks while 0.1 <= |v| < 1000."""
    assert _gfmt(0.0, 11, 3) == "   0.00    "        # zero is treated as N=1
    assert _gfmt(465.4, 11, 3) == "   465.    "
    assert _gfmt(2067.0, 11, 3) == "  0.207E+04"
    assert all(len(_gfmt(v, 11, 3)) == 11
               for v in (0.0, 0.5, 9.9, 465.4, 2067.0, 1.6e6))
