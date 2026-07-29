"""The ``.out`` report (output.f) against the real TASOPT 737 run.

This is a **byte-for-byte diff against the reference program's own
``runs/737/737.out``**, not a comparison against numbers somebody chose to
check. Every value in the report is compared at its printed precision, in its
printed column, including the padding of the fixed-length name fields and the
zero-length records Fortran writes for ``write(lu,*)``.

Coverage
--------
The whole file: 4565 lines, of which **4553 match byte for byte**. The twelve
that do not are the three rows of the ``Noise...`` table in each of the two
missions, and only the decibel column of those -- the observer positions on
the same lines are computed and do match. The decibels need ``tfnoise.f``, a
full ESDU/Heidmann fan-and-jet acoustic model, which is not ported.

Those six lines are identified by content, skipped and counted.

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


def _needs_tfnoise(want, i):
    """True if reference line ``i`` carries a decibel value."""
    for k in range(1, 4):
        if i - k >= 0 and want[i - k].startswith("    x [m]   z [m]"):
            return True
    return False


def test_report_matches_the_reference_byte_for_byte(lines):
    got, want = lines
    assert len(got) == len(want) == 4565

    compared = skipped = 0
    for i in range(len(want)):
        if _needs_tfnoise(want, i):
            skipped += 1
            # The observer position on the same line *is* computed.
            assert got[i][:16] == want[i][:16], f"line {i + 1} position"
            continue
        assert got[i] == want[i], (
            f"line {i + 1}\n  port: {got[i]!r}\n  ref : {want[i]!r}")
        compared += 1
    assert (compared, skipped) == (4559, 6)


def test_the_noise_table_positions_are_computed(lines):
    """noise.f's geometry needs no acoustics, so only the dB column is
    missing -- the sideline, cutback and flyover positions are all there."""
    got, want = lines
    i = want.index("    x [m]   z [m]     dB")
    for k, where in enumerate(("sideline", "cutback", "flyover"), start=1):
        assert got[i + k][:16] == want[i + k][:16], where


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
