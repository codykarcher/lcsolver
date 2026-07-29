"""Optimiser restart files (getsave.f) against real Fortran output.

Both halves of the format are checked against files the shipped program
actually wrote, running the 737 case with ``Lopt = T`` and ``Lsavwrite = T``:

* ``tests/data/sav_header.txt`` is the ``.sav`` it produced -- two header
  lines and nothing else;
* ``tests/data/sav_body.txt`` is the ``fort.8`` it left in the working
  directory at the same time, holding every simplex it should have written
  into the ``.sav``.

That split is not a convenience. ``tasopt.f`` opens the save file, writes the
header, and closes it again *before* the optimisation runs, so every later
``wrtsave1`` writes to a closed unit 8 and gfortran redirects it to
``fort.8``. The shipped program cannot produce a readable restart file; see
``DISCREPANCIES.md`` §42.

This port writes both halves to the same file, which is what the routines
were plainly meant to do. The formats themselves are reproduced exactly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tasopt_py.model import indices as I
from tasopt_py.savefile import (SaveMismatch, _g22, _i4, _iovar_indices,
                                read_sav, write_sav)

DATA = Path(__file__).parent / "data"
HEADER = DATA / "sav_header.txt"
BODY = DATA / "sav_body.txt"

#: The 737's active optimisation variables, in cparo order.
VARS = ["CL", "AR", "sweep", "hboxo", "hboxs", "lambdat", "rcls", "rclt",
        "alt", "Tt4CR", "Tt4TO"]

pytestmark = pytest.mark.skipif(not HEADER.exists() or not BODY.exists(),
                                reason="save-file dumps not present")


def test_header_matches_the_fortran_byte_for_byte():
    ref = HEADER.read_text().splitlines()
    assert len(ref) == 2, "the shipped program writes only the header"
    assert _i4([len(VARS), 0, 0]) == ref[0]
    assert _i4(_iovar_indices(VARS)) == ref[1]


def test_the_variable_list_is_stored_as_cparo_indices():
    """So a file records *which* variables were optimised, not just how
    many. CL is 1, AR 2, sweep 3, ... lambdas (6) is absent here."""
    assert _iovar_indices(VARS) == [1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 14]
    assert I.CPARO[5] == "lambdas"          # the one that is skipped


def test_body_rows_match_the_fortran_byte_for_byte():
    """Every simplex row the Fortran wrote, reformatted from its own
    values -- so the g22.14 edit descriptor is right for all of them."""
    rows = [ln for ln in BODY.read_text().splitlines()
            if ln.strip() and len(ln.split()) == len(VARS)]
    assert len(rows) >= len(VARS) + 1
    for ln in rows:
        assert _g22([float(x) for x in ln.split()]) == ln


def test_round_trip_through_a_file(tmp_path):
    simplex = [[float(i + j) for i in range(len(VARS))]
               for j in range(len(VARS) + 1)]
    p = tmp_path / "case.sav"
    write_sav(p, VARS, "", "", {(1, 1): simplex})

    got = read_sav(p, VARS, "", "", ni=1, nj=1)
    assert got.variables == VARS
    assert got.ni == 1 and got.nj == 1
    for a, b in zip(got.simplexes[(1, 1)], simplex):
        assert a == pytest.approx(b)


def test_round_trip_over_a_parameter_grid(tmp_path):
    simplexes = {(i, j): [[float(i * 100 + j * 10 + k) for k in range(2)]
                          for _ in range(3)]
                 for i in (1, 2, 3) for j in (1, 2)}
    p = tmp_path / "grid.sav"
    write_sav(p, ["CL", "AR"], "AR", "Mach", simplexes)

    got = read_sav(p, ["CL", "AR"], "AR", "Mach", ni=3, nj=2)
    assert got.ni == 3 and got.nj == 2
    assert set(got.simplexes) == set(simplexes)


def test_a_file_for_a_different_variable_set_is_refused(tmp_path):
    p = tmp_path / "case.sav"
    write_sav(p, ["CL", "AR"], "", "", {(1, 1): [[1.0, 2.0]] * 3})
    with pytest.raises(SaveMismatch, match="variable selection"):
        read_sav(p, ["CL", "sweep"], "", "")


def test_a_file_with_a_different_variable_count_is_refused(tmp_path):
    p = tmp_path / "case.sav"
    write_sav(p, ["CL", "AR"], "", "", {(1, 1): [[1.0, 2.0]] * 3})
    with pytest.raises(SaveMismatch, match="number of optimisation"):
        read_sav(p, ["CL"], "", "")


def test_a_file_for_a_different_sweep_is_refused(tmp_path):
    p = tmp_path / "case.sav"
    write_sav(p, ["CL"], "AR", "", {(1, 1): [[1.0]] * 2})
    with pytest.raises(SaveMismatch, match="i parameter"):
        read_sav(p, ["CL"], "Mach", "")


def test_a_grid_size_mismatch_is_refused(tmp_path):
    p = tmp_path / "case.sav"
    write_sav(p, ["CL"], "", "", {(1, 1): [[1.0]] * 2})
    with pytest.raises(SaveMismatch, match="i number mismatch"):
        read_sav(p, ["CL"], "", "", ni=5, nj=1)


def test_a_truncated_file_is_refused(tmp_path):
    p = tmp_path / "short.sav"
    p.write_text("   2   0   0\n")
    with pytest.raises(SaveMismatch, match="end of save file"):
        read_sav(p, ["CL", "AR"], "", "")


def test_the_header_only_file_the_shipped_program_writes(tmp_path):
    """Reading the real ``.sav`` back gives no grid points at all -- which is
    what the ``fort.8`` bug costs. It reads cleanly if the caller does not ask
    for a grid size, and fails the check if it does."""
    got = read_sav(HEADER, VARS, "", "")
    assert got.simplexes == {}
    with pytest.raises(SaveMismatch, match="i number mismatch"):
        read_sav(HEADER, VARS, "", "", ni=1, nj=1)


def test_a_wrong_sized_simplex_is_refused(tmp_path):
    with pytest.raises(ValueError, match="expected 3"):
        write_sav(tmp_path / "x.sav", ["CL", "AR"], "", "",
                  {(1, 1): [[1.0, 2.0], [3.0, 4.0]]})
