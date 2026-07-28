"""The parameter arrays and the generated index constants.

The constants are generated from ``index.inc`` by ``tools/gen_indices.py``;
these tests pin the values that the rest of the port depends on, so a bad
regeneration is caught rather than silently shifting every array position.
"""
from __future__ import annotations

import pytest

from tasopt_py.model import Aircraft, ParamArray, ParamMatrix
from tasopt_py.model import indices as I


def test_array_totals_match_index_inc():
    assert (I.IITOTAL, I.IMTOTAL, I.IGTOTAL) == (10, 32, 254)
    assert (I.IATOTAL, I.IETOTAL) == (61, 195)
    assert (I.IPTOTAL, I.ISTOTAL, I.IOTOTAL) == (17, 16, 15)


def test_cooling_row_count_is_derived_not_declared():
    """ncrowx is ieTmet1 - ieepsc1, not a literal -- check the arithmetic."""
    assert I.NCROWX == I.IETMET1 - I.IEEPSC1
    assert I.NCROWX == 4


def test_mission_points_are_ordered():
    """Static, rotate, takeoff, cutback, 5 climb, 2 cruise, 5 descent, test.

    Note the cruise gets only two points against five each for climb and
    descent, and that iptest is a 17th slot outside the mission proper --
    ipdescentn is 16, not iptotal.
    """
    assert I.IPSTATIC < I.IPROTATE < I.IPTAKEOFF < I.IPCUTBACK
    assert I.IPCLIMB1 < I.IPCLIMBN < I.IPCRUISE1
    assert I.IPCLIMBN - I.IPCLIMB1 == 4
    assert I.IPCRUISE1 < I.IPCRUISEN < I.IPDESCENT1
    assert I.IPCRUISEN - I.IPCRUISE1 == 1
    assert I.IPDESCENTN - I.IPDESCENT1 == 4
    assert I.IPDESCENTN == 16
    assert I.IPTEST == I.IPTOTAL == 17


def test_every_index_is_in_range_for_its_array():
    limits = [("II", I.IITOTAL), ("IM", I.IMTOTAL), ("IG", I.IGTOTAL),
              ("IA", I.IATOTAL), ("IE", I.IETOTAL), ("IP", I.IPTOTAL)]
    for name, value in vars(I).items():
        if not name.isupper() or not isinstance(value, int):
            continue
        for prefix, total in limits:
            if name.startswith(prefix):
                assert 1 <= value <= total, f"{name} = {value} > {total}"
                break


def test_generated_names_are_unique():
    """Uppercasing the Fortran names must not alias two different slots."""
    seen = {}
    for name, value in vars(I).items():
        if name.isupper() and isinstance(value, int):
            seen.setdefault(name, value)
    assert len(seen) == len([n for n in vars(I) if n.isupper()
                             and isinstance(vars(I)[n], int)])


def test_param_array_is_one_based():
    p = ParamArray(5, "test")
    assert len(p) == 5
    p[1] = 3.0
    p[5] = 7.0
    assert (p[1], p[5]) == (3.0, 7.0)
    for bad in (0, 6, -1):
        with pytest.raises(IndexError, match="out of range"):
            p[bad]


def test_param_matrix_indexing_and_columns():
    m = ParamMatrix(4, 3, "test")
    m[2, 3] = 9.0
    assert m[2, 3] == 9.0
    assert m.shape == (4, 3)
    col = m.column(3)
    assert col[2] == 9.0
    col[1] = 1.5
    m.set_column(3, col)
    assert m[1, 3] == 1.5
    with pytest.raises(IndexError, match="point out of range"):
        m[1, 4]
    with pytest.raises(IndexError, match="index out of range"):
        m[5, 1]


def test_aircraft_arrays_are_sized_from_index_inc():
    ac = Aircraft()
    assert len(ac.pari) == I.IITOTAL
    assert len(ac.parg) == I.IGTOTAL
    assert ac.para.shape == (I.IATOTAL, I.IPTOTAL)
    assert ac.pare.shape == (I.IETOTAL, I.IPTOTAL)


def test_aircraft_copy_is_deep():
    ac = Aircraft()
    ac.parg[I.IGB] = 35.0
    ac.pare[I.IETT4, I.IPCRUISE1] = 1450.0
    other = ac.copy()
    other.parg[I.IGB] = 40.0
    other.pare[I.IETT4, I.IPCRUISE1] = 1500.0
    assert ac.parg[I.IGB] == 35.0
    assert ac.pare[I.IETT4, I.IPCRUISE1] == 1450.0
