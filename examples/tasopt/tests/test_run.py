"""``run_case`` -- the port's equivalent of a plain ``tasopt 737`` run.

The physics is covered by ``test_wsize.py`` (the sizing loop),
``test_woper.py`` (the off-design loop) and ``test_tasfile.py`` (the reader
producing exactly the state the program hands to ``wsize``). What is left to
check is the *wiring* between them, and that is done here with stubs rather
than by sizing a 737 a fourth time -- the real run takes 25 seconds.

Run for real with::

    python -m tasopt_py /path/to/737.tas

which reproduces both of the convergence tables the shipped program prints,
ending at WTO = 174979.1500 lbf against its 174979.1499, and
PFEI = 7.849124 against the 7.8491 in ``737.out``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tasopt_py import run as R
from tasopt_py.model import indices as I

TAS = Path("/Users/codykarcher/Desktop/Tasopt2.16/runs/737/737.tas")

pytestmark = pytest.mark.skipif(not TAS.exists(), reason="737.tas not present")


class _Stub:
    """Records the calls made to it and returns something result-shaped."""

    def __init__(self, **attrs):
        self.calls = []
        self._attrs = attrs

    def __call__(self, *a, **kw):
        self.calls.append((a, kw))
        return type("R", (), dict(self._attrs))()


@pytest.fixture
def stubbed(monkeypatch):
    ws = _Stub(converged=True, iterations=18,
               mission=type("M", (), {"PFEI": 7.849124})())
    wo = _Stub(converged=True, iterations=3)
    monkeypatch.setattr(R, "wsize", ws)
    monkeypatch.setattr(R, "woper", wo)
    monkeypatch.setattr(R, "airtable", lambda p: ("table", p))
    return ws, wo


def test_sizes_the_design_mission_with_the_files_own_settings(stubbed):
    ws, _ = stubbed
    r = R.run_case(TAS)
    assert len(ws.calls) == 1
    args, kw = ws.calls[0]
    # wsize takes (pari, parg, parm, para, pare) positionally.
    assert args == r.case.design
    assert kw["iterwmax"] == r.case.settings.iterwmax == 50
    assert (kw["wrlx1"], kw["wrlx2"], kw["wrlx3"]) == (0.5, 0.9, 0.5)
    assert kw["initwgt"] == 0 and kw["initeng"] == 0
    assert kw["table"] == ("table", r.case.airfoil_file)


def test_off_design_missions_start_from_the_design_state(stubbed):
    _, wo = stubbed
    r = R.run_case(TAS)
    # The 737 has two missions, so exactly one off-design run.
    assert len(wo.calls) == 1 == len(r.off_design)
    args, kw = wo.calls[0]
    design, second = r.case.missions
    assert args[0] is r.case.pari and args[1] is r.case.parg
    assert args[2] is second.parm and args[3] is second.para
    assert args[4] is second.pare
    # ...and are handed the *design* mission's matrices as the reference.
    assert args[5] is design.para and args[6] is design.pare
    assert kw["iterfmax"] == 15
    # tasopt.f sets initeng = 1 before its woper loop: the engine state is
    # already there from the sizing.
    assert kw["initeng"] == 1


def test_off_design_can_be_skipped(stubbed):
    _, wo = stubbed
    r = R.run_case(TAS, off_design=False)
    assert wo.calls == [] and r.off_design == []


def test_reported_weights_come_from_the_design_mission(stubbed):
    r = R.run_case(TAS)
    r.case.missions[0].parm[I.IMWTO] = 4.44822e5
    r.case.missions[0].parm[I.IMWFUEL] = 4.44822e4
    assert r.WTO_lbf == pytest.approx(1.0e5)
    assert r.Wfuel_lbf == pytest.approx(1.0e4)
    assert r.PFEI == pytest.approx(7.849124)


def test_cli_usage_message(capsys):
    assert R.main([]) == 2
    assert "usage" in capsys.readouterr().err
