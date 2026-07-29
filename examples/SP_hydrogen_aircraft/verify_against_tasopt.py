"""Verify the SP optimum against the models it was derived from.

The derivation chain is: SP constraints <- TASOPT v3 Python port <- TASOPT.jl,
with the port verified against TASOPT.jl to machine precision *at the fit
points* (see ../tasopt/DISCREPANCIES.md). This script closes the remaining
gap: it solves the SP, then feeds the SOLVED DESIGN back through the port's
high-fidelity routines and measures what each reformulation costs at the
optimum -- which is where the model is actually used, and not where it was
fitted.

The fuel-cell chain has additionally been run through TASOPT.jl itself
(HT_PEMFC_voltage / PEMsize in Julia) and agrees with the port to every
printed digit, so "port" and "TASOPT.jl" are interchangeable below.

Run: ``python -m examples.SP_hydrogen_aircraft.verify_against_tasopt``
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "examples" / "convexengineering",
           _ROOT / "examples" / "tasopt"):
    sys.path.insert(0, str(_p))

#: The E175-class mission of the README's second case study.
MISSION = dict(W_pay=8.63e4, R_req=3.7e6, R_fuse=1.55, l_cabin=19.5)
G = 9.81


def run(mission=None, tee=True):
    warnings.filterwarnings("ignore")
    import pyomo.environ as pyo

    from edi.structure.structureDetector import structure_detector
    from edi.solvers.ipopt.slcp_bridge import solve_sia
    from edi.solvers.ipopt.sia import SIAOptions
    from edi.units.unitCorrector import unit_corrector
    from examples.SP_hydrogen_aircraft.model import build

    fm = build(**(mission or MISSION))
    unit_corrector(fm)
    st = structure_detector(fm)
    res = solve_sia(st, options=SIAOptions())
    if not res.converged:
        raise RuntimeError(res.status)
    for v, val in zip(st['variables'], res.x):
        v.set_value(float(val))
    V = {v.name: pyo.value(v) for v in fm.component_data_objects(pyo.Var)}

    rows = []

    def cmp(name, sp, hf, cls):
        rows.append((name, sp, hf, (sp - hf) / hf * 100, cls))

    # --- fuel cell: exact 1-D model at the solved current density ----------
    from tasopt_py.engine_v3.fuelcell_1d import (PEMInputs, ht_pemfc_voltage,
                                                 pem_size)
    u = PEMInputs(j=V["FC_j[0]"], T=453.15, p_A=3e5, p_C=3e5, x_H2O_A=0.1,
                  x_H2O_C=0.1, lam_H2=3.0, lam_O2=3.0, t_M=100e-6,
                  t_A=250e-6, t_C=250e-6, kind="HT-PEMFC")
    cmp("cell voltage [V]", V["FC_V_cell[0]"], ht_pemfc_voltage(u),
        "monomial fit, at its cap")
    n_hf, A_hf, Q_hf = pem_size(V["FC_P_e[0]"], 800.0, u)
    cmp("stack cells", V["FC_n_cells"], n_hf, "follows the fit")
    cmp("cell area [m2]", V["FC_A_cell"], A_hf, "follows the fit")
    cmp("stack heat [MW]", V["FC_Q_fc[0]"] / 1e6, Q_hf / 1e6,
        "follows the fit")

    # --- tank: the port's inner-vessel sizing at the solved load -----------
    from tasopt_py.cryo.tank import FuselageTank, size_inner_tank
    from tasopt_py.cryo.geometry import CrossSection
    R_fuse = (mission or MISSION)["R_fuse"]
    xs = CrossSection(radius=R_fuse, bubble_lower_downward_shift=0.0,
                      bubble_center_y_offset=0.0, n_webs=0)
    tk = FuselageTank(
        Wfuelintank=V["Tank_W_fuel"], rhofuel=70.0, rhofuelgas=1.3,
        ullage_frac=0.05, Tfuel=20.0, pvent=1.3e5, clearance_fuse=0.0,
        ARtank=2.0, ew=0.9, ftankadd=0.0, theta_inner=1.0,
        inner_material="Al-2219-T87", outer_material="Al-2219-T87",
        theta_outer=[1.0], t_insul=[V["Tank_t_insul"]],
        material_insul=["polyurethane32"])
    it = size_inner_tank(R_fuse, xs, tk)
    cmp("tank dry mass [kg]", V["Tank_W_tank"] / G, it.Wtank / G,
        "lumped insulation, AR-fixed heads")
    cmp("tank length [m]", V["Tank_l_tank"], it.l_tank, "same")

    # --- ducted fan: the port's full gas path at matched thrust and u_j ----
    from tasopt_py.atmosphere import atmos
    from tasopt_py.engine_v3.ducted_fan_cycle import ducted_fan_size
    a11 = atmos(11.0)
    F_per, uj = V["PT_F_net[0]"] / 2.0, V["PT_u_j[0]"]
    lo, hi = 1.05, 2.2
    for _ in range(50):
        pif = 0.5 * (lo + hi)
        r = ducted_fan_size(G, 0.78, a11.T, a11.p, a11.a, 0.6, F_per,
                            0, 0, 0, pif, 1.0, 1.0, 0.9)
        lo, hi = (pif, hi) if r.u[8] < uj else (lo, pif)
    cmp("fan shaft power [MW]", V["PT_P_shaft[0]"] / 1e6, 2 * r.Pfan / 1e6,
        "ideal disc vs epf=0.9 gas path")
    cmp("fan mass flow [kg/s]", V["PT_mdot_a[0]"], 2 * r.mfan,
        "momentum closure")
    cmp("fan diameter [m]", V["PT_D_fan"],
        2 * math.sqrt(r.A[2] / math.pi), "disc area vs M2=0.6 fan face")

    # --- motor: the port's PMSM sizing, direct drive and geared ------------
    from tasopt_py.propsys.motor import MotorDesign, size_motor
    P_mot = V["PT_P_shaft_max"] / 2.0
    rpm_dd = 300.0 / (V["PT_D_fan"] / 2.0) * 60.0 / (2 * math.pi)
    m_dd = size_motor(MotorDesign(), rpm_dd, P_mot).mass
    cmp("motor mass [kg]", V["PT_W_motor"] / 2.0 / G, m_dd,
        "10 kW/kg flat vs PMSM, DIRECT drive")
    # Find what speed the SP's 10 kW/kg implies -- the port answers with a
    # limit instead: sweeping rpm upward, the SHAFT binds (the P*Omega^2
    # group from the motor verification) before the mass ever gets there.
    rpm, m_hs, best = rpm_dd, m_dd, (rpm_dd, m_dd)
    shaft_bound = None
    while P_mot / m_hs < 10e3 and rpm < 6e4:
        rpm *= 1.1
        try:
            m_hs = size_motor(MotorDesign(), rpm, P_mot).mass
            best = (rpm, m_hs)
        except ValueError:
            shaft_bound = rpm
            rpm, m_hs = best
            break

    if tee:
        print("=" * 80)
        print(" SP optimum vs the high-fidelity models it was derived from")
        print(" (port == TASOPT.jl: fuel-cell chain re-run in Julia, "
              "identical to every digit)")
        print("=" * 80)
        print(f"{'quantity':22s} {'SP':>10s} {'TASOPT':>10s} {'diff':>8s}"
              f"  what the difference is")
        for nm, sp, hf, d, cls in rows:
            print(f"{nm:22s} {sp:10.4g} {hf:10.4g} {d:+7.1f}%  {cls}")
        print(f"\n  fan comparison at matched u_j: pif = {pif:.3f}")
        if shaft_bound is not None:
            print(f"  the SP's 10 kW/kg per-fan motor is INFEASIBLE per the "
                  f"port's PMSM limits:")
            print(f"  sweeping speed, the shaft binds near "
                  f"{shaft_bound:,.0f} rpm at {P_mot / m_hs / 1e3:.1f} kW/kg "
                  f"max ({m_hs:,.0f} kg);")
            print(f"  10 kW/kg therefore implies SEVERAL smaller motors per "
                  f"fan, which the SP does not model.")
        else:
            print(f"  the SP's 10 kW/kg motor is a PMSM at ~{rpm:,.0f} rpm "
                  f"({rpm / rpm_dd:.1f}:1 gearbox it does not model).")
        print(f"  direct drive at {rpm_dd:,.0f} rpm would weigh "
              f"{m_dd:,.0f} kg, not {V['PT_W_motor'] / 2 / G:,.0f} kg.")
    return rows


if __name__ == "__main__":
    run()
