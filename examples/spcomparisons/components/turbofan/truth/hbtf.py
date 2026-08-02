"""Two-spool separate-flow turbofan cycle, parameterized.

The ``HBTF`` cycle group is pycycle's own high-bypass turbofan example
(``example_cycles/high_bypass_turbofan.py``), reproduced here nearly verbatim
so the truth harness is self-contained. That example is pycycle's validated
CFM56-class regression case (``benchmark_hbtf.py`` pins DESIGN W=344.303
lbm/s, OPR=30.094, FAR=0.0249199, TSFC=0.63072 at M0.8/35kft/5900 lbf), which
is exactly the provenance we want for an anchor. Do not "improve" the physics
here -- fidelity changes belong in the SP rows, and this file's job is to
stay pinned to pycycle.

What IS different from the example:

* ``AnchorCycle`` (the MPCycle) takes an ``EngineSpec`` (see ``engines.py``)
  instead of hard-coding one engine, so the same architecture serves both the
  CFM56-class and the GEnx-class anchors.
* Off-design points are built from the spec: T4-throttled points for rated
  conditions (takeoff, rolling takeoff, top of climb) and a percent-thrust
  point for part-power cruise, hung off a named full-power point exactly the
  way the example hangs OD_part_pwr off OD_full_pwr.
* The Newton solver prints one line per iteration (iprint=1) instead of the
  example's full table, because run_anchors.py drives many warm-started
  re-runs during the walk out to static sea level.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import openmdao.api as om
import pycycle.api as pyc


@dataclass(frozen=True)
class ODPoint:
    """One off-design anchor point."""
    name: str
    MN: float
    alt_ft: float
    #: 'T4' throttles to a turbine rating; 'PC' to a fraction of another
    #: point's net thrust.
    mode: str
    #: T4 target, degR ('T4' mode).
    T4_R: float | None = None
    #: Thrust fraction and the full-power point it references ('PC' mode).
    PC: float | None = None
    pc_of: str | None = None


@dataclass(frozen=True)
class EngineSpec:
    """Everything that distinguishes one anchor engine from another."""
    name: str
    # -- design flight condition and sizing targets --
    MN_des: float
    alt_des_ft: float
    Fn_des_lbf: float
    T4_des_R: float
    # -- cycle design variables --
    BPR: float
    FPR: float
    LPC_PR: float
    HPC_PR: float
    eff_fan: float
    eff_lpc: float
    eff_hpc: float
    eff_hpt: float
    eff_lpt: float
    LP_Nmech: float
    HP_Nmech: float
    # -- off-design anchor points (exactly four for the milestone) --
    od_points: tuple = ()
    # -- overrides of the CFM56-class cycle parameters below --
    params: dict = field(default_factory=dict)
    station_MNs: dict = field(default_factory=dict)
    # -- initial guesses, scaled to the engine's size --
    W_guess: float = 300.0
    # -- public reference figures: name -> (value, units, source, approx) --
    refs: dict = field(default_factory=dict)
    notes: str = ""


#: Station Mach numbers at the design point, from the pycycle example.
#: These size the flowpath areas; they are geometry choices, not physics.
STATION_MNS = {
    "inlet.MN": 0.751, "fan.MN": 0.4578, "splitter.BPR": None,
    "splitter.MN1": 0.3104, "splitter.MN2": 0.4518, "duct4.MN": 0.3121,
    "lpc.MN": 0.3059, "duct6.MN": 0.3563, "hpc.MN": 0.2442,
    "bld3.MN": 0.3000, "burner.MN": 0.1025, "hpt.MN": 0.3650,
    "duct11.MN": 0.3063, "lpt.MN": 0.4127, "duct13.MN": 0.4463,
    "byp_bld.MN": 0.4489, "duct15.MN": 0.4589,
}

#: Cycle parameters (pressure losses, bleeds, nozzle coefficients, power
#: extraction), from the pycycle example's CFM56-class values. A spec can
#: override any of them via ``EngineSpec.params``.
CYCLE_PARAMS = {
    "inlet.ram_recovery": 0.9990,
    "duct4.dPqP": 0.0048,
    "duct6.dPqP": 0.0101,
    "burner.dPqP": 0.0540,
    "duct11.dPqP": 0.0051,
    "duct13.dPqP": 0.0107,
    "duct15.dPqP": 0.0149,
    "core_nozz.Cv": 0.9933,
    "byp_bld.bypBld:frac_W": 0.005,
    "byp_nozz.Cv": 0.9939,
    "hpc.cool1:frac_W": 0.050708,
    "hpc.cool1:frac_P": 0.5,
    "hpc.cool1:frac_work": 0.5,
    "hpc.cool2:frac_W": 0.020274,
    "hpc.cool2:frac_P": 0.55,
    "hpc.cool2:frac_work": 0.5,
    "bld3.cool3:frac_W": 0.067214,
    "bld3.cool4:frac_W": 0.101256,
    "hpc.cust:frac_P": 0.5,
    "hpc.cust:frac_work": 0.5,
    "hpc.cust:frac_W": 0.0445,
    "hpt.cool3:frac_P": 1.0,
    "hpt.cool4:frac_P": 0.0,
    "lpt.cool1:frac_P": 1.0,
    "lpt.cool2:frac_P": 0.0,
    ("hp_shaft.HPX", "hp"): 250.0,
}


class HBTF(pyc.Cycle):
    """The pycycle example cycle, verbatim physics."""

    def initialize(self):
        self.options.declare('throttle_mode', default='T4',
                             values=['T4', 'percent_thrust'])
        super().initialize()

    def setup(self):
        design = self.options['design']

        self.options['thermo_method'] = 'CEA'
        self.options['thermo_data'] = pyc.species_data.janaf
        FUEL_TYPE = 'Jet-A(g)'

        self.add_subsystem('fc', pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())
        self.add_subsystem('fan', pyc.Compressor(map_data=pyc.FanMap,
                           bleed_names=[], map_extrap=True),
                           promotes_inputs=[('Nmech', 'LP_Nmech')])
        self.add_subsystem('splitter', pyc.Splitter())
        self.add_subsystem('duct4', pyc.Duct())
        self.add_subsystem('lpc', pyc.Compressor(map_data=pyc.LPCMap,
                           map_extrap=True),
                           promotes_inputs=[('Nmech', 'LP_Nmech')])
        self.add_subsystem('duct6', pyc.Duct())
        self.add_subsystem('hpc', pyc.Compressor(map_data=pyc.HPCMap,
                           bleed_names=['cool1', 'cool2', 'cust'],
                           map_extrap=True),
                           promotes_inputs=[('Nmech', 'HP_Nmech')])
        self.add_subsystem('bld3', pyc.BleedOut(bleed_names=['cool3', 'cool4']))
        self.add_subsystem('burner', pyc.Combustor(fuel_type=FUEL_TYPE))
        self.add_subsystem('hpt', pyc.Turbine(map_data=pyc.HPTMap,
                           bleed_names=['cool3', 'cool4'], map_extrap=True),
                           promotes_inputs=[('Nmech', 'HP_Nmech')])
        self.add_subsystem('duct11', pyc.Duct())
        self.add_subsystem('lpt', pyc.Turbine(map_data=pyc.LPTMap,
                           bleed_names=['cool1', 'cool2'], map_extrap=True),
                           promotes_inputs=[('Nmech', 'LP_Nmech')])
        self.add_subsystem('duct13', pyc.Duct())
        self.add_subsystem('core_nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv'))
        self.add_subsystem('byp_bld', pyc.BleedOut(bleed_names=['bypBld']))
        self.add_subsystem('duct15', pyc.Duct())
        self.add_subsystem('byp_nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv'))
        self.add_subsystem('lp_shaft', pyc.Shaft(num_ports=3),
                           promotes_inputs=[('Nmech', 'LP_Nmech')])
        self.add_subsystem('hp_shaft', pyc.Shaft(num_ports=2),
                           promotes_inputs=[('Nmech', 'HP_Nmech')])
        self.add_subsystem('perf', pyc.Performance(num_nozzles=2, num_burners=1))

        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('hpc.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('core_nozz.Fg', 'perf.Fg_0')
        self.connect('byp_nozz.Fg', 'perf.Fg_1')

        self.connect('fan.trq', 'lp_shaft.trq_0')
        self.connect('lpc.trq', 'lp_shaft.trq_1')
        self.connect('lpt.trq', 'lp_shaft.trq_2')
        self.connect('hpc.trq', 'hp_shaft.trq_0')
        self.connect('hpt.trq', 'hp_shaft.trq_1')
        self.connect('fc.Fl_O:stat:P', 'core_nozz.Ps_exhaust')
        self.connect('fc.Fl_O:stat:P', 'byp_nozz.Ps_exhaust')

        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:
            balance.add_balance('W', units='lbm/s', eq_units='lbf')
            self.connect('balance.W', 'fc.W')
            self.connect('perf.Fn', 'balance.lhs:W')
            self.promotes('balance', inputs=[('rhs:W', 'Fn_DES')])

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')
            self.promotes('balance', inputs=[('rhs:FAR', 'T4_MAX')])

            balance.add_balance('lpt_PR', val=1.5, lower=1.001, upper=8,
                                eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.lpt_PR', 'lpt.PR')
            self.connect('lp_shaft.pwr_in_real', 'balance.lhs:lpt_PR')
            self.connect('lp_shaft.pwr_out_real', 'balance.rhs:lpt_PR')

            balance.add_balance('hpt_PR', val=1.5, lower=1.001, upper=8,
                                eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.hpt_PR', 'hpt.PR')
            self.connect('hp_shaft.pwr_in_real', 'balance.lhs:hpt_PR')
            self.connect('hp_shaft.pwr_out_real', 'balance.rhs:hpt_PR')
        else:
            if self.options['throttle_mode'] == 'T4':
                balance.add_balance('FAR', val=0.017, lower=1e-4,
                                    eq_units='degR')
                self.connect('balance.FAR', 'burner.Fl_I:FAR')
                self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')
                self.promotes('balance', inputs=[('rhs:FAR', 'T4_MAX')])
            elif self.options['throttle_mode'] == 'percent_thrust':
                balance.add_balance('FAR', val=0.017, lower=1e-4,
                                    eq_units='lbf', use_mult=True)
                self.connect('balance.FAR', 'burner.Fl_I:FAR')
                self.connect('perf.Fn', 'balance.rhs:FAR')
                self.promotes('balance', inputs=[('mult:FAR', 'PC'),
                                                 ('lhs:FAR', 'Fn_max')])

            # Upper bound must clear a GEnx-class SLS mass flow (~2,600
            # lbm/s); at 2500 the TO point "converged" with W pinned on the
            # bound and the core-area balance quietly violated.
            balance.add_balance('W', units='lbm/s', lower=10., upper=4000.,
                                eq_units='inch**2')
            self.connect('balance.W', 'fc.W')
            self.connect('core_nozz.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('BPR', lower=2., upper=15., eq_units='inch**2')
            self.connect('balance.BPR', 'splitter.BPR')
            self.connect('byp_nozz.Throat:stat:area', 'balance.lhs:BPR')

            balance.add_balance('lp_Nmech', val=1.5, units='rpm', lower=500.,
                                eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.lp_Nmech', 'LP_Nmech')
            self.connect('lp_shaft.pwr_in_real', 'balance.lhs:lp_Nmech')
            self.connect('lp_shaft.pwr_out_real', 'balance.rhs:lp_Nmech')

            balance.add_balance('hp_Nmech', val=1.5, units='rpm', lower=500.,
                                eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.hp_Nmech', 'HP_Nmech')
            self.connect('hp_shaft.pwr_in_real', 'balance.lhs:hp_Nmech')
            self.connect('hp_shaft.pwr_out_real', 'balance.rhs:hp_Nmech')

        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I')
        self.pyc_connect_flow('inlet.Fl_O', 'fan.Fl_I')
        self.pyc_connect_flow('fan.Fl_O', 'splitter.Fl_I')
        self.pyc_connect_flow('splitter.Fl_O1', 'duct4.Fl_I')
        self.pyc_connect_flow('duct4.Fl_O', 'lpc.Fl_I')
        self.pyc_connect_flow('lpc.Fl_O', 'duct6.Fl_I')
        self.pyc_connect_flow('duct6.Fl_O', 'hpc.Fl_I')
        self.pyc_connect_flow('hpc.Fl_O', 'bld3.Fl_I')
        self.pyc_connect_flow('bld3.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'hpt.Fl_I')
        self.pyc_connect_flow('hpt.Fl_O', 'duct11.Fl_I')
        self.pyc_connect_flow('duct11.Fl_O', 'lpt.Fl_I')
        self.pyc_connect_flow('lpt.Fl_O', 'duct13.Fl_I')
        self.pyc_connect_flow('duct13.Fl_O', 'core_nozz.Fl_I')
        self.pyc_connect_flow('splitter.Fl_O2', 'byp_bld.Fl_I')
        self.pyc_connect_flow('byp_bld.Fl_O', 'duct15.Fl_I')
        self.pyc_connect_flow('duct15.Fl_O', 'byp_nozz.Fl_I')

        self.pyc_connect_flow('hpc.cool1', 'lpt.cool1', connect_stat=False)
        self.pyc_connect_flow('hpc.cool2', 'lpt.cool2', connect_stat=False)
        self.pyc_connect_flow('bld3.cool3', 'hpt.cool3', connect_stat=False)
        self.pyc_connect_flow('bld3.cool4', 'hpt.cool4', connect_stat=False)

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-8
        newton.options['rtol'] = 1e-99
        newton.options['iprint'] = 1
        newton.options['maxiter'] = 50
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 1000
        newton.options['reraise_child_analysiserror'] = False
        ls = newton.linesearch = om.ArmijoGoldsteinLS()
        ls.options['maxiter'] = 3
        ls.options['rho'] = 0.75

        self.linear_solver = om.DirectSolver()

        super().setup()


class AnchorCycle(pyc.MPCycle):
    """DESIGN + the spec's off-design points, wired the example's way."""

    def initialize(self):
        self.options.declare('spec', types=EngineSpec)
        super().initialize()

    def setup(self):
        spec = self.options['spec']

        self.pyc_add_pnt('DESIGN', HBTF(thermo_method='CEA'))

        mns = dict(STATION_MNS)
        mns.update(spec.station_MNs)
        for key, val in mns.items():
            if val is None:
                continue
            self.set_input_defaults(f'DESIGN.{key}', val)
        self.set_input_defaults('DESIGN.splitter.BPR', spec.BPR)
        self.set_input_defaults('DESIGN.LP_Nmech', spec.LP_Nmech, units='rpm')
        self.set_input_defaults('DESIGN.HP_Nmech', spec.HP_Nmech, units='rpm')

        params = dict(CYCLE_PARAMS)
        params.update(spec.params)
        for key, val in params.items():
            if isinstance(key, tuple):
                self.pyc_add_cycle_param(key[0], val, units=key[1])
            else:
                self.pyc_add_cycle_param(key, val)

        for od in spec.od_points:
            self.pyc_add_pnt(od.name, HBTF(
                design=False, thermo_method='CEA',
                throttle_mode=('percent_thrust' if od.mode == 'PC' else 'T4')))
            # Started AT the design condition; run_anchors.py walks them out
            # to their target conditions on warm restarts.
            self.set_input_defaults(f'{od.name}.fc.MN', spec.MN_des)
            self.set_input_defaults(f'{od.name}.fc.alt', spec.alt_des_ft,
                                    units='ft')
            self.set_input_defaults(f'{od.name}.fc.dTs', 0., units='degR')
            if od.mode == 'PC':
                self.connect(f'{od.pc_of}.perf.Fn', f'{od.name}.Fn_max')

        self.pyc_use_default_des_od_conns()
        self.pyc_connect_des_od('core_nozz.Throat:stat:area', 'balance.rhs:W')
        self.pyc_connect_des_od('byp_nozz.Throat:stat:area', 'balance.rhs:BPR')

        super().setup()


def build_problem(spec: EngineSpec) -> om.Problem:
    """A ready-to-run problem at the design point, off-design points parked
    at the design condition with the example's initial guesses."""
    prob = om.Problem()
    prob.model = AnchorCycle(spec=spec)
    prob.setup()

    prob.set_val('DESIGN.fan.PR', spec.FPR)
    prob.set_val('DESIGN.fan.eff', spec.eff_fan)
    prob.set_val('DESIGN.lpc.PR', spec.LPC_PR)
    prob.set_val('DESIGN.lpc.eff', spec.eff_lpc)
    prob.set_val('DESIGN.hpc.PR', spec.HPC_PR)
    prob.set_val('DESIGN.hpc.eff', spec.eff_hpc)
    prob.set_val('DESIGN.hpt.eff', spec.eff_hpt)
    prob.set_val('DESIGN.lpt.eff', spec.eff_lpt)
    prob.set_val('DESIGN.fc.alt', spec.alt_des_ft, units='ft')
    prob.set_val('DESIGN.fc.MN', spec.MN_des)
    prob.set_val('DESIGN.T4_MAX', spec.T4_des_R, units='degR')
    prob.set_val('DESIGN.Fn_DES', spec.Fn_des_lbf, units='lbf')

    prob['DESIGN.balance.FAR'] = 0.025
    prob['DESIGN.balance.W'] = spec.W_guess
    prob['DESIGN.balance.lpt_PR'] = 4.0
    prob['DESIGN.balance.hpt_PR'] = 3.0
    prob['DESIGN.fc.balance.Pt'] = 5.2
    prob['DESIGN.fc.balance.Tt'] = 440.0

    for od in spec.od_points:
        if od.mode == 'T4':
            prob.set_val(f'{od.name}.T4_MAX', spec.T4_des_R, units='degR')
        else:
            prob.set_val(f'{od.name}.PC', 1.0)
        prob[f'{od.name}.balance.FAR'] = 0.02467
        prob[f'{od.name}.balance.W'] = spec.W_guess
        prob[f'{od.name}.balance.BPR'] = spec.BPR
        prob[f'{od.name}.balance.lp_Nmech'] = spec.LP_Nmech
        prob[f'{od.name}.balance.hp_Nmech'] = spec.HP_Nmech
        prob[f'{od.name}.hpt.PR'] = 3.0
        prob[f'{od.name}.lpt.PR'] = 4.0
        prob[f'{od.name}.fan.map.RlineMap'] = 2.0
        prob[f'{od.name}.lpc.map.RlineMap'] = 2.0
        prob[f'{od.name}.hpc.map.RlineMap'] = 2.0

    return prob
