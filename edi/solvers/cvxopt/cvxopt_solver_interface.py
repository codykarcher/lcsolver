#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2023
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
import logging

import pyomo.environ as pyo

from pyomo.contrib.appsi.base import (
    Solver, 
    SolverConfig, 
    SolutionLoader, 
    Results, 
    TerminationCondition
)

from pyomo.core.base.block import _BlockData
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.config import ConfigValue, NonNegativeInt
from pyomo.core.base.var import ScalarVar, _GeneralVarData, IndexedVar
from pyomo.common.log import LogStream
from pyomo.common.tee import capture_output, TeeStream

# from edi.presolve.structureDetector import structure_detector
# from edi.solvers.cvxopt.LP import solve_LP
# from edi.solvers.cvxopt.QP import solve_QP
# from edi.solvers.cvxopt.GP import solve_GP

from edi.presolve.structureDetector import structure_detector
from edi.solvers.cvxopt.LP import solve_LP
from edi.solvers.cvxopt.QP import solve_QP
from edi.solvers.cvxopt.GP import solve_GP
from edi.solvers.cvxopt.SP import solve_SP

from pyomo.common.dependencies import attempt_import
cvxopt, cvxopt_available = attempt_import( "cvxopt" )
# if not cvxopt_available:
#     raise ImportError('The CVXOPT solver requires cvxopt')



logger = logging.getLogger(__name__)

# inherit from SovlerConfig
class CVXOPTConfig(SolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super(CVXOPTConfig, self).__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.declare('logfile', ConfigValue(domain=str))
        self.declare('solver_output_logger', ConfigValue())
        self.declare('log_level', ConfigValue(domain=NonNegativeInt))

        self.logfile = ''
        self.solver_output_logger = logger
        self.log_level = logging.INFO


class CVXOPTResults(Results):
    def __init__(self):
        super().__init__()
        self.wallclock_time = None
        self.messages = None


class CVXOPT(Solver):
    def __init__(self, only_child_vars: bool = False) -> None:
        super().__init__()
        # Things specific to pyomo, or things pyomo will make unified across all options
        # ex: time limit options
        # Also: streamsolver->to terminal or not, loadsolution-> load, report timing->to print timing results (in base solver config)
        # pyomo specific: symbolic sover labels (when translating to something sovler can understand, use x1,x2,x3 or based on pyomo names)
        self._config = CVXOPTConfig()

        # sovler specific options
        # ex: maxiters
        self._options = dict()
        self._symbol_map = None

        # ignore for now
        self._only_child_vars = only_child_vars

    def available(self):
        if cvxopt_available:
            return self.Availability.FullLicense
        else:
            return self.Availability.NotFound
        
    def version(self):
        #need
        verString = cvxopt.__version__
        verAr = verString.split('.')
        verNums = [int(v) for v in verAr]
        return tuple(verNums)

    @property
    def config(self):
        return self._config
    
    @property
    def options(self):
        return self._options
    
    @property
    def symbol_map(self):
        # raise NotImplementedError
        # will give fake one later
        raise NotImplementedError('symbol_map not currently supported')

    def _set_options(self):
        #map from config to scip things
        # CVXOPT does not have any of these, keeping as reference
        # if self.config.time_limit is not None:
        #     scip_model.setParam('limits/time', self.config.time_limit)

        # cvxopt.solvers.options['show_progress'] 
        #     Whether to print or not
        #     True/False (default: True)
        # cvxopt.solvers.options['maxiters']
        #     Maximum number of convex iterations
        #     positive integer (default: 100)
        # cvxopt.solvers.options['refinement']    = self.options['refinement']      
        #     number of iterative refinement steps when solving KKT equations (only SOCP I think)
        #     nonnegative integer (default: 1)
        # cvxopt.solvers.options['abstol'] 
        #     Required primal/dual gap tolerance to return as optimal
        #     scalar (default: 1e-7)
        # cvxopt.solvers.options['reltol'] 
        #     Required RELATIVE primal/dual gap tolerance to return as optimal
        #     scalar (default: 1e-6)
        # cvxopt.solvers.options['feastol'] 
        #     Required feasibility tolerance for primal and dual to return as optimal
        #     scalar (default: 1e-7).

        # ensure default baseline because these are global
        defaults = {
            'show_progress' : False ,
            'maxiters'      : 100   ,
            'refinement'    : 1     ,
            'abstol'        : 1e-7  ,
            'reltol'        : 1e-6  ,
            'feastol'       : 1e-7  ,
        }

        for key, val in defaults.items():
            cvxopt.solvers.options[key] = val

        #map from options to cvxopt things
        for key, val in self.options.items():
            if ky in ['show_progress', 'maxiters', 'refinement', 'abstol', 'reltol', 'feastol']:
                cvxopt.solvers.options[key] = val

    def _postsolve(self, res, unwrappedVariables):
        results = Results()

        status = res['status']
        if status == 'optimal':
            results.termination_condition = TerminationCondition.optimal
        elif status == 'dual infeasible':
            results.termination_condition = TerminationCondition.unbounded
        elif status == 'primal infeasible':
            results.termination_condition = TerminationCondition.infeasible
        elif status == 'unknown':
            results.termination_condition = TerminationCondition.unknown
        # elif status == 'userinterrupt':
        #     results.termination_condition = TerminationCondition.interrupted
        # elif status == 'terminate':
        #     results.termination_condition = TerminationCondition.interrupted
        # elif status == 'timelimit':
        #     results.termination_condition = TerminationCondition.maxTimeLimit
        # elif status == 'inforunbd':
        #     results.termination_condition = TerminationCondition.infeasibleOrUnbounded
        else:
            results.termination_condition = TerminationCondition.unknown

        # x y z s 

        #unconverged, these are none
        if res['status'] == 'optimal':
            results.best_feasible_objective = res['primal objective'] + res.get('objective_shift',0.0)
            results.best_objective_bound    = res['dual objective']   + res.get('objective_shift',0.0)
        else:
            results.best_feasible_objective = None
            results.best_objective_bound    = None

        results.solve_type = res['problem_structure']

        if res['status'] == 'optimal':
            primals = {}
            for i in range(0,len(unwrappedVariables)):
                #key is id of variable, val is tuple of var and val
                primals[id(unwrappedVariables[i])] = (unwrappedVariables[i], res['x'][i])
        else:
            primals = None

        # reduced costs are duals for variable bounds
        # need to seperate out the variable bounds into the reduced cost bin

        # Parameters
        # ----------
        # primals: dict
        #     maps id(Var) to (var, value)
        # duals: dict
        #     maps Constraint to dual value
        # slacks: dict
        #     maps Constraint to slack value
        # reduced_costs: dict
        #     maps id(Var) to (var, reduced_cost)

        results.solution_loader = SolutionLoader(primals=primals, duals=None, slacks=None, reduced_costs=None)

        if self.config.load_solution:
            if res['status'] == 'optimal':
                if results.termination_condition != TerminationCondition.optimal:
                    logger.warning(
                        'Loading a feasible but suboptimal solution. '
                        'Please set load_solution=False and check '
                        'results.termination_condition and '
                        'results.found_feasible_solution() before loading a solution.'
                    )
                results.solution_loader.load_vars()
            else:
                raise RuntimeError(
                    'A feasible solution was not found, so no solution can be loaded.'
                    'Please set opt.config.load_solution=False and check '
                    'results.termination_condition and '
                    'results.best_feasible_objective before loading a solution.'
                )
            
        return results

    def solve( self, model: _BlockData, timer: HierarchicalTimer = None ) -> Results:
        if timer is None:
            timer = HierarchicalTimer()
        timer.start('solve')

        variableList = [ vr for vr in model.component_objects( pyo.Var, descend_into=True, active=True ) ]

        unwrappedVariables = []
        vrIdx = -1
        for i in range(0, len(variableList)):
            vr = variableList[i]
            vrIdx += 1
            if isinstance(vr, ScalarVar):
                unwrappedVariables.append(vr)
            elif isinstance(vr, IndexedVar):
                unwrappedVariables.append(vr)
                for sd in vr.index_set().data():
                    vrIdx += 1
                    unwrappedVariables.append(vr[sd])
            else:
                raise DeveloperError( 'Variable is not a variable.  Should not happen.  Contact developers' )


        self._set_options()

        # copy paste
        ostreams = [ LogStream( level=self.config.log_level, logger=self.config.solver_output_logger ) ]
        if self.config.stream_solver:
            ostreams.append(sys.stdout)
        if self.config.logfile:
            f = open(self.config.logfile, 'w')
            ostreams.append(f)

        try:
            # take output and send to logger and term
            with TeeStream(*ostreams) as t:
                with capture_output(output=t.STDOUT, capture_fd=True):
                    timer.start('structure walker')
                    structures = structure_detector(model)
                    timer.stop('structure walker')

                    timer.start('cvxopt optimize')
                    if structures['Linear_Program'][0]:
                        cvxoptRes = solve_LP(structures)
                        cvxoptRes['problem_structure'] = 'linear_program'
                    elif structures['Quadratic_Program'][0]:
                        cvxoptRes = solve_QP(structures)
                        cvxoptRes['problem_structure'] = 'quadratic_program'
                    elif structures['Geometric_Program'][0]:
                        cvxoptRes = solve_GP(structures)
                        cvxoptRes['problem_structure'] = 'geometric_program'
                    else:
                        raise ValueError('Could not convert the formulation to a valid CVXOPT structure (LP,QP,GP)')
                    timer.stop('cvxopt optimize')
        finally:
            if self.config.logfile:
                f.close()

        res = self._postsolve(cvxoptRes,unwrappedVariables)

        timer.stop('solve')

        if self.config.report_timing:
            logger.info('\n' + str(timer))

        return res