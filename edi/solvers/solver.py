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

from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.dependencies import attempt_import
# from pyomo.contrib.edi.tools.structureDetector import structure_detector
from edi.structure.structureDetector import structure_detector
from edi.units.unitCorrector import unit_corrector


cvxopt, cvxopt_available = attempt_import( "cvxopt" )
if not cvxopt_available:
    raise ImportError('The CVXOPT solver requires cvxopt')

def cvxopt_solve(m):
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['maxiters'] = 100
    cvxopt.solvers.options['feastol'] = 1e-6
    cvxopt.printing.options['width'] = -1

    # print('walking units')
    m_corrected_units = unit_corrector(m)
    # print('walking structure')
    structures = structure_detector(m_corrected_units)
    # print('Detected problem structure')
    
    # structures = structure_detector(m)
    # print(structures)
    if structures['Linear_Program'][0]:
        # from pyomo.contrib.edi.solvers.cvxopt import solve_LP
        from edi.solvers.cvxopt.LP import solve_LP
        res = solve_LP(structures)
        res['problem_structure'] = 'linear_program'
    elif structures['Quadratic_Program'][0]:
        # from pyomo.contrib.edi.solvers.cvxopt import solve_QP
        from edi.solvers.cvxopt.QP import solve_QP
        res = solve_QP(structures)
        res['problem_structure'] = 'quadratic_program'
    elif structures['Geometric_Program'][0]:
        # from pyomo.contrib.edi.solvers.cvxopt import solve_GP
        from edi.solvers.cvxopt.GP import solve_GP
        res = solve_GP(structures)
        res['problem_structure'] = 'geometric_program'
    elif structures['Signomial_Program'][0]:
        # from pyomo.contrib.edi.solvers.native.SP import solve_SP
        from edi.solvers.cvxopt.SP import solve_SP
        res = solve_SP(structures,m)
        res['problem_structure'] = 'signomial_program_pccp'
    else:
        raise ValueError('Could not convert the formulation to a valid CVXOPT structure (LP,QP,GP,SP)')

    return res






