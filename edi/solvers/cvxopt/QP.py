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

from pyomo.common.dependencies import attempt_import

cvxopt, cvxopt_available = attempt_import( "cvxopt" )
if not cvxopt_available:
    raise ImportError('The CVXOPT solver requires cvxopt')


def solve_QP(structures):

    P = cvxopt.matrix( structures['Quadratic_Program'][1][0] )
    q = cvxopt.matrix(structures['Quadratic_Program'][1][1])
    objective_shift = structures['Quadratic_Program'][1][2]
    AG = structures['Quadratic_Program'][1][3]
    bh = structures['Quadratic_Program'][1][4]
    operators = structures['Quadratic_Program'][2]

    A = None
    G = None
    b = None
    h = None

    eqUnscramble = []
    iqUnscramble = []
    if AG is not None:
        for i in range(0,len(AG)):
            if operators[i] == '==':
                eqUnscramble.append(i)
                if A is None:
                    A = [AG[i].tolist()]
                    b = [-1*bh[i]]
                else:
                    A += [AG[i].tolist()]
                    b += [-1*bh[i]]
            else:
                iqUnscramble.append(i)
                if G is None:
                    G = [AG[i].tolist()]
                    h = [-1*bh[i]]
                else:
                    G += [AG[i].tolist()]
                    h += [-1*bh[i]]

    if A is not None:
        A = cvxopt.matrix(A).T
    if b is not None:
        b = cvxopt.matrix(b)
    if G is not None:
        G = cvxopt.matrix(G).T
    if h is not None:
        h = cvxopt.matrix(h)

    # cvxopt adds a 1/2 to the quadratic term, need to multiply by 2
    res = cvxopt.solvers.qp(2.0*P,q,G,h,A,b)
    res['objective_shift'] = objective_shift 
    res['inequality_unscramble'] = iqUnscramble
    res['equality_unscramble']   = eqUnscramble

    res['N_cons_total']    = structures['info']['N_cons_total']   
    res['N_cons_noBounds'] = structures['info']['N_cons_noBounds']
    res['N_cons_bounds']   = structures['info']['N_cons_bounds']  

    return res
