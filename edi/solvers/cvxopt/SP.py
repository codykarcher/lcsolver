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

import copy

import pyomo.environ as pyo
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.dependencies import attempt_import
from pyomo.core.base.var import ScalarVar, _GeneralVarData, IndexedVar

if numpy_available:
    import numpy as np
else:
    raise ImportError('The Signomial Programming solver requires numpy')

cvxopt, cvxopt_available = attempt_import( "cvxopt" )
if not cvxopt_available:
    raise ImportError('The Signomial Programming solver currently requires cvxopt')

# from edi.structure.detectorSupportFunctions import (
#      # gpRow_add,
#      # gpRow_subtract,
#      gpRow_multiply,
#      gpRow_divide,
#      # collapseGProws,
#      # parseDict_GP,
#      # checkObjectiveHessian_PD,
#      # checkLinear,
#      # unstructured_dict,
# )
from edi.structure.detectorSupportFunctions import (
     gpRow_multiply,
     gpRow_divide,
)

# from edi.solvers.cvxopt.GP import solve_GP
from edi.solvers.cvxopt.GP import solve_GP

def evaluate_posynomial(gpRows,x_star):
    N_vars = len(gpRows[0])-2
    ucis = np.unique([rw[0] for rw in gpRows])
    if len(ucis) != 1:
        raise ValueError('A non-posynomial object was detected')
    N_rows = len(ucis)
    subVals = np.zeros([len(gpRows),N_vars+1])
    for i in range(0,len(gpRows)):
        subVals[i][0] = gpRows[i][1]
        for j in range(0,N_vars):
            subVals[i][j+1] = x_star[j]**gpRows[i][j+2]
    val = 0.0
    for i in range(0,len(subVals)):
        val += np.prod(subVals[i])

    grad = np.zeros([N_vars])
    for i in range(0,N_vars):
        for j in range(0,len(subVals)):
            if subVals[j][i+1] == 0:
                subGrad = 0
            else:
                subGrad = np.prod(subVals[j])/subVals[j][i+1]
            subGrad *= gpRows[j][i+2] * x_star[i]**(gpRows[j][i+2] - 1)
            grad[i] += subGrad

    return val, grad

def monomial_approximation(gpRows,x_star):
    N_vars = len(gpRows[0])-2
    ucis = np.unique([rw[0] for rw in gpRows])
    if len(ucis) != 1:
        raise ValueError('A non-posynomial object was detected')

    f, dfdx = evaluate_posynomial(gpRows,x_star)

    lc = f
    a = [x_star[i]/f * dfdx[i] for i in range(0,N_vars)]
    for i in range(0,N_vars):
        lc *= (1/x_star[i])**a[i]

    return [[None, lc] + a]


def TemplateDict(num=None, dem=None, app=False):
    return {'numerator':              num, 
            'denominator':            dem, 
            'approximateNumerator':   app,
            'numeratorIndices':       []}


def pccp_modification(constraintList,penalty_exponent=5.0):

    slackStartIndex = len(constraintList[0]['numerator'][0])
    
    # count the number of sp constraints
    spCounter = 0
    for i in range(0,len(constraintList)):
        if constraintList[i]['denominator'] is not None or constraintList[i]['approximateNumerator']==True:
            spCounter += 1

    # allocate empty slots for slack variables
    for i in range(0,len(constraintList)):
        for j in range(0,len(constraintList[i]['numerator'])): 
            constraintList[i]['numerator'][j] += np.zeros(spCounter).tolist()
        if constraintList[i]['denominator'] is not None:
            for j in range(0,len(constraintList[i]['denominator'])): 
                constraintList[i]['denominator'][j] += np.zeros(spCounter).tolist()

    # create slack monomial
    slackMonomial = [0.0]*len(constraintList[0]['numerator'][0])
    for i in range(0,len(slackMonomial)):
        if i >= slackStartIndex:
            slackMonomial[i] = penalty_exponent
    slackMonomial[0] = None
    slackMonomial[1] = 1.0

    # add slacks to each constraint
    spIndex = 0
    for i in range(0,len(constraintList)):
        if i == 0:
            constraintList[i]['numerator'] = gpRow_multiply(constraintList[i]['numerator'], [slackMonomial])
        else:
            if constraintList[i]['denominator'] is not None:
                # is a SP constraint
                slackRow = np.zeros(len(constraintList[0]['numerator'][0])).tolist()
                slackRow[0] = None
                slackRow[1] = 1.0
                slackRow[slackStartIndex+spIndex] = 1.0
                constraintList[i]['denominator'] = gpRow_multiply(constraintList[i]['denominator'], [slackRow])
                spIndex += 1
            elif constraintList[i]['approximateNumerator']==True:
                # is a posynomial equality
                slackRow = np.zeros(len(constraintList[0]['numerator'][0])).tolist()
                slackRow[0] = None
                slackRow[1] = 1.0
                slackRow[slackStartIndex+spIndex] = 1.0
                constraintList[i]['numerator'] = gpRow_divide(constraintList[i]['numerator'], [slackRow])
                spIndex += 1
            else:
                # constraint needs no slack
                pass

    # add s>=1
    for i in range(0,spCounter):
        # currentCon += 1
        nextRow = np.zeros(len(constraintList[0]['numerator'][0])).tolist()
        nextRow[0] = None
        nextRow[1] = 1.0
        nextRow[i+slackStartIndex] = -1
        constraintList.append(TemplateDict([nextRow],None,False))
        
    return constraintList, spCounter


def solve_SP(structures, m, reltol=1e-4, var_reltol = 1, max_iter = 50, use_pccp = True, penalty_exponent=5.0):
    variableList = [ vr for vr in m.component_objects( pyo.Var, descend_into=True, active=True ) ]

    unwrappedVariables = []
    vrIdx = -1
    for i in range(0, len(variableList)):
        vr = variableList[i]
        vrIdx += 1
        if isinstance(vr, ScalarVar):
            unwrappedVariables.append(vr)
        elif isinstance(vr, IndexedVar):
            # unwrappedVariables.append(vr)
            for sd in vr.index_set().data():
                vrIdx += 1
                unwrappedVariables.append(vr[sd])
        else:
            raise DeveloperError( 'Variable is not a variable.  Should not happen.  Contact developers' )

    x_star = [uv.value for uv in unwrappedVariables]

    spRows = structures['Signomial_Program'][1]
    operators = structures['Signomial_Program'][2]
    # print(spRows)

    conIxs,conCounts = np.unique([spRows[i][0] for i in range(0,len(spRows))],return_counts=True)
    conIxs = conIxs.tolist()
    conCounts = conCounts.tolist()

    constraintList = []

    constraintCounter = 0
    lastConstraintIndex = 0
    numeratorBuffer = []
    denominatorBuffer = []
    newOperators = []

    for i in range(0,len(spRows)+1):
        if i == len(spRows):
            lastConstraintIndex = 1
            rw = [-10]
        else:
            rw = spRows[i]
            
        if rw[0] == lastConstraintIndex:
            numeratorBuffer.append(rw)
        elif -1*rw[0]-1 == lastConstraintIndex:
            denominatorBuffer.append(rw)
        else:
            if constraintCounter > 0:
                operator = operators[constraintCounter-1]
            else:
                #is an objective, 
                operator = '<='                

            N_monomials_num = conCounts[conIxs.index(constraintCounter)]
            if -1*constraintCounter-1 in conIxs:
                N_monomials_dem = conCounts[conIxs.index(-1*constraintCounter-1)]
            else:
                N_monomials_dem = 0
            
            if operator == '==' and N_monomials_dem >= 1 :
                # is signomial equality
                if denominatorBuffer == []:
                    denominatorBuffer = None
                constraintList.append(TemplateDict(copy.deepcopy(numeratorBuffer), copy.deepcopy(denominatorBuffer), True))
                newOperators.append('<=')

                if denominatorBuffer == None:
                    denominatorBuffer = [None, 1.0] + [0.0]*len(x_star)
                constraintList.append(TemplateDict(copy.deepcopy(denominatorBuffer), copy.deepcopy(numeratorBuffer), True))
                newOperators.append('<=')   

            elif operator == '==':
                # monomial equality
                constraintList.append(TemplateDict(copy.deepcopy(numeratorBuffer), None, False))
                newOperators.append('<=')

                flippedRows = []
                for srw in numeratorBuffer:
                    newRow = [ srw[0], 1.0/srw[1] ] + [-1*vl for vl in srw[2:]]
                    flippedRows.append(newRow)
                constraintList.append(TemplateDict(copy.deepcopy(flippedRows), None, False))
                newOperators.append('<=')
                
            else:

                if denominatorBuffer == []:
                    denominatorBuffer = None
                constraintList.append(TemplateDict(copy.deepcopy(numeratorBuffer), copy.deepcopy(denominatorBuffer), False))
                newOperators.append(operator)

            if i != len(spRows):
                numeratorBuffer = [rw]
                denominatorBuffer = []
                constraintCounter += 1
                lastConstraintIndex = rw[0]

    for i in range(0,len(constraintList)):
        for j in range(0,len(constraintList[i]['numerator'])): 
            constraintList[i]['numerator'][j][0] = 1
        if constraintList[i]['denominator'] is not None:
            for j in range(0,len(constraintList[i]['denominator'])): 
                constraintList[i]['denominator'][j][0] = 1

    if use_pccp:
        constraintList, N_slacks = pccp_modification(constraintList,penalty_exponent)
        x_star += [1.0]*N_slacks

    # print(constraintList)
    gpRows = []
    nextRowIndex = 0
    approximatedConstraintList = []
    for i in range(0,len(constraintList)):
        num = constraintList[i]['numerator']
        dem = constraintList[i]['denominator']
        app = constraintList[i]['approximateNumerator']
        
        for j in range(0,len(num)):
            constraintList[i]['numerator'][j][0] = i

        if dem is not None:
            for j in range(0,len(dem)):
                constraintList[i]['denominator'][j][0] = -1*i - 1
        
        if dem is None and not app:
            # gp compatible
            gpRows += copy.deepcopy(num)
            nextRowIndex += len(num)
            
        elif dem is None:
            # posynomial equality
            gpRows.append(None)
            constraintList[i]['numeratorIndices'] = [nextRowIndex]
            approximatedConstraintList.append(constraintList[i])
            nextRowIndex += 1

        elif not app:
            # signomial inequality, dont approximate numerator
            gpRows += [None]*len(num)
            constraintList[i]['numeratorIndices'] = list(range(nextRowIndex,nextRowIndex+len(num)))
            approximatedConstraintList.append(constraintList[i])
            nextRowIndex += len(num)

        else:
            # signomial equality, approximate numerator
            gpRows.append(None)
            constraintList[i]['numeratorIndices'] = [nextRowIndex]
            approximatedConstraintList.append(constraintList[i])
            nextRowIndex += 1

        # for rw in gpRows:
        #     print(rw)
        # print()

    for itr in range(0,max_iter):
        for i in range(0,len(approximatedConstraintList)):
            num = approximatedConstraintList[i]['numerator']
            dem = approximatedConstraintList[i]['denominator']
            numix = approximatedConstraintList[i]['numeratorIndices']
            if approximatedConstraintList[i]['denominator'] is None:
                num_ma = monomial_approximation(copy.deepcopy(num),x_star)
                num_ma[0][0] = num[0][0]
                gpRows[numix[0]] = num_ma[0]
            else:
                if approximatedConstraintList[i]['approximateNumerator']:
                    num_ma = monomial_approximation(copy.deepcopy(num),x_star)
                    dem_ma = monomial_approximation(copy.deepcopy(dem),x_star)
                    num_ma[0][0] = num[0][0]
                    dem_ma[0][0] = dem[0][0]
                    frac = gpRow_divide(num_ma,dem_ma)
                    gpRows[numix[0]] = frac[0]
                else:
                    dem_ma = monomial_approximation(copy.deepcopy(dem),x_star)
                    dem_ma[0][0] = dem[0][0]
                    posy = gpRow_divide(copy.deepcopy(num), dem_ma)
                    for j in numix:
                        gpRows[j] = posy[j-numix[0]]

        pack = {}
        pack['Linear_Program'] = [False]
        pack['Quadratic_Program'] = [False]
        pack['Geometric_Program'] = [True, gpRows, ['<=']*gpRows[-1][0]]
        # print(len(newOperators))
        # print(newOperators)
        # print(gpRows[-1][0])

        pack['info']={}
        pack['info']['N_cons_total']    = structures['info']['N_cons_total']
        pack['info']['N_cons_noBounds'] = structures['info']['N_cons_noBounds']
        pack['info']['N_cons_bounds']   = structures['info']['N_cons_bounds']
        
        if itr == 0:
            prevObj = np.inf
        else:
            prevObj = res['primal objective']

        # print(pack)
        # print('solving GP approximation')
        res = solve_GP(pack)

        # new_x_star = x_star + 0.5*(np.array(res['x']).reshape(len(x_star))-x_star)
        new_x_star = np.array(res['x']).reshape(len(x_star))
        # if res['primal objective'] == 0:
        #     reltol_check = abs(prevObj - res['primal objective']) <= reltol
        # else:

        # print('=========')
        # print(itr)
        # print(res['primal objective'])
        # print(res)
        # print(abs(prevObj - res['primal objective'])/res['primal objective'])
        # # print(reltol_check)
        # for i in range(0,len(x_star)):
        #     # print('new_x: ',new_x_star[i])
        #     print(abs(new_x_star[i]-x_star[i])/new_x_star[i])
        # # print(var_reltol_check)

        reltol_check = abs(prevObj - res['primal objective'])/res['primal objective'] <= reltol

        var_reltol_check = all([abs(new_x_star[i]-x_star[i])/new_x_star[i] <= var_reltol for i in range(0,len(x_star))])

        if var_reltol_check and reltol_check:
            # break
            return res
        else:
            # x_star = [ nxv[0] for nxv in new_x_star.tolist()]
            x_star = new_x_star
            # print(x_star)
        
    raise RuntimeError('Reached the maximum iteration count and failed to find a solution.  There is likely a problem with your formulation')

    # print(res)
    # return res