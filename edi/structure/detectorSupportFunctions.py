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

import math
import random
import copy
import re
import io
import pyomo.environ as pyo

from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.dependencies import attempt_import


# from edi.structure.walkerSupportFunctions import (
#     processMonomial,
# )
from edi.structure.walkerSupportFunctions import processMonomial

if numpy_available:
    import numpy as np
else:
    raise ImportError('The stucture detector requires numpy')

def gpRow_add(gr1, gr2):
    gr1_isFrac = any([g[0] <= 0 for g in gr1])
    gr2_isFrac = any([g[0] <= 0 for g in gr2])
    if gr1_isFrac or gr2_isFrac:
        # if gr1_isFrac and gr2_isFrac:
        raise RuntimeError('Signomial Fraction addition should not be occurring here')
    #     elif gr2_isFrac: 
    #         gr1_temp = copy.deepcopy(gr2)
    #         gr2 = gr1
    #         gr1 = gr1_temp
    #     else:
    #         pass

    #     if len(gr2) > 1 :
    #         raise RuntimeError('Adding a posynomial to a fraction, shouldnt happen but not sure')

    #     for i in range(0,len(gr1)):
    #         gr1[i][1] *= gr2[0][1]
    #         for j in range(2,len(gr1[0])):
    #             gr1[i][j] += gr2[0][j]

    #     return gr1
    # else:
    return collapseGProws(gr1+gr2)

def gpRow_subtract(gr1, gr2):
    for i in range(0,len(gr2)):
        if gr2[i][0] >= 0:
            gr2[i][1] *= -1
    return gpRow_add(gr1,gr2)

def gpRow_multiply(gr1, gr2):
    if len(gr1) == 1 and len(gr2) != 1:
        gr1_temp = gr2
        gr2 = gr1
        gr1 = gr1_temp

    if len(gr2) == 1:
        for i in range(0,len(gr1)):
            if gr1[i][0] >= 0:
                gr1[i][1] *= gr2[0][1]
                for j in range(2, len(gr2[0])):
                    gr1[i][j] += gr2[0][j]
        return collapseGProws(gr1)
    else:
        raise RuntimeError('Posynomial multiplication should have already been performed')

def gpRow_divide(gr1, gr2):
    if len(gr2) == 1:
        for i in range(2, len(gr2[0])):
            gr2[0][i] *= -1
        gr2[0][1] = 1/gr2[0][1]
        return gpRow_multiply(gr1, gr2)
    else:
        denom_ix = -1*gr1[0][0] - 1
        for i in range(0, len(gr2)):
            gr2[i][0] = denom_ix
        return gr1+gr2


def collapseGProws(gpRows):
    # this function was improved by AI for speed
    if not gpRows:
        return []
    
    # Group terms by (constraint_index, exponents) tuple for efficient lookup
    # Key: (constraint_index, tuple(exponents)), Value: list of indices with same key
    term_groups = {}
    
    for i, row in enumerate(gpRows):
        if row[1] == 0:  # Skip zero coefficient terms early
            continue
            
        # Create key from constraint index and exponents
        key = (row[0], tuple(row[2:]))
        
        if key not in term_groups:
            term_groups[key] = []
        term_groups[key].append(i)
    
    # Build result by combining coefficients for each group
    gpRows_new = []
    for key, indices in term_groups.items():
        if len(indices) == 1:
            # Single term, just copy it
            gpRows_new.append(gpRows[indices[0]][:])  # Make a copy to avoid modifying original
        else:
            # Multiple terms, sum coefficients
            total_coeff = sum(gpRows[idx][1] for idx in indices)
            if total_coeff != 0:  # Only add if non-zero
                # Use first row as template and update coefficient
                new_row = gpRows[indices[0]][:]  # Make a copy
                new_row[1] = total_coeff
                gpRows_new.append(new_row)
    
    return gpRows_new


# def collapseGProws(gpRows):
#     similarTerms = []
#     skipList = []
#     for i in range(0,len(gpRows)):
#         if i not in skipList:
#             for j in range(0,len(gpRows)):
#                 if j>i and j not in skipList:
#                     if gpRows[i][2:] == gpRows[j][2:] and gpRows[i][0] == gpRows[j][0]:
#                         skipList.append(i)
#                         skipList.append(j)
#                         doAppend = True
#                         for ii in range(0,len(similarTerms)):
#                             if i in similarTerms[ii]:
#                                 similarTerms[ii].append(j)
#                                 dontAppend = False
#                                 break
#                         if doAppend:
#                             similarTerms.append([i,j])
    
#     gpRows_new = []
#     skipList = []
#     for i in range(0,len(gpRows)):
#         doAppend = True
#         if i in skipList:
#             doAppend = False
#         for ii in range(0,len(similarTerms)):
#             if i in similarTerms[ii] and i not in skipList:
#                 gpRow_new = gpRows[i]
#                 gpRow_new[1] = sum([gpRows[c][1] for c in similarTerms[ii]])
#                 if gpRow_new[1] != 0:
#                     gpRows_new.append(gpRow_new)
#                 skipList += similarTerms[ii]
#                 doAppend = False
#                 break
#         if doAppend:
#             if gpRows[i][1] != 0:
#                 gpRows_new.append(gpRows[i])

#     return gpRows_new


def parseDict_GP(ix,rv,N_vars_unwrapped,variableMap):
    if rv['monomial']['status'] == 'yes':
        lcs = rv['monomial']['leadingConstant']
        bss = rv['monomial']['bases']
        exs = rv['monomial']['exponents']
        gpRow = processMonomial(ix,lcs,bss,exs,N_vars_unwrapped,variableMap)
        return [gpRow]

    if rv['signomial']['status'] == 'yes':
        gpRows = []
        for i in range(0,len(rv['signomial']['leadingCoefficients'])):
            lcs = rv['signomial']['leadingCoefficients'][i]
            bss = rv['signomial']['bases'][i]
            exs = rv['signomial']['exponents'][i]
            gpRow = processMonomial(ix,lcs,bss,exs,N_vars_unwrapped,variableMap)
            gpRows.append(gpRow)
        return collapseGProws(gpRows)

    # if neither other thing flags, it is a signomial fraction
    # Denominators have the index -ix-1, except objective which is -1
    gpRows = []
    for j in [0,1]:
        if j == 0:
            sf_ix = ix
            sf_ele = rv['signomial_fraction']['numerator']
        else:
            sf_ix = -ix-1
            sf_ele = rv['signomial_fraction']['denominator']
        for i in range(0,len(sf_ele['leadingCoefficients'])):
            lcs = sf_ele['leadingCoefficients'][i]
            bss = sf_ele['bases'][i]
            exs = sf_ele['exponents'][i]
            gpRow = processMonomial(sf_ix,lcs,bss,exs,N_vars_unwrapped,variableMap)
            gpRows.append(gpRow)

    return collapseGProws(gpRows)


def checkObjectiveHessian_PD(gpRows):
    N_vars = len(gpRows[0])-2
    hessian = np.zeros([N_vars,N_vars])
    P = np.zeros([N_vars,N_vars])
    q = np.zeros([N_vars])
    r = 0.0
    onlyZeros = True
    for rw in gpRows:
        if rw[0] == 0:
            exponents = rw[2:]
            if any([vl not in [0.0,1.0,2.0] for vl in exponents]) or sum([abs(vl) for vl in exponents]) > 2:
                return [False,None,None,None]
            elif any([vl == 2.0 for vl in exponents]) :
                ix = exponents.index(2.0)
                hessian[ix,ix] = 2.0 * rw[1]
                P[ix,ix] = rw[1]
                onlyZeros = False
            elif sum([abs(vl) for vl in exponents]) == 2:
                # has 1 and 1
                firstIndex  = exponents.index(1.0)
                secondIndex = exponents.index(1.0, firstIndex + 1)
                hessian[firstIndex,secondIndex] = rw[1]
                hessian[secondIndex,firstIndex] = rw[1]
                P[firstIndex,secondIndex] = rw[1]/2.0
                P[secondIndex,firstIndex] = rw[1]/2.0
                onlyZeros = False
            elif sum([abs(vl) for vl in exponents]) == 1: 
                # has one 1
                ix = exponents.index(1.0)
                q[ix] = rw[1]
            else:
                # is a constant, exps=0
                r = rw[1]

        else:
            break

    if onlyZeros:
        return [False,None,None,None]

    eigenvalues, eigenvectors = np.linalg.eig(hessian)
    if min(eigenvalues) > 0:
        return [True,P,q,r]
    else:
        return [False,None,None,None]

def checkLinear(gpRows):
    N_vars = len(gpRows[0])-2
    ucis = np.unique([rw[0] for rw in gpRows])
    N_rows = len(ucis)

    A = np.zeros([N_rows,N_vars])
    b = np.zeros([N_rows])

    for rw in gpRows:
        uci = rw[0]
        exponents = rw[2:]
        if any([vl not in [0.0,1.0] for vl in exponents]) or sum([abs(vl) for vl in exponents]) > 1:
            return [False,None,None]
        elif sum([abs(vl) for vl in exponents]) == 1: 
            # has one 1
            ix = exponents.index(1.0)
            A[uci-ucis[0]][ix] = rw[1]
        else:
            # is a constant, exps=0
            b[uci-ucis[0]] = rw[1]

    return [True,A,b]


def unstructured_dict():
    return {"Linear_Program"   :[False, None, None], 
            "Quadratic_Program":[False, None, None],
            "Geometric_Program":[False, None, None], 
            "Signomial_Program":[False, None, None],} 
