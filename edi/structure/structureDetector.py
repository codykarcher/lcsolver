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
import pyomo
import pyomo.environ as pyo

from pyomo.core.base.block import BlockData
from pyomo.common.collections.component_map import ComponentMap
from pyomo.core.base.var import ScalarVar, VarData, IndexedVar
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.numeric_types import RegisterNumericType
RegisterNumericType(pyomo.common.enums.ObjectiveSense)

# from pyomo.contrib.edi.tools.structureWalker import _StructureVisitor
from edi.structure.structureWalker import _StructureVisitor

from pyomo.common.dependencies import numpy, numpy_available

if numpy_available:
    import numpy as np
else:
    raise ImportError('The stucture detector requires numpy')

# from pyomo.contrib.edi.tools.detectorSupportFunctions import (
#      gpRow_add,
#      gpRow_subtract,
#      # gpRow_multiply,
#      gpRow_divide,
#      # collapseGProws,
#      parseDict_GP,
#      checkObjectiveHessian_PD,
#      checkLinear,
#      unstructured_dict,
# )
from edi.structure.detectorSupportFunctions import (
     gpRow_add,
     gpRow_subtract,
     # gpRow_multiply,
     gpRow_divide,
     # collapseGProws,
     parseDict_GP,
     checkObjectiveHessian_PD,
     checkLinear,
     unstructured_dict,
)

def implementVariableBound(vr,pyomo_component,N_bound_cons):
    """
    This function finds any upper or lower bounds declared in the variable declaration and implemnts 
    them as constraints in the optimization problem

    if variables are not continuous, returns [False, None, None] which is detected and parsed at the layer above

    otherwise, returns [True, updated_pyomo_component, N_bounds_cons]
    """
    # Reject if variable is not continuous
    if vr.domain.name not in ['Reals','NonNegativeReals','NonPositiveReals']:
        return [False, None, None]
        # return unstructured_dict() | { "message":"A non-continuous variable (%s) was detected"%(vr.name) } 

    # Walk through the variables, and add the bounds from the variable declaration in to the constraint list
    # Get the bounds
    var_lower_bound, var_upper_bound = vr.bounds
    # handle a non-negative real domain, ie x>=0
    if vr.domain.name == 'NonNegativeReals':
        if var_lower_bound is None:
            # set to zero if nothing else defined
            var_lower_bound = 0
        else:
            # Otherwise, take the maximum of the two (eg, the higer lower bound)
            var_lower_bound = max([0,var_lower_bound])

    # Handle if x<=0
    if vr.domain.name == 'NonPositiveReals':
        if var_upper_bound is None:
            # set to zero if not otherwise defined
            var_upper_bound = 0
        else:
            # otherwise take the tighter upper bound
            var_upper_bound = min([0,var_upper_bound])

    # Now that bounds are set, need to add them to the pyomo object
    # only do if lower bound is present
    if var_lower_bound is not None:
        # Need to come up with a key for the pyomo object
        # Will be x_lowerBound.  However, if x_lowerBound exists, we will do something different
        proposedKey = vr.name + '_lowerBound'
        # Check if the proposed key is in the pyomo object already
        if proposedKey in list(pyomo_component.__dict__.keys()):
            # if it is, we will create a key x_lowerBound_randomSeed_######
            for i in range(0,10):
                # Generate 10 random tries for ###### and see if it is in the pyomo dict
                proposedKey = vr.name + '_lowerBound_randomSeed_' + str(int(math.floor(random.random()*1e6)))
                # if it isn't in the pyomo dict, we use the key and break the loop
                if proposedKey not in list(pyomo_component.__dict__.keys()):
                    break
            # This should never happen, but if you can't find a unique key then we notify the user
            raise ValueError('Could not found a unique identifier for the lower bound on variable '+vr.name)
        # Now that we have a key, add the new constraint to the pyomo object
        setattr(pyomo_component, proposedKey, pyo.Constraint(expr = vr >= var_lower_bound))
        # Increment the number of bounding constraints
        N_bound_cons += 1

    # only do if upper bound is present
    if var_upper_bound is not None:
        # again will default to x_upperBound
        proposedKey = vr.name + '_upperBound'
        # check to see if this key already exists
        if proposedKey in list(pyomo_component.__dict__.keys()):
            # if it does, then we generate a random key
            for i in range(0,10):
                proposedKey = vr.name + '_upperBound_randomSeed_' + str(int(math.floor(random.random()*1e6)))
                if proposedKey not in list(pyomo_component.__dict__.keys()):
                    break
            raise ValueError('Could not found a unique identifier for the upper bound on variable '+vr.name)
        # and add the new constraint
        setattr(pyomo_component, proposedKey, pyo.Constraint(expr = vr <= var_lower_bound))
        # increment the number of bounding constraints
        N_bound_cons += 1

    return [True, pyomo_component, N_bound_cons]

def structure_detector(pyomo_component):
    # Various setup things

    if not isinstance(pyomo_component, BlockData):
        raise ValueError( "Invalid type %s passed into the convexity detector"%(str(type(pyomo_component))))

    # get all of the variables in the pyomo object (optimization problem)
    variableList = [ vr for vr in pyomo_component.component_objects(pyo.Var, descend_into=True, active=True) ]

    # Walk through all the variables and ensure that their domains are compatible with optimization structure
    # Eg, no discrete, no weird sets, etc
    N_bound_cons = 0
    for vr in variableList:
        # check if it's a vector, matrix, etc...
        if isinstance(vr,pyomo.core.base.var.IndexedVar):
            # Get the set the variable is indexed over, this would be [1,2,3,4...] for a vector 
            # or [(1,1), (1,2)... ] for a matrix
            ix_st = list(vr.index_set())
            # Iterate for all the variables in the indexed set (eg elements in the vector)
            for ix in ix_st:
                [success, pyomo_component, N_bound_cons] = implementVariableBound(vr[ix],pyomo_component,N_bound_cons)
                if not success:
                    return unstructured_dict() | { "message":"A non-continuous variable (%s) was detected"%(vr[ix].name) } 

        else: #variable is scalar
            [success, pyomo_component, N_bound_cons] = implementVariableBound(vr,pyomo_component,N_bound_cons)
            if not success:
                return unstructured_dict() | { "message":"A non-continuous variable (%s) was detected"%(vr[ix].name) } 

    # get all the objectives
    objectives    = [ obj for obj in pyomo_component.component_data_objects(pyo.Objective , descend_into=True, active=True ) ]
    # get all the constraints
    constraints   = [ con for con in pyomo_component.component_objects(     pyo.Constraint, descend_into=True, active=True ) ]
    # get all the parameters
    parameterList = [ pm for pm in pyomo_component.component_objects(       pyo.Param     , descend_into=True, active=True ) ]
    # variables were extracted above
    ## variableList = [ vr for vr in pyomo_component.component_objects(pyo.Var, descend_into=True, active=True) ]

    ### TODO ===================================
    # Need to handle when variables are present in the formulation but not the objectives/constraints
    # For now, trust the user not to do this.

    # obj_con_vars = []
    # for i, obj in enumerate(objectives):
    #     objvrs = list(identify_variables(obj))
    #     # print(objvrs)
    #     for v in objvrs:
    #         if v.name not in [vv.name for vv in obj_con_vars]:
    #             obj_con_vars.append(v)
    # if len(constraints) > 0:
    #     # print(structures)
    #     # conCounter = 0
    #     # operatorList = []
    #     # Iterate over the constraints
    #     for i, con in enumerate(constraints):
    #         convrs = list(identify_variables(con)) #[ vr for vr in con.component_objects(pyo.Var, descend_into=True, active=True) ]
    #         for v in convrs:
    #             if v.name not in [vv.name for vv in obj_con_vars]:
    #                 obj_con_vars.append(v)
    # # print(obj_con_vars)

    # variableList = obj_con_vars


    # Need to create a mapping between the variable and the index as stored in the optimization problem
    # x    -> 0
    # y[1] -> 1
    # y[2] -> 2
    # z    -> 3 
    # etc...
    variableMap = ComponentMap()
    # Also will have all the unwrapped variables to track any unused in the constraints
    unwrappedVariables = []
    # start so with a +1 becomes 0
    vrIdx = -1
    # iterate through the variable list (which is not unwrapped)
    for i,vr in enumerate(variableList):
        # Check if scalar
        if isinstance(vr, ScalarVar):
            # if so, simple to just add to gthe map
            vrIdx += 1
            variableMap[vr] = vrIdx
            unwrappedVariables.append(vr)
        # if indexed, little more complicated
        elif isinstance(vr, IndexedVar):
            # only append the actual indexed vars, not the top level variable
            for sd in vr.index_set().data():
                # Iterate through all the variables in the indexed set and add these
                vrIdx += 1
                variableMap[vr[sd]] = vrIdx
                unwrappedVariables.append(vr[sd])
        else:
            # somehow a variable in the variable list is no longer a variable
            raise RuntimeError( 'Variable is not a variable.  Should not happen.  Contact developers' )

    # Get the total number of unwrapped variables
    N_vars_unwrapped = len(variableMap.keys())

    # Declare a visitor/walker
    visitor = _StructureVisitor()

    # starts building the output dict
    # Structured as : [strucure_present, the gp matrix stack, the constraint operator list]
    structures = {"Linear_Program"   :[True,[],[]], 
                  "Quadratic_Program":[True,[],[]],
                  "Geometric_Program":[True,[],[]], 
                  "Signomial_Program":[True,[],[]],} # Convex, LogConvex, Convex_QCQP, 

    # Current implementation only allows for one objective
    if len(objectives) != 1:
        return unstructured_dict() | { "message":"Current implementation only allows for a single objective, multiple objectives detected"} 

    # Iterate over the objectives
    for obj in objectives:
        # Walk the expression, returns the full breakdown of the constraint in dictionary form
        rv = visitor.walk_expression(obj.sense * obj)
        # parses into a gp-solver like matrix/vector
        gpRows = parseDict_GP(0,rv,N_vars_unwrapped,variableMap)
        # Should be in the form [constraint_number, leading constant, exponent for var_1, exponent for var_2...]
        # constraint_number for objectives will be either 0 for numerator or -1 for denomonator

        # check that all of the first entries (constraint number) are 0, otherwise this is a signomial fraction
        if not all([rw[0]==0.0 for rw in gpRows]):
            # is sp with fractional objective
            structures['Linear_Program'][0] = False
            structures['Quadratic_Program'][0] = False
            structures['Geometric_Program'][0] = False
            if not all([rw[1]>0.0 for rw in gpRows]):
                # has subtraction in the objective, which is not allowed
                return unstructured_dict()  | { "message":"Current implementation only allows for objectives to be fractions of posynomials, not signomials"} 
            else:
                # is a valid posynomial / posynomial objective for a signomial program
                structures['Signomial_Program'][1] += gpRows
        else:
            # this is a single signomial expression
            # Check to see if all of the leading constants are positive for GP/SP
            if not all([rw[1]>0.0 for rw in gpRows]):
                # has subtraction in the objective
                structures['Geometric_Program'][0] = False
                structures['Signomial_Program'][0] = False
            else:
                # has monomial or posynomial objective 
                # cannot yet make determination on LP/QP
                structures['Geometric_Program'][1] += gpRows
                structures['Signomial_Program'][1] += gpRows  


            # check hessian of objective
            # Checks for positive definate, returns [true/false, P, q, r], only true if quadratic, not if linear
            quadraticCheck = checkObjectiveHessian_PD(gpRows)

            # if its positive definite
            if quadraticCheck[0]:
                # If PD, then it is a QP and is not an LP
                structures['Quadratic_Program'][1] = quadraticCheck[1:] + [None,None]
                structures['Linear_Program'][0] = False
            else:
                # if not PD, then it is not a QP
                structures['Quadratic_Program'][0] = False

                # And may be an LP, need to check for linearity (all eigenvalues would be zero)
                linearCheck = checkLinear(gpRows)
                if linearCheck[0]:
                    structures['Linear_Program'][1] = linearCheck[1:] + [None,None]
                else:
                    structures['Linear_Program'][0] = False  

    # check that there are constraints
    N_cons = 0
    if len(constraints) > 0:
        # print(structures)
        conCounter = 0
        operatorList = []
        # Iterate over the constraints
        for i, con in enumerate(constraints):
            # Need to extract from pyomo model
            for c in con.values():
                # increment constraint count
                N_cons += 1
                # Extract the expression, which includes the operator
                cexpr = c.expr
                # print('========================================================================')
                # print(cexpr)
                # print('------------------------------------------------------------------------')
                # Walk the expression
                rv = visitor.walk_expression(cexpr)
                # print(rv)
                # print('========================================================================')
                # Rv includes all constant, monomial, signomial, etc...  Walk through each of these
                for rvv in rv:
                    # Extract all of the GP style matricies
                    gpRows_lhs = parseDict_GP(i+1,rvv['lhs'],N_vars_unwrapped,variableMap)
                    gpRows_rhs = parseDict_GP(i+1,rvv['rhs'],N_vars_unwrapped,variableMap)
                    operator = rvv['operator']
                    operatorList.append(operator)

                    if not any([rw[0] < 0 for rw in gpRows_lhs]):
                        lhs_zeroed = gpRow_subtract(copy.deepcopy(gpRows_lhs), copy.deepcopy(gpRows_rhs))
                        # Do LP/QP stuff
                        if not all([rw[0]>=0.0 for rw in lhs_zeroed]):
                            # has fraction
                            structures['Linear_Program'][0] = False
                            structures['Linear_Program'][1] = None
                            structures['Quadratic_Program'][0] = False
                            structures['Quadratic_Program'][1] = None         
                        else:               
                            linearCheck = checkLinear(lhs_zeroed)
                            if linearCheck[0]:
                                if structures['Linear_Program'][0] != False:
                                    if structures['Linear_Program'][1][2] is None:
                                        structures['Linear_Program'][1][2] = linearCheck[1]
                                        structures['Linear_Program'][1][3] = linearCheck[2]
                                    else:
                                        structures['Linear_Program'][1][2] = np.append( structures['Linear_Program'][1][2], linearCheck[1] , axis=0)
                                        structures['Linear_Program'][1][3] = np.append( structures['Linear_Program'][1][3], linearCheck[2] )
                                if structures['Quadratic_Program'][0] != False:
                                    if structures['Quadratic_Program'][1][3] is None:
                                        structures['Quadratic_Program'][1][3] = linearCheck[1]
                                        structures['Quadratic_Program'][1][4] = linearCheck[2]
                                    else:
                                        structures['Quadratic_Program'][1][3] = np.append( structures['Quadratic_Program'][1][3], linearCheck[1] , axis=0)
                                        structures['Quadratic_Program'][1][4] = np.append( structures['Quadratic_Program'][1][4], linearCheck[2] )
                            else:
                                structures['Linear_Program'][0] = False
                                structures['Linear_Program'][1] = None
                                structures['Quadratic_Program'][0] = False
                                structures['Quadratic_Program'][1] = None   
                    else:
                        # signomial fraction present
                        structures['Linear_Program'][0] = False
                        structures['Linear_Program'][1] = None
                        structures['Quadratic_Program'][0] = False
                        structures['Quadratic_Program'][1] = None   


                    # Do GP/SP Stuff

                    # this was trying to be smart, but had an issue with non-zero rhs terms
                    # if len(gpRows_rhs) == 1 :
                    #     if gpRows_rhs[0][1] < 0:
                    #         # constraint is bound to less than a negative number, infeasible
                    #         structures['Geometric_Program'][0] = False
                    #         structures['Geometric_Program'][1] = None
                    #         structures['Signomial_Program'][0] = False
                    #         structures['Signomial_Program'][1] = None                            
                    #     else:
                    #         print(gpRows_lhs)
                    #         print(gpRows_rhs)  
                    #         if gpRows_rhs[0][1] == 0.0:
                    #             lhs_final = gpRows_lhs
                    #         else:
                    #             lhs_final = gpRow_divide(gpRows_lhs, gpRows_rhs)
                    # else:
                    lhs_zeroed = gpRow_subtract(copy.deepcopy(gpRows_lhs), copy.deepcopy(gpRows_rhs))
                    unique, counts = numpy.unique([lhz[0] for lhz in lhs_zeroed], return_counts=True)
                    countDict = dict(zip(unique, counts))
                    if len(unique) > 1:
                        # have a fraction, this shouldnt be possible at this stage
                        raise RuntimeError('Encountered an unexpected signomial fraction, shouldnt happen but not sure')
                    negative_monomial_indices = []
                    for ii in range(0,len(lhs_zeroed)):
                        if lhs_zeroed[ii][1] < 0:
                            negative_monomial_indices.append(ii)
                    if len(negative_monomial_indices) == 0 or len(lhs_zeroed)==1 :
                        lhs_final = gpRow_add(lhs_zeroed, [[lhs_zeroed[0][0],1.0]+[0.0]*N_vars_unwrapped])
                    elif len(negative_monomial_indices) == 1:
                        negMonomial = copy.deepcopy(lhs_zeroed[negative_monomial_indices[0]])
                        posMonomial = copy.deepcopy(negMonomial)
                        posMonomial[1] *= -1
                        lhs_inter = gpRow_subtract(lhs_zeroed, [negMonomial])
                        lhs_final = gpRow_divide(lhs_inter, [posMonomial])
                    else:
                        negPosynomial = [ copy.deepcopy(lhs_zeroed[nmi]) for nmi in negative_monomial_indices ]
                        posPosynomial = copy.deepcopy(negPosynomial)
                        for ii in range(0,len(posPosynomial)):
                            posPosynomial[ii][1] *= -1
                        lhs_inter = gpRow_subtract(lhs_zeroed, negPosynomial)
                        lhs_final = gpRow_divide(lhs_inter, posPosynomial)
                    # this is where the else indent should be if present

                    if not all([rw[1]>0.0 for rw in lhs_final]):
                        # has subtraction, which is not allowed under this definition of SP
                        structures['Geometric_Program'][0] = False
                        structures['Geometric_Program'][1] = None
                        structures['Signomial_Program'][0] = False
                        structures['Signomial_Program'][1] = None

                    if not all([rw[0]>=0.0 for rw in lhs_final]):
                        # is sp with fraction
                        structures['Geometric_Program'][0] = False
                        structures['Geometric_Program'][1] = None
                        if structures['Signomial_Program'][0] != False:
                            structures['Signomial_Program'][1] += lhs_final
                    else:
                        # monomial or valid posynomial   
                        if structures['Geometric_Program'][0] != False:
                            structures['Geometric_Program'][1] += lhs_final
                        if structures['Signomial_Program'][0] != False:
                            structures['Signomial_Program'][1] += lhs_final

        if structures['Linear_Program'][0] != False:
            structures['Linear_Program'][2] = operatorList
        if structures['Quadratic_Program'][0] != False:
            structures['Quadratic_Program'][2] = operatorList
        if structures['Geometric_Program'][0] != False:
            structures['Geometric_Program'][2] = operatorList
        if structures['Signomial_Program'][0] != False:
            structures['Signomial_Program'][2] = operatorList

        if structures['Geometric_Program'][0] != False:
            conIxs = [ structures['Geometric_Program'][1][i][0] for i in range(0,len(structures['Geometric_Program'][1])) ]
            unique, counts = numpy.unique(conIxs, return_counts=True)
            countDict = dict(zip(unique, counts))
            for i in range(0,len(operatorList)):
                conIx = i+1
                if operatorList[i] == '==':
                    if countDict[conIx] > 1:
                        structures['Geometric_Program'][0] = False
                        structures['Geometric_Program'][1] = None
                        structures['Geometric_Program'][2] = None
                        break

    structures['info'] = {}
    structures['info']['N_cons_total']    = N_cons
    structures['info']['N_cons_noBounds'] = N_cons - N_bound_cons
    structures['info']['N_cons_bounds']   = N_bound_cons
    # print(structures)
    return structures








