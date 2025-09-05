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

from pyomo.environ import Var ##########

from pyomo.core.base.block import BlockData
from pyomo.common.collections.component_map import ComponentMap
from pyomo.core.base.var import ScalarVar, VarData, IndexedVar
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.numeric_types import RegisterNumericType
RegisterNumericType(pyomo.common.enums.ObjectiveSense)

# from pyomo.contrib.edi.tools.structureWalker import _StructureVisitor
from edi.units.unitWalker import _UnitVisitor

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

def unit_corrector(pyomo_component):
    if not isinstance(pyomo_component, BlockData):
        raise ValueError( "Invalid type %s passed into the convexity detector"%(str(type(pyomo_component))))
    
    corrected_model = pyomo_component.clone()
    corrected_model.pprint()
    # get all the variables ### WTF AM I DOING#######################################################
    #variableList = [ vr for vr in  corrected_model.component_objects(pyo.Var, descend_into=True, active=True) ]
    #print(str(variableList[0]))
    #corrected_model.variableList = variableList
    #corrected_model.variableList = Var()

    # JK :D variables not needed if copied over from pyomo clone
 
    ###########################################################################################    

    # get all the objectives
    objectives    = [ obj for obj in corrected_model.component_data_objects(pyo.Objective , descend_into=True, active=True ) ]
    # get all the constraints
    constraints   = [ con for con in corrected_model.component_objects(     pyo.Constraint, descend_into=True, active=True ) ]

    #corrected_model.clear() #clears out corrected_model
    visitor = _UnitVisitor()



    for obj in objectives:
        # Walk the expression, returns the full breakdown of the constraint in dictionary form
        rv = visitor.walk_expression(obj.sense * obj)

########################## Delete Old and Add Corrected Objective ##########################

        # need to put rv into new pyomo model
        corrected_model.__delattr__(obj.name)  # remove existing constraint ### FIX THIS
        corrected_model.__setattr__(obj.name, pyo.Objective(expr=rv))  # define a new one ### FIX THIS

#######################################################################################

    # check that there are constraints
    if len(constraints) > 0:
        # print(structures)
        # Iterate over the constraints
        for i, con in enumerate(constraints):
            # Need to extract from pyomo model
            for c in con.values():
                # Extract the expression, which includes the operator
                cexpr = c.expr
                # Walk the expression
                #print(str(cexpr))
                rv = visitor.walk_expression(cexpr)

######################## Delete Old and Add Corrected Constraint ########################

                # need to put rv into new pyomo model
                #print(dir(corrected_model.ConstraintList))
                corrected_model.__delattr__(con.name)  # remove existing constraint
                corrected_model.__setattr__(con.name, pyo.Constraint(expr=rv))  # define a new one
                #print(rv)
                #corrected_model.del_component(con.name)
                #corrected_model.add_component(con.name, pyo.Constraint(expr=rv))
                
                
###################################################################################

    #corrected_model._constructed = False
    #corrected_model.construct()

    print('\n\n\nUnit Corrected Pyomo Objective:\n\n\n')
    corrected_model.pprint()
    
    return corrected_model