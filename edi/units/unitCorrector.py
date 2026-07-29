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

# from edi.structure.structureWalker import _StructureVisitor
from edi.units.unitWalker import _UnitVisitor

from pyomo.common.dependencies import numpy, numpy_available

if numpy_available:
    import numpy as np
else:
    raise ImportError('The stucture detector requires numpy')

# from edi.structure.detectorSupportFunctions import (
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


class UnitMismatch(RuntimeError):
    """A constraint or objective whose sides do not agree dimensionally."""


def _units_of(expr):
    """The units of a sub-expression, or None if they cannot be determined."""
    try:
        u = pyo.units.get_units(expr)
    except Exception:
        return None
    s = str(u)
    return 'dimensionless' if s in ('None', '', 'dimensionless') else s


def _describe_mismatch(name, cexpr, exc):
    """Say what is wrong, in the terms the author wrote it in.

    The previous message was `Error with constraint: <expression>` -- it named
    the constraint and stopped there, leaving the author to work out which side
    was wrong and by how much. Almost always the answer is a single missing
    factor, and the checker already knows what it is.
    """
    lines = [f"Unit mismatch in {name}:", '', f'    {cexpr}', '']

    left = right = None
    try:
        args = getattr(cexpr, 'args', None)
        if args and len(args) >= 2:
            left, right = _units_of(args[0]), _units_of(args[-1])
    except Exception:
        pass

    if left is None and right is None:
        lines.append('  The two sides could not be reduced to units at all, '
                     'which usually means')
        lines.append('  something in the expression is not a quantity -- a '
                     'bare Python float where')
        lines.append('  a Constant was meant, say.')
    else:
        lines.append(f'  left  side : {left if left else "?"}')
        lines.append(f'  right side : {right if right else "?"}')
        lines.append('')
        if left and right and left != right:
            factor = None
            try:
                a, b = pyo.units.get_units(cexpr.args[0]), \
                    pyo.units.get_units(cexpr.args[-1])
                factor = str(pyo.units.get_units(b / a))
            except Exception:
                factor = None
            if factor and factor not in ('None', ''):
                lines.append(f'  The two differ by a factor of [{factor}]. '
                             'Either multiply the left')
                lines.append(f'  side by [{factor}], or divide the right side '
                             'by it.')
            else:
                lines.append(f'  The left side is [{left}] and the right is '
                             f'[{right}]; they must match.')
        elif left == right:
            lines.append('  The two sides agree, so the failure is elsewhere '
                         'in the expression --')
            lines.append('  most often a term inside it that is itself '
                         'inconsistent.')

    # The raw error carries object reprs and is several hundred characters of
    # pointer addresses; keep the head of it for anyone debugging the walker
    # itself, and no more.
    detail = ' '.join(str(exc).split())
    # Pyomo appends a repr of every offending node, which is pointer addresses
    # and nothing a modeller can act on. The sentence before them is the part
    # worth keeping.
    if '<' in detail:
        detail = detail.split('<', 1)[0].rstrip(' :,(')
    if len(detail) > 140:
        detail = detail[:140] + '...'
    lines += ['', f'  (underlying: {type(exc).__name__}: {detail})']
    return '\n'.join(lines)


def unit_corrector(pyomo_component):
    if not isinstance(pyomo_component, BlockData):
        raise ValueError( "Invalid type %s passed into the convexity detector"%(str(type(pyomo_component))))
    
    corrected_model = pyomo_component.clone()
    # corrected_model.pprint()
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
        try:
            rv = visitor.walk_expression(obj.sense * obj).expr
        except Exception as exc:
            raise UnitMismatch(_describe_mismatch(
                f"objective {obj.name!r}", obj.expr, exc)) from exc

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
            # print('============')
            # print(con.expr)
            # print()
            # Need to extract from pyomo model
            for c in con.values():
                # Extract the expression, which includes the operator
                cexpr = c.expr
                # Walk the expression
                #print(str(cexpr))
                try:
                    rv = visitor.walk_expression(cexpr)
                except Exception as exc:
                    raise UnitMismatch(_describe_mismatch(
                        f"constraint {con.name!r}", cexpr, exc)) from exc

######################## Delete Old and Add Corrected Constraint ########################

                # need to put rv into new pyomo model
                # print(dir(corrected_model.ConstraintList))
                corrected_model.__delattr__(con.name)  # remove existing constraint
                corrected_model.__setattr__(con.name, pyo.Constraint(expr=rv))  # define a new one
                # print(rv)
                #corrected_model.del_component(con.name)
                #corrected_model.add_component(con.name, pyo.Constraint(expr=rv))
                
                
###################################################################################

    # corrected_model._constructed = False
    # corrected_model.construct()

    # print('\n\n\nUnit Corrected Pyomo Objective:\n\n\n')
    # corrected_model.pprint()
    
    return corrected_model