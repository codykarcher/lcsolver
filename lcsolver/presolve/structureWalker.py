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
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor

from pyomo.core.expr import (
    NegationExpression,
    ProductExpression,
    DivisionExpression,
    PowExpression,
    AbsExpression,
    UnaryFunctionExpression,
    MonomialTermExpression,
    LinearExpression,
    SumExpression,
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
    Expr_ifExpression,
    ExternalFunctionExpression,
)

from pyomo.core.expr.visitor import identify_components
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import ScalarObjective, _GeneralObjectiveData
import pyomo.core.kernel as kernel
from pyomo.core.expr.template_expr import (
    GetItemExpression,
    GetAttrExpression,
    TemplateSumExpression,
    IndexTemplate,
    Numeric_GetItemExpression,
    templatize_constraint,
    resolve_template,
    templatize_rule,
)
# from pyomo.core.base.var import ScalarVar, _GeneralVarData, IndexedVar, VarData
from pyomo.core.base.var import ScalarVar, IndexedVar, VarData
from pyomo.core.base.param import ParamData, ScalarParam, IndexedParam
from pyomo.core.base.set import _SetData
from pyomo.core.base.constraint import ScalarConstraint, IndexedConstraint
from pyomo.common.collections.component_map import ComponentMap
from pyomo.common.collections.component_set import ComponentSet
from pyomo.core.expr.template_expr import (
    NPV_Numeric_GetItemExpression,
    NPV_Structural_GetItemExpression,
    Numeric_GetAttrExpression,
)
from pyomo.core.expr.numeric_expr import (
    NPV_SumExpression, 
    NPV_DivisionExpression, 
    NPV_ProductExpression,
    NPV_PowExpression,
    NPV_NegationExpression,
    PowExpression as NE_PowExpression,
    DivisionExpression as NE_DivisionExpression,
) 

from pyomo.core.base.block import IndexedBlock

from pyomo.core.base.external import _PythonCallbackFunctionID

# from pyomo.core.base.block import _BlockData
from pyomo.core.base.block import BlockData

from pyomo.repn.util import ExprType

from pyomo.common import DeveloperError

from pyomo.core.base.units_container import _PyomoUnit

_CONSTANT = ExprType.CONSTANT
_MONOMIAL = ExprType.MONOMIAL
_GENERAL = ExprType.GENERAL

from pyomo.common.dependencies import numpy, numpy_available
if numpy_available:
    import numpy as np

# from lcsolver.presolve.walkerSupportFunctions import (
#     unarySignomial,
#     no_structure_dict,
#     monomial_multiplication,
#     signomial_multiplication,
#     signomial_fraction_multiplication,
#     signomial_power_evaluation,
#     # processMonomial,
# )
from lcsolver.presolve.walkerSupportFunctions import (
    no_structure_dict,
    monomial_multiplication,
    signomial_multiplication,
    signomial_fraction_multiplication,
    signomial_power_evaluation,
    StructureDictionary,
    # processMonomial,
)


def handle_sumExpression_node(visitor, node, *args):
    # print('handling node handle_sumExpression_node(visitor, node, *args):')
    if any([ a['signomial_fraction']['status']!='yes' for a in args]):
        return no_structure_dict()

    if any([ a['signomial']['status']!='yes' for a in args]):
        # at least one signomial fraction
 
        signomial_denominator = handle_num_node(visitor, 1.0)
        for a in args:
            a_denom = StructureDictionary()
            a_denom['signomial']['status'] = 'yes'
            a_denom['signomial']['leadingCoefficients'] = a['signomial_fraction']['denominator']['leadingCoefficients']
            a_denom['signomial']['bases']               = a['signomial_fraction']['denominator']['bases']
            a_denom['signomial']['exponents']           = a['signomial_fraction']['denominator']['exponents']
            a_denom.propagate()
            signomial_denominator = signomial_multiplication(signomial_denominator, a_denom)

        signomial_numerator = StructureDictionary()
        signomial_numerator['signomial']['status'] = 'yes'
        signomial_numerator['signomial']['leadingCoefficients'] = args[0]['signomial_fraction']['denominator']['leadingCoefficients']
        signomial_numerator['signomial']['bases']               = args[0]['signomial_fraction']['denominator']['bases']
        signomial_numerator['signomial']['exponents']           = args[0]['signomial_fraction']['denominator']['exponents']
        for a in args[1:]:
            a_denom = StructureDictionary()
            a_denom['signomial']['status'] = 'yes'
            a_denom['signomial']['leadingCoefficients'] = a['signomial_fraction']['denominator']['leadingCoefficients']
            a_denom['signomial']['bases']               = a['signomial_fraction']['denominator']['bases']
            a_denom['signomial']['exponents']           = a['signomial_fraction']['denominator']['exponents']
            a_denom.propagate()
            signomial_numerator = signomial_multiplication(signomial_numerator,a_denom)
        
        for i in range(1,len(args)):
            # sn_temp = args[i]
            sn_temp = StructureDictionary()
            sn_temp['signomial']['status'] = 'yes'
            sn_temp['signomial']['leadingCoefficients'] = args[i]['signomial_fraction']['numerator']['leadingCoefficients']
            sn_temp['signomial']['bases']               = args[i]['signomial_fraction']['numerator']['bases']
            sn_temp['signomial']['exponents']           = args[i]['signomial_fraction']['numerator']['exponents']
            sn_temp.propagate()
            for j in range(0,len(args)):
                if j != i:
                    aj = StructureDictionary()
                    aj['signomial']['status'] = 'yes'
                    aj['signomial']['leadingCoefficients'] = args[j]['signomial_fraction']['denominator']['leadingCoefficients']
                    aj['signomial']['bases']               = args[j]['signomial_fraction']['denominator']['bases']
                    aj['signomial']['exponents']           = args[j]['signomial_fraction']['denominator']['exponents']
                    aj.propagate()
                    sn_temp = signomial_multiplication(sn_temp, aj)

            signomial_numerator = handle_sumExpression_node(visitor, node, signomial_numerator, sn_temp)

        nsd = StructureDictionary()
        nsd['signomial_fraction']['status'] = 'yes'
        nsd['signomial_fraction']['numerator']['leadingCoefficients']   = signomial_numerator['signomial']['leadingCoefficients']
        nsd['signomial_fraction']['numerator']['bases']                 = signomial_numerator['signomial']['bases']
        nsd['signomial_fraction']['numerator']['exponents']             = signomial_numerator['signomial']['exponents']
        nsd['signomial_fraction']['denominator']['leadingCoefficients'] = signomial_denominator['signomial']['leadingCoefficients']
        nsd['signomial_fraction']['denominator']['bases']               = signomial_denominator['signomial']['bases']
        nsd['signomial_fraction']['denominator']['exponents']           = signomial_denominator['signomial']['exponents']
        nsd.propagate()

        return nsd

    if  any([ a['monomial']['status']!='yes' for a in args]) or any([ a['constant']['status']!='yes' for a in args]):
        # at least one monomial

        signomialDict = StructureDictionary()
        signomialDict['signomial']['status'] = 'yes'
        signomialDict['signomial']['leadingCoefficients'] = args[0]['signomial']['leadingCoefficients']
        signomialDict['signomial']['bases']               = args[0]['signomial']['bases']
        signomialDict['signomial']['exponents']           = args[0]['signomial']['exponents']
        signomialDict.propagate()

        for a_all in args[1:]:
            a = StructureDictionary()
            a['signomial']['status'] = 'yes'
            a['signomial']['leadingCoefficients'] = a_all['signomial']['leadingCoefficients']
            a['signomial']['bases']               = a_all['signomial']['bases']
            a['signomial']['exponents']           = a_all['signomial']['exponents']
            a.propagate()

            signomialDict['signomial']['leadingCoefficients'] += a['signomial']['leadingCoefficients']
            signomialDict['signomial']['bases']               += a['signomial']['bases']
            signomialDict['signomial']['exponents']           += a['signomial']['exponents']

        # should call simplify here, but there are significant issues with this function
        # signomialDict.simplify()

        signomialDict.propagate()
        return signomialDict 

    # is still constant
    vl = sum([a['constant']['value'] for a in args])
    return handle_num_node(visitor,float(vl))

def handle_product_node(visitor, node, arg1, arg2):
    # print('handling node handle_product_node(visitor, node, arg1, arg2):')
    if arg1["constant"]["status"] == "yes" and arg2["constant"]["status"] == "yes":
        vl = float(arg1['constant']['value']*arg2['constant']['value'])
        return handle_num_node(visitor,vl)

    if arg1["monomial"]["status"] == "yes" and arg2["constant"]["status"] == "yes":
        # swap and delay
        arg2_temp = arg2
        arg2 = arg1
        arg1 = arg2_temp

    if ( (arg1["constant"]["status"] == "yes" and arg2["monomial"]["status"] == "yes") or 
         (arg1["monomial"]["status"] == "yes" and arg2["monomial"]["status"] == "yes") ):

        # print('mon')
        return monomial_multiplication(arg1,arg2)

    if ( (arg1["signomial"]["status"] == "yes" and arg2["constant"]["status"] == "yes") or
         (arg1["signomial"]["status"] == "yes" and arg2["monomial"]["status"] == "yes") ):
        # swap and delay
        arg2_temp = arg2
        arg2 = arg1
        arg1 = arg2_temp

    if ( (arg1["constant"]["status"]  == "yes" and arg2["signomial"]["status"] == "yes") or
         (arg1["monomial"]["status"]  == "yes" and arg2["signomial"]["status"] == "yes") or 
         (arg1["signomial"]["status"] == "yes" and arg2["signomial"]["status"] == "yes") ):

        # print('sig')
        return signomial_multiplication(arg1, arg2)

    if ( (arg1["signomial_fraction"]["status"] == "yes" and arg2["constant"]["status"] == "yes") or 
         (arg1["signomial_fraction"]["status"] == "yes" and arg2["monomial"]["status"] == "yes") or
         (arg1["signomial_fraction"]["status"] == "yes" and arg2["signomial"]["status"] == "yes") ):
        # swap and delay
        arg2_temp = arg2
        arg2 = arg1
        arg1 = arg2_temp

    if ( (arg1["constant"]["status"]           == "yes" and arg2["signomial_fraction"]["status"] == "yes") or
         (arg1["monomial"]["status"]           == "yes" and arg2["signomial_fraction"]["status"] == "yes") or 
         (arg1["signomial"]["status"]          == "yes" and arg2["signomial_fraction"]["status"] == "yes") or 
         (arg1["signomial_fraction"]["status"] == "yes" and arg2["signomial_fraction"]["status"] == "yes") ):

        # print('sigfrac')
        return signomial_fraction_multiplication(arg1, arg2)

    return no_structure_dict()

def handle_division_node(visitor, node, arg1, arg2):
    # print('handling node handle_division_node(visitor, node, arg1, arg2):')
    if arg2["constant"]["status"] == "yes":
        arg2['constant']['value'] = 1/arg2['constant']['value']
        arg2.propagate()
        return handle_product_node(visitor, node, arg1, arg2)
    if arg2["monomial"]["status"] == "yes":
        arg2['monomial']['leadingConstant'] = 1/arg2['monomial']['leadingConstant']
        arg2['monomial']['exponents'] = [ -c for c in arg2['monomial']['exponents'] ]
        arg2.propagate()
        return handle_product_node(visitor, node, arg1, arg2)
    if arg2["signomial"]["status"] == "yes":
        a2 = no_structure_dict()
        a2['signomial_fraction']['status'] = 'yes'  
        a2['signomial_fraction']['numerator']['leadingCoefficients'] = [1.0]
        a2['signomial_fraction']['numerator']['bases'] = [[]]
        a2['signomial_fraction']['numerator']['exponents'] = [[]]
        a2['signomial_fraction']['denominator']['leadingCoefficients'] = arg2["signomial"]['leadingCoefficients']
        a2['signomial_fraction']['denominator']['bases'] = arg2["signomial"]['bases']
        a2['signomial_fraction']['denominator']['exponents'] = arg2["signomial"]['exponents']
        a2.propagate()
        return handle_product_node(visitor, node, arg1, a2)

    return no_structure_dict()

def handle_pow_node(visitor, node, arg1, arg2):
    # print('handling node handle_pow_node(visitor, node, arg1, arg2):')
    if arg2["constant"]["value"] is None:
        return no_structure_dict()
    elif arg2["constant"]["status"]=='no' and arg2["constant"]["value"] is not None:
        # is a param, substitute value first by setting to value
        arg2 = handle_num_node(visitor,arg2["constant"]["value"])
    elif arg2["constant"]["status"] == "yes":
        arg2 = handle_num_node(visitor,arg2["constant"]["value"])
    else:
        raise ValueError("Power node requires a constant value for exponentiation")

    expVal = arg2["constant"]["value"]
    if arg1["constant"]["status"] == "yes":
        arg1["constant"]["value"] = arg1["constant"]["value"]**expVal
        arg1.propagate()
        return arg1
    if arg1["monomial"]["status"] == "yes":
        arg1["monomial"]["leadingConstant"] = arg1["monomial"]["leadingConstant"]**expVal
        arg1["monomial"]["exponents"] = [ vl * expVal for vl in arg1["monomial"]["exponents"] ]
        arg1.propagate()
        return arg1
    if arg1["signomial"]["status"] == "yes":
        if float(expVal).is_integer():
            rv = signomial_power_evaluation(arg1,expVal)
            rv.propagate()
            return rv
        else:
            return no_structure_dict()   
    if arg1["signomial_fraction"]["status"] == "yes":
        if float(expVal).is_integer():
            num = StructureDictionary()
            num['signomial']['status'] = 'yes'
            num['signomial']['leadingCoefficients'] = arg1["signomial_fraction"]['numerator']["leadingCoefficients"]
            num['signomial']['bases'] = arg1["signomial_fraction"]['numerator']["bases"]
            num['signomial']['exponents'] = arg1["signomial_fraction"]['numerator']["exponents"]
            num.propagate()
            num_exp = signomial_power_evaluation(num, expVal)

            dem = StructureDictionary()
            dem['signomial']['status'] = 'yes'
            dem['signomial']['leadingCoefficients'] = arg1["signomial_fraction"]['denominator']["leadingCoefficients"]
            dem['signomial']['bases'] = arg1["signomial_fraction"]['denominator']["bases"]
            dem['signomial']['exponents'] = arg1["signomial_fraction"]['denominator']["exponents"]
            dem.propagate()
            dem_exp = signomial_power_evaluation(dem, expVal)

            elementDict = no_structure_dict()
            elementDict['signomial_fraction']['status'] = 'yes'
            elementDict['signomial_fraction']['numerator']['leadingCoefficients']   = num_exp['leadingCoefficients']
            elementDict['signomial_fraction']['numerator']['bases']                 = num_exp['bases']
            elementDict['signomial_fraction']['numerator']['exponents']             = num_exp['exponents']
            elementDict['signomial_fraction']['denominator']['leadingCoefficients'] = dem_exp['leadingCoefficients']
            elementDict['signomial_fraction']['denominator']['bases']               = dem_exp['bases']
            elementDict['signomial_fraction']['denominator']['exponents']           = dem_exp['exponents']
            elementDict.propagate()
            return elementDict

        else:
            return no_structure_dict()   

    return no_structure_dict()      

def handle_negation_node(visitor, node, arg1):
    # print('handling node handle_negation_node(visitor, node, arg1):')
    arg2 = handle_num_node(visitor, -1.0)
    return handle_product_node(visitor,node,arg2,arg1)

def handle_var_node(visitor, node):
    # print('handling node handle_var_node(visitor, node):')
    # TODO: should exempt the case of a fixed variable
    elementDict = StructureDictionary()
    elementDict['monomial']['status'] = 'yes'
    elementDict['monomial']['leadingConstant'] = 1.0
    elementDict['monomial']['bases'] = [node]
    elementDict['monomial']['exponents'] = [1.0]
    elementDict.propagate()
    return elementDict

def handle_param_node(visitor, node):
    # print('handling node handle_param_node(visitor, node):')
    elementDict = StructureDictionary()
    elementDict['constant']['value'] = node
    elementDict['monomial']['status'] = 'yes'
    elementDict['monomial']['leadingConstant'] = 1.0
    elementDict['monomial']['bases'] = [node.value]
    elementDict['monomial']['exponents'] = [1.0]
    elementDict.propagate()
    return elementDict

def handle_unary_node(visitor, node, arg1):
    # print('handling node handle_unary_node(visitor, node, arg1):')
    fcn_handle = node.getname()
    if fcn_handle == 'sqrt':
        arg2 = handle_num_node(visitor, 0.5)
        return handle_pow_node(visitor, node, arg1, arg2)
    else:
        # has no structure
        return no_structure_dict()

def handle_abs_node(visitor, node, arg1):
    # print('handling node handle_abs_node(visitor, node, arg1):')
    if arg1['constant']['status']=='yes':
        vl = abs(arg1['constant']['value'])
        return handle_num_node(visitor,float(vl))
    else:
        # has no structure
        return no_structure_dict()  

def handle_num_node(visitor, node):
    # print('handling node handle_num_node(visitor, node):')
    elementDict = StructureDictionary()
    elementDict['constant']['status'] = 'yes'
    # pyo.value resolves a mutable Param (or an expression over Params) to its
    # number; a plain float passes through. Bare float() raises on a Pyomo
    # object, which is how a Constant used as an EXPONENT -- legal, and the
    # only way to get a sensitivity to an exponent -- used to reach the
    # walker's generic error handler.
    elementDict['constant']['value'] = float(pyo.value(node))
    elementDict.propagate()
    return elementDict

def handle_monomialTermExpression_node(visitor, node, arg1, arg2):
    # print('handling node handle_monomialTermExpression_node(visitor, node, arg1, arg2):')
    return handle_product_node(visitor,node,arg1,arg2)

def handle_named_expression_node(visitor, node, arg1):
    # print('handling node handle_named_expression_node(visitor, node, arg1):')
    # needed to preserve consistency with the exitNode function call
    # prevents the need to type check in the exitNode function
    return arg1

def handle_exprif_node(visitor, node, arg1, arg2, arg3):
    # print('handling node handle_exprif_node(visitor, node, arg1, arg2, arg3):')
    # has no structure
    return no_structure_dict()

def handle_external_function_node(visitor, node, *args):
    # print('handling node handle_external_function_node(visitor, node, *args):')
    # has no structure
    return no_structure_dict()

def handle_functionID_node(visitor, node, *args):
    # print('handling node handle_functionID_node(visitor, node, *args):')
    # seems to just be a placeholder empty wrapper object
    return handle_external_function_node(visitor, node, *args)

def handle_equality_node(visitor, node, arg1, arg2):
    # print('handling node handle_equality_node(visitor, node, arg1, arg2):')
    return [ {'lhs':arg1, 'operator':'==', 'rhs':arg2} ]

def handle_inequality_node(visitor, node, arg1, arg2):
    # print('handling node handle_inequality_node(visitor, node, arg1, arg2):')
    return [ {'lhs':arg1, 'operator':'<=', 'rhs':arg2} ]

def handle_ranged_inequality_node(visitor, node, arg1, arg2, arg3):
    # print('handling node handle_ranged_inequality_node(visitor, node, arg1, arg2, arg3):')
    return [ {'lhs':arg1, 'operator':'<=', 'rhs':arg2},
             {'lhs':arg2, 'operator':'<=', 'rhs':arg3}  ]

# TODO: fix this
def handle_unit_node(visitor, node):
    # print('handling node handle_unit_node(visitor, node):')
    elementDict = StructureDictionary()
    elementDict['constant']['status'] = 'yes'
    elementDict['constant']['value'] = 1.0
    elementDict.propagate()
    return elementDict

class _StructureVisitor(StreamBasedExpressionVisitor):
    def __init__(self):
        super().__init__()

        self._operator_handles = {
            ScalarVar: handle_var_node,
            int: handle_num_node,
            float: handle_num_node,
            SumExpression: handle_sumExpression_node,
            NegationExpression: handle_negation_node,
            ProductExpression: handle_product_node,
            DivisionExpression: handle_division_node,
            PowExpression: handle_pow_node,
            AbsExpression: handle_abs_node,
            UnaryFunctionExpression: handle_unary_node,
            Expr_ifExpression: handle_exprif_node,
            EqualityExpression: handle_equality_node,
            InequalityExpression: handle_inequality_node,
            RangedExpression: handle_ranged_inequality_node,
            _GeneralExpressionData: handle_named_expression_node,
            ScalarExpression: handle_named_expression_node,
            kernel.expression.expression: handle_named_expression_node,
            kernel.expression.noclone: handle_named_expression_node,
            _GeneralObjectiveData: handle_named_expression_node,
            # _GeneralVarData: handle_var_node,
            ScalarObjective: handle_named_expression_node,
            kernel.objective.objective: handle_named_expression_node,
            ExternalFunctionExpression: handle_external_function_node,
            _PythonCallbackFunctionID: handle_functionID_node,
            LinearExpression: handle_sumExpression_node,
            MonomialTermExpression: handle_monomialTermExpression_node,
            IndexedVar: handle_var_node,
            ScalarParam: handle_param_node,
            ParamData: handle_param_node,
            IndexedParam: handle_param_node,
            NPV_SumExpression: handle_sumExpression_node,
            NPV_DivisionExpression: handle_division_node,
            NPV_ProductExpression: handle_product_node,
            NPV_PowExpression: handle_pow_node,
            NPV_NegationExpression: handle_negation_node,
            NE_DivisionExpression: handle_division_node,
            NE_PowExpression: handle_pow_node,
            VarData: handle_var_node,
            _PyomoUnit: handle_unit_node,
        }
        if numpy_available:
            self._operator_handles[np.float64] = handle_num_node
            self._operator_handles[np.int64] = handle_num_node

    def exitNode(self, node, data):
        try:
            handles = self._operator_handles
            handler = handles.get(node.__class__)
            if handler is None:
                # A subclass of a known component (LCScalarVar, LCScalarParam,
                # a user's Var subclass) dispatches to its nearest base class's
                # handler, cached so the walk stays one dict lookup per node.
                for klass in node.__class__.__mro__:
                    handler = handles.get(klass)
                    if handler is not None:
                        handles[node.__class__] = handler
                        break
                else:
                    raise KeyError(node.__class__)
            return handler(self, node, *data)
        except:
            raise RuntimeError(
                'Structure walker encountered an error when processing type %s, contact the LCsolver developers'
                % (node.__class__)
            )
