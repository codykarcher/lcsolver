# ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2023
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# import math
# import random
# import copy
# import re
# import io
# import pyomo.environ as pyo
from collections import namedtuple
unitsPack = namedtuple('unitsPack', ['expr','units'])

from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.environ import units, as_quantity

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

# from pyomo.core.expr.visitor import identify_components
# from pyomo.core.expr.base import ExpressionBase
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import ScalarObjective, _GeneralObjectiveData
import pyomo.core.kernel as kernel
# from pyomo.core.expr.template_expr import (
#     GetItemExpression,
#     GetAttrExpression,
#     TemplateSumExpression,
#     IndexTemplate,
#     Numeric_GetItemExpression,
#     templatize_constraint,
#     resolve_template,
#     templatize_rule,
# )
# from pyomo.core.base.var import ScalarVar, _GeneralVarData, IndexedVar, VarData
from pyomo.core.base.var import ScalarVar, IndexedVar, VarData
from pyomo.core.base.param import ParamData, ScalarParam, IndexedParam
# from pyomo.core.base.set import _SetData
# from pyomo.core.base.constraint import ScalarConstraint, IndexedConstraint
# from pyomo.common.collections.component_map import ComponentMap
# from pyomo.common.collections.component_set import ComponentSet
# from pyomo.core.expr.template_expr import (
#     NPV_Numeric_GetItemExpression,
#     NPV_Structural_GetItemExpression,
#     Numeric_GetAttrExpression,
# )
from pyomo.core.expr.numeric_expr import (
    NPV_SumExpression, 
    NPV_DivisionExpression, 
    NPV_ProductExpression,
    NPV_PowExpression,
    NPV_UnaryFunctionExpression,
    NPV_NegationExpression,
    DivisionExpression as NE_DivisionExpression,
) 

# from pyomo.core.base.block import IndexedBlock

from pyomo.core.base.external import _PythonCallbackFunctionID

# from pyomo.core.base.block import _BlockData
# from pyomo.core.base.block import BlockData

from pyomo.repn.util import ExprType

# from pyomo.common import DeveloperError

from pyomo.core.base.units_container import _PyomoUnit

from pyomo.environ import value

_CONSTANT = ExprType.CONSTANT
_MONOMIAL = ExprType.MONOMIAL
_GENERAL = ExprType.GENERAL

from pyomo.common.dependencies import numpy_available
if numpy_available:
    import numpy as np

# from pyomo.core.base.units_container import pint_available
# if pint_available:
#     import pint
#     ureg = pint.UnitRegistry(system='mks')


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
    # unarySignomial,
    no_structure_dict,
    # monomial_multiplication,
    # signomial_multiplication,
    # signomial_fraction_multiplication,
    # signomial_power_evaluation,
    # processMonomial,
)

def _units_match(u1, u2, tol=1e-9):
    """Whether two pint units (already in base units) are the same dimension.

    NOT ``u1 == u2``.  Pint compares the exponent floats exactly, and a
    fractional exponent poisons them: with a Constant in [N/W^0.803] times a
    power [W^0.803], the walker builds ``kg^0.197*m^-0.606*s^0.409`` times
    ``kg^0.803*m^1.606*s^-2.409`` and the seconds exponent comes out
    ``-1.9999999999999998`` -- not ``-2.0``, so ``==`` said the product was
    not newtons.  Worse, pint's *formatter* rounds for display, so the
    resulting report printed ``[N] =/= [N]``.  Exponents are order-ten
    numbers, so an absolute tolerance is enough.
    """
    if u1 == u2:
        return True
    e1 = dict(getattr(u1, '_units', {}))
    e2 = dict(getattr(u2, '_units', {}))
    return all(abs(e1.get(k, 0.0) - e2.get(k, 0.0)) <= tol
               for k in set(e1) | set(e2))


def _is_dimensionless(u, tol=1e-9):
    """Dimensionless within the same tolerance -- a residual ``kg^1e-16``
    left by fractional-exponent arithmetic is still dimensionless."""
    return all(abs(e) <= tol for e in dict(getattr(u, '_units', {})).values())


def handle_var_node(visitor,node):
    var_units = units.get_units(node)
    # K = as_quantity(1.0*var_units).to_base_units().magnitude #correction factor
    K = as_quantity(1.0*var_units).to_base_units() #correction factor
    return unitsPack(expr=K.magnitude*node, units=K.units)

def handle_param_node(visitor, node):
    param_units = units.get_units(node)
    # K = as_quantity(1.0*param_units).to_base_units().magnitude
    K = as_quantity(1.0*param_units).to_base_units()
    return unitsPack(expr=K.magnitude*node, units=K.units)

def handle_num_node(visitor, node):    
    #return node*units.dimensionless
    # print('nd: ', node)
    # print('ndu: ', units.get_units(node))
    return unitsPack(expr=node, units=units.pint_registry('').units)

def handle_negation_node(visitor,node,arg1):
    # WARNING: PYOMO CONVERTS 1 and -1 TO UNITS (replaces value with a unary sign)
    if isinstance(node.args[0],_PyomoUnit): #checks to see if node is a Pyomo unit (for cases like -1*units and 1*units)
        # Read the child, not `node.expr` -- a negation node carries `args` and
        # has no `expr`, so this raised AttributeError for every expression it
        # was meant to handle. It is reached whenever a term's coefficient is
        # exactly 1, because Pyomo folds `1.0*units.m` down to the bare unit
        # and negates that: `a*m - 1.0*m` hits it and `a*m - 1.5*m` does not,
        # which is why the failure looked like it depended on the numbers.
        #
        # Converted to base units like every other leaf, rather than returned
        # in its declared units. `units` here is a pint unit throughout the
        # walker, and the sum node compares those for equality -- handing back
        # a Pyomo units container made every sum containing a negated unit
        # report mismatching units instead.
        K = as_quantity(1.0 * node.args[0]).to_base_units()
        return unitsPack(expr = value(node) * K.magnitude, units = K.units)
    else:
        # Negate the *rebuilt* child, not the original node. Returning `node`
        # here silently discarded every unit conversion performed inside a
        # negated subexpression: in `A*((x - y)**2 - (x - z)**2)` with z in
        # feet and everything else in metres, the first difference was
        # rebuilt as `x - 0.3048*z` but the second, sitting under the
        # negation, came back as the untouched `- (x - z)**2`. The constraint
        # then evaluated with feet read as metres -- no error, just a wrong
        # number. Swapping the two operands used to "fix" it, which is how
        # this was found.
        return unitsPack(expr=-arg1.expr, units=arg1.units)

def handle_sumExpression_node(visitor,node, *args):
    arg_checker = []
    for arg in args:
        arg_checker.append(arg.units)
    if all(_units_match(ag, arg_checker[0]) for ag in arg_checker):
        handled_sum = sum(ag.expr for ag in args)
    else: 
        raise ValueError('Function cannot handle mismatching units in SumNode: %s'%('+'.join(str(arg) for arg in args)))
    return unitsPack(expr=handled_sum, units=arg_checker[0])

def handle_pow_node(visitor, node, arg1, arg2):
    dimensionless = units.pint_registry('').units
    if  ( (isinstance(arg2.expr, int) or isinstance(arg2.expr, float)) and _is_dimensionless(arg2.units) ): #checks to make sure the power is only a number (yay)
        return unitsPack(expr=arg1.expr**arg2.expr, units=arg1.units**arg2.expr)

    if not _is_dimensionless(arg2.units): # units in power is a nono
        raise ValueError('Function handle_pow_node cannot handle units %s in the exponent'%(arg2))

    # A SYMBOLIC exponent: a Constant (mutable Param), or an expression over
    # Constants. This is legal, and it is the only way to get a sensitivity to
    # an exponent -- but only over a dimensionless base. Were the base
    # dimensional, the units of the result would depend on a number a deck can
    # change, so the model would not have fixed units at all; that is refused
    # here rather than left to surface as a pint error deeper down.
    if _is_dimensionless(arg1.units):
        return unitsPack(expr=arg1.expr**arg2.expr, units=dimensionless)

    raise ValueError(
        'Function handle_pow_node cannot raise a quantity with units %s to a '
        'non-constant exponent: the units of the result would depend on the '
        "exponent's value. Make the base dimensionless (divide by a reference "
        'quantity) if the exponent has to stay a Constant.' % (arg1.units,))

def handle_product_node(visitor, node, arg1, arg2): #units * units will probably not create any strange cases
    return unitsPack(expr=arg1.expr * arg2.expr, units=units.pint_registry(str(arg1.units) + '*' +str(arg2.units)).units)

def handle_division_node(visitor, node, arg1, arg2): #same as product
    #if (str(arg1)==str(arg2)): #
    #   return 1
    #else: return arg1/arg2
    return unitsPack(expr=arg1.expr / arg2.expr, units=units.pint_registry(str(arg1.units) + '/(' +str(arg2.units)+')').units) ### Note: will fail case x/x in power (ex: x^(x/x))

def handle_abs_node(visitor, node):
    return unitsPack(expr=abs(node.expr), units=node.units)

def handle_unit_node(visitor, node): 
    K = as_quantity(1.0 * node).to_base_units() # correction factor
    return unitsPack(expr=K.magnitude, units=K.units)

#: What each unary function does to its argument's units. Mirrors Pyomo's own
#: table in ``pyomo.core.base.units_container`` so that LCsolver and
#: ``pyomo.environ.units.get_units`` agree rather than inventing a second
#: convention.
#:
#: ``'dimensionless'``
#:     the argument must be dimensionless and the result is dimensionless.
#:     Covers log/log10/exp, and the trigonometric functions too: their
#:     argument is an angle, and an angle's base unit is the radian, which pint
#:     reports as dimensionless. The inverse functions return radians, again
#:     dimensionless in base units. Since this walker converts everything to
#:     base units before comparing, the one rule covers both directions.
#: ``'same'``
#:     the result carries the argument's units (ceil, floor).
#: ``'sqrt'``
#:     the result carries the argument's units to the one-half power.
_UNARY_UNITS = {
    'log': 'dimensionless', 'log10': 'dimensionless', 'exp': 'dimensionless',
    'sin': 'dimensionless', 'cos': 'dimensionless', 'tan': 'dimensionless',
    'sinh': 'dimensionless', 'cosh': 'dimensionless', 'tanh': 'dimensionless',
    'asin': 'dimensionless', 'acos': 'dimensionless', 'atan': 'dimensionless',
    'asinh': 'dimensionless', 'acosh': 'dimensionless',
    'atanh': 'dimensionless',
    'ceil': 'same', 'floor': 'same',
    'sqrt': 'sqrt',
}


def handle_unary_node(visitor, node, arg1):
    """``sin(x)``, ``exp(x)``, ``sqrt(x)`` and friends.

    This used to call ``units.get_units(arg1)`` unconditionally on its way in.
    ``arg1`` is a :data:`unitsPack`, not a Pyomo expression, so that raised
    ``AttributeError: 'unitsPack' object has no attribute
    'is_expression_type'`` for *every* unary function -- including ``sqrt``,
    which the branch below was written to support. The AttributeError then
    surfaced through the unit reporter as a mismatch whose two sides agreed,
    which is a contradiction on the face of it.

    ``x ** 0.5`` was unaffected, being a power node rather than a unary one,
    which is why models that spell their roots that way never hit this.
    """
    fcn_handle = node.getname()
    rule = _UNARY_UNITS.get(fcn_handle)
    if rule is None:
        raise ValueError(
            'unit checking does not know the function %r. Known functions: %s'
            % (fcn_handle, ', '.join(sorted(_UNARY_UNITS))))

    if rule == 'sqrt':
        return handle_pow_node(
            visitor, node, arg1,
            unitsPack(expr=0.5, units=units.pint_registry('').units))

    if rule == 'same':
        return unitsPack(expr=node.create_node_with_local_data((arg1.expr,)),
                         units=arg1.units)

    if not _is_dimensionless(arg1.units):
        raise ValueError(
            '%s() requires a dimensionless argument, but its argument has '
            'units of [%s]. Divide it by a reference quantity in those units '
            'first.' % (fcn_handle, arg1.units))
    return unitsPack(expr=node.create_node_with_local_data((arg1.expr,)),
                     units=units.pint_registry('').units)

def handle_monomialTermExpression_node(visitor, node, arg1, arg2): #?
    return handle_product_node(visitor,node,arg1,arg2)

def handle_named_expression_node(visitor, node, arg1):
    # needed to preserve consistencency with the exitNode function call
    # prevents the need to type check in the exitNode function
    return arg1

def handle_exprif_node(visitor, node, arg1, arg2, arg3): #?
    # has no structure
    return no_structure_dict()

def handle_external_function_node(visitor, node, *args): #?
    # has no structure
    return no_structure_dict()

def handle_functionID_node(visitor, node, *args): #?
    # seems to just be a placeholder empty wrapper object
    return handle_external_function_node(visitor, node, *args)

def handle_equality_node(visitor, node, arg1, arg2):
    if not _units_match(arg1.units, arg2.units):
        raise ValueError('Function cannot handle mismatching units in EqualityNode: %s == %s'%(str(arg1),str(arg2)))

    LHS = arg1.expr
    if hasattr(LHS, 'units'): # if it has units, strip LHS of units
        LHS = arg1.expr.magnitude
    else: 
        pass

    RHS = arg2.expr
    if hasattr(RHS, 'units'):
        RHS = arg2.expr.magnitude # if it has units, strip RHS of units
    else: 
        pass

    # converts everything to a pyomo object (everything is in base units at this point)
    LHS = LHS*units.dimensionless
    RHS = RHS*units.dimensionless
    #rewrite constraint
    expr1 = LHS == RHS
    return expr1

def handle_inequality_node(visitor, node, arg1, arg2):
    if not _units_match(arg1.units, arg2.units):
        raise ValueError('Function cannot handle mismatching units in InequalityNode: %s >= %s'%(str(arg1),str(arg2)))

    LHS = arg1.expr
    RHS = arg2.expr
    
    if hasattr(LHS, 'units'): # if it has units, strip LHS of units
        LHS = arg1.expr.magnitude
    else: 
        pass

    if hasattr(RHS, 'units'):
        RHS = arg2.expr.magnitude # if it has units, strip RHS of units
    else: 
        pass
    
    # converts everything to a pyomo object (everything is in base units at this point)
    LHS = LHS*units.dimensionless
    RHS = RHS*units.dimensionless
    
    #rewrite constraint
    expr1 = LHS <= RHS
    return expr1

def handle_ranged_inequality_node(visitor, node, arg1, arg2, arg3):
    if not (_units_match(arg1.units, arg2.units)
            and _units_match(arg1.units, arg3.units)):
        raise ValueError('Function cannot handle mismatching units in RangedInequalityNode: %s <= %s <= %s'%(str(arg1),str(arg2),str(arg3)))

    LHS = arg1.expr
    if hasattr(LHS, 'units'): # if it has units, strip LHS of units
        LHS = arg1.expr.magnitude
    else: 
        pass

    MID = arg2.expr
    if hasattr(MID, 'units'):
        MID = arg2.expr.magnitude # if it has units, strip MID of units
    else: 
        pass

    RHS = arg3.expr
    if hasattr(RHS, 'units'): # if it has units, strip RHS of units
        RHS = arg3.expr.magnitude # if it has units, strip RHS of units
    else: 
        pass

    expr1 = LHS <= MID
    expr2 = MID <= RHS

    return [expr1,expr2]

class _UnitVisitor(StreamBasedExpressionVisitor):
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
            NPV_UnaryFunctionExpression: handle_unary_node,
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
            VarData: handle_var_node,
            _PyomoUnit: handle_unit_node,
            # ScalarConstraint: handle_constraint_node,
            # IndexedConstraint: handle_indexedconstraint_node,
        }
        if numpy_available:
            self._operator_handles[np.float64] = handle_num_node
            self._operator_handles[np.int64] = handle_num_node

    def exitNode(self, node, data):
        handles = self._operator_handles
        handler = handles.get(node.__class__)
        if handler is None:
            # A subclass of a known component (LCScalarVar, LCScalarParam, a
            # user's Var subclass) dispatches to its nearest base class's
            # handler, cached so the walk stays one dict lookup per node.
            for klass in node.__class__.__mro__:
                handler = handles.get(klass)
                if handler is not None:
                    handles[node.__class__] = handler
                    break
            else:
                raise KeyError(node.__class__)
        return handler(self, node, *data)