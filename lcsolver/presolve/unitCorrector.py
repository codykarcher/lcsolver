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

# from lcsolver.presolve.structureWalker import _StructureVisitor
from lcsolver.presolve.unitWalker import _UnitVisitor

from pyomo.common.dependencies import numpy, numpy_available

if numpy_available:
    import numpy as np
else:
    raise ImportError('The stucture detector requires numpy')

# from lcsolver.presolve.detectorSupportFunctions import (
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

    The old message named the constraint and stopped; the answer is almost
    always a single missing factor, and the checker already knows it.
    """
    lines = [f'Error in units for {name}:', '', f'    {cexpr}', '']

    left = right = None
    try:
        args = getattr(cexpr, 'args', None)
        if args and len(args) >= 2:
            left, right = _units_of(args[0]), _units_of(args[-1])
    except Exception:
        pass

    if left is None and right is None:
        lines.append('    the two sides do not reduce to units at all')
        lines.append('')
        lines.append('  Usually something in the expression is not a quantity '
                     '-- a bare Python')
        lines.append('  float where a Constant was meant, say.')
    else:
        lines.append(f'    [{left or "?"}]  =/=  [{right or "?"}]')
        lines.append('')
        if left and right and left != right:
            # Pyomo converts freely within a dimension, so reaching here means
            # different dimensions; the ratio is what the short side is missing
            factor = None
            try:
                a, b = pyo.units.get_units(cexpr.args[0]), \
                    pyo.units.get_units(cexpr.args[-1])
                factor = str(pyo.units.get_units(b / a))
            except Exception:
                factor = None
            lines.append('  These are different dimensions, so no conversion '
                         'between them exists.')
            if factor and factor not in ('None', ''):
                lines.append(f'  The left side is short by [{factor}]: either '
                             f'multiply the left side')
                lines.append(f'  by [{factor}], or divide the right side by it.')
            else:
                lines.append(f'  The left side is [{left}] and the right is '
                             f'[{right}]; they must match.')
        elif left == right:
            lines.append('  The two sides agree, so the failure is inside the '
                         'expression rather')
            lines.append('  than between its sides -- most often a sum whose '
                         'terms disagree.')

    # keep only the head of the raw error, for anyone debugging the walker
    detail = ' '.join(str(exc).split())
    # Pyomo appends reprs of every offending node (pointer addresses); drop them
    if '<' in detail:
        detail = detail.split('<', 1)[0]
    # cutting at the repr can land mid-argument-list; back up to the sentence end
    if detail.count('(') > detail.count(')'):
        detail = (detail.rsplit(':', 1)[0] if ':' in detail
                  else detail.split('(', 1)[0])
    detail = detail.rstrip(' :,(')
    if len(detail) > 140:
        detail = detail[:140] + '...'
    lines += ['', f'  (underlying: {type(exc).__name__}: {detail})']
    return '\n'.join(lines)


def _join_failures(failures):
    """Every mismatch in one message, in the order they were written.

    The walk carries on past a failure and reports the lot; one error per
    round trip is expensive and the same missing conversion usually repeats.
    """
    if len(failures) == 1:
        return failures[0]
    rule = '\n' + '-' * 70 + '\n'
    return (f'{len(failures)} unit errors:\n' + rule.join(failures)
            + '\n' + '-' * 70)


def unit_corrector(pyomo_component):
    if not isinstance(pyomo_component, BlockData):
        raise ValueError( "Invalid type %s passed into the convexity detector"%(str(type(pyomo_component))))

    # IDEMPOTENCY: correcting twice is the identity, but the second walk used
    # to crash on detector-added bound rows (bracketed .name differs from the
    # attribute key), breaking sensitivities() on corrected models. Return a
    # fresh clone to keep the fresh-copy contract every caller holds.
    if getattr(pyomo_component, '_lc_unit_corrected', False):
        return pyomo_component.clone()

    corrected_model = pyomo_component.clone()
    # Stamp the clone with the identity of what it was cloned FROM: structures
    # from a DIFFERENT formulation handed back to solve() would write that
    # model's numbers onto this one with no error. solve() compares the token.
    token = getattr(pyomo_component, '_edi_identity', None)
    if token is None:
        import uuid
        token = uuid.uuid4().hex
        try:
            pyomo_component._edi_identity = token
        except Exception:
            pass
    try:
        corrected_model._edi_source_identity = token
        # ... and WHICH REVISION: a later load_constants makes the clone
        # stale, and solving a stale clone answers the previous deck
        corrected_model._edi_source_revision = getattr(
            pyomo_component, '_edi_revision', 0)
    except Exception:
        pass
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

    # every mismatch found, reported together at the end
    failures = []



    for obj in objectives:
        # Walk the expression, returns the full breakdown of the constraint in dictionary form
        try:
            rv = visitor.walk_expression(obj.sense * obj).expr
        except Exception as exc:
            failures.append(_describe_mismatch(
                f"objective {obj.name!r}", obj.expr, exc))
            continue

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
                    failures.append(_describe_mismatch(
                        f"constraint {con.name!r}", cexpr, exc))
                    continue

######################## Delete Old and Add Corrected Constraint ########################

                # need to put rv into new pyomo model
                # replace via the component HANDLE and its storage key: .name
                # renders quoted for bracketed names (FS_M[0]_lowerBound) and
                # __delattr__ on the rendered form can't find the attribute
                _key = con.local_name
                corrected_model.del_component(con)  # remove existing constraint
                corrected_model.add_component(_key, pyo.Constraint(expr=rv))  # define a new one
                
                
###################################################################################

    # corrected_model._constructed = False
    # corrected_model.construct()

    # print('\n\n\nUnit Corrected Pyomo Objective:\n\n\n')
    # corrected_model.pprint()

    if failures:
        raise UnitMismatch(_join_failures(failures))

    corrected_model._lc_unit_corrected = True  # idempotency marker, see top
    return corrected_model


class UnitCheck:
    """The result of :func:`unit_check`: did the units balance, and the model.

    Carries a summary() like the rest of the pre-solve chain. model is the
    corrected clone to hand to structure_detector; truthy when balanced.
    """

    __slots__ = ('ok', 'model', 'failures', 'n_objectives', 'n_constraints')

    def __init__(self, ok, model, failures=(), n_objectives=0,
                 n_constraints=0):
        self.ok = bool(ok)
        self.model = model
        self.failures = list(failures)
        self.n_objectives = int(n_objectives)
        self.n_constraints = int(n_constraints)

    def __bool__(self):
        return self.ok

    def __str__(self):
        return self.summary()

    def summary(self) -> str:
        L = ['units', '-----']
        if self.ok:
            L.append(f'  balanced: {self.n_objectives} objective'
                     f'{"s" if self.n_objectives != 1 else ""} and '
                     f'{self.n_constraints} constraint'
                     f'{"s" if self.n_constraints != 1 else ""} check out, and '
                     f'the model has been converted to base units.')
            L.append('  Nothing downstream is meaningful until this passes, '
                     'so it is the first step of the chain.')
        else:
            L.append(_join_failures(self.failures))
            L.append('')
            L.append('  Nothing further can be checked until the units '
                     'balance: the structure detector reads the')
            L.append('  unit-corrected model, and this one has no correction.')
        return '\n'.join(L)


def unit_check(pyomo_component, raise_on_error=True):
    """Check that a model's units balance; the first step of the chain.

    Same walk as unit_corrector but returns a UnitCheck, so the units step
    answers to summary() like the rest of the chain:
    structure_detector(unit_check(f).model). raise_on_error=False reports a
    mismatch instead of raising, for diagnostic callers.
    """
    import pyomo.environ as pyo

    n_obj = len(list(pyomo_component.component_data_objects(
        ctype=pyo.Objective, descend_into=True, active=True)))
    n_con = len(list(pyomo_component.component_data_objects(
        ctype=pyo.Constraint, descend_into=True, active=True)))
    try:
        model = unit_corrector(pyomo_component)
    except UnitMismatch as exc:
        if raise_on_error:
            raise
        return UnitCheck(False, None, [str(exc)], n_obj, n_con)
    return UnitCheck(True, model, (), n_obj, n_con)