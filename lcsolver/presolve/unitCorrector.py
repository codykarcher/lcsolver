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

    The previous message was `Error with constraint: <expression>` -- it named
    the constraint and stopped there, leaving the author to work out which side
    was wrong and by how much. Almost always the answer is a single missing
    factor, and the checker already knows what it is.
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
            # Pyomo converts freely between units of the same dimension, so
            # reaching here means the two sides are not the same dimension at
            # all and no conversion between them exists. The ratio is still
            # worth printing: it is exactly what the short side is missing.
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

    # The raw error carries object reprs and is several hundred characters of
    # pointer addresses; keep the head of it for anyone debugging the walker
    # itself, and no more.
    detail = ' '.join(str(exc).split())
    # Pyomo appends a repr of every offending node, which is pointer addresses
    # and nothing a modeller can act on. The sentence before them is the part
    # worth keeping.
    if '<' in detail:
        detail = detail.split('<', 1)[0]
    # Cutting at the repr can land mid-argument-list; back up to the end of the
    # sentence before it, which is the part that names the failing node.
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

    Checking stops at the first failure only if the checker raises there, and
    a model whose units are wrong in one place is usually wrong in several --
    the same missing conversion repeated. Reporting them one solve at a time
    makes the author pay a full round trip per constraint, so the walk carries
    on past a failure and reports the lot, as the preconditioner does.
    """
    if len(failures) == 1:
        return failures[0]
    rule = '\n' + '-' * 70 + '\n'
    return (f'{len(failures)} unit errors:\n' + rule.join(failures)
            + '\n' + '-' * 70)


def unit_corrector(pyomo_component):
    if not isinstance(pyomo_component, BlockData):
        raise ValueError( "Invalid type %s passed into the convexity detector"%(str(type(pyomo_component))))
    
    corrected_model = pyomo_component.clone()
    # Stamp the clone with the identity of what it was cloned FROM.  Detected
    # structures carry this clone, and a caller may hand those structures back
    # to solve() to skip re-detecting.  Handed structures belonging to a
    # DIFFERENT formulation, the backends would solve that other model's clone
    # and write its numbers onto this one -- the same shape, so no error, just
    # the wrong answer.  solve() compares this token and refuses.
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
        # ... and WHICH REVISION of it.  The clone freezes the constants as they
        # were; a later load_constants makes it stale, and solving a stale clone
        # answers the previous deck.
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

    #: Every mismatch found, reported together at the end.
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

    if failures:
        raise UnitMismatch(_join_failures(failures))

    return corrected_model


class UnitCheck:
    """The result of :func:`unit_check`: did the units balance, and the model.

    Carries a ``summary()`` like every other object in the pre-solve chain, so
    a reader does not have to remember which step returns what. ``model`` is
    the corrected clone -- the thing to hand to
    :func:`~lcsolver.presolve.structureDetector.structure_detector` -- and the
    result is truthy when the units balance.
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

    The same walk as :func:`unit_corrector` -- this is the name to use -- but
    it returns a :class:`UnitCheck` rather than the bare corrected model, so
    the units step answers to ``summary()`` like the rest of the chain::

        check = unit_check(f)
        print(check.summary())
        structures = structure_detector(check.model)

    ``raise_on_error=False`` reports a mismatch instead of raising, which is
    what a diagnostic caller wants: asking what is wrong with a model is
    exactly when it is most likely to be wrong.
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