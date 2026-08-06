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
try:
    from pyomo.core.expr.visitor import identify_mutable_parameters
except Exception:  # pragma: no cover - older Pyomo
    def identify_mutable_parameters(expr):
        return []
from pyomo.common.numeric_types import RegisterNumericType
RegisterNumericType(pyomo.common.enums.ObjectiveSense)

# from lcsolver.presolve.structureWalker import _StructureVisitor
from lcsolver.presolve.detected import Detected, Term, as_detected
from lcsolver.presolve.structureWalker import _StructureVisitor

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
from lcsolver.presolve.detectorSupportFunctions import (
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

def _divide_or_disqualify(structures, numerator, denominator):
    """``gpRow_divide``, unless there is no numerator left to divide.

    Subtracting the negative monomials leaves NOTHING when every monomial is
    negative, which is what a bound like ``x >= -10`` becomes. Such a
    constraint has no positive part, so it is not a posynomial ratio and is not
    expressible as a geometric or signomial program at all -- but it is a
    perfectly ordinary linear constraint, and the model may well be a valid LP
    or QP.

    Before this, ``gpRow_divide`` indexed ``gr1[0]`` on the empty list and the
    whole detector raised, so **any LP or QP with a negative variable bound
    could not be detected** through the default path where bounds are
    materialized as rows. Disqualifying the log-space structures and carrying
    on lets the linear ones through.
    """
    if not numerator:
        for k in ('Geometric_Program', 'Signomial_Program'):
            structures[k][0] = False
            structures[k][1] = None
        return []
    return gpRow_divide(numerator, denominator)


def implementVariableBound(vr,pyomo_component,N_bound_cons,collect=None):
    """
    This function finds any upper or lower bounds declared in the variable declaration and implemnts 
    them as constraints in the optimization problem

    if variables are not continuous, returns [False, None, None] which is detected and parsed at the layer above

    otherwise, returns [True, updated_pyomo_component, N_bounds_cons]

    If ``collect`` is a ComponentMap, the resolved ``(lower, upper)`` pair is
    recorded there and **no** Pyomo constraint is built. The bounds then travel
    as bounds rather than as rows, which is what a solver actually wants: IPOPT
    and cvxopt both take variable bounds natively, and writing ``x <= c`` as a
    row makes every downstream stage pay for it -- the expression walker parses
    it, the row list carries it, and an SP sub-problem rebuilds it as a
    log-sum-exp once per iteration. On SPaircraft that is 4859 of 6126 rows.
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

    # Hand the bounds back as bounds instead of materializing them as rows.
    if collect is not None:
        collect[vr] = (var_lower_bound, var_upper_bound)
        return [True, pyomo_component, N_bound_cons]

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
        setattr(pyomo_component, proposedKey, pyo.Constraint(expr = vr <= var_upper_bound))
        # increment the number of bounding constraints
        N_bound_cons += 1

    return [True, pyomo_component, N_bound_cons]

#: What each backend can read. A structure carries features -- bounds split
#: out, equalities condensed, columns substituted away -- and a backend that
#: cannot read one of them will not fail, it will quietly solve a DIFFERENT
#: problem. That is the failure mode this repository keeps rediscovering, so
#: the capability is declared rather than remembered.
CONSUMES = {
    'solve_LP':   {'bounds_in_rows'},
    'solve_QP':   {'bounds_in_rows'},
    'solve_GP':   {'bounds_in_rows'},
    'solve_SP':   {'bounds_in_rows'},
    'solve_slcp': {'bounds_in_rows', 'bounds_split'},
    'solve_sia':  {'bounds_in_rows', 'bounds_split'},
}


def features(structures):
    """The set of representation features a detected structure carries."""
    out = set()
    out.add('bounds_split' if structures.get('bounds') is not None
            else 'bounds_in_rows')
    info = structures.get('info') or {}
    if info.get('N_vars_substituted'):
        out.add('columns_substituted')
    if info.get('N_vars_removed'):
        out.add('columns_removed')
    return out


def require(structures, who):
    """Refuse a structure carrying a feature ``who`` cannot read.

    Cheap, and it turns "this backend silently ignored half the problem" into
    an error naming the feature and the backend.
    """
    can = CONSUMES.get(who)
    if can is None:
        return
    missing = features(structures) - can - {'columns_substituted',
                                            'columns_removed'}
    if not missing:
        return
    # Say what to do about it, not merely what is wrong. The remedy is the
    # part a caller actually needs.
    remedy = {
        'bounds_split': "re-run structure_detector with bounds_as_rows=True",
    }
    how = "; ".join(remedy[m] for m in sorted(missing) if m in remedy)
    raise ValueError(
        f"{who} cannot read a structure with {sorted(missing)}. It understands "
        f"{sorted(can)}. Solving would ignore that part of the problem and "
        f"return an answer to a different question."
        + (f" To fix: {how}." if how else ""))


def require_bounds_as_rows(structures, who):
    """Deprecated alias for :func:`require`, kept for external callers."""
    return require(structures, who)


def _require_bounds_as_rows_legacy(structures, who):
    """Refuse structures whose bounds a backend is about to ignore.

    A backend that reads only the rows would silently solve an unbounded
    relaxation if handed ``bounds_as_rows=False`` output -- the bounds are in
    ``structures['bounds']`` and nothing would look at them. Failing loudly is
    the only safe default; a silently relaxed problem still returns an answer,
    and that answer can look entirely reasonable.
    """
    if structures.get('bounds') is not None:
        raise ValueError(
            f"{who} reads variable bounds from the constraint rows, but these "
            "structures were built with bounds_as_rows=False, which puts them "
            "in structures['bounds'] instead. Solving would ignore every "
            "bound. Re-run structure_detector with bounds_as_rows=True.")



def _drop_zero_terms(gpRows):
    """Remove terms whose leading coefficient is exactly zero.

    ``0 * x`` contributes nothing to a sum, so a zero-coefficient term is not
    a modelling error -- it is an absent term. But the GP/SP tests below ask
    ``coefficient > 0``, which a zero fails exactly as a NEGATIVE coefficient
    does, so the classifier read it as a subtraction and rejected the whole
    model with "structure is neither a GP nor an SP".

    That matters because ``0 * some_variable`` is the natural way to write a
    units-correct absent term in a parameterised model -- an architecture with
    no fuel tank contributing no tank weight, say. Dropping the term here is
    exact and makes that idiom work.

    A row that is ALL zero terms is left alone: an identically-zero expression
    is a real problem and should be reported as one further down, not silently
    turned into an empty posynomial.
    """
    if not gpRows:
        return gpRows
    kept = [rw for rw in gpRows if rw[1] != 0.0]
    return kept if kept else gpRows



def _blame(structures, classes, name, reason, row=None):
    """Record which row ruled out which problem class, and why.

    The detector otherwise only flips a global flag, so a model that "is an
    SP" cannot say what stopped it being a GP -- which is the first thing
    anyone wants to know, because a signomial constraint is usually one term
    away from a posynomial one. Cheap: a few tuples, appended where the flag
    was already being cleared.
    """
    blk = structures.setdefault('blockers', {})
    for cls in classes:
        rows = blk.setdefault(cls, [])
        if (name, reason, row) not in rows:
            rows.append((name, reason, row))


def structure_detector(pyomo_component, bounds_as_rows=True):
    """Detect the optimization structure of a Pyomo model.

    ``bounds_as_rows`` selects how bounds declared on a variable are carried.

    ``True`` (default)
        Each bound becomes a Pyomo constraint and then a row, as it always
        has. Every existing caller sees exactly what it saw before.

    ``False``
        Bounds are published as ``structures['bounds']`` -- a list of
        ``(lower, upper)`` aligned with ``structures['variables']`` -- and no
        rows are emitted for them. A solver that takes variable bounds
        natively should prefer this: it is the same problem with far fewer
        rows, and the reduction is large (SPaircraft goes from 6126 rows to
        1267).

    Nothing is tightened or dropped when bounds are split out; the values pass
    through unchanged. That matters here, because SPaircraft needs the full
    1e-30..1e30 box for the reference solution to be inside it, and a presolve
    that "cleaned up" those limits would cut off the answer.
    """
    # Accept the previous step's result directly, so the chain composes:
    #   structures = structure_detector(unit_check(f))
    # rather than making the caller reach inside for `.model`.
    if hasattr(pyomo_component, 'model') and hasattr(pyomo_component, 'ok'):
        if not pyomo_component.ok:
            raise ValueError(
                'the units do not balance, so there is no corrected model to '
                'detect structure on. Read unit_check(...).summary().')
        pyomo_component = pyomo_component.model

    # Various setup things

    if not isinstance(pyomo_component, BlockData):
        raise ValueError( "Invalid type %s passed into the convexity detector"%(str(type(pyomo_component))))

    # get all of the variables in the pyomo object (optimization problem)
    variableList = [ vr for vr in pyomo_component.component_objects(pyo.Var, descend_into=True, active=True) ]

    # Walk through all the variables and ensure that their domains are compatible with optimization structure
    # Eg, no discrete, no weird sets, etc
    N_bound_cons = 0
    boundCollector = None if bounds_as_rows else ComponentMap()
    for vr in variableList:
        # check if it's a vector, matrix, etc...
        if isinstance(vr,pyomo.core.base.var.IndexedVar):
            # Get the set the variable is indexed over, this would be [1,2,3,4...] for a vector 
            # or [(1,1), (1,2)... ] for a matrix
            ix_st = list(vr.index_set())
            # Iterate for all the variables in the indexed set (eg elements in the vector)
            for ix in ix_st:
                [success, pyomo_component, N_bound_cons] = implementVariableBound(vr[ix],pyomo_component,N_bound_cons,boundCollector)
                if not success:
                    return unstructured_dict() | { "message":"A non-continuous variable (%s) was detected"%(vr[ix].name) } 

        else: #variable is scalar
            [success, pyomo_component, N_bound_cons] = implementVariableBound(vr,pyomo_component,N_bound_cons,boundCollector)
            if not success:
                return unstructured_dict() | { "message":"A non-continuous variable (%s) was detected"%(vr[ix].name) } 

    # get all the objectives
    objectives    = [ obj for obj in pyomo_component.component_data_objects(pyo.Objective , descend_into=True, active=True ) ]
    # get all the constraints
    constraints   = [ con for con in pyomo_component.component_objects(     pyo.Constraint, descend_into=True, active=True ) ]

    # Drop constraints that contain no Vars before the numbering below is
    # established. A constant-only constraint -- `Qmax >= Q` where both were
    # substituted, say -- carries no structure, but feeding it to the
    # posynomial machinery zeroes it to a bare negative number
    # (10 - 100 = -90, then +1 -> -89); the negative leading coefficient
    # reads as a subtraction and silently declares the whole model
    # unstructured, so a model that IS a GP gets misrouted to a general NLP
    # solver with no diagnostic.
    #
    # This must happen here rather than inside the loop: `parseDict_GP` is
    # handed `i+1` from `enumerate(constraints)` and uses it to group
    # monomials by constraint, so skipping one mid-loop leaves a gap in the
    # numbering and later indexing walks off the end of the operator list.
    _kept = []
    for con in constraints:
        datas = list(con.values())
        if datas and not any(list(identify_variables(c.expr)) for c in datas):
            try:
                if not all(bool(pyo.value(c.expr)) for c in datas):
                    # A constraint with no variables that evaluates false is a
                    # proof of infeasibility, and the cheapest one available --
                    # available before any solve is attempted. Flag it as such
                    # rather than only as "unstructured": callers otherwise
                    # read the absence of structure as "send it to a general
                    # NLP solver", which then reports a bare
                    # termination_condition=infeasible and loses the sentence
                    # that says WHICH constraint and why.
                    # Only the model's own constants; the expression also
                    # carries Pyomo unit parameters, and "dimensionless = 1"
                    # is noise in a message meant to say what is wrong.
                    try:
                        own = {c.name for c in pyomo_component.get_constants()}
                    except Exception:
                        own = None
                    seen, parts = set(), []
                    for c in datas:
                        for v in identify_mutable_parameters(c.expr):
                            nm = getattr(v, 'name', None)
                            if nm is None or nm in seen:
                                continue
                            if own is not None and nm not in own:
                                continue
                            seen.add(nm)
                            try:
                                parts.append(f'{nm} = {pyo.value(v):g}')
                            except Exception:
                                pass
                    vals = ', '.join(parts)
                    return unstructured_dict() | {
                        "infeasible": True,
                        "message": "Constraint %s involves no variables and is "
                                   "false as written, so the model has no "
                                   "feasible point%s" % (
                                       con.name,
                                       f' ({vals})' if vals else '')}
            except Exception:
                # Not evaluable (mismatched units, say) -- that is the unit
                # checker's job to report, not something to guess at here.
                pass
            continue
        _kept.append(con)
    constraints = _kept
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
    structures = {"blockers": {},
                  "Linear_Program"   :[True,[],[]], 
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
        gpRows = _drop_zero_terms(
            parseDict_GP(0,rv,N_vars_unwrapped,variableMap))
        if gpRows is None:
            return unstructured_dict() | { "message":"The objective is not expressible in the GP algebra (it contains an operation outside the monomial/signomial/signomial-fraction forms, such as a transcendental function)"}
        # Should be in the form [constraint_number, leading constant, exponent for var_1, exponent for var_2...]
        # constraint_number for objectives will be either 0 for numerator or -1 for denomonator

        # check that all of the first entries (constraint number) are 0, otherwise this is a signomial fraction
        if not all([rw[0]==0.0 for rw in gpRows]):
            # is sp with fractional objective
            _blame(structures, ['Linear_Program', 'Quadratic_Program',
                                'Geometric_Program'], 'the objective',
                   'is a ratio of posynomials', row=0)
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
                _blame(structures, ['Geometric_Program', 'Signomial_Program'],
                       'the objective', 'has a negative term (a true signomial)',
                       row=0)
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
                _blame(structures, ['Linear_Program'], 'the objective',
                       'is quadratic, not affine', row=0)
                structures['Linear_Program'][0] = False
            else:
                # if not PD, then it is not a QP
                structures['Quadratic_Program'][0] = False

                # And may be an LP, need to check for linearity (all eigenvalues would be zero)
                linearCheck = checkLinear(gpRows)
                if linearCheck[0]:
                    structures['Linear_Program'][1] = linearCheck[1:] + [None,None]
                else:
                    _blame(structures, ['Linear_Program'], 'the objective',
                           'is neither affine nor a positive-definite quadratic',
                           row=0)
                    structures['Linear_Program'][0] = False  

    # check that there are constraints
    N_cons = 0
    if len(constraints) > 0:
        # print(structures)
        conCounter = 0
        operatorList = []
        #: (row index, name, rows produced) for every '==' constraint, kept
        #: regardless of which flags are still alive. The monomial-equality
        #: check below used to read the GP row list, which is None once
        #: anything else has cleared the GP flag -- so on a model that was
        #: already non-GP for another reason, posynomial equalities were never
        #: examined and never blamed. That made the blame list incomplete, and
        #: an incomplete blame list is worse than none: `structure_report`
        #: infers "simplifies to a GP after presolve" from it.
        equality_rows = {}
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
                    gpRows_lhs = _drop_zero_terms(parseDict_GP(
                        i+1,rvv['lhs'],N_vars_unwrapped,variableMap))
                    gpRows_rhs = _drop_zero_terms(parseDict_GP(
                        i+1,rvv['rhs'],N_vars_unwrapped,variableMap))
                    if gpRows_lhs is None or gpRows_rhs is None:
                        side = 'lhs' if gpRows_lhs is None else 'rhs'
                        return unstructured_dict() | { "message":"The %s of constraint %s is not expressible in the GP algebra (it contains an operation outside the monomial/signomial/signomial-fraction forms, such as a transcendental function)"%(side,c.name)}
                    operator = rvv['operator']
                    operatorList.append(operator)

                    if not any([rw[0] < 0 for rw in gpRows_lhs]):
                        lhs_zeroed = gpRow_subtract(copy.deepcopy(gpRows_lhs), copy.deepcopy(gpRows_rhs))
                        # Do LP/QP stuff
                        if not all([rw[0]>=0.0 for rw in lhs_zeroed]):
                            # has fraction
                            _blame(structures, ['Linear_Program', 'Quadratic_Program'], c.name,
                                   'divides by an expression, so it is not affine', row=i + 1)
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
                                _blame(structures, ['Linear_Program', 'Quadratic_Program'], c.name,
                                       'is nonlinear in the design variables', row=i + 1)
                                structures['Linear_Program'][0] = False
                                structures['Linear_Program'][1] = None
                                structures['Quadratic_Program'][0] = False
                                structures['Quadratic_Program'][1] = None   
                    else:
                        # signomial fraction present
                        _blame(structures, ['Linear_Program', 'Quadratic_Program'], c.name,
                               'contains a signomial fraction', row=i + 1)
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
                        # lhs - rhs came out as a fraction N/D, which happens
                        # whenever the constraint divides by a multi-term
                        # expression -- `z == 1 - a*(1-y)/(1+y)` and the like.
                        #
                        # Once the constraint has been moved to the form
                        # N/D {<=,==,>=} 0 the denominator can be dropped:
                        # every variable in a GP or SP is strictly positive, so
                        # a denominator whose monomials all carry positive
                        # coefficients is itself strictly positive, and
                        # dividing through by it preserves the relation and its
                        # direction. What remains, N {<=,==,>=} 0, is exactly
                        # what the rest of this branch expects.
                        #
                        # A denominator with a negative coefficient could
                        # change sign over the domain, so its direction is not
                        # safe to assume; that case is still rejected.
                        numerator_rows = [r for r in lhs_zeroed if r[0] >= 0]
                        denominator_rows = [r for r in lhs_zeroed if r[0] < 0]
                        if (not numerator_rows or not denominator_rows
                                or not all(r[1] > 0.0 for r in denominator_rows)):
                            raise RuntimeError(
                                'Encountered a signomial fraction whose denominator '
                                'is not provably positive; cannot clear it')
                        lhs_zeroed = numerator_rows
                        unique, counts = numpy.unique(
                            [lhz[0] for lhz in lhs_zeroed], return_counts=True)
                        countDict = dict(zip(unique, counts))
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
                        lhs_final = _divide_or_disqualify(structures, lhs_inter,
                                                          [posMonomial])
                    else:
                        negPosynomial = [ copy.deepcopy(lhs_zeroed[nmi]) for nmi in negative_monomial_indices ]
                        posPosynomial = copy.deepcopy(negPosynomial)
                        for ii in range(0,len(posPosynomial)):
                            posPosynomial[ii][1] *= -1
                        lhs_inter = gpRow_subtract(lhs_zeroed, negPosynomial)
                        lhs_final = _divide_or_disqualify(structures, lhs_inter,
                                                          posPosynomial)
                    # this is where the else indent should be if present

                    if not all([rw[1]>0.0 for rw in lhs_final]):
                        # has subtraction, which is not allowed under this definition of SP
                        _blame(structures, ['Geometric_Program', 'Signomial_Program'],
                               c.name, 'has a negative term that cannot be moved '
                                       'to the other side (a true signomial)',
                               row=i + 1)
                        structures['Geometric_Program'][0] = False
                        structures['Geometric_Program'][1] = None
                        structures['Signomial_Program'][0] = False
                        structures['Signomial_Program'][1] = None

                    if operator == '==':
                        prev = equality_rows.get(i + 1, [c.name, 0])
                        prev[1] += len(lhs_final)
                        equality_rows[i + 1] = prev

                    if not all([rw[0]>=0.0 for rw in lhs_final]):
                        # is sp with fraction
                        _blame(structures, ['Geometric_Program'], c.name,
                               'is a ratio of posynomials, not a posynomial '
                               '(this is what makes the model an SP)', row=i + 1)
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

        # A geometric program admits only MONOMIAL equalities: `log-sum-exp == 0`
        # is not a convex set, so a posynomial equality is a signomial
        # constraint however it is written. Any '==' row that expanded to more
        # than one term is one of those.
        #
        # Every offender is recorded, not just the first, and the scan runs
        # whatever the GP flag currently says. Both matter: it used to `break`
        # after one and to be skipped entirely once GP was already False, so
        # the three drag-fit equalities in the Hoburg UAV went unblamed on a
        # model that another constraint had already made non-GP.
        for conIx, (name, n_rows) in sorted(equality_rows.items()):
            if n_rows > 1:
                _blame(structures, ['Geometric_Program'], name,
                       'is a posynomial equality; a GP admits only monomial '
                       'equalities, so this is a signomial constraint',
                       row=conIx)
                structures['Geometric_Program'][0] = False
                structures['Geometric_Program'][1] = None
                structures['Geometric_Program'][2] = None

    structures['info'] = {}
    structures['info']['N_cons_total']    = N_cons
    structures['info']['N_cons_noBounds'] = N_cons - N_bound_cons
    structures['info']['N_cons_bounds']   = N_bound_cons
    # Publish the variable ordering used for the exponent/coefficient columns.
    # The solution vector returned by the cvxopt backends is indexed in exactly
    # this order, so downstream code (notably solution write-back) needs it.
    structures['variables'] = list(unwrappedVariables)
    # Variable bounds, in the same order, when they were not turned into rows.
    # `None` distinguishes "bounds are in the rows, as always" from "bounds are
    # here and there are none on this variable", which is an empty list.
    structures['bounds'] = (
        None if boundCollector is None
        else [boundCollector.get(v, (None, None)) for v in unwrappedVariables])
    # Keep a strong reference to the model the variables came from. Callers
    # routinely write `structure_detector(unit_corrector(m))`, which leaves the
    # clone unreferenced; once it is collected, the IndexedVar components die
    # with it and every VarData in `variables` reports its name as
    # '[Unattached VarData]', breaking name-based write-back.
    structures['model'] = pyomo_component
    # A dict subclass with named fields. Every existing consumer indexes it
    # exactly as before; new code can read `.kind`, `.space`, `.terms(i)`
    # instead of learning the positional row format.
    return as_detected(structures)








