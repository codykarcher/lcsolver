#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2023
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#
#  Development of this module was conducted as part of the Institute for
#  the Design of Advanced Energy Systems (IDAES) with support through the
#  Simulation-Based Engineering, Crosscutting Research Program within the
#  U.S. Department of Energy’s Office of Fossil Energy and Carbon Management.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import keyword

import numpy as _np
import pyomo
import pyomo.environ as pyo
from pyomo.util.check_units import assert_units_consistent
from pyomo.environ import ConcreteModel
from pyomo.environ import Var, Param, Objective, Constraint, Set
from pyomo.environ import maximize, minimize
from pyomo.environ import units as pyomo_units
from pyomo.core.base.var import IndexedVar, ScalarVar
from pyomo.core.base.param import IndexedParam, ScalarParam

from lcsolver.objects.vector import (
    VectorComponent,
    VectorArray,
    as_array,
    broadcast_cols,
    broadcast_rows,
)
from pyomo.common.dependencies import attempt_import

egb, egb_available = attempt_import(
    "pyomo.contrib.pynumero.interfaces.external_grey_box"
)
if egb_available:
    from pyomo.contrib.pynumero.interfaces.external_grey_box import (
        ExternalGreyBoxModel,
        ExternalGreyBoxBlock,
    )
else:
    # Make pynumero optional - only needed for black box functionality
    ExternalGreyBoxModel = None
    ExternalGreyBoxBlock = None

from pyomo.environ import (
    Reals,
    PositiveReals,
    NonPositiveReals,
    NegativeReals,
    NonNegativeReals,
    Integers,
    PositiveIntegers,
    NonPositiveIntegers,
    NegativeIntegers,
    NonNegativeIntegers,
    Boolean,
    Binary,
    Any,
    AnyWithNone,
    EmptySet,
    UnitInterval,
    PercentFraction,
    RealInterval,
    IntegerInterval,
)

domainList = [
    Reals,
    PositiveReals,
    NonPositiveReals,
    NegativeReals,
    NonNegativeReals,
    Integers,
    PositiveIntegers,
    NonPositiveIntegers,
    NegativeIntegers,
    NonNegativeIntegers,
    Boolean,
    Binary,
    Any,
    AnyWithNone,
    EmptySet,
    UnitInterval,
    PercentFraction,
    RealInterval,
    IntegerInterval,
]


def decodeUnits(u_val):
    if isinstance(u_val, str):
        if u_val in ['', '-', 'None', ' ', 'dimensionless']:
            return pyomo_units.__getattr__('dimensionless')
        else:
            return pyomo_units.__getattr__(u_val)
    else:
        return u_val



def _reject_flexible_size(what, name, size):
    """Refuse np.inf as a size. Flexible length is for black-box inputs;
    a Variable/Constant must state its own length."""
    for entry in (size if isinstance(size, (list, tuple)) else [size]):
        if isinstance(entry, str) or (
            isinstance(entry, (float, _np.floating)) and _np.isinf(entry)
        ):
            raise ValueError(
                f"{what} {name!r} cannot be declared of flexible length. "
                "np.inf is for a black-box input, which takes its length from "
                f"the component it is given; a {what} must state how many "
                "elements it has.")


def _index_map(what, name, values, size, keyword):
    """Convert a nested list of initial values to the index-keyed dict Pyomo
    wants, numpy layout (first index outermost). Shape must match size exactly:
    a transposed array is a different model that solves without complaint.
    A single number or a dict is passed through unchanged.
    """
    if values is None or isinstance(values, (dict, _Unset)):
        return values
    if not isinstance(values, (list, tuple, _np.ndarray)):
        return values

    # Non-concrete shapes (np.inf, strings, floats) are left to
    # _reject_flexible_size and Pyomo, which have better messages
    dims = list(size) if isinstance(size, (list, tuple)) else [size]
    if any(not isinstance(dim, (int, _np.integer)) or isinstance(dim, bool)
           for dim in dims if dim is not None):
        return values

    arr = _np.asarray(values, dtype=object)
    shape = tuple(size) if isinstance(size, (list, tuple)) else (
        () if size in (None, 0) else (int(size),))

    if shape == ():
        want = arr.shape[0] if arr.ndim == 1 else list(arr.shape)
        raise ValueError(
            f"{what} {name!r} was declared without a size, so it is a single "
            f"quantity, but {keyword} holds {arr.size} values. Declare "
            f"size={want} to make it a vector.")
    if arr.shape != shape:
        declared = list(shape) if len(shape) > 1 else shape[0]
        raise ValueError(
            f"{what} {name!r} was declared size={declared} but {keyword} has "
            f"shape {list(arr.shape)}. LCsolver will not reshape it: a "
            "transposed array is a different model and it would solve without "
            "complaint. Give the values in the declared shape, first index "
            "outermost.")
    return {(idx[0] if len(idx) == 1 else idx): arr[idx]
            for idx in _np.ndindex(arr.shape)}


class _Unset:
    """Sentinel for an omitted guess, distinct from a guess of None."""
    def __repr__(self):
        return '<unset>'


UNSET = _Unset()


def _unwrap_0d(result):
    """A full reduction gives a 0-d array; hand back the expression itself.

    numpy keeps the array type through `.sum()`, so `f.sum(x)` would otherwise
    return a 0-d array that Pyomo tries to treat as an indexed rule.
    """
    if isinstance(result, _np.ndarray) and result.ndim == 0:
        return result.item()
    return result


class EDIVar(VectorComponent, IndexedVar):
    """An indexed Variable that also reads as a vector. Behavior lives in
    VectorComponent; this is just the pairing with Pyomo's class."""


class EDIParam(VectorComponent, IndexedParam):
    """An indexed Constant that also reads as a vector."""


class _ScalarVectorOps:
    """Comparisons for a scalar component against a vector operand.

    Python asks the left operand first and Pyomo raises on an indexed operand
    instead of returning NotImplemented, so `scalar >= vector` would never
    reach the vector's broadcasting. Route vector operands back through the
    array machinery; leave everything else to Pyomo.
    """

    def __ge__(self, other):
        if isinstance(other, (VectorComponent, _np.ndarray)):
            return as_array(other) <= self
        return super().__ge__(other)

    def __le__(self, other):
        if isinstance(other, (VectorComponent, _np.ndarray)):
            return as_array(other) >= self
        return super().__le__(other)

    def __eq__(self, other):
        if isinstance(other, (VectorComponent, _np.ndarray)):
            return as_array(other) == self
        return super().__eq__(other)

    # __eq__ is overridden, so the inherited identity hash must be restated
    __hash__ = object.__hash__


class LCScalarVar(_ScalarVectorOps, ScalarVar):
    """A scalar Variable whose comparisons broadcast against a vector."""


class LCScalarParam(_ScalarVectorOps, ScalarParam):
    """A scalar Constant whose comparisons broadcast against a vector."""


def _is_black_box(entry):
    """Is this ConstraintList entry a black box, [outputs, operators, inputs,
    box]? Detected by shape (4 parts, last is the box), not by being a list,
    so a helper returning several rows can be dropped in whole."""
    from lcsolver.objects.blackBoxFunctionModel import BlackBoxFunctionModel
    return (isinstance(entry, (tuple, list)) and len(entry) == 4
            and isinstance(entry[3], BlackBoxFunctionModel))


def _flatten_rows(items):
    """Flatten ConstraintList entries. Black boxes and dicts are leaves;
    numpy arrays (elementwise vector comparisons) ravel to one entry per
    element instead of handing a matrix row to Constraint."""
    if isinstance(items, _np.ndarray):
        items = items.ravel().tolist()
    for item in items:
        if isinstance(item, _np.ndarray):
            yield from item.ravel().tolist()
        elif isinstance(item, dict) or _is_black_box(item):
            yield item
        elif isinstance(item, (tuple, list)):
            yield from _flatten_rows(item)
        else:
            yield item


class Group:
    """A named region of a formulation: prefixed declarations, positional args.

        wing = f.group('wing')
        AR = wing.Variable('AR', 11.0, '-', 'aspect ratio')     # -> wing_AR
        t = wing.group('box').Variable('t_cap', 0.01, 'm', 'cap thickness')

    Names stay flat (wing_box_t_cap), not Pyomo sub-blocks, so the detector,
    unit walker, write-back and saved reference solutions all work unchanged.
    """

    __slots__ = ('_formulation', '_prefix', '_name', '_groups', '_path')

    def __init__(self, formulation, name, prefix, path=None):
        object.__setattr__(self, '_formulation', formulation)
        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_prefix', prefix)
        object.__setattr__(self, '_path', path or name)
        object.__setattr__(self, '_groups', {})

    @property
    def name(self):
        return self._name

    @property
    def prefix(self):
        return self._prefix

    @property
    def path(self):
        """Dotted path for printing, e.g. 'wing.box'. Carried, not derived:
        'landing_gear' would otherwise render as 'landing.gear'."""
        return self._path

    def group(self, name, prefix=None):
        """A nested group, named <this>_<name>. `prefix` overrides the flat
        name and is taken as written, not appended to this group's prefix."""
        if name not in self._groups:
            self._groups[name] = Group(self._formulation, name,
                                       prefix if prefix is not None
                                       else f'{self._prefix}{name}_',
                                       f'{self._path}.{name}')
        return self._groups[name]

    def Variable(self, name, guess=UNSET, units=None, description='', **kw):
        return self._formulation.Variable(f'{self._prefix}{name}', guess,
                                          units, description, **kw)

    def Constant(self, name, value, units=None, description='', **kw):
        return self._formulation.Constant(f'{self._prefix}{name}', value,
                                          units, description, **kw)

    def Constraint(self, expr, holographic=False):
        return self._formulation.Constraint(expr, holographic=holographic)

    def ConstraintList(self, conList, holographic=False):
        return self._formulation.ConstraintList(conList,
                                                holographic=holographic)

    def HolographicConstraint(self, expr):
        return self._formulation.HolographicConstraint(expr)

    def HolographicConstraintList(self, conList):
        return self._formulation.HolographicConstraintList(conList)

    def sum(self, vector, axis=None):
        return self._formulation.sum(vector, axis=axis)

    def prod(self, vector, axis=None):
        return self._formulation.prod(vector, axis=axis)

    def scalar_sum(self, parts):
        return self._formulation.scalar_sum(parts)

    def retype_to_float(self, x):
        return self._formulation.retype_to_float(x)

    def broadcast_rows(self, vector, n):
        return self._formulation.broadcast_rows(vector, n)

    def broadcast_cols(self, vector, n):
        return self._formulation.broadcast_cols(vector, n)

    def __getattr__(self, item):
        if item.startswith('_'):
            raise AttributeError(item)
        groups = object.__getattribute__(self, '_groups')
        if item in groups:
            return groups[item]
        # `wing.lambda` is a syntax error but taper ratio really is named
        # `lambda` (the gpkit cross-check depends on it), so accept the
        # PEP 8 trailing underscore: `wing.lambda_`
        if item.endswith('_') and keyword.iskeyword(item[:-1]):
            item = item[:-1]
        # Otherwise fall through to the component this group named.
        return getattr(self._formulation, f'{self._prefix}{item}')

    def __getitem__(self, item):
        """g['name'] as an alias for g.name, for names held in a variable."""
        try:
            return getattr(self, item)
        except AttributeError as exc:
            raise KeyError(item) from exc

    def __contains__(self, item):
        try:
            getattr(self, item)
            return True
        except AttributeError:
            return False

    def __repr__(self):
        return f"<Group {self._name!r} -> {self._prefix!r}>"


class Formulation(ConcreteModel):
    """A Pyomo ConcreteModel plus a declaration API that records everything.

    Variable/Constant require units and (by default) a guess; vectors read as
    vectors; constraints can be holographic; a black box enters as a
    RuntimeConstraint. Everything declared is recorded in order with units and
    description, which is what get_variables, solution, and sensitivities read.
    Declarations return the component and attach it as f.<name>; after a solve
    the values are written back, so pyo.value(f.x) answers with the optimum.
    """

    def __init__(self):
        super(Formulation, self).__init__()
        # self._variable_counter = 1
        # self._constant_counter = 1
        self._objective_counter = 0
        self._constraint_counter = 0

        self._groups = {}
        # names of constraints declared holographic, see HolographicConstraint
        self._holographic = set()
        self._sensitivity_cache = None
        # set False to let Variable omit its guess, see require_guesses
        self._require_guesses = True
        self._defaulted_guesses = []

        self._variable_keys = []
        self._constant_keys = []
        self._objective_keys = []
        self._runtimeObjective_keys = []
        self._objective_keys = []
        self._runtimeConstraint_keys = []
        self._constraint_keys = []
        self._allConstraint_keys = []

    # Reserved names: a component so named would shadow the attribute and
    # make it unreachable
    RESERVED_NAMES = ('solution', 'sensitivities', 'group')

    def __setattr__(self, key, value):
        """Set the attribute, and let a SubModel learn its own name.
        `f.ferry_model = FerryModel(...)` -- the attribute IS the name, so the
        group, deck prefix and sensitivity label can never drift apart."""
        ConcreteModel.__setattr__(self, key, value)
        if key.startswith('_'):
            return
        attach = getattr(value, '_attach', None)
        if callable(attach):
            attach(self, key)

    @property
    def solution(self):
        """The current values as a printable Solution. Reads THIS model, the
        one a solve writes back to -- a detected structure's clone is never
        solved and would return the initial guess."""
        from lcsolver.objects.solution import Solution

        return Solution.from_model(self, sensitivities=self._sensitivity_cache,
                                   ambiguous=getattr(self, '_ambiguous_cache',
                                                     None),
                                   holographic=getattr(self,
                                                       '_holographic_cache',
                                                       None),
                                   report=getattr(self, '_solve_report',
                                                  None))

    def solution_with_sensitivities(self, **kwargs):
        """The solution, with sensitivities computed and attached."""
        from lcsolver.objects.solution import Solution

        ambiguous = None
        try:
            res = self.sensitivities(**kwargs)
            sens = res['sensitivities']
            ambiguous = res.get('ambiguous')
        except Exception:
            sens = None
        self._sensitivity_cache = sens
        self._ambiguous_cache = ambiguous
        return Solution.from_model(self, sensitivities=sens,
                                   ambiguous=ambiguous)

    def _check_name_available(self, name, what='component'):
        if name in self.RESERVED_NAMES:
            raise ValueError(
                f"{name!r} is reserved: a {what} of that name would shadow "
                f"`f.{name}`, which is how the {name} is reached. Choose "
                "another name.")

    # -- grouping -----------------------------------------------------------
    def group(self, name, prefix=None):
        """A named region of the model; see Group. `f.group('wing')` returns
        it and `f.wing` reaches it afterwards. `prefix` overrides the default
        `<name>_` member prefix, for models whose component names are already
        published against a saved reference solution."""
        self._check_name_available(name, 'group')
        if name not in self._groups:
            # A component of the same name wins attribute lookup before
            # __getattr__ is reached, so the group would be permanently
            # unreachable as f.<name> -- refuse rather than shadow
            if self.component(name) is not None:
                raise ValueError(
                    f"cannot create a group named {name!r}: this formulation "
                    f"already has a component called {name!r}, and it would "
                    f"shadow the group -- `f.{name}` would return the "
                    "component. Choose a different group name.")
            self._groups[name] = Group(self, name,
                                       prefix if prefix is not None
                                       else f'{name}_')
        return self._groups[name]

    @property
    def require_guesses(self):
        """Whether Variable insists on an initial guess. Default True: for a
        signomial or black-box model the starting point decides which optimum
        is reached. Set False for a known GP, where the solve is global in log
        space; defaulted guesses are still reported by optimization_check."""
        return self._require_guesses

    @require_guesses.setter
    def require_guesses(self, value):
        self._require_guesses = bool(value)

    @property
    def defaulted_guesses(self):
        """Names of variables whose guess was supplied by default."""
        return list(self._defaulted_guesses)

    def __getattr__(self, item):
        # Pyomo resolves components here; a group is not a component, so it is
        # looked up only once the normal path has failed.
        try:
            return super().__getattr__(item)
        except AttributeError:
            groups = self.__dict__.get('_groups') or {}
            if item in groups:
                return groups[item]
            raise

    def Variable(
        self, name, guess=UNSET, units=None, description='', size=None,
        bounds=None, domain=None
    ):
        """Declare a quantity the solver is free to choose; returns the Var
        and attaches it as f.<name>. `units` is required ('-' for a pure
        number) and so is `guess` unless require_guesses is False. `size`
        declares a vector (size=3) or array (size=[3, 4]) that reads as numpy;
        `guess` may then be a number or an array of exactly the declared
        shape. `bounds` is (lower, upper) in the declared units, `domain` a
        Pyomo domain -- both hard limits; a well-posedness or fit-validity
        limit is better written as a HolographicConstraint."""
        self._check_name_available(name, 'variable')
        if guess is UNSET:
            if self._require_guesses:
                raise ValueError(
                    f"Variable {name!r} needs a guess. It is required so that "
                    "the author states what they expect the quantity to be, "
                    "and because for a signomial or black-box model the "
                    "starting point decides which optimum you reach. If this "
                    "model is a geometric program, where the solve is global "
                    "in log space and the guess cannot change the answer, set "
                    "`f.require_guesses = False` first.")
            # a GP is scale-free in log space; any positive start works
            guess = 1.0
            self._defaulted_guesses.append(name)
        if units is None:
            raise ValueError(
                f"Variable {name!r} needs units. Use '-' for a dimensionless "
                "quantity; LCsolver requires them so that unit errors are caught "
                "rather than propagated.")
        if domain is None:
            domain = Reals
        else:
            if domain not in domainList:
                raise RuntimeError("Invalid domain")

        if bounds is not None:
            if not isinstance(bounds, (list, tuple)):
                raise ValueError(
                    'The keyword bounds must be a 2 length list or tuple of floats'
                )
            if len(bounds) != 2:
                raise ValueError(
                    'The keyword bounds must be a 2 length list or tuple of floats'
                )
            if not isinstance(bounds[0], (float, int)):
                raise ValueError(
                    'The keyword bounds must be a 2 length list or tuple of floats'
                )
            if not isinstance(bounds[1], (float, int)):
                raise ValueError(
                    'The keyword bounds must be a 2 length list or tuple of floats'
                )
            if bounds[0] > bounds[1]:
                raise ValueError("Lower bound is higher than upper bound")

        guess = _index_map('Variable', name, guess, size, 'guess')

        if size is not None:
            _reject_flexible_size('Variable', name, size)
            if isinstance(size, (list, tuple)):
                for i in range(0, len(size)):
                    if not isinstance(size[i], int):
                        raise ValueError(
                            'Invalid size.  Must be an integer or list/tuple of integers'
                        )
                    # if size[i] == 1 or size[i] == 0:
                    #     raise ValueError(
                    #         'A value of 0 or 1 is not valid for defining size.  Use fewer dimensions.'
                    #     )
                    if i == 0:
                        st = pyo.Set(initialize=list(range(0, size[i])))
                    else:
                        st *= pyo.Set(initialize=list(range(0, size[i])))
                st.construct()
                self.add_component(
                    name,
                    EDIVar(
                        st,
                        name=name,
                        initialize=guess,
                        domain=domain,
                        bounds=bounds,
                        doc=description,
                        units=decodeUnits(units),
                    ),
                )
                self.component(name)._edi_shape = tuple(size)
            else:
                if isinstance(size, int):
                    # if size == 1 or size == 0:
                    if size == 0:
                        self.add_component(
                            name,
                            LCScalarVar(
                                name=name,
                                initialize=guess,
                                domain=domain,
                                bounds=bounds,
                                doc=description,
                                units=decodeUnits(units),
                            ),
                        )
                    else:
                        st = pyo.Set(initialize=list(range(0, size)))
                        st.construct()
                        self.add_component(
                            name,
                            EDIVar(
                                st,
                                name=name,
                                initialize=guess,
                                domain=domain,
                                bounds=bounds,
                                doc=description,
                                units=decodeUnits(units),
                            ),
                        )
                        self.component(name)._edi_shape = (size,)
                else:
                    raise ValueError(
                        'Invalid size.  Must be an integer or list/tuple of integers'
                    )
        else:
            self.add_component(
                name,
                LCScalarVar(
                    name=name,
                    initialize=guess,
                    domain=domain,
                    bounds=bounds,
                    doc=description,
                    units=decodeUnits(units),
                ),
            )
        self.__dict__[name].construct()
        self._variable_keys.append(name)
        return self.__dict__[name]

    def Constant(self, name, value, units, description='', size=None, within=None):
        """Declare a number the model depends on but does not choose: a
        mutable Param, so it can be swapped via load_constants and ranked by
        sensitivities -- a literal in an expression can be neither. Same rules
        as Variable for units and size, with `value` in place of `guess` and
        `within` in place of `domain`. One asymmetry: for a Constant size=1
        declares a scalar, not a length-1 vector, so a single-element list
        value is rejected by Pyomo; use size=2+ or omit size."""
        self._check_name_available(name, 'constant')
        if within is None:
            within = Reals
        else:
            if within not in domainList:
                raise RuntimeError("Invalid within")

        value = _index_map('Constant', name, value, size, 'value')

        if size is not None:
            _reject_flexible_size('Constant', name, size)
            if isinstance(size, (list, tuple)):
                for i in range(0, len(size)):
                    if not isinstance(size[i], int):
                        raise ValueError(
                            'Invalid size.  Must be an integer or list/tuple of integers'
                        )
                    # if size[i] == 1:
                    #     raise ValueError(
                    #         'A value of 1 is not valid for defining size.  Use fewer dimensions.'
                    #     )
                    if i == 0:
                        st = pyo.Set(initialize=list(range(0, size[i])))
                    else:
                        st *= pyo.Set(initialize=list(range(0, size[i])))
                st.construct()
                self.add_component(
                    name,
                    EDIParam(
                        st,
                        name=name,
                        initialize=value,
                        within=within,
                        doc=description,
                        units=decodeUnits(units),
                        mutable=True,
                    ),
                )
                self.component(name)._edi_shape = tuple(size)
            else:
                if isinstance(size, int):
                    if size == 1 or size == 0:
                        self.add_component(
                            name,
                            LCScalarParam(
                                name=name,
                                initialize=value,
                                within=within,
                                doc=description,
                                units=decodeUnits(units),
                                mutable=True,
                            ),
                        )
                    else:
                        st = pyo.Set(initialize=list(range(0, size)))
                        st.construct()
                        self.add_component(
                            name,
                            EDIParam(
                                st,
                                name=name,
                                initialize=value,
                                within=within,
                                doc=description,
                                units=decodeUnits(units),
                                mutable=True,
                            ),
                        )
                        self.component(name)._edi_shape = (size,)
                else:
                    raise ValueError(
                        'Invalid size.  Must be an integer or list/tuple of integers'
                    )
        else:
            self.add_component(
                name,
                LCScalarParam(
                    name=name,
                    initialize=value,
                    within=within,
                    doc=description,
                    units=decodeUnits(units),
                    mutable=True,
                ),
            )

        self.__dict__[name].construct()
        self._constant_keys.append(name)
        return self.__dict__[name]

    def Objective(self, expr, sense=minimize):
        """Declare the quantity to be minimized (sense=maximize for the other
        direction). Attached as objective_1, objective_2, ... in order.
        More than one is allowed but classifies as unstructured; declare one."""
        self._objective_counter += 1
        self.add_component(
            'objective_' + str(self._objective_counter),
            pyo.Objective(expr=expr, sense=sense),
        )
        self._objective_keys.append('objective_' + str(self._objective_counter))
        self.__dict__['objective_' + str(self._objective_counter)].construct()

    # def RuntimeObjective(self):
    #     pass

    def Constraint(self, expr, holographic=False):
        """Declare one constraint; returns the name it was given
        ('constraint_4'), the counter shared with RuntimeConstraint so
        numbering follows the model as written. Most models should use
        ConstraintList instead. holographic=True: see HolographicConstraint."""
        self._constraint_counter += 1
        conName = 'constraint_' + str(self._constraint_counter)
        self.add_component(conName, pyo.Constraint(expr=expr))
        self._constraint_keys.append(conName)
        self._allConstraint_keys.append(conName)
        self.__dict__[conName].construct()
        if holographic:
            self._holographic.add(conName)
        return conName

    def HolographicConstraint(self, expr):
        """A constraint that must hold but must not bind: a well-posedness box,
        the edge of a fit's data. An optimum sitting on one is the solver
        extrapolating, and the solve still converges -- so these are declared,
        and every solve checks whether any of them bound. The constraint
        itself is imposed exactly as an ordinary one."""
        return self.Constraint(expr, holographic=True)

    def HolographicConstraintList(self, conList):
        """`ConstraintList`, with every entry declared holographic."""
        return self.ConstraintList(conList, holographic=True)

    def RuntimeConstraint(self, outputs, operators, inputs, black_box,
                          constants=None):
        """Declare outputs == black_box(inputs) as a constraint, via an
        ExternalGreyBoxBlock (so solve takes the cyipopt route). Inputs and
        outputs must be Variables, not expressions; indexed ones are unwrapped
        elementwise. The pairing with the box's declarations is POSITIONAL --
        units convert at the boundary, so a wrong order does not raise, it
        converts the wrong quantity. `operators` is '==' or a list per output;
        '>='/'<=' validate but the cyipopt route still imposes equalities.
        Usually written inside ConstraintList: [z, '==', [x, y], UnitCircle()].
        """
        self._constraint_counter += 1
        conName = 'constraint_' + str(self._constraint_counter)
        self._runtimeConstraint_keys.append(conName)
        self._allConstraint_keys.append(conName)

        self.add_component(conName, ExternalGreyBoxBlock())
        self.__dict__[conName].construct()

        # TODO:  Need to include operators after Michael fixes things

        inputs_raw = inputs
        outputs_raw = outputs
        operators_raw = operators

        if isinstance(
            inputs_raw, (pyomo.core.base.var.IndexedVar, pyomo.core.base.var.VarData)
        ):
            inputs_raw = [inputs_raw]
        elif isinstance(inputs_raw, (list, tuple)):
            inputs_raw = list(inputs_raw)
        else:
            raise ValueError("Invalid type for input variables")

        if isinstance(
            outputs_raw, (pyomo.core.base.var.IndexedVar, pyomo.core.base.var.VarData)
        ):
            outputs_raw = [outputs_raw]
        elif isinstance(outputs_raw, (list, tuple)):
            outputs_raw = list(outputs_raw)
        else:
            raise ValueError("Invalid type for output variables")
        for lst in [outputs_raw, inputs_raw]:
            for vr in lst:
                if not isinstance(
                    vr, (pyomo.core.base.var.IndexedVar, pyomo.core.base.var.VarData)
                ):
                    raise ValueError("Invalid type when checking inputs and outputs")

        if isinstance(operators_raw, (list, tuple)):
            operators_raw = list(operators_raw)
        elif isinstance(operators_raw, str):
            operators_raw = [operators_raw]
        else:
            raise ValueError("Invalid type for operators")
        for opr in operators_raw:
            if opr not in ["==", ">=", "<="]:
                raise ValueError("Invalid operator")

        # Formulation Constants the box consumes, wired positionally against
        # its own constants declarations.  Never optimizer columns; their
        # jacobian columns feed only the sensitivity report
        constants_raw = constants
        if constants_raw is None:
            constants_raw = []
        elif isinstance(constants_raw, (pyomo.core.base.param.ScalarParam,
                                        pyomo.core.base.param.ParamData)):
            constants_raw = [constants_raw]
        elif isinstance(constants_raw, (list, tuple)):
            constants_raw = list(constants_raw)
        else:
            raise ValueError('Invalid type for runtime constraint constants')
        for cp in constants_raw:
            if isinstance(cp, pyomo.core.base.param.IndexedParam):
                raise NotImplementedError(
                    'indexed Constants are not yet supported as black-box '
                    'constants; pass scalar Constants')
            if not isinstance(cp, (pyomo.core.base.param.ScalarParam,
                                   pyomo.core.base.param.ParamData)):
                raise ValueError(
                    'runtime constraint constants must be formulation '
                    'Constants (pyomo Params); got %s'
                    % type(cp).__name__)
        n_declared = len(getattr(black_box, 'constants', []) or [])
        if len(constants_raw) != n_declared:
            raise ValueError(
                'the black box %s declares %d constant(s) but %d were wired '
                'in; the pairing is positional, so the counts must match'
                % (type(black_box).__name__, n_declared, len(constants_raw)))

        black_box.setOptimizationVariables(inputs_raw, outputs_raw)
        black_box.setOptimizationConstants(constants_raw)

        outputs_raw_length = len(outputs_raw)
        operators_raw_length = len(operators_raw)

        outputs_unwrapped = []
        for ovar in outputs_raw:
            if isinstance(ovar, pyomo.core.base.var.IndexedVar):
                validIndices = list(ovar.index_set().data())
                for vi in validIndices:
                    outputs_unwrapped.append(ovar[vi])
            else:  # a ScalarVar or a bare VarData element, validated above
                outputs_unwrapped.append(ovar)

        inputs_unwrapped = []
        for ivar in inputs_raw:
            if isinstance(ivar, pyomo.core.base.var.IndexedVar):
                validIndices = list(ivar.index_set().data())
                for vi in validIndices:
                    inputs_unwrapped.append(ivar[vi])
            else:  # a ScalarVar or a bare VarData element, validated above
                inputs_unwrapped.append(ivar)

        black_box._NunwrappedOutputs = len(outputs_unwrapped)
        black_box._NunwrappedInputs = len(inputs_unwrapped)
        black_box.post_init_setup()

        self.__dict__[conName].set_external_model(
            black_box, inputs=inputs_unwrapped, outputs=outputs_unwrapped
        )
        # Operators, one per UNWRAPPED output (broadcast a single entry).
        # Recorded for the sequential bridge, which emits '>='/'<=' as
        # one-sided rows: when the model presses the output onto the box,
        # the inequality binds at the optimum without a black-box equality
        # manifold for the solver to fall off. The cyipopt route still
        # imposes equalities and ignores this record (its TODO stands).
        ops = (operators_raw * len(outputs_unwrapped)
               if len(operators_raw) == 1 else list(operators_raw))
        if len(ops) != len(outputs_unwrapped):
            ops = ['=='] * len(outputs_unwrapped)
        self.__dict__[conName]._lc_operators = ops

    # -- vector operations ---------------------------------------------
    # Explicit functions rather than more operator overloading: a reduction or
    # a broadcast has to say which axis it means, and an operator cannot.
    def sum(self, vector, axis=None):
        """Sum a vector or matrix, optionally along one axis. Exists because
        iterating an indexed Pyomo component yields its index KEYS, so plain
        sum(x) would return 0 + 1 + 2."""
        return _unwrap_0d(as_array(vector).sum(axis=axis))

    def prod(self, vector, axis=None):
        """Product of a vector or matrix, optionally along one axis."""
        return _unwrap_0d(_np.prod(as_array(vector), axis=axis))

    def scalar_sum(self, parts):
        """Add scalar quantities, refusing anything vector-valued. A vector
        slipping into a rollup quietly becomes one constraint per element
        downstream; the seed is the first part rather than a bare 0, so units
        are checked on every addition."""
        parts = list(parts)
        for part in parts:
            if hasattr(part, 'shape') and getattr(part, 'shape', ()) != ():
                raise TypeError(
                    f"scalar_sum() adds scalars, and got a vector quantity "
                    f"({part}). A rollup over a vector must say which axis it "
                    f"means -- use f.sum(x, axis=...) for that -- because "
                    f"summing it here would silently build one row per "
                    f"element.")
        if not parts:
            return 0
        out = parts[0]
        for part in parts[1:]:
            out = out + part
        return out

    def retype_to_float(self, x):
        """The float behind a number or a declared Constant, for places that
        need a VALUE at build time -- notably GP exponents, which are numbers,
        never parameters."""
        if isinstance(x, (int, float)):
            return float(x)
        return float(pyo.value(x))

    def broadcast_rows(self, vector, n):
        """`vector` repeated as each of `n` rows -> (n, len(vector)). LCsolver
        never broadcasts silently; this compares a per-column limit against a
        matrix."""
        return broadcast_rows(vector, n)

    def broadcast_cols(self, vector, n):
        """``vector`` repeated as each of ``n`` columns -> ``(len(vector), n)``."""
        return broadcast_cols(vector, n)

    def ConstraintList(self, conList, holographic=False):
        """Declare all the constraints at once. Entries dispatch on type: an
        algebraic comparison goes to Constraint; a vector comparison flattens
        to one constraint per element; a black box -- [outputs, operators,
        inputs, box] or a dict of the same keywords -- goes to
        RuntimeConstraint; a plain list of rows (a generator's output) is
        flattened in place. The three can be mixed freely in one list.
        holographic=True marks every algebraic entry; nothing is returned."""
        for con in _flatten_rows(conList):
            if isinstance(con, dict):
                self.RuntimeConstraint(**con)
            elif isinstance(con, (tuple, list)):
                self.RuntimeConstraint(*con)
            else:
                self.Constraint(con, holographic=holographic)

    def get_variables(self):
        """The variables declared through Variable, in declaration order.
        A Var added by hand via add_component is not here -- these lists are
        what LCsolver can report on, and why a hand-added variable is missing
        from the solution table."""
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._variable_keys
        ]

    def get_constants(self):
        """The constants declared through Constant, in declaration order.
        Same restriction as get_variables; these are what sensitivities
        reports against."""
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._constant_keys
        ]

    def load_constants(self, constants):
        """Set declared constants from an input deck (name -> value), after
        the build; a sweep is a loop over loads and solves. Names include
        group prefixes ('wing_area'). An unknown name raises KeyError with
        near-misses suggested -- a deck key that silently does nothing sizes
        the wrong thing. A sized Constant takes a sequence of that length.
        Returns the formulation, so a load can be chained onto a build."""
        import difflib

        known = {c.name: c for c in self.get_constants()}
        for name, value in constants.items():
            if name not in known:
                near = difflib.get_close_matches(name, known.keys(), n=3)
                hint = f"; nearest declared: {', '.join(near)}" if near else ""
                raise KeyError(
                    f"'{name}' is not a constant of this formulation{hint}"
                )
            comp = known[name]
            if comp.is_indexed():
                values = list(value)
                keys = sorted(comp.keys())
                if len(values) != len(keys):
                    raise ValueError(
                        f"'{name}' has {len(keys)} entries; the deck supplied "
                        f"{len(values)}"
                    )
                for k, v in zip(keys, values):
                    comp[k].set_value(float(v))
            else:
                comp.set_value(float(value))
        # Structures detected earlier hold a clone with the OLD values, which
        # would answer the previous deck's question -- bump the revision so
        # solve() refuses stale structures
        self._edi_revision = getattr(self, '_edi_revision', 0) + 1

        return self

    def get_objectives(self):
        """The objectives declared through Objective, in declaration order."""
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._objective_keys
        ]

    def get_constraints(self):
        """Every declared constraint, algebraic and runtime alike, in order.
        Walkers must expect both Constraint and ExternalGreyBoxBlock; use
        get_explicitConstraints / get_runtimeConstraints for one kind."""
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._allConstraint_keys
        ]

    def get_explicitConstraints(self):
        """The algebraic constraints only -- the ones with an expression that
        unit correction, structure detection and duals can work on."""
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._constraint_keys
        ]

    def get_runtimeConstraints(self):
        """The black-box constraints only, as their grey-box blocks.
        Non-empty means solve must take the cyipopt route -- the AMPL-based
        IPOPT route cannot call Python at an iterate."""
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._runtimeConstraint_keys
        ]

    def sensitivities(self, normalized=True, **kwargs):
        """How strongly the optimum responds to each Constant; call after a
        solve. Default is the log-log sensitivity d log(f*) / d log(c)
        (normalized=False for the raw derivative). Comes from the duals via
        the envelope theorem -- one solve, all symbolic. Returns the dict from
        lcsolver.postsolve.sensitivity.sensitivities."""
        from lcsolver.postsolve.sensitivity import sensitivities as _sens

        return _sens(self, normalized=normalized, **kwargs)

    def print_sensitivities(self, **kwargs):
        """Print the sensitivity table, sorted by magnitude."""
        from lcsolver.postsolve.sensitivity import format_sensitivities

        print(format_sensitivities(self.sensitivities(**kwargs)))

    def _detected(self):
        """This formulation's detected structure, units corrected first --
        classifying the uncorrected form can name the same model differently."""
        from lcsolver.presolve.structureDetector import structure_detector
        from lcsolver.presolve.unitCorrector import unit_corrector
        return structure_detector(unit_corrector(self), bounds_as_rows=False)

    def structure_report(self, top=5):
        """What kind of problem this is and which constraints stop it being a
        simpler one, named with their bodies. `top` caps the constraints
        listed per class (None for all); returns the text, prints nothing."""
        from lcsolver.presolve.reductions import structure_report as _report
        return _report(self, top=top)

    def optimization_check(self, top=5):
        """Every structural check in one call, needing no solution. Pass the
        solved model to lcsolver.presolve.optimization_check directly for the
        degeneracy and cancellation checks, which do."""
        from lcsolver.presolve.reductions import optimization_check
        return optimization_check(self, structure_top=top)

    def check_units(self):
        """Assert every declared objective and constraint balances
        dimensionally; raises UnitsError naming the first that does not.
        Runtime constraints are skipped -- a grey-box block has no expression
        to walk; their units are checked at the boundary on every evaluation.
        Optional: solve's unit correction catches this too. This is the
        version to reach for while a model is still being written."""
        for i in range(1, self._objective_counter + 1):
            assert_units_consistent(self.__dict__['objective_' + str(i)])

        for i in range(1, self._constraint_counter + 1):
            if not isinstance(
                self.__dict__['constraint_' + str(i)],
                pyomo.contrib.pynumero.interfaces.external_grey_box.ExternalGreyBoxBlock,
            ):
                assert_units_consistent(self.__dict__['constraint_' + str(i)])
