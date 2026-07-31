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
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.param import IndexedParam

from edi.objects.vector import (
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
    """An indexed Variable that also reads as a vector.

    The behaviour is in :class:`~edi.objects.vector.VectorComponent`; this is
    only the pairing with Pyomo's class. Declared here rather than there so
    that the vector module stays free of Pyomo component internals.
    """


class EDIParam(VectorComponent, IndexedParam):
    """An indexed Constant that also reads as a vector."""


class Group:
    """A named region of a formulation.

    Every multi-part model in this repository was namespacing by hand::

        def add_wing(f, ..., prefix="Wing_"):
            V = lambda n, g, u, d: f.Variable(name=f"{prefix}{n}", guess=g,
                                              units=u, description=d)

    Nine of twenty-six model files open with a shim of that shape, which is the
    API reporting a defect: when every author independently invents the same
    abbreviation, the canonical form is wrong for the thing they do fifty times
    a file. The prefix is half of what those shims are for; the other half is
    the length of ``f.Variable(name=..., guess=..., units=..., description=...)``.

    A group supplies both::

        wing = f.group('wing')
        AR = wing.Variable('AR', 11.0, '-', 'aspect ratio')     # -> wing_AR
        box = wing.group('box')
        t   = box.Variable('t_cap', 0.01, 'm', 'cap thickness') # -> wing_box_t_cap

    and ``f.wing`` reaches it afterwards, so a builder no longer has to thread a
    prefix string through its signature and back out again.

    Names stay **flat** -- ``wing_box_t_cap``, joined by underscores -- rather
    than becoming Pyomo sub-blocks. That keeps the detector, the unit walker,
    write-back and every saved reference solution working exactly as they do
    now; the hierarchy is in how you write the model, not in a second component
    tree to keep consistent with the first.
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
        """The dotted path used when printing, e.g. ``wing.box``.

        Carried rather than derived from the prefix: a group named
        ``landing_gear`` has prefix ``landing_gear_``, and turning underscores
        into dots would render it ``landing.gear``.
        """
        return self._path

    def group(self, name, prefix=None):
        """A nested group, named ``<this>_<name>``.

        ``prefix`` overrides the flat name, as on
        :meth:`Formulation.group`; it is taken as written rather than
        appended to this group's own prefix, so a nested group can carry a
        name a model already publishes.
        """
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

    def Constraint(self, expr):
        return self._formulation.Constraint(expr)

    def ConstraintList(self, conList):
        return self._formulation.ConstraintList(conList)

    def sum(self, vector, axis=None):
        return self._formulation.sum(vector, axis=axis)

    def prod(self, vector, axis=None):
        return self._formulation.prod(vector, axis=axis)

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
        # A quantity may legitimately be named for a Python keyword -- a wing
        # taper ratio is `lambda` in every reference this repository is
        # checked against, and the name is load-bearing because the gpkit
        # cross-check maps `\lambda` onto it. `wing.lambda` is a syntax
        # error, so the trailing underscore PEP 8 prescribes for exactly this
        # collision is accepted: `wing.lambda_`.
        if item.endswith('_') and keyword.iskeyword(item[:-1]):
            item = item[:-1]
        # Otherwise fall through to the component this group named.
        return getattr(self._formulation, f'{self._prefix}{item}')

    def __getitem__(self, item):
        """``g['name']`` for the same thing as ``g.name``.

        Attribute access is how a group is meant to be read, but a name held
        in a variable has to be looked up somehow, and a group is often passed
        where a dictionary of quantities used to be.
        """
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
    def __init__(self):
        super(Formulation, self).__init__()
        # self._variable_counter = 1
        # self._constant_counter = 1
        self._objective_counter = 0
        self._constraint_counter = 0

        self._groups = {}
        self._sensitivity_cache = None
        #: Set False to let `Variable` omit its guess. See `require_guesses`.
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

    #: Names a formulation reserves for itself. A component of one of these
    #: names would shadow the attribute, so `f.solution` would return a
    #: variable and the real solution would be unreachable.
    RESERVED_NAMES = ('solution', 'sensitivities', 'group')

    @property
    def solution(self):
        """The current values, as a :class:`~edi.objects.solution.Solution`.

        Pyomo reloads a solution onto the model, and EDI keeps doing that, so
        `pyo.value(f.x)` answers after a solve. This is the same information
        with somewhere to live: objective, every variable and constant with its
        units and description, and the sensitivities once computed, printable
        as a table.

        It reads THIS model, which is the one a solve writes back to. A
        detected structure holds the unit-corrected clone's variables, and that
        clone is never solved -- reading it returns the initial guess.
        """
        from edi.objects.solution import Solution

        return Solution.from_model(self, sensitivities=self._sensitivity_cache,
                                   ambiguous=getattr(self, '_ambiguous_cache',
                                                     None))

    def solution_with_sensitivities(self, **kwargs):
        """The solution, with sensitivities computed and attached."""
        from edi.objects.solution import Solution

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
        """A named region of the model; see :class:`Group`.

        ``f.group('wing')`` returns it and ``f.wing`` reaches it afterwards, so
        a builder can stop threading a prefix string through its signature.

        ``prefix`` overrides the flat name each member gets, which defaults to
        ``<name>_``. That is for a model whose component names are already
        published -- ``Wing_AR`` verified against a saved reference solution,
        say. Such a model can adopt groups for what they give it (the shorter
        declaration, `f.wing`, dotted output) without renaming anything, which
        would otherwise mean rewriting the reference alongside it and losing
        the check.
        """
        self._check_name_available(name, 'group')
        if name not in self._groups:
            # A component of the same name wins attribute lookup, since Pyomo
            # resolves it before __getattr__ is ever reached -- the group would
            # be created, then be permanently unreachable as `f.<name>`. Say so
            # rather than leaving a shadowed object behind.
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
        """Whether :meth:`Variable` insists on an initial guess. Default True.

        The guess is required on purpose: it makes the author state what they
        expect a quantity to be, it mirrors ``Constant``, and it tells the
        backend the intended scale -- which is real information when the
        problem is signomial or carries a black box, where the starting point
        decides which optimum you reach.

        Turning it off is for someone who knows their model is a geometric
        program, where the solve is global in log space and the guess cannot
        change the answer::

            f.require_guesses = False
            AR = f.Variable('AR', units='-', description='aspect ratio')

        Variables that took a default are recorded and reported by
        ``edi.presolve.optimization_check``, so the omission stays visible
        rather than
        becoming invisible.
        """
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
            # A geometric program is scale-free in log space, so 1 is as good a
            # starting point as any; what matters is that it is positive.
            guess = 1.0
            self._defaulted_guesses.append(name)
        if units is None:
            raise ValueError(
                f"Variable {name!r} needs units. Use '-' for a dimensionless "
                "quantity; EDI requires them so that unit errors are caught "
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

        if size is not None:
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
                            pyo.Var(
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
                pyo.Var(
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
        self._check_name_available(name, 'constant')
        if within is None:
            within = Reals
        else:
            if within not in domainList:
                raise RuntimeError("Invalid within")

        if size is not None:
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
                            pyo.Param(
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
                pyo.Param(
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
        self._objective_counter += 1
        self.add_component(
            'objective_' + str(self._objective_counter),
            pyo.Objective(expr=expr, sense=sense),
        )
        self._objective_keys.append('objective_' + str(self._objective_counter))
        self.__dict__['objective_' + str(self._objective_counter)].construct()

    # def RuntimeObjective(self):
    #     pass

    def Constraint(self, expr):
        self._constraint_counter += 1
        conName = 'constraint_' + str(self._constraint_counter)
        self.add_component(conName, pyo.Constraint(expr=expr))
        self._constraint_keys.append(conName)
        self._allConstraint_keys.append(conName)
        self.__dict__[conName].construct()

    def RuntimeConstraint(self, outputs, operators, inputs, black_box):
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
            inputs_raw, (pyomo.core.base.var.IndexedVar, pyomo.core.base.var.ScalarVar)
        ):
            inputs_raw = [inputs_raw]
        elif isinstance(inputs_raw, (list, tuple)):
            inputs_raw = list(inputs_raw)
        else:
            raise ValueError("Invalid type for input variables")

        if isinstance(
            outputs_raw, (pyomo.core.base.var.IndexedVar, pyomo.core.base.var.ScalarVar)
        ):
            outputs_raw = [outputs_raw]
        elif isinstance(outputs_raw, (list, tuple)):
            outputs_raw = list(outputs_raw)
        else:
            raise ValueError("Invalid type for output variables")
        for lst in [outputs_raw, inputs_raw]:
            for vr in lst:
                if not isinstance(
                    vr, (pyomo.core.base.var.IndexedVar, pyomo.core.base.var.ScalarVar)
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

        black_box.setOptimizationVariables(inputs_raw, outputs_raw)

        outputs_raw_length = len(outputs_raw)
        operators_raw_length = len(operators_raw)

        outputs_unwrapped = []
        for ovar in outputs_raw:
            if isinstance(ovar, pyomo.core.base.var.ScalarVar):
                outputs_unwrapped.append(ovar)
            else:  # isinstance(ovar, pyomo.core.base.var.IndexedVar), validated above
                validIndices = list(ovar.index_set().data())
                for vi in validIndices:
                    outputs_unwrapped.append(ovar[vi])

        inputs_unwrapped = []
        for ivar in inputs_raw:
            if isinstance(ivar, pyomo.core.base.var.ScalarVar):
                inputs_unwrapped.append(ivar)
            else:  # isinstance(ivar, pyomo.core.base.var.IndexedVar), validated above
                validIndices = list(ivar.index_set().data())
                for vi in validIndices:
                    inputs_unwrapped.append(ivar[vi])

        black_box._NunwrappedOutputs = len(outputs_unwrapped)
        black_box._NunwrappedInputs = len(inputs_unwrapped)
        black_box.post_init_setup()

        # TODO:  Need to unwrap operators

        self.__dict__[conName].set_external_model(
            black_box, inputs=inputs_unwrapped, outputs=outputs_unwrapped
        )  # ,operators=operators_unwrapped)

    # -- vector operations ---------------------------------------------
    # Explicit functions rather than more operator overloading: a reduction or
    # a broadcast has to say which axis it means, and an operator cannot.
    def sum(self, vector, axis=None):
        """Sum a vector or matrix, optionally along one axis.

        The reason this exists rather than plain ``sum``: iterating an indexed
        Pyomo component yields its index KEYS, so ``sum(x)`` returns
        ``0 + 1 + 2``. EDI refuses that now, and this is what it points at.
        """
        return _unwrap_0d(as_array(vector).sum(axis=axis))

    def prod(self, vector, axis=None):
        """Product of a vector or matrix, optionally along one axis."""
        return _unwrap_0d(_np.prod(as_array(vector), axis=axis))

    def broadcast_rows(self, vector, n):
        """``vector`` repeated as each of ``n`` rows -> ``(n, len(vector))``.

        EDI never broadcasts silently, so this is how a per-column limit is
        compared against a matrix.
        """
        return broadcast_rows(vector, n)

    def broadcast_cols(self, vector, n):
        """``vector`` repeated as each of ``n`` columns -> ``(len(vector), n)``."""
        return broadcast_cols(vector, n)

    def ConstraintList(self, conList):
        # An elementwise comparison produces an array of constraints, of
        # whatever shape the operands had. Flatten it, so that
        # `f.ConstraintList(M >= f.broadcast_rows(cap, n))` reads the way it
        # should rather than handing a matrix row to Constraint.
        if isinstance(conList, _np.ndarray):
            conList = list(conList.ravel())
        else:
            conList = [c for item in conList
                       for c in (item.ravel().tolist()
                                 if isinstance(item, _np.ndarray) else [item])]
        for i in range(0, len(conList)):
            con = conList[i]
            if isinstance(con, (tuple, list)):
                self.RuntimeConstraint(*con)
            elif isinstance(con, dict):
                self.RuntimeConstraint(**con)
            else:
                self.Constraint(con)

    def get_variables(self):
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._variable_keys
        ]

    def get_constants(self):
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._constant_keys
        ]

    def get_objectives(self):
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._objective_keys
        ]

    def get_constraints(self):
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._allConstraint_keys
        ]

    def get_explicitConstraints(self):
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._constraint_keys
        ]

    def get_runtimeConstraints(self):
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._runtimeConstraint_keys
        ]

    def sensitivities(self, normalized=True, **kwargs):
        """How strongly the optimum responds to each Constant.

        Call this after a solve. By default it returns the log-log sensitivity
        ``d log(f*) / d log(c)`` for every Constant, which is unitless and so
        comparable across constants with different physical units. Pass
        ``normalized=False`` for the raw derivative ``d f* / d c``.

        The numbers come from the constraint duals via the envelope theorem, so
        they cost one solve regardless of how many constants the model has, and
        every partial derivative is taken symbolically rather than by
        differencing. See :mod:`edi.solvers.sensitivity` for the details.

        Returns
        -------
        dict
            See :func:`edi.solvers.sensitivity.sensitivities`.
        """
        from edi.solvers.sensitivity import sensitivities as _sens

        return _sens(self, normalized=normalized, **kwargs)

    def print_sensitivities(self, **kwargs):
        """Print the sensitivity table, sorted by magnitude."""
        from edi.solvers.sensitivity import format_sensitivities

        print(format_sensitivities(self.sensitivities(**kwargs)))

    def _detected(self):
        """This formulation's detected structure, units corrected first.

        Detection reads the corrected twin, not the declared model: a
        constraint stated in feet against one in metres is a different set of
        exponents, and classifying the uncorrected form can call the same
        model by a different name.
        """
        from edi.presolve.structureDetector import structure_detector
        from edi.presolve.unitCorrector import unit_corrector
        return structure_detector(unit_corrector(self), bounds_as_rows=False)

    def structure_report(self, top=5):
        """What kind of problem this is, and what stops it being a simpler one.

        Answers the question a class name raises rather than settles: an SP is
        an SP *because of specific constraints*, and they are usually a
        reformulation away from posynomial. Names them, with their bodies.

        ``top`` caps the constraints listed per class; ``top=None`` lists all.
        Returns the text and prints nothing.
        """
        from edi.presolve.reductions import structure_report as _report
        return _report(self, top=top)

    def optimization_check(self, top=5):
        """Every structural check, in one call: structure, then presolve.

        The pre-solve half of the report -- it needs no solution. Pass the
        solved model to `edi.presolve.optimization_check` directly for
        the degeneracy and cancellation checks, which do.
        """
        from edi.presolve.reductions import optimization_check
        return optimization_check(self, structure_top=top)

    def check_units(self):
        for i in range(1, self._objective_counter + 1):
            assert_units_consistent(self.__dict__['objective_' + str(i)])

        for i in range(1, self._constraint_counter + 1):
            if not isinstance(
                self.__dict__['constraint_' + str(i)],
                pyomo.contrib.pynumero.interfaces.external_grey_box.ExternalGreyBoxBlock,
            ):
                assert_units_consistent(self.__dict__['constraint_' + str(i)])
