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
    """A component of the model cannot be of unknown length.

    ``np.inf`` is how a *black box* declares an input whose length it takes
    from whatever it is handed, which is what lets one box serve a three
    element vector in one model and a ten element one in another. A Variable
    or a Constant is the thing doing the handing, so it has to say how many
    elements it has. Worth its own message because the two declarations sit
    a few lines apart in a model file and read alike.
    """
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
    """An array of initial values, laid out onto the component's index set.

    Pyomo initializes an indexed component from a dict keyed by the index --
    ``(i, j)`` for a matrix -- and a nested list handed to it instead fails
    with ``KeyError: "Index '0' is not valid for indexed component 'e'"``,
    which names neither the shape it wanted nor the shape it got. Writing the
    dict comprehension by hand is the workaround, and it is noise in a file
    whose whole purpose is to be read::

        value={(i, j): dist[i][j] for i in range(3) for j in range(4)}
        value=[[15.0, 40.0, 90.0, 130.0], ...]          # this, instead

    Numpy's layout is used, because everything else about an LCsolver vector
    already follows it: the outer level is the FIRST index, so ``values[i][j]``
    is element ``(i, j)``.

    The shape has to match ``size`` exactly. That is the part that matters: a
    3x4 array quietly accepted for a 4x3 declaration is a transposed model
    that solves and answers a different question -- the same reason
    :mod:`lcsolver.objects.vector` refuses to broadcast silently.

    A single number is left alone, since Pyomo gives it to every element, and
    so is a dict, which is already in the form Pyomo wants.
    """
    if values is None or isinstance(values, (dict, _Unset)):
        return values
    if not isinstance(values, (list, tuple, _np.ndarray)):
        return values

    # Anything that is not a concrete shape -- np.inf, a string, a float --
    # is left to `_reject_flexible_size` and Pyomo's own size validation,
    # which have better messages for it than a shape comparison would.
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
    """An indexed Variable that also reads as a vector.

    The behaviour is in :class:`~lcsolver.objects.vector.VectorComponent`; this is
    only the pairing with Pyomo's class. Declared here rather than there so
    that the vector module stays free of Pyomo component internals.
    """


class EDIParam(VectorComponent, IndexedParam):
    """An indexed Constant that also reads as a vector."""


class _ScalarVectorOps:
    """Comparisons for a SCALAR component against a vector operand.

    Python hands the LEFT operand the comparison first, and Pyomo raises on
    an indexed operand rather than returning NotImplemented -- so a scalar on
    the left of a vector row (``P_installed >= P`` with ``P`` a sized
    Variable) would never reach the vector's own broadcasting.  These
    overrides route a vector operand back through the array machinery, one
    row per element, and leave every other comparison to Pyomo.
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
    """Is this `ConstraintList` entry a black box rather than a list of rows?

    A black box is ``[outputs, operators, inputs, box]``, so it is told apart
    by what it IS -- four parts whose last one is the box -- and not merely by
    being a list. Anything else that arrives as a list is a list of rows, and
    gets flattened. Without this a helper returning several rows could not be
    dropped into a list whole: four rows would be read as a runtime
    constraint, and any other count would raise.
    """
    from lcsolver.objects.blackBoxFunctionModel import BlackBoxFunctionModel
    return (isinstance(entry, (tuple, list)) and len(entry) == 4
            and isinstance(entry[3], BlackBoxFunctionModel))


def _flatten_rows(items):
    """Every `ConstraintList` entry, with nesting and arrays flattened away.

    Black boxes and dicts are leaves; numpy arrays -- what an elementwise
    comparison between two vectors produces -- ravel to one entry per element,
    so ``f.ConstraintList(M >= f.broadcast_rows(cap, n))`` declares a
    constraint per element instead of handing a matrix row to `Constraint`.
    """
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
    """An optimization model, and a record of what it is made of.

    A Formulation *is* a Pyomo ``ConcreteModel`` -- every Pyomo idiom keeps
    working, and anything Pyomo can do to a model it can do to this one. What
    it adds is a declaration API that refuses, at the point of writing, the
    things an engineering model otherwise gets wrong silently:

    * `Variable` and `Constant` will not be declared without units (``'-'``
      for a pure number), so comparing feet against metres is a dimensional
      error rather than a plausible wrong answer;
    * `Variable` will not be declared without a guess either, because for a
      signomial or black-box model the starting point decides which optimum
      is reached -- see `require_guesses` for the exemption;
    * a vector quantity reads as a vector: ``x[-1]``, ``x >= y`` elementwise,
      `sum` (plain ``sum(x)`` is refused, because iterating a Pyomo indexed
      component yields its index keys and would silently add ``0 + 1 + 2``);
    * a constraint can be declared *holographic* -- it must hold, but if it
      binds the answer is not an answer (see `HolographicConstraint`);
    * an analysis code enters as a `RuntimeConstraint` and is thereafter an
      ordinary constraint.

    Everything declared is also recorded, in declaration order, with its units
    and description. That record is what `get_variables` and its siblings hand
    back, what makes `solution` a printable table rather than a dictionary of
    floats, and what lets `sensitivities` report against constants by name.

    A model is written top down::

        f = Formulation()
        x = f.Variable('x', guess=1.0, units='m', description='the x variable')
        c = f.Constant('c', value=2.0, units='m', description='a constant')
        f.Objective(x)
        f.ConstraintList([x >= c])
        sol = lcsolver.solve(f)

    Declarations return the component and also attach it as ``f.<name>``,
    because that is how Pyomo names things. `group` gives a model with more
    than a handful of them somewhere to put the prefixes that its authors were
    otherwise threading through every builder signature by hand.

    After a solve the values are written back onto this model, so
    ``pyo.value(f.x)`` answers with the optimum rather than the guess.

    Names in `RESERVED_NAMES` are refused: a component so named would shadow
    the attribute of the same name and make it unreachable.
    """

    def __init__(self):
        super(Formulation, self).__init__()
        # self._variable_counter = 1
        # self._constant_counter = 1
        self._objective_counter = 0
        self._constraint_counter = 0

        self._groups = {}
        #: Names of constraints declared holographic -- there to bound the
        #: problem, not to shape the answer. See `HolographicConstraint`.
        self._holographic = set()
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

    def __setattr__(self, key, value):
        """Set the attribute, and let a :class:`SubModel` learn its own name.

        Attaching a block is how it gets both a formulation and a name::

            f.ferry_model = FerryModel(n_segments=20)

        so the attribute IS the name -- the group its components live in, the
        prefix on every deck key, the row label in the sensitivity table. There
        is no second place to spell it and so no way for the two to drift.
        Anything else assigned to a formulation is set exactly as before.
        """
        ConcreteModel.__setattr__(self, key, value)
        if key.startswith('_'):
            return
        attach = getattr(value, '_attach', None)
        if callable(attach):
            attach(self, key)

    @property
    def solution(self):
        """The current values, as a :class:`~lcsolver.objects.solution.Solution`.

        Pyomo reloads a solution onto the model, and LCsolver keeps doing that, so
        `pyo.value(f.x)` answers after a solve. This is the same information
        with somewhere to live: objective, every variable and constant with its
        units and description, and the sensitivities once computed, printable
        as a table.

        It reads THIS model, which is the one a solve writes back to. A
        detected structure holds the unit-corrected clone's variables, and that
        clone is never solved -- reading it returns the initial guess.
        """
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
        ``lcsolver.presolve.optimization_check``, so the omission stays visible
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
        """Declare a quantity the solver is free to choose.

        Returns the Pyomo ``Var``, and attaches it as ``f.<name>``, so the
        declaration can be used directly in the constraints that follow::

            x = f.Variable('x', guess=1.0, units='m', description='the x variable')

        ``units`` has no default and omitting it is an error, even for a pure
        number -- write ``'-'`` (or ``''``, or ``'dimensionless'``) for that
        case. A quantity that never said what it is cannot be checked against
        the one it is compared to, and an unchecked model does not fail, it
        answers.

        ``guess`` is required for a related reason: it makes the author state
        the scale they expect, and where the problem is signomial or carries a
        black box the starting point selects the optimum. A model that is a
        geometric program is genuinely exempt -- the solve is global in log
        space -- and says so once, via ``f.require_guesses = False``, after
        which the guess defaults to 1.0 and the omission is still reported by
        ``lcsolver.presolve.optimization_check``.

        ``size`` declares a vector or an array: ``size=3`` for a length-3
        vector, ``size=[3, 4]`` for a 3x4 one, indexed from zero. The result
        behaves as a numpy array of the individual ``VarData`` objects (see
        :mod:`lcsolver.objects.vector`), so slicing, negative indexing and
        elementwise comparison all work, and ``x >= y`` over two vectors yields
        one constraint per element for `ConstraintList` to take. ``guess`` may
        then be a single number, given to every element, or an array of exactly
        the declared shape -- exactly, because a transposed array is a
        different model and it would solve without complaint. Omitting ``size``
        (or ``size=0``) declares a scalar.

        ``np.inf`` is not a size here. Flexible length belongs to a black-box
        input, which takes its length from whatever component it is handed; a
        Variable is the thing doing the handing and has to state its own.

        ``bounds`` is a ``(lower, upper)`` pair of plain numbers, read in the
        declared units, and ``domain`` is a Pyomo domain such as
        ``NonNegativeReals``. Both are hard limits the solver may sit on
        without saying so; a limit that exists only to keep the problem well
        posed, or to mark where a fit stops being valid, is better written as a
        `HolographicConstraint`, which is checked after every solve for having
        bound.
        """
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
        """Declare a number the model depends on but does not choose.

        The same declaration as `Variable`, with ``value`` in place of
        ``guess`` and the same rules for ``units`` (required; ``'-'`` for
        dimensionless) and ``size``::

            c = f.Constant('c', value=[1.0, 2.0], units='-', size=2,
                           description='a constant c')

        What is declared is a *mutable* Pyomo ``Param``, and that is the reason
        to prefer it over writing the number into the constraint. A literal is
        invisible: it cannot be changed and the model re-solved without
        rebuilding it, and nothing can report against it. A Constant can, and
        `sensitivities` does -- every one of them is ranked by
        ``d log(f*) / d log(c)`` after a solve, from one solve, symbolically.
        So anything a designer might later want the price of should be a
        Constant rather than a number in an expression.

        ``within`` is the Pyomo domain the value must lie in (default
        ``Reals``); it is the ``Param`` spelling of `Variable`'s ``domain``.

        One asymmetry with `Variable` to know about: for a Constant, ``size=1``
        declares a scalar rather than a length-1 vector, so a single-element
        list passed as its ``value`` will be rejected by Pyomo. Use ``size=2``
        or more for a genuine vector, and omit ``size`` for a scalar.
        """
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
        """Declare the quantity to be minimized -- ``f.Objective(c / x)``.

        ``sense=maximize`` for the other direction; both names come from Pyomo
        and are re-exported by lcsolver.

        Nothing is returned. The objective is attached as ``objective_1``,
        ``objective_2``, ... in declaration order, which is the order
        `get_objectives` reports and the reason `check_units` can walk them.

        More than one may be declared, but a model with more than one is not a
        structured problem: `lcsolver.presolve.structureDetector` classifies it
        as unstructured and says so, since there is no single quantity for a
        GP, SP or QP to be about. Declare one.
        """
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
        """Declare one constraint, and return the name it was given.

        ``f.Constraint(x + y <= 1.0 * units.m)``. The comparison operator has
        already built the expression; this attaches it to the formulation as
        ``constraint_1``, ``constraint_2``, ... The counter is shared with
        `RuntimeConstraint`, so the numbering follows the model as written and
        a black box does not create a gap in it.

        Most models should be using `ConstraintList` instead: it takes them all
        at once, accepts black-box entries beside algebraic ones, and flattens
        a vector comparison into its elements. This is the one-at-a-time form
        underneath, and the name it returns -- ``'constraint_4'`` -- is the
        handle for the component afterwards, via ``f.constraint_4``.

        ``holographic=True`` records the constraint as one that must hold but
        must not bind; `HolographicConstraint` is the way to say that, and
        explains why it is worth saying.
        """
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
        """A constraint that must hold but must not *bind*.

        Some constraints are not part of the design problem; they are there to
        keep it well posed. A 1e-30..1e30 box that stops a variable running to
        zero. The edges of the data a fit was made from. A limit that says
        "beyond here I am not modelling anything, I am extrapolating".

        The answer is only meaningful if none of them is active. An optimum
        sitting on the edge of a fit's validity is not an optimum, it is the
        solver telling you it wanted to go somewhere you have no data for, and
        the number it returned is whatever the fit happened to extrapolate to.
        That is easy to miss, because the solve converges and the answer looks
        like any other.

        So they are declared, not merely written, and every solve checks them
        and says so. Nothing about the constraint itself changes -- it is
        imposed exactly as an ordinary one -- only that LCsolver knows to watch it.
        """
        return self.Constraint(expr, holographic=True)

    def HolographicConstraintList(self, conList):
        """`ConstraintList`, with every entry declared holographic."""
        return self.ConstraintList(conList, holographic=True)

    def RuntimeConstraint(self, outputs, operators, inputs, black_box):
        """Declare ``outputs == black_box(inputs)`` as a constraint.

        This is how an analysis code enters a model. The body is not an
        expression that can be written down and differentiated in advance; it
        is a routine the solver calls at every iterate, which is why it is
        called a *runtime* constraint. Underneath it is a Pyomo
        ``ExternalGreyBoxBlock``, so a model containing one has to be solved
        through cyipopt -- ``solve`` switches to that route by itself.

        ``outputs`` and ``inputs`` are Pyomo variables, singly or in a list;
        indexed ones are unwrapped element by element for the solver. They must
        be Variables, not expressions: the box is handed values and its
        derivatives land on these columns of the jacobian, and there is nothing
        to differentiate an expression through.

        The pairing with the box's own declarations is **positional**, not by
        name. The i-th entry of ``inputs`` supplies the i-th input the box
        declared, and it is that declaration that says what units the box
        wants; the conversion happens at the boundary, so a model in metres can
        drive a box that thinks in feet without either side knowing. Getting
        the order wrong therefore does not raise, it converts the wrong
        quantity -- keep the lists in declaration order.

        ``operators`` is ``'=='``, or a list of them, one per output. ``'>='``
        and ``'<='`` are accepted and validated but not yet applied: every
        runtime constraint is currently imposed as an equality.

        Usually written inside `ConstraintList` rather than called directly::

            f.ConstraintList([
                [z, '==', [x, y], UnitCircle()],
                x + y <= 1.0 * units.m,
            ])
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

        self.__dict__[conName].set_external_model(
            black_box, inputs=inputs_unwrapped, outputs=outputs_unwrapped
        )
        # Operators, one per UNWRAPPED output (broadcast a single entry).
        # Recorded on the block for the sequential bridge, which emits
        # '>=' / '<=' runtime constraints as ONE-SIDED rows --- the form
        # the Hoburg/helicopter free-section results use: when the model
        # itself presses the output onto the box (minimized drag, capped
        # ood, stall-margined clmax), an inequality binds at the optimum
        # WITHOUT creating a black-box equality manifold for the solver
        # to fall off.  The cyipopt route still imposes equalities and
        # ignores this record (its TODO stands).
        ops = (operators_raw * len(outputs_unwrapped)
               if len(operators_raw) == 1 else list(operators_raw))
        if len(ops) != len(outputs_unwrapped):
            ops = ['=='] * len(outputs_unwrapped)
        self.__dict__[conName]._lc_operators = ops

    # -- vector operations ---------------------------------------------
    # Explicit functions rather than more operator overloading: a reduction or
    # a broadcast has to say which axis it means, and an operator cannot.
    def sum(self, vector, axis=None):
        """Sum a vector or matrix, optionally along one axis.

        The reason this exists rather than plain ``sum``: iterating an indexed
        Pyomo component yields its index KEYS, so ``sum(x)`` returns
        ``0 + 1 + 2``. LCsolver refuses that now, and this is what it points at.
        """
        return _unwrap_0d(as_array(vector).sum(axis=axis))

    def prod(self, vector, axis=None):
        """Product of a vector or matrix, optionally along one axis."""
        return _unwrap_0d(_np.prod(as_array(vector), axis=axis))

    def scalar_sum(self, parts):
        """Add SCALAR quantities, refusing anything vector-valued.

        A rollup -- a weight statement, a power budget -- is a scalar sum by
        construction.  Both Python's ``sum`` and :meth:`sum` accept a vector
        among the parts and quietly return an ARRAY, which downstream becomes
        one constraint per element instead of one row: a different model that
        still solves.  This says the intent instead, and the seed is the first
        part rather than a dimensionless zero, so units are checked on every
        addition rather than against a bare 0.
        """
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
        """The float behind a number or a declared Constant.

        For the handful of places that need a VALUE at build time rather than
        a symbol in a row: a station guess grid, a branch on whether a term
        exists at all, and every GP EXPONENT -- an exponent is a number, never
        a parameter, so anything used as one has to come through here.
        """
        if isinstance(x, (int, float)):
            return float(x)
        return float(pyo.value(x))

    def broadcast_rows(self, vector, n):
        """``vector`` repeated as each of ``n`` rows -> ``(n, len(vector))``.

        LCsolver never broadcasts silently, so this is how a per-column limit is
        compared against a matrix.
        """
        return broadcast_rows(vector, n)

    def broadcast_cols(self, vector, n):
        """``vector`` repeated as each of ``n`` columns -> ``(len(vector), n)``."""
        return broadcast_cols(vector, n)

    def ConstraintList(self, conList, holographic=False):
        """Declare all the constraints at once; the way a model should be written.

        An entry may be any of three things, and which one it is decides how it
        is handled:

        * an algebraic comparison, ``x + y <= 1.0 * units.m``, which goes to
          `Constraint`;
        * a numpy array of them -- what an elementwise comparison between two
          vectors produces -- which is flattened first, so
          ``f.ConstraintList([M >= f.broadcast_rows(cap, n)])`` declares one
          constraint per element instead of handing a matrix row to
          `Constraint`;
        * a black box, as the four positional parts
          ``[outputs, operators, inputs, box]`` or a dict of the same keywords,
          which goes to `RuntimeConstraint`.

        The three can be mixed freely in one list, which is the point::

            f.ConstraintList([
                [z, '==', [x, y], UnitCircle()],
                x + y <= 1.0 * units.m,
                v >= w,                            # vectors: one per element
            ])

        A runtime constraint has to arrive as its parts rather than as an
        expression, since there is no expression -- that is why entries are
        dispatched on type instead of simply being added.

        A PLAIN LIST OF ROWS IS ALSO AN ENTRY, flattened in place. A helper
        that returns several rows -- a `ConstraintGenerator`, a point maker --
        can be dropped in whole::

            f.ConstraintList([
                cl * solidity == 6. * CT,
                f.polar.generate_rows(cl, tau, Re, cd),      # however many rows it is
            ])

        A black box is told apart from a list of rows by what it IS rather
        than by being a list: four parts whose last is a
        `BlackBoxFunctionModel`. Nothing else about a list is load-bearing, so
        a generator returning one row and a generator returning five read the
        same at the call site.

        ``holographic=True`` marks every algebraic entry as holographic; see
        `HolographicConstraintList`. Nothing is returned.
        """
        for con in _flatten_rows(conList):
            if isinstance(con, dict):
                self.RuntimeConstraint(**con)
            elif isinstance(con, (tuple, list)):
                self.RuntimeConstraint(*con)
            else:
                self.Constraint(con, holographic=holographic)

    def get_variables(self):
        """The variables declared through `Variable`, in declaration order.

        Only those. A Pyomo ``Var`` put on the model with ``add_component`` is
        a perfectly good variable and the solver will treat it as one, but it
        will not appear here, because these lists are the record of what
        LCsolver knows the provenance of -- what has units, a description and a
        stated guess, and can therefore be reported on. If a variable is
        missing from a solution table, this is why.
        """
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._variable_keys
        ]

    def get_constants(self):
        """The constants declared through `Constant`, in declaration order.

        The same restriction as `get_variables`: a ``Param`` added by hand is
        not here. These are the constants `sensitivities` reports against, so
        a number that does not appear in this list is a number no one will ever
        be told the price of.
        """
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._constant_keys
        ]

    def load_constants(self, constants):
        """Set declared constants from an input deck, after the build.

        ``constants`` is a mapping of constant name to value::

            f = build_my_model()
            f.load_constants({'weight_payload': 600.0, 'v_cruise': 62.0})
            result = lcsolver.solve(f)

        This is the counterpart to declaring a number as a `Constant` rather
        than writing it into a constraint. A Constant is a mutable Pyomo
        ``Param``, so a model is built once with its defaults declared inline
        and a deck is applied to the built model, instead of threading a
        configuration dictionary through the constructor and rebuilding for
        every case. Re-solving after a load is the ordinary cycle: nothing is
        reconstructed, so a sweep is a loop over loads and solves.

        Names are the ones `Constant` was called with, group prefixes
        included -- ``'wing_area'`` for a constant declared on
        ``f.group('wing')``. A name that is not a declared constant raises
        ``KeyError`` rather than being ignored, with near-misses suggested,
        because a deck key that silently does nothing is a model that
        silently sizes the wrong thing. A `Constant` declared with ``size``
        takes a sequence of that length.

        Returns the formulation, so a load can be chained onto a build.
        """
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
        # Constants changed, so any structures detected earlier hold a clone
        # with the OLD values.  Solving with those would answer the previous
        # deck's question and report it as this one's, with no error, so the
        # revision is bumped and solve() refuses stale structures.
        self._edi_revision = getattr(self, '_edi_revision', 0) + 1

        return self

    def get_objectives(self):
        """The objectives declared through `Objective`, in declaration order.

        A list because more than one can be declared, not because more than one
        is useful; see `Objective`.
        """
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._objective_keys
        ]

    def get_constraints(self):
        """Every declared constraint, algebraic and runtime alike, in order.

        The two kinds are different objects -- a ``Constraint`` component and
        an ``ExternalGreyBoxBlock`` -- so anything that walks this list has to
        expect both, which is exactly what `check_units` has to do. Use
        `get_explicitConstraints` or `get_runtimeConstraints` to get one kind
        and not the other.
        """
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._allConstraint_keys
        ]

    def get_explicitConstraints(self):
        """The algebraic constraints only -- the ones with an expression.

        These are the constraints anything symbolic can work on: unit
        correction, structure detection, the duals a sensitivity is read from.
        A runtime constraint has no body to read, so it is not here.
        """
        return [
            self.__dict__[nm]
            for nm in self.__dict__.keys()
            if nm in self._constraint_keys
        ]

    def get_runtimeConstraints(self):
        """The black-box constraints only, as their grey-box blocks.

        Non-empty is the fact that decides how the model must be solved: a
        formulation with one of these cannot go through the AMPL-based IPOPT
        route, which has no way to call Python at an iterate, and ``solve``
        takes the cyipopt route instead.
        """
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
        differencing. See :mod:`lcsolver.postsolve.sensitivity` for the details.

        Returns
        -------
        dict
            See :func:`lcsolver.postsolve.sensitivity.sensitivities`.
        """
        from lcsolver.postsolve.sensitivity import sensitivities as _sens

        return _sens(self, normalized=normalized, **kwargs)

    def print_sensitivities(self, **kwargs):
        """Print the sensitivity table, sorted by magnitude."""
        from lcsolver.postsolve.sensitivity import format_sensitivities

        print(format_sensitivities(self.sensitivities(**kwargs)))

    def _detected(self):
        """This formulation's detected structure, units corrected first.

        Detection reads the corrected twin, not the declared model: a
        constraint stated in feet against one in metres is a different set of
        exponents, and classifying the uncorrected form can call the same
        model by a different name.
        """
        from lcsolver.presolve.structureDetector import structure_detector
        from lcsolver.presolve.unitCorrector import unit_corrector
        return structure_detector(unit_corrector(self), bounds_as_rows=False)

    def structure_report(self, top=5):
        """What kind of problem this is, and what stops it being a simpler one.

        Answers the question a class name raises rather than settles: an SP is
        an SP *because of specific constraints*, and they are usually a
        reformulation away from posynomial. Names them, with their bodies.

        ``top`` caps the constraints listed per class; ``top=None`` lists all.
        Returns the text and prints nothing.
        """
        from lcsolver.presolve.reductions import structure_report as _report
        return _report(self, top=top)

    def optimization_check(self, top=5):
        """Every structural check, in one call: structure, then presolve.

        The pre-solve half of the report -- it needs no solution. Pass the
        solved model to `lcsolver.presolve.optimization_check` directly for
        the degeneracy and cancellation checks, which do.
        """
        from lcsolver.presolve.reductions import optimization_check
        return optimization_check(self, structure_top=top)

    def check_units(self):
        """Assert that every declared objective and constraint balances dimensionally.

        Raises ``pyomo.core.base.units_container.UnitsError`` naming the first
        one that does not, and returns nothing otherwise. It is Pyomo's
        ``assert_units_consistent``, applied to what this formulation declared
        -- which is a check that can exist at all only because units are
        mandatory on every `Variable` and `Constant`, so both sides of every
        comparison have something to say.

        Runtime constraints are skipped: a grey-box block has no expression to
        walk. Their units are checked instead where they can be, at the
        boundary, every time the box is evaluated -- inputs converted into the
        units the box declared, outputs and jacobian converted back out of them
        -- and a conversion that cannot be made raises there.

        Calling this is optional, not a step before solving. ``solve`` will not
        run a dimensionally inconsistent model either: unit correction raises
        ``UnitMismatch`` with a report saying which constraint and what the
        correction would be. This is the version to reach for while a model is
        still being written, when there is nothing to solve yet.
        """
        for i in range(1, self._objective_counter + 1):
            assert_units_consistent(self.__dict__['objective_' + str(i)])

        for i in range(1, self._constraint_counter + 1):
            if not isinstance(
                self.__dict__['constraint_' + str(i)],
                pyomo.contrib.pynumero.interfaces.external_grey_box.ExternalGreyBoxBlock,
            ):
                assert_units_consistent(self.__dict__['constraint_' + str(i)])
