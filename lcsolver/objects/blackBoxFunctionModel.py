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

import copy
import pyomo
import pyomo.environ as pyo
import pyomo.core.expr.ndarray
from pyomo.core.expr.numvalue import NumericValue
from pyomo.environ import units as pyomo_units
from pyomo.common.dependencies import attempt_import


egb, egb_available = attempt_import(
    "pyomo.contrib.pynumero.interfaces.external_grey_box"
)

from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.dependencies import scipy, scipy_available


if numpy_available:
    import numpy as np
else:
    raise ImportError(
        "lcsolver requires numpy to enable black box capability, fix with 'pip install numpy' "
    )

if scipy_available:
    import scipy.sparse as sps
else:
    raise ImportError(
        "lcsolver requires scipy to enable black box capability, fix with 'pip install scipy' "
    )

if egb_available:
    from pyomo.contrib.pynumero.interfaces.external_grey_box import (
        ExternalGreyBoxModel,
        ExternalGreyBoxBlock,
    )
else:
    raise ImportError(
        "lcsolver requires pyomo.contrib.pynumero to be installed to enable black box capability, this should have installed with base pyomo"
    )


#: How a dimension of flexible length is stored, and what ``sizeCheck`` looks
#: for when deciding to skip a dimension.
FLEXIBLE_LENGTH = -1

_SIZE_ERROR = (
    'Invalid size %r.  A dimension must be an integer, or np.inf '
    "(equivalently the string 'inf') for a vector of flexible length."
)


def _decode_size_entry(val):
    """One declared dimension, as the integer stored internally.

    ``np.inf`` and the string ``'inf'`` both mean flexible length. Any other
    string is still accepted as flexible: that was the only way to declare it
    before, and a model written against it should keep working.
    """
    if isinstance(val, str):
        return FLEXIBLE_LENGTH
    if isinstance(val, (float, np.floating)):
        if val == np.inf:
            return FLEXIBLE_LENGTH
        raise ValueError(_SIZE_ERROR % (val,))
    if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
        raise ValueError(_SIZE_ERROR % (val,))
    return int(val)


class BlackBoxFunctionModel_Variable(object):
    def __init__(self, name, units, description='', size=0):
        # Order matters
        self.name = name
        self.units = units
        self.size = size
        self.description = description

    # =====================================================================================================================
    # The printing function
    # =====================================================================================================================
    def __repr__(self):
        return self.name

    # =====================================================================================================================
    # Define the name
    # =====================================================================================================================
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        if isinstance(val, str):
            self._name = val
        else:
            raise ValueError('Invalid name.  Must be a string.')

    # =====================================================================================================================
    # Define the units
    # =====================================================================================================================
    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, val):
        # set dimensionless if a null string is passed in
        if isinstance(val, str):
            if val in ['-', '', ' ']:
                val = 'dimensionless'
        if val is None:
            val = 'dimensionless'

        if isinstance(val, str):
            self._units = pyomo_units.__getattr__(val)
        elif isinstance(val, pyomo.core.base.units_container._PyomoUnit):
            self._units = val
        else:
            raise ValueError(
                'Invalid units.  Must be a string compatible with pint or a unit instance.'
            )

    # =====================================================================================================================
    # Define the size
    # =====================================================================================================================
    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, val):
        """The declared shape: 0 for a scalar, an integer, or one per dimension.

        A dimension may be declared of flexible length with ``np.inf``, in
        which case the box takes whatever the model hands it and that
        dimension's length check is skipped -- so one model can serve a
        three element vector here and a ten element one there. The string
        ``'inf'`` means the same thing, which is what this accepted before
        numpy's constant was allowed.
        """
        if isinstance(val, (list, tuple)):
            sizeTemp = []
            for x in val:
                x = _decode_size_entry(x)
                if x == 1:
                    raise ValueError(
                        'A value of 1 is not valid for defining size.  Use fewer dimensions.'
                    )
                sizeTemp.append(x)
            # Store the decoded dimensions rather than the declaration as
            # written. sizeCheck skips a dimension by comparing it against -1,
            # so leaving an np.inf or an 'inf' in the list made a
            # multidimensional flexible size fail its own length check.
            self._size = sizeTemp
        elif val is None:
            self._size = 0  # set to scalar
        else:
            val = _decode_size_entry(val)
            if val == 1:
                raise ValueError(
                    'A value of 1 is not valid for defining size.  Use 0 to indicate a scalar value.'
                )
            self._size = val

    # =====================================================================================================================
    # Define the description
    # =====================================================================================================================
    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, val):
        if isinstance(val, str):
            self._description = val
        else:
            raise ValueError('Invalid description.  Must be a string.')


class TypeCheckedList(list):
    def __init__(self, checkItem, itemList=None):
        super(TypeCheckedList, self).__init__()
        self.checkItem = checkItem

        if itemList is not None:
            if isinstance(itemList, list) or isinstance(itemList, tuple):
                for itm in itemList:
                    self.append(itm)
            else:
                raise ValueError('Input to itemList is not iterable')

    def __setitem__(self, key, val):
        if isinstance(val, self.checkItem):
            super(TypeCheckedList, self).__setitem__(key, val)
        elif isinstance(val, (tuple, list)):
            cks = [isinstance(vl, self.checkItem) for vl in val]
            if sum(cks) == len(cks):
                super(TypeCheckedList, self).__setitem__(key, val)
            else:
                raise ValueError('Input must be an instance of the defined type')
        else:
            raise ValueError('Input must be an instance of the defined type')

    def append(self, val):
        if isinstance(val, self.checkItem):
            super(TypeCheckedList, self).append(val)
        else:
            raise ValueError('Input must be an instance of the defined type')


class BBList(TypeCheckedList):
    def __init__(self):
        super(BBList, self).__init__(BlackBoxFunctionModel_Variable, [])
        self._lookupDict = {}
        self._counter = 0

    def __deepcopy__(self, memo):
        # The generic list-subclass protocol restores the instance __dict__
        # (including the already-populated _lookupDict) and THEN replays the
        # items through our custom ``append``, which trips the duplicate-name
        # guard on the first item -- so deepcopying any populated BBList
        # always raised, and Pyomo's clone() responded by silently setting
        # the enclosing grey-box block's `_ex_model` to None. Rebuild through
        # ``append`` on an empty instance instead; it reconstructs
        # _lookupDict and _counter itself.
        new = self.__class__()
        memo[id(self)] = new
        for item in list(self):
            new.append(copy.deepcopy(item, memo))
        return new

    def __getitem__(self, val):
        if isinstance(val, int):
            return super(BBList, self).__getitem__(val)
        elif isinstance(val, str):
            return super(BBList, self).__getitem__(self._lookupDict[val])
        else:
            raise ValueError('Input must be an integer or a valid variable name')

    def append(*args, **kwargs):
        """Declare one input or output of a black box.

        The call every `BlackBoxFunctionModel` subclass makes in its
        ``__init__``, and the reason a box can be given a model's variables at
        all. Takes ``name, units, description, size`` positionally or by
        keyword, in that order, and builds the
        `BlackBoxFunctionModel_Variable` itself::

            self.inputs.append(name='x', units='ft', description='the x variable')
            self.inputs.append('r', 'm', 'radii', np.inf)

        ``description`` defaults to empty and ``size`` to 0, meaning a scalar;
        ``np.inf`` declares a dimension of flexible length. An already-built
        variable object may be appended instead of its parts.

        Order is what identifies these afterwards -- the model's variables are
        paired with them by position -- so a duplicate name is refused rather
        than silently shadowing: two inputs called ``'x'`` would make the
        by-name lookup wrong for one of them, and `parseInputs` keys its run
        cases by name.
        """
        args = list(args)
        self = args.pop(0)

        if len(args) + len(kwargs.values()) == 1:
            if len(args) == 1:
                inputData = args[0]
            if len(kwargs.values()) == 1:
                inputData = list(kwargs.values())[0]

            if isinstance(inputData, self.checkItem):
                if inputData.name in self._lookupDict.keys():
                    raise ValueError(
                        "Key '%s' already exists in the input list" % (inputData.name)
                    )
                self._lookupDict[inputData.name] = self._counter
                self._counter += 1
                super(BBList, self).append(inputData)
            else:
                if isinstance(inputData, str):
                    raise ValueError(
                        "Key '%s' not passed in to the black box variable constructor"
                        % ('units')
                    )
                else:
                    raise ValueError('Invalid (single) input type')

        elif len(args) + len(kwargs.values()) <= 4:
            argKeys = ['name', 'units', 'description', 'size']
            ipd = dict(zip(argKeys[0 : len(args)], args))
            for ky, vl in kwargs.items():
                if ky in ipd:
                    raise ValueError(
                        "Key '%s' declared after non-keyword arguments and is out of order"
                        % (ky)
                    )
                else:
                    ipd[ky] = vl

            for ak in argKeys:
                if ak not in ipd.keys():
                    if ak == 'description':
                        ipd['description'] = ''
                    elif ak == 'size':
                        ipd['size'] = 0
                    else:
                        raise ValueError(
                            "Key '%s' not passed in to the black box variable constructor"
                            % (ak)
                        )

            if ipd['name'] in self._lookupDict.keys():
                raise ValueError(
                    "Key '%s' already exists in the input list" % (ipd['name'])
                )
            self._lookupDict[ipd['name']] = self._counter
            self._counter += 1
            super(BBList, self).append(BlackBoxFunctionModel_Variable(**ipd))

        else:
            raise ValueError('Too many inputs to a black box variable')


errorString = 'This function is calling to the base class and has not been defined.'


class BlackBoxFunctionModel(ExternalGreyBoxModel):
    """Base class for wrapping an analysis code so it can be a constraint.

    Subclass it, declare what goes in and what comes out, and write
    ``BlackBox``. The model can then constrain the outputs to equal the box
    evaluated at the inputs, and the solver calls it at every iterate::

        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):
                super().__init__()
                self.description = 'This model evaluates z = x**2 + y**2'

                self.inputs.append(name='x', units='ft', description='the x variable')
                self.inputs.append(name='y', units='ft', description='the y variable')
                self.outputs.append(name='z', units='ft**2', description='the result')

                self.availableDerivative = 1

            def BlackBox(self, x, y):
                x, y = self.sanitizeInputs(x, y, strip_units=True)
                return self.packOutputs(x**2 + y**2, [2 * x, 2 * y])

        f.ConstraintList([[z, '==', [x, y], UnitCircle()]])

    The declarations are what makes this more than a function pointer. An
    analysis code has units it works in and shapes it expects, and they are
    routinely not the model's -- the box above thinks in feet while the model
    is in metres, and neither side's source says so anywhere. Because both
    declared, the conversion happens at the boundary in both directions, values
    and jacobian alike, and a shape that does not match its declaration raises
    there rather than being quietly reshaped somewhere deeper. The alternative,
    which is what wrapping an analysis code by hand looks like, is a factor of
    0.3048 living in a comment.

    Three attributes are set in ``__init__``:

    ``inputs``, ``outputs``
        `BBList` of declarations, appended in the order the box takes them.
        ``append`` accepts ``name, units, description, size`` positionally or
        by keyword. Order matters: the model's variables are matched to these
        by position, not by name (see `Formulation.RuntimeConstraint`).
    ``description``
        A sentence about the model, printed by `getSummary`.
    ``availableDerivative``
        The highest derivative the box can return -- 0 for values only, 1 for
        values and a jacobian. `parseInputs` reports it back to the box so one
        implementation can serve both. A box used inside a solve must supply
        the jacobian: the solver asks for it separately from the values, and
        LCsolver has nothing to fall back on if it is absent.

    A box can be evaluated on its own -- ``UnitCircle().BlackBox(0.5*units.m,
    0.5*units.m)`` -- which is worth doing before putting it in a model, since
    it is where a units or shape mistake announces itself with the shortest
    stack.

    The rest of the class is either machinery the solver calls (`input_names`,
    `set_input_values`, `evaluate_outputs`, `evaluate_jacobian_outputs`, from
    Pyomo's ``ExternalGreyBoxModel``, all driven by `fillCache`) or helpers a
    ``BlackBox`` implementation calls: `sanitizeInputs`, `packOutputs`,
    `parseInputs`, `convert`, `pyomo_value`.
    """

    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super(BlackBoxFunctionModel, self).__init__()

        # List of the inputs and outputs
        self.inputs = BBList()
        self.outputs = BBList()

        self.inputVariables_optimization = None
        self.outputVariables_optimization = None

        # A simple description of the model
        self.description = None

        # Defines the order of derivative available in the black box
        self.availableDerivative = 0

        self._cache = None
        self._NunwrappedOutputs = None
        self._NunwrappedInputs = None

    def __deepcopy__(self, memo):
        # A black box routinely holds a handle to the analysis it drives -- a
        # pyCAPS Problem, a CFD session, a ctypes/SWIG wrapper -- and such
        # handles refuse deepcopy ("ctypes objects containing pointers cannot
        # be pickled"). Under the generic protocol that single attribute
        # aborted the whole model clone() in unit_corrector, killing every
        # solve of a formulation whose box stored its analysis object. Copy
        # attribute-by-attribute instead, and share by reference anything that
        # refuses: the clone must drive the SAME external analysis -- a
        # stateful resource cannot be meaningfully duplicated anyway.
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, val in self.__dict__.items():
            try:
                new.__dict__[key] = copy.deepcopy(val, memo)
            except Exception:
                new.__dict__[key] = val
        return new

    def setOptimizationVariables(
        self, inputVariables_optimization, outputVariables_optimization
    ):
        """Record which of the model's variables this box is wired to.

        Called by `Formulation.RuntimeConstraint` when the box is attached, not
        by a model author. The two lists are matched to `inputs` and `outputs`
        by position, and that pairing is what the unit conversion at each
        evaluation reads: entry ``i`` of this list is in the model's units,
        ``self.inputs[i]`` says what units the box wants.
        """
        self.inputVariables_optimization = inputVariables_optimization
        self.outputVariables_optimization = outputVariables_optimization

    # ---------------------------------------------------------------------------------------------------------------------
    # pyomo things
    # ---------------------------------------------------------------------------------------------------------------------
    def input_names(self):
        """The model-side input names, one per scalar, for pynumero.

        Part of Pyomo's ``ExternalGreyBoxModel`` interface rather than
        something to call. An indexed variable is unwrapped here into one name
        per element -- ``x[0]``, ``x[1]``, ... -- because the solver deals in a
        flat vector and needs a name for each column of it.
        """
        inputs_unwrapped = []
        for ivar in self.inputVariables_optimization:
            if isinstance(ivar, pyomo.core.base.var.ScalarVar):
                inputs_unwrapped.append(ivar)
            elif isinstance(ivar, pyomo.core.base.var.IndexedVar):
                validIndices = list(ivar.index_set().data())
                for vi in validIndices:
                    inputs_unwrapped.append(ivar[vi])
            else:
                raise ValueError("Invalid type for input variable")

        return [ip.__str__() for ip in inputs_unwrapped]

    def output_names(self):
        """The model-side output names, one per scalar. See `input_names`."""
        outputs_unwrapped = []
        for ovar in self.outputVariables_optimization:
            if isinstance(ovar, pyomo.core.base.var.ScalarVar):
                outputs_unwrapped.append(ovar)
            elif isinstance(ovar, pyomo.core.base.var.IndexedVar):
                validIndices = list(ovar.index_set().data())
                for vi in validIndices:
                    outputs_unwrapped.append(ovar[vi])
            else:
                raise ValueError("Invalid type for output variable")

        return [op.__str__() for op in outputs_unwrapped]

    def set_input_values(self, input_values):
        """The solver's flat vector for the next iterate. Invalidates the cache.

        Part of Pyomo's ``ExternalGreyBoxModel`` interface. Dropping the cache
        here is what ties the box to the iterate: the outputs and the jacobian
        are asked for in separate calls, so they must be recomputed once and
        only once per point, and never carried across one.

        A re-set to the bit-identical point keeps the cache. The SIA machinery
        (sub-problem, restore, line search) routinely hands back the same
        iterate it just evaluated; on the capsPhase AVL demo 260 of 510
        BlackBox calls were exact repeats, each costing three AVL runs. Only
        exact equality qualifies -- any numerical difference, however small, is
        a new point and must be re-evaluated.
        """
        if self._cache is not None and np.array_equal(
                np.asarray(input_values), np.asarray(self._input_values)):
            self._input_values = input_values
            return
        self._input_values = input_values
        self._cache = None

    def evaluate_outputs(self):
        """The output values at the current iterate. Calls `fillCache`."""
        self.fillCache()
        opts = self._cache['pyomo_outputs']
        return opts

    def evaluate_jacobian_outputs(self):
        """The jacobian at the current iterate, sparse. Calls `fillCache`.

        Asked for separately from the values but computed with them, in one
        call to `BlackBox` -- an analysis code is usually expensive enough that
        evaluating it twice per iterate would be the dominant cost.
        """
        self.fillCache()
        jac = self._cache['pyomo_jacobian']
        return jac

    def post_init_setup(self, defaultVal=1.0):
        """Give the box a starting input vector, once its length is known.

        Called by `Formulation.RuntimeConstraint` after the box is attached,
        which is the first moment the number of scalar inputs exists -- it
        comes from the model's variables, not from the declarations, since a
        declared input may be of flexible length.
        """
        # _NunwrappedInputs is assigned when the black box is attached to a
        # Formulation (see formulation.py); at construction time it is still
        # None. numpy >= 2.0 rejects None as a shape, so fall back to a scalar
        # placeholder (the numpy 1.x behavior) until the real size is known.
        n = self._NunwrappedInputs
        self._input_values = np.ones(() if n is None else n) * defaultVal

    def attachUnits(self, val, unts):
        """Give a bare number the units it is understood to already be in.

        A plain float or ndarray coming back from a box is taken to be a
        magnitude in the units declared for that output, so attaching ``unts``
        loses no information and puts the value on the normal conversion path.
        Anything that already carries units is returned untouched, so this is
        safe to apply to either.
        """
        # A black box that works in dimensionless quantities naturally returns
        # plain numbers: pyomo collapses expressions such as
        # 'float * dimensionless' back to a float, and numpy operations on
        # dimensionless arrays return plain ndarrays.  Such a value is taken to
        # already be in the units declared for that input/output, so the units
        # are attached here and the normal conversion path is used from there.
        if isinstance(val, np.ndarray):
            if isinstance(val, pyomo.core.expr.ndarray.NumericNDArray):
                return val
            return val * unts
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float, np.integer, np.floating)):
            return val * unts
        return val

    def fillCache(self):
        """Cross the boundary once: call `BlackBox` and store what came back.

        This is where the model's world and the box's world are reconciled, and
        it is the whole reason the inputs and outputs are declared. Going in,
        the solver's flat vector is cut back up into the shapes the box's
        variables have, each piece converted from the model's units into the
        units the box declared, and size-checked. Coming out, values and every
        jacobian entry are converted the other way -- an entry of
        ``d(output)/d(input)`` being converted in the compound units
        ``output units / input units``, which is the part no one gets right by
        hand -- and laid into a dense array that becomes the sparse block the
        solver reads.

        Cached because the solver asks for values and jacobian separately and
        an analysis code is usually the expensive thing in the loop;
        `set_input_values` drops the cache, so there is exactly one call to
        `BlackBox` per iterate.

        The result is published only once every step has succeeded. Filling
        ``self._cache`` in place left a partial but non-None cache behind
        whenever anything raised, and the next call then skipped the rebuild
        and failed with a ``KeyError`` that hid the original error.
        """
        if self._cache is None:
            # Build into a local dict and only publish it once every step has
            # succeeded.  Populating self._cache in place leaves a partially
            # filled (but non-None) cache behind if anything below raises, and
            # the next call then skips the rebuild and fails with a spurious
            # KeyError that hides the original error.
            cache = {}

            raw_inputs = self._input_values
            bb_inputs = []

            ptr = 0

            for i in range(0, len(self.inputVariables_optimization)):
                optimizationInput = self.inputVariables_optimization[i]
                if not isinstance(
                    optimizationInput,
                    (pyomo.core.base.var.IndexedVar, pyomo.core.base.var.ScalarVar),
                ):
                    raise ValueError(
                        "Invalid input variable type for input %d ('%s'): expected a "
                        "pyomo ScalarVar or IndexedVar, received %s"
                        % (i, self.inputs[i].name, type(optimizationInput).__name__)
                    )

                ipt = self.inputs[i]

                shape = [len(idx) for idx in optimizationInput.index_set().subsets()]
                localShape = ipt.size

                optimizationUnits = self.inputVariables_optimization[i].get_units()
                localUnits = ipt.units

                if isinstance(optimizationInput, pyomo.core.base.var.IndexedVar):
                    value = np.zeros(shape)
                    for vix in list(optimizationInput.index_set().data()):
                        raw_val = float(raw_inputs[ptr]) * optimizationUnits
                        raw_val_correctedUnits = pyomo_units.convert(
                            raw_val, localUnits
                        )
                        value[vix] = pyo.value(raw_val_correctedUnits)
                        ptr += 1
                    self.sizeCheck(localShape, value * localUnits)
                    bb_inputs.append(value * localUnits)

                else:  # isinstance(optimizationInput, pyomo.core.base.var.ScalarVar):, just checked this
                    value = raw_inputs[ptr] * optimizationUnits
                    ptr += 1
                    self.sizeCheck(localShape, value)
                    value_correctedUnits = pyomo_units.convert(value, localUnits)
                    bb_inputs.append(value_correctedUnits)

            bbo = self.BlackBox(*bb_inputs)

            cache['raw'] = bbo
            cache['raw_value'] = bbo[0]
            cache['raw_jacobian'] = bbo[1]

            outputVector = []
            if not isinstance(bbo[0], (list, tuple)):
                valueList = [bbo[0]]
                jacobianList = [bbo[1]]
            else:
                valueList = bbo[0]
                jacobianList = bbo[1]
            for i in range(0, len(valueList)):
                optimizationOutput = self.outputVariables_optimization[i]
                if not isinstance(
                    optimizationOutput,
                    (pyomo.core.base.var.IndexedVar, pyomo.core.base.var.ScalarVar),
                ):
                    raise ValueError(
                        "Invalid output variable type for output %d ('%s'): expected a "
                        "pyomo ScalarVar or IndexedVar, received %s"
                        % (i, self.outputs[i].name, type(optimizationOutput).__name__)
                    )
                opt = self.outputs[i]

                modelOutputUnits = opt.units
                outputOptimizationUnits = optimizationOutput.get_units()
                vl = self.attachUnits(valueList[i], modelOutputUnits)
                if isinstance(vl, pyomo.core.expr.ndarray.NumericNDArray):
                    validIndexList = optimizationOutput.index_set().data()
                    for j in range(0, len(validIndexList)):
                        vi = validIndexList[j]
                        corrected_value = pyo.value(
                            pyomo_units.convert(vl[vi], outputOptimizationUnits)
                        )  # now unitless in correct units
                        outputVector.append(corrected_value)

                elif isinstance(vl, NumericValue):
                    # Any united scalar: _PyomoUnit, or one of the NPV_*
                    # expressions that unit arithmetic produces (a product for
                    # 'val * units.ft', but a division for 'val * units.ft/units.s'
                    # and a power for '1.0 * units.ft**2').
                    corrected_value = pyo.value(
                        pyomo_units.convert(vl, outputOptimizationUnits)
                    )  # now unitless in correct units
                    outputVector.append(corrected_value)

                else:
                    raise ValueError(
                        "Invalid value returned by the black box for output %d ('%s'): "
                        "expected a pyomo united scalar or array, or a plain number in "
                        "units of %s, received %s"
                        % (
                            i,
                            self.outputs[i].name,
                            str(modelOutputUnits),
                            type(valueList[i]).__name__,
                        )
                    )

            cache['pyomo_outputs'] = outputVector

            outputJacobian = (
                np.ones([self._NunwrappedOutputs, self._NunwrappedInputs]) * -1
            )
            ptr_row = 0
            ptr_col = 0

            for i in range(0, len(jacobianList)):
                oopt = self.outputVariables_optimization[i]
                # Checked about 20 lines above
                # if not isinstance(oopt, (pyomo.core.base.var.ScalarVar,pyomo.core.base.var.IndexedVar)):
                #     raise ValueError("Invalid type for output variable")
                lopt = self.outputs[i]
                oounits = oopt.get_units()
                lounits = lopt.units
                # oshape  = [len(idx) for idx in oopt.index_set().subsets()]
                ptr_col = 0
                for j in range(0, len(self.inputs)):
                    oipt = self.inputVariables_optimization[j]
                    # This is checked about 80 lines up
                    # if not isinstance(oipt, (pyomo.core.base.var.ScalarVar,pyomo.core.base.var.IndexedVar)):
                    #     raise ValueError("Invalid type for output variable")
                    lipt = self.inputs[j]
                    oiunits = oipt.get_units()
                    liunits = lipt.units
                    # ishape  = [len(idx) for idx in oipt.index_set().subsets()]

                    jacobianValue_raw = self.attachUnits(
                        jacobianList[i][j], lounits / liunits
                    )

                    if isinstance(jacobianValue_raw, NumericValue):
                        corrected_value = pyo.value(
                            pyomo_units.convert(jacobianValue_raw, oounits / oiunits)
                        )  # now unitless in correct units
                        outputJacobian[ptr_row, ptr_col] = corrected_value
                        ptr_col += 1
                        ptr_row_step = 1

                    elif isinstance(
                        jacobianValue_raw, pyomo.core.expr.ndarray.NumericNDArray
                    ):
                        jshape = jacobianValue_raw.shape

                        if isinstance(oopt, pyomo.core.base.var.ScalarVar):
                            oshape = 0
                        else:  # isinstance(oopt, pyomo.core.base.var.IndexedVar), checked above
                            oshape = [len(idx) for idx in oopt.index_set().subsets()]

                        if isinstance(oipt, pyomo.core.base.var.ScalarVar):
                            ishape = 0
                        else:  # isinstance(oipt, pyomo.core.base.var.IndexedVar), checked above
                            ishape = [len(idx) for idx in oipt.index_set().subsets()]

                        if oshape == 0:
                            validIndices = list(oipt.index_set().data())
                            for vi in validIndices:
                                corrected_value = pyo.value(
                                    pyomo_units.convert(
                                        jacobianValue_raw[vi], oounits / oiunits
                                    )
                                )  # now unitless in correct units
                                outputJacobian[ptr_row, ptr_col] = corrected_value
                                ptr_col += 1
                            ptr_row_step = 1

                        elif ishape == 0:
                            ptr_row_cache = ptr_row
                            validIndices = list(oopt.index_set().data())
                            for vi in validIndices:
                                corrected_value = pyo.value(
                                    pyomo_units.convert(
                                        jacobianValue_raw[vi], oounits / oiunits
                                    )
                                )  # now unitless in correct units
                                outputJacobian[ptr_row, ptr_col] = corrected_value
                                ptr_row += 1
                            ptr_row = ptr_row_cache
                            # A scalar input occupies one column of the block,
                            # which the next input must start after.
                            ptr_col += 1
                            ptr_row_step = len(validIndices)

                        # elif ishape == 0 and oshape == 0: # Handled by the scalar case above

                        else:
                            # both are dimensioned vectors
                            # oshape, ishape, jshape
                            ptr_row_cache = ptr_row
                            ptr_col_cache = ptr_col
                            validIndices_o = list(oopt.index_set().data())
                            validIndices_i = list(oipt.index_set().data())

                            for vio in validIndices_o:
                                if isinstance(vio, (float, int)):
                                    vio = (vio,)
                                for vii in validIndices_i:
                                    if isinstance(vii, (float, int)):
                                        vii = (vii,)
                                    corrected_value = pyo.value(
                                        pyomo_units.convert(
                                            jacobianValue_raw[vio + vii],
                                            oounits / oiunits,
                                        )
                                    )  # now unitless in correct units
                                    outputJacobian[ptr_row, ptr_col] = corrected_value
                                    ptr_col += 1
                                ptr_col = ptr_col_cache
                                ptr_row += 1
                            ptr_row = ptr_row_cache
                            # The block spans one column per input element; the
                            # inner loop rewinds ptr_col for the next row, so
                            # step past the whole block once the rows are done.
                            ptr_col = ptr_col_cache + len(validIndices_i)
                            ptr_row_step = len(validIndices_o)

                    else:
                        raise ValueError(
                            "Invalid jacobian type for d(output %d, '%s')/d(input %d, "
                            "'%s'): expected a pyomo united scalar or array, or a plain "
                            "number in units of %s, received %s"
                            % (
                                i,
                                lopt.name,
                                j,
                                lipt.name,
                                str(lounits / liunits),
                                type(jacobianList[i][j]).__name__,
                            )
                        )

                ptr_row += ptr_row_step

            cache['pyomo_jacobian'] = sps.coo_matrix(outputJacobian)
            self._cache = cache

    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    # These models must be defined in each individual model, just placeholders here
    def BlackBox(*args, **kwargs):
        """The analysis itself. Every subclass defines this; the base refuses.

        LCsolver calls it with one argument per declared input, in declaration
        order, each carrying the units that input was declared with. So the
        first line of a ``BlackBox`` is almost always
        ``self.sanitizeInputs(..., strip_units=True)``, which converts and
        size-checks them and hands back plain numbers to compute with.

        What comes back has to be in the declared *output* units, and this is
        the half that is easy to get wrong, because the jacobian's units are
        not written down anywhere -- ``d z / d x`` is ``ft**2 / ft`` and
        nothing in the code says so. `packOutputs` derives them and is the
        recommended way to return::

            return self.packOutputs(z, [dzdx, dzdy])

        Returning them by hand is equally valid, in which case the contract is:
        a single-output box returns ``(value, [dv_din0, dv_din1, ...])``, and a
        multi-output box returns ``([v0, v1, ...], [[...], [...]])`` -- one
        jacobian row per output, one entry per input. Drop the second element
        entirely for a box that returns values only.

        The signature is the subclass's to choose. ``def BlackBox(self, x, y)``
        is the ordinary form. ``def BlackBox(self, *args, **kwargs)`` fronted by
        `parseInputs` is what a box is written as when it should also accept a
        batch of run cases, keyword inputs, or options of its own.

        The base class raises ``AttributeError``: reaching it means the
        subclass never defined the one method it exists to define.
        """
        raise AttributeError(errorString)

    def convert(self, val, unts):
        """Convert a value to ``unts``, whether it is a scalar or an array.

        ``pyomo_units.convert`` handles a united scalar and nothing else; an
        array of them has to be converted element by element, which is what
        this adds. A bare number is treated as dimensionless first, so it
        converts only into a dimensionless target and raises otherwise rather
        than being assumed to be in whatever was wanted.
        """
        try:
            val = val * pyomo_units.dimensionless
        except:
            pass  ## will handle later

        if isinstance(
            val,
            (
                pyomo.core.base.units_container._PyomoUnit,
                pyomo.core.expr.numeric_expr.NPV_ProductExpression,
            ),
        ):
            return pyomo_units.convert(val, unts)
        elif isinstance(val, pyomo.core.expr.ndarray.NumericNDArray):
            shp = val.shape
            ix = np.ndindex(*shp)
            opt = np.zeros(shp)
            for i in range(0, np.prod(shp)):
                ixt = next(ix)
                opt[ixt] = pyo.value(pyomo_units.convert(val[ixt], unts))
            return opt * unts
        else:
            raise ValueError('Invalid type passed to unit conversion function')

    def pyomo_value(self, val):
        """The number behind a united value, elementwise for an array.

        ``pyo.value`` on an array of united elements raises; this walks it.
        The magnitude is in whatever units the value was carrying, so this is
        only safe once the value has been converted -- which is why
        `sanitizeInputs` converts before it strips.
        """
        try:
            return pyo.value(val)
        except:
            if isinstance(val, pyomo.core.expr.ndarray.NumericNDArray):
                shp = val.shape
                ix = np.ndindex(*shp)
                opt = np.zeros(shp)
                for i in range(0, np.prod(shp)):
                    ixt = next(ix)
                    opt[ixt] = pyo.value(val[ixt])
                return opt
            else:
                raise ValueError('Invalid type passed to pyomo_value function')

    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    def parseInputs(self, *args, **kwargs):
        """Normalise however a ``BlackBox`` was called into a list of run cases.

        Writing a black box is expensive, so it is worth having one serve every
        way it might be reached: the solver calling it with one value per
        input, a script sweeping a vector of them, a caller passing a dict, a
        caller passing options the box understands but the optimizer knows
        nothing about. This turns all of those into the same thing. A box that
        wants that generality opens with::

            def BlackBox(self, *args, **kwargs):
                runCases, returnMode, extras = self.parseInputs(*args, **kwargs)

        Every one of these is understood, and each yields the run cases you
        would expect::

            bb(x, y)                        # positional, in declaration order
            bb(x=x, y=y)                    # by name, or a mix of the two
            bb({'x': x, 'y': y})            # one case as a dict
            bb({'x': xs, 'y': ys})          # one case per element of xs, ys
            bb([{'x': x, 'y': y}, {...}])   # a list of cases as dicts
            bb([[x, y], [x, y]])            # a list of cases as sequences

        ``bb([x, y])`` is deliberately *not* in that list. It is ambiguous
        between one case and a batch, and rather than guess, the error names
        the four ways to say which was meant.

        Returns ``(runCases, returnMode, extras)``.

        ``runCases`` is a list of dicts keyed by declared input name, values
        already converted into the declared input units and size-checked, since
        each case goes through `sanitizeInputs` on the way through. There is
        always a list, even for a single case: ``runCases[0]['x']``.

        ``returnMode`` tells the box what shape its return should take, and is
        the only way it can know, because the call ``bb(x, y)`` and the call
        ``bb({'x': xs, 'y': ys})`` produce identical run-case lists when the
        vectors have length one. It is ``availableDerivative`` when the call
        described several cases, and ``-availableDerivative - 1`` -- so,
        negative -- when it described exactly one. Negative therefore means
        "one fewer level of indexing on the way out", and the derivative order
        is recovered as ``-(returnMode + 1)``, which is what the idiom
        ``if returnMode < 0: returnMode = -1 * (returnMode + 1)`` at the bottom
        of a batch-capable box is doing.

        ``extras`` is everything passed that was not a declared input: extra
        keywords as themselves, extra positional arguments collected under
        ``extras['remainingArgs']``. This is where a box finds its own options
        -- a tolerance, a mesh level, a run mode -- and it is only ever
        populated for the positional/keyword call forms; the dict and list-of-
        cases forms return ``{}``, having no room for anything else.
        """
        args = list(args)  # convert tuple to list

        inputNames = [self.inputs[i].name for i in range(0, len(self.inputs))]

        # ------------------------------
        # ------------------------------
        if len(args) + len(kwargs.values()) == 1:
            if len(args) == 1:
                inputData = args[0]
            if len(kwargs.values()) == 1:
                inputData = list(kwargs.values())[0]

            if len(inputNames) == 1:
                try:
                    rs = self.sanitizeInputs(inputData)
                    return (
                        [dict(zip(inputNames, [rs]))],
                        -self.availableDerivative - 1,
                        {},
                    )  # one input being passed in
                except:
                    pass  # otherwise, proceed

            if isinstance(inputData, (list, tuple)):
                dataRuns = []
                for idc in inputData:
                    if isinstance(idc, dict):
                        sips = self.sanitizeInputs(**idc)
                        if len(inputNames) == 1:
                            sips = [sips]
                        runDictS = dict(zip(inputNames, sips))
                        dataRuns.append(
                            runDictS
                        )  # the BlackBox([{'x1':x1, 'x2':x2},{'x1':x1, 'x2':x2},...]) case
                    elif isinstance(idc, (list, tuple)):
                        if len(idc) == len(inputNames):
                            sips = self.sanitizeInputs(*idc)
                            if len(inputNames) == 1:
                                sips = [sips]
                            runDictS = dict(zip(inputNames, sips))
                            dataRuns.append(
                                runDictS
                            )  # the BlackBox([ [x1, x2], [x1, x2],...]) case
                        else:
                            raise ValueError(
                                'Entry in input data list has improper length'
                            )
                    else:
                        raise ValueError(
                            "Invalid data type in the input list.  Note that BlackBox([x1,x2,y]) must be passed in as BlackBox([[x1,x2,y]]) or "
                            + "BlackBox(*[x1,x2,y]) or BlackBox({'x1':x1,'x2':x2,'y':y}) or simply BlackBox(x1, x2, y) to avoid processing singularities.  "
                            + "Best practice is BlackBox({'x1':x1,'x2':x2,'y':y})"
                        )
                return dataRuns, self.availableDerivative, {}

            elif isinstance(inputData, dict):
                if set(list(inputData.keys())) == set(inputNames):
                    try:
                        inputLengths = [len(inputData[kw]) for kw in inputData.keys()]
                    except:
                        sips = self.sanitizeInputs(**inputData)
                        if len(inputNames) == 1:
                            sips = [sips]
                        return (
                            [dict(zip(inputNames, sips))],
                            -self.availableDerivative - 1,
                            {},
                        )  # the BlackBox(*{'x1':x1, 'x2':x2}) case, somewhat likely

                    if not all(
                        [
                            inputLengths[i] == inputLengths[0]
                            for i in range(0, len(inputLengths))
                        ]
                    ):
                        sips = self.sanitizeInputs(**inputData)
                        return (
                            [dict(zip(inputNames, sips))],
                            -self.availableDerivative - 1,
                            {},
                        )  # the BlackBox(*{'x1':x1, 'x2':x2}) case where vectors for x1... are passed in (likely to fail on previous line)
                    else:
                        try:
                            sips = self.sanitizeInputs(**inputData)
                            return (
                                [dict(zip(inputNames, sips))],
                                -self.availableDerivative - 1,
                                {},
                            )  # the BlackBox(*{'x1':x1, 'x2':x2}) case where vectors all inputs have same length intentionally (likely to fail on previous line)
                        except:
                            dataRuns = []
                            for i in range(0, inputLengths[0]):
                                runDict = {}
                                for ky, vl in inputData.items():
                                    runDict[ky] = vl[i]
                                sips = self.sanitizeInputs(**runDict)
                                if len(inputNames) == 1:
                                    sips = [sips]
                                runDictS = dict(zip(inputNames, sips))
                                dataRuns.append(runDictS)
                            return (
                                dataRuns,
                                self.availableDerivative,
                                {},
                            )  # the BlackBox({'x1':x1_vec, 'x2':x2_vec}) case, most likely
                else:
                    raise ValueError('Keywords did not match the expected list')
            else:
                raise ValueError('Got unexpected data type %s' % (str(type(inputData))))
        # ------------------------------
        # ------------------------------
        else:
            if any(
                [
                    list(kwargs.keys())[i] in inputNames
                    for i in range(0, len(list(kwargs.keys())))
                ]
            ):
                # some of the inputs are defined in the kwargs
                if len(args) >= len(inputNames):
                    raise ValueError(
                        'A keyword input is defining an input, but there are too many unkeyed arguments for this to occur.  Check the inputs.'
                    )
                else:
                    if len(args) != 0:
                        availableKeywords = inputNames[-len(args) :]
                    else:
                        availableKeywords = inputNames

                    valList = args + [None] * (len(inputNames) - len(args))
                    for ky in availableKeywords:
                        ix = inputNames.index(ky)
                        valList[ix] = kwargs[ky]

                    # Not possible to reach due to other checks
                    # if any([valList[i]==None for i in range(0,len(valList))]):
                    #     raise ValueError('Keywords did not properly fill in the remaining arguments. Check the inputs.')

                    sips = self.sanitizeInputs(*valList)

                    if len(inputNames) == 1:
                        sips = [sips]

                    remainingKwargs = copy.deepcopy(kwargs)
                    for nm in inputNames:
                        try:
                            del remainingKwargs[nm]
                        except:
                            # was in args
                            pass

                    return (
                        [dict(zip(inputNames, sips))],
                        -self.availableDerivative - 1,
                        remainingKwargs,
                    )  # Mix of args and kwargs define inputs
            else:
                # all of the inputs are in the args
                try:
                    sips = self.sanitizeInputs(*args[0 : len(inputNames)])

                    if len(inputNames) == 1:
                        sips = [sips]

                    remainingKwargs = copy.deepcopy(kwargs)
                    remainingKwargs['remainingArgs'] = args[len(inputNames) :]
                    return (
                        [dict(zip(inputNames, sips))],
                        -self.availableDerivative - 1,
                        remainingKwargs,
                    )  # all inputs are in args
                # except:
                except Exception as e:
                    # not possible due to other checks
                    # if str(e) == 'Too many inputs':
                    #     raise ValueError(e)
                    if str(e) == 'Not enough inputs':
                        raise ValueError(e)
                    else:  # otherwise, proceed
                        runCases, returnMode, extra_singleInput = self.parseInputs(
                            args[0]
                        )
                        remainingKwargs = copy.deepcopy(kwargs)
                        remainingKwargs['remainingArgs'] = args[len(inputNames) :]
                        return runCases, returnMode, remainingKwargs

    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    def sizeCheck(self, size, ipval_correctUnits):
        """Check a value against a declared size, and raise if it disagrees.

        ``size`` is the decoded form held on a
        `BlackBoxFunctionModel_Variable`: ``0`` for a scalar, an integer, or a
        list with one entry per dimension. A dimension stored as
        ``FLEXIBLE_LENGTH`` (-1) was declared ``np.inf`` and is skipped, which
        is what lets one box serve a three-element vector in one model and a
        ten-element one in another.

        The rank has to match as well as the lengths. A value of the right
        total size but the wrong shape is a different quantity, and numpy would
        take it without comment.
        """
        if size is not None:
            szVal = ipval_correctUnits
            if isinstance(
                szVal,
                (
                    pyomo.core.expr.numeric_expr.NPV_ProductExpression,
                    pyomo.core.base.units_container._PyomoUnit,
                ),
            ):
                if size != 0 and size != 1:
                    raise ValueError(
                        'Size did not match the expected size %s (ie: Scalar)'
                        % (str(size))
                    )
            elif isinstance(szVal, pyomo.core.expr.ndarray.NumericNDArray):
                shp = szVal.shape
                if isinstance(size, (int, float)):
                    size = [size]
                # else:
                if len(shp) != len(size):
                    raise ValueError(
                        'Shapes/Sizes of %s does not match the expected %s'
                        % (str(shp), str(size))
                    )
                for j in range(0, len(shp)):
                    if size[j] != -1:  # was declared of flexible length
                        if size[j] != shp[j]:
                            raise ValueError(
                                'Shapes/Sizes of %s does not match the expected %s'
                                % (str(shp), str(size))
                            )
            else:
                raise ValueError('Invalid type detected when checking size')

    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    def sanitizeInputs(self, *args, **kwargs):
        """Validate and unit-convert the values handed to ``BlackBox``.

        Accepts the inputs positionally (in declaration order) or by name,
        converts each to its declared input units, and size-checks it. By
        default the returned values keep their units. Pass
        ``strip_units=True`` to get plain floats/arrays instead -- these are
        magnitudes IN THE DECLARED INPUT UNITS (conversion happens first,
        then the units are stripped), which is what makes stripping safe.
        A single declared input is returned bare; several come back as a
        list: ``x, y = self.sanitizeInputs(x, y, strip_units=True)``.
        """
        nameList = [self.inputs[i].name for i in range(0, len(self.inputs))]

        strip_units = False
        if 'strip_units' in kwargs:
            if 'strip_units' in nameList:
                raise ValueError(
                    "an input named 'strip_units' collides with the "
                    "strip_units keyword of sanitizeInputs; rename the input")
            strip_units = kwargs.pop('strip_units')

        if len(args) + len(kwargs.values()) > len(nameList):
            raise ValueError('Too many inputs')
        if len(args) + len(kwargs.values()) < len(nameList):
            raise ValueError('Not enough inputs')
        inputDict = {}
        for i in range(0, len(args)):
            rg = args[i]
            inputDict[nameList[i]] = rg

        for ky, vl in kwargs.items():
            if ky in nameList:
                inputDict[ky] = vl
            else:
                raise ValueError(
                    'Unexpected input keyword argument %s in the inputs' % (ky)
                )

        opts = []

        for i in range(0, len(nameList)):
            name = nameList[i]
            nameCheck = self.inputs[i].name
            unts = self.inputs[i].units
            size = self.inputs[i].size

            # should be impossible
            # if name != nameCheck:
            #     raise RuntimeError('Something went wrong and values are not consistent.  Check your defined inputs.')

            ipval = inputDict[name]

            if isinstance(ipval, pyomo.core.expr.ndarray.NumericNDArray):
                for ii in range(0, len(ipval)):
                    try:
                        ipval[ii] = self.convert(ipval[ii], unts)  # ipval.to(unts)
                    except:
                        raise ValueError(
                            'Could not convert %s of %s to %s'
                            % (name, str(ipval), str(unts))
                        )
                ipval_correctUnits = ipval
            else:
                try:
                    ipval_correctUnits = self.convert(ipval, unts)  # ipval.to(unts)
                except:
                    raise ValueError(
                        'Could not convert %s of %s to %s'
                        % (name, str(ipval), str(unts))
                    )

            # superseded by the custom convert function
            # if not isinstance(ipval_correctUnits, (pyomo.core.expr.numeric_expr.NPV_ProductExpression,
            #                                        pyomo.core.expr.ndarray.NumericNDArray,
            #                                        pyomo.core.base.units_container._PyomoUnit)):
            #     ipval_correctUnits = ipval_correctUnits * pyomo_units.dimensionless

            self.sizeCheck(size, ipval_correctUnits)

            opts.append(ipval_correctUnits)
        if strip_units:
            opts = [self.pyomo_value(o) for o in opts]
        if len(opts) == 1:
            opts = opts[0]

        return opts

    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    def packOutputs(self, values, jacobian=None):
        """Attach declared units to outputs and shape the ``BlackBox`` return.

        The inverse of ``sanitizeInputs(strip_units=True)``: a plain number
        or array is taken to be a magnitude in the DECLARED OUTPUT UNITS and
        gets them attached; a value already carrying units is converted to
        the declared units instead (catching unit mistakes). Jacobian entries
        get ``output units / input units`` derived automatically -- the
        error-prone part of the return contract.

        ``values`` is the single output value, or a list with one entry per
        declared output. ``jacobian`` is the list ``[d(out)/d(in_0), ...]``
        for a single output, or a list of such lists (one per output). The
        return is exactly what ``BlackBox`` must produce: ``values`` alone
        when ``jacobian`` is None, else ``(values, jacobian)``, with the
        single-output case unwrapped::

            def BlackBox(self, x, y):
                x, y = self.sanitizeInputs(x, y, strip_units=True)
                return self.packOutputs(x**2 + y**2, [2*x, 2*y])
        """
        n_out = len(self.outputs)
        n_in = len(self.inputs)
        multi = isinstance(values, (list, tuple))
        vals = list(values) if multi else [values]
        if len(vals) != n_out:
            raise ValueError(
                'packOutputs received %d value(s) for %d declared output(s)'
                % (len(vals), n_out))

        packed_vals = []
        for k in range(n_out):
            unts = self.outputs[k].units
            packed_vals.append(self.convert(self.attachUnits(vals[k], unts),
                                            unts))

        if jacobian is None:
            return packed_vals if multi else packed_vals[0]

        jac_rows = ([list(r) for r in jacobian] if multi
                    else [list(jacobian)])
        if len(jac_rows) != n_out or any(len(r) != n_in for r in jac_rows):
            raise ValueError(
                'packOutputs expected a jacobian of %d row(s) with %d '
                'entries each (d output / d input)' % (n_out, n_in))

        packed_jac = []
        for k in range(n_out):
            row = []
            for j in range(n_in):
                dunits = self.outputs[k].units / self.inputs[j].units
                row.append(self.convert(
                    self.attachUnits(jac_rows[k][j], dunits), dunits))
            packed_jac.append(row)

        if multi:
            return packed_vals, packed_jac
        return packed_vals[0], packed_jac[0]

    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    def checkOutputs(self, *args, **kwargs):
        """Not implemented; raises ``NotImplementedError``.

        The output-side counterpart of `sanitizeInputs`, left unfinished
        because `packOutputs` covers what a box actually needs -- attaching or
        converting the declared units on the way out -- and the checking half
        of it happens in `fillCache`, where the value has to be converted
        anyway. The body is kept, commented, for whoever finishes it.
        """
        raise NotImplementedError('Contact developers to use this function')
        # nameList = [self.outputs[i].name for i in range(0,len(self.outputs))]
        # if len(args) + len(kwargs.values()) > len(nameList):
        #     raise ValueError('Too many outputs')
        # if len(args) + len(kwargs.values()) < len(nameList):
        #     raise ValueError('Not enough outputs')
        # inputDict = {}
        # for i in range(0,len(args)):
        #     rg = args[i]
        #     inputDict[nameList[i]] = rg

        # for ky, vl in kwargs.items():
        #     if ky in nameList:
        #         inputDict[ky] = vl
        #     else:
        #         raise ValueError('Unexpected output keyword argument %s in the outputs'%(ky))

        # opts = []

        # for i in range(0,len(nameList)):
        #     name = nameList[i]
        #     nameCheck = self.outputs[i].name
        #     unts = self.outputs[i].units
        #     size = self.outputs[i].size

        #     if name != nameCheck:
        #         raise RuntimeError('Something went wrong and values are not consistent.  Check your defined inputs.')

        #     ipval = inputDict[name]

        #     if unts is not None:
        #         try:
        #             ipval_correctUnits = self.convert(ipval, unts)
        #         except:
        #             raise ValueError('Could not convert %s of %s to %s'%(name, str(ipval),str(unts)))
        #     else:
        #         ipval_correctUnits = ipval

        #     if not isinstance(ipval_correctUnits, (pyomo.core.expr.numeric_expr.NPV_ProductExpression,
        #                                            pyomo.core.expr.ndarray.NumericNDArray,
        #                                            pyomo.core.base.units_container._PyomoUnit)):
        #         ipval_correctUnits = ipval_correctUnits * pyomo_units.dimensionless

        #     self.sizeCheck(size, ipval_correctUnits)

        #     opts.append(ipval_correctUnits)
        # if len(opts) == 1:
        #     opts = opts[0]

        # return opts

    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    def getSummary(self, whitespace=6):
        """The box's declarations as a printable table, returned as a string.

        Description, then inputs and outputs with their units, sizes and
        descriptions. This is what a box knows about itself and all a caller
        can know about it, so it is the answer to "what does this thing take?"
        -- particularly for a box that came from somewhere else. ``whitespace``
        is the padding between columns; `summary` is the same thing as a
        property.
        """
        pstr = '\n'
        pstr += 'Model Description\n'
        pstr += '=================\n'
        descr_str = self.description.__repr__()
        pstr += descr_str[1:-1] + '\n\n'

        longestName = 0
        longestUnits = 0
        longestSize = 0
        for ipt in self.inputs:
            nml = len(ipt.name)
            if nml > longestName:
                longestName = nml
            unts = ipt.units.__str__()  # _repr_html_()
            # unts = unts.replace('<sup>','^')
            # unts = unts.replace('</sup>','')
            # unts = unts.replace('\[', '[')
            # unts = unts.replace('\]', ']')
            unl = len(unts)
            if unl > longestUnits:
                longestUnits = unl
            if type(ipt.size) == list:
                lsz = len(ipt.size.__repr__())
            else:
                lsz = len(str(ipt.size))
            if lsz > longestSize:
                longestSize = lsz
        namespace = max([4, longestName]) + whitespace
        unitspace = max([5, longestUnits]) + whitespace
        sizespace = max([4, longestSize]) + whitespace
        fulllength = namespace + unitspace + sizespace + 11
        pstr += 'Inputs\n'
        pstr += '=' * fulllength
        pstr += '\n'
        pstr += 'Name'.ljust(namespace)
        pstr += 'Units'.ljust(unitspace)
        pstr += 'Size'.ljust(sizespace)
        pstr += 'Description'
        pstr += '\n'
        pstr += '-' * (namespace - whitespace)
        pstr += ' ' * whitespace
        pstr += '-' * (unitspace - whitespace)
        pstr += ' ' * whitespace
        pstr += '-' * (sizespace - whitespace)
        pstr += ' ' * whitespace
        pstr += '-----------'
        pstr += '\n'
        for ipt in self.inputs:
            pstr += ipt.name.ljust(namespace)
            unts = ipt.units.__str__()  # _repr_html_()
            # unts = unts.replace('<sup>','^')
            # unts = unts.replace('</sup>','')
            # unts = unts.replace('\[', '[')
            # unts = unts.replace('\]', ']')
            pstr += unts.ljust(unitspace)
            lnstr = '%s' % (ipt.size.__repr__())
            pstr += lnstr.ljust(sizespace)
            pstr += ipt.description
            pstr += '\n'
        pstr += '\n'

        longestName = 0
        longestUnits = 0
        longestSize = 0
        for opt in self.outputs:
            nml = len(opt.name)
            if nml > longestName:
                longestName = nml
            unts = opt.units.__str__()  # _repr_html_()
            # unts = unts.replace('<sup>','^')
            # unts = unts.replace('</sup>','')
            # unts = unts.replace('\[', '[')
            # unts = unts.replace('\]', ']')
            unl = len(unts)
            if unl > longestUnits:
                longestUnits = unl
            if type(opt.size) == list:
                lsz = len(opt.size.__repr__())
            else:
                lsz = len(str(opt.size))
            if lsz > longestSize:
                longestSize = lsz
        namespace = max([4, longestName]) + whitespace
        unitspace = max([5, longestUnits]) + whitespace
        sizespace = max([4, longestSize]) + whitespace
        fulllength = namespace + unitspace + sizespace + 11
        pstr += 'Outputs\n'
        pstr += '=' * fulllength
        pstr += '\n'
        pstr += 'Name'.ljust(namespace)
        pstr += 'Units'.ljust(unitspace)
        pstr += 'Size'.ljust(sizespace)
        pstr += 'Description'
        pstr += '\n'
        pstr += '-' * (namespace - whitespace)
        pstr += ' ' * whitespace
        pstr += '-' * (unitspace - whitespace)
        pstr += ' ' * whitespace
        pstr += '-' * (sizespace - whitespace)
        pstr += ' ' * whitespace
        pstr += '-----------'
        pstr += '\n'
        for opt in self.outputs:
            pstr += opt.name.ljust(namespace)
            unts = opt.units.__str__()  # _repr_html_()
            # unts = unts.replace('<sup>','^')
            # unts = unts.replace('</sup>','')
            # unts = unts.replace('\[', '[')
            # unts = unts.replace('\]', ']')
            pstr += unts.ljust(unitspace)
            lnstr = '%s' % (opt.size.__repr__())
            pstr += lnstr.ljust(sizespace)
            pstr += opt.description
            pstr += '\n'
        pstr += '\n'

        return pstr

    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------

    @property
    def summary(self):
        """`getSummary` with the default spacing -- ``print(box.summary)``."""
        return self.getSummary()

    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    def __repr__(self):
        pstr = 'AnalysisModel( ['
        for i in range(0, len(self.outputs)):
            pstr += self.outputs[i].name
            pstr += ','
        pstr = pstr[0:-1]
        pstr += ']'
        pstr += ' == '
        pstr += 'f('
        for ipt in self.inputs:
            pstr += ipt.name
            pstr += ', '
        pstr = pstr[0:-2]
        pstr += ')'
        pstr += ' , '
        pstr = pstr[0:-2]
        pstr += '])'
        return pstr
