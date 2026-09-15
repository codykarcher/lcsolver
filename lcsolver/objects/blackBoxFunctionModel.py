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


# How a flexible-length dimension is stored; sizeCheck skips it
FLEXIBLE_LENGTH = -1

_SIZE_ERROR = (
    'Invalid size %r.  A dimension must be an integer, or np.inf '
    "(equivalently the string 'inf') for a vector of flexible length."
)


def _decode_size_entry(val):
    """Decode one declared dimension to the stored integer. np.inf and 'inf'
    mean flexible length; any string is accepted as flexible for backwards
    compatibility."""
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
        """Declared shape: 0 for a scalar, an integer, or one entry per
        dimension. np.inf (or 'inf') declares a flexible-length dimension
        whose length check is skipped."""
        if isinstance(val, (list, tuple)):
            sizeTemp = []
            for x in val:
                x = _decode_size_entry(x)
                if x == 1:
                    raise ValueError(
                        'A value of 1 is not valid for defining size.  Use fewer dimensions.'
                    )
                sizeTemp.append(x)
            # Store decoded dimensions: sizeCheck compares against -1, so a
            # leftover np.inf/'inf' made a multidimensional flexible size
            # fail its own length check
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
        # The list-subclass protocol restores _lookupDict and then replays
        # items through our append, tripping the duplicate-name guard --
        # Pyomo's clone() then silently set the grey-box block's _ex_model
        # to None. Rebuild through append on an empty instance instead
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
        """Declare one black box input/output: name, units, description, size
        (positional or keyword, in that order). description defaults to '',
        size to 0 (scalar); np.inf declares a flexible-length dimension. An
        already-built variable object may be appended directly. Duplicate
        names are refused -- variables pair by position but are looked up by
        name, and parseInputs keys its run cases on them.
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

    Subclass, append input/output declarations (order matters -- the model's
    variables pair with them by position), set description and
    availableDerivative (0 values only, 1 values + jacobian; a box in a solve
    must supply the jacobian), and write BlackBox. Units and shapes are
    declared on both sides, so conversion and size checks happen at the
    boundary in both directions, values and jacobian alike. The pyomo
    ExternalGreyBoxModel machinery is driven through fillCache; a BlackBox
    implementation leans on sanitizeInputs, packOutputs, and parseInputs.
    A box can be evaluated standalone -- worth doing before putting it in a
    model, where a units or shape mistake has the shortest stack.
    """

    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super(BlackBoxFunctionModel, self).__init__()

        # List of the inputs and outputs
        self.inputs = BBList()
        self.outputs = BBList()

        # Formulation CONSTANTS this box consumes, declared like inputs and
        # wired positionally via f.RuntimeConstraint(..., constants=[f.rho]).
        # Values arrive as trailing BlackBox arguments; jacobian columns for
        # them (trailing entries of each packOutputs row) feed ONLY the
        # reported d(objective)/d(constant) sensitivities
        self.constants = BBList()

        self.inputVariables_optimization = None
        self.outputVariables_optimization = None

        # Formulation-side Params matched to self.constants, set by RuntimeConstraint
        self.constantParams_optimization = []

        # A simple description of the model
        self.description = None

        # Defines the order of derivative available in the black box
        self.availableDerivative = 0

        # Permission to finite-difference a missing jacobian (2 extra BlackBox
        # calls per scalar input per iterate).  Off by default; set through
        # solve(f, allow_blackbox_finite_difference=True), never silently
        self.allow_finite_difference = False

        # Central difference step: fd_relative_step of each input's magnitude,
        # floored at fd_absolute_step near zero
        self.fd_relative_step = 1e-6
        self.fd_absolute_step = 1e-8

        self._cache = None
        self._NunwrappedOutputs = None
        self._NunwrappedInputs = None

    # Attributes shared BY REFERENCE with every clone -- how a box holds a live
    # analysis handle (a pyCAPS Problem, a CFD session, a ctypes wrapper):
    #     reference_attributes = ('capsProblem', 'mses')
    # Every clone then drives the SAME external analysis.  Declarations
    # accumulate over the class hierarchy
    reference_attributes = ()

    # Working state reset (not copied, not shared) on clone; rebuilt on demand
    _reset_on_copy = ('_cache',)

    def __deepcopy__(self, memo):
        # Analysis handles refuse deepcopy ("ctypes objects containing pointers
        # cannot be pickled") and used to abort the whole model clone(), so copy
        # attribute by attribute.  Sharing is DECLARED via reference_attributes;
        # an undeclared attribute that refuses to copy still shares (old models
        # keep working) but warns [LC-W311] -- the old bare except also aliased
        # mutable state between clones, silently corrupting both
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        declared = set()
        for klass in cls.__mro__:
            declared |= set(getattr(klass, 'reference_attributes', ()) or ())
        reset = set()
        for klass in cls.__mro__:
            reset |= set(getattr(klass, '_reset_on_copy', ()) or ())
        for key, val in self.__dict__.items():
            if key in declared:
                new.__dict__[key] = val
                continue
            if key in reset:
                new.__dict__[key] = None
                continue
            try:
                new.__dict__[key] = copy.deepcopy(val, memo)
            except Exception as exc:
                import warnings
                warnings.warn(
                    "[LC-W311] attribute %r of %s cannot be deep-copied "
                    "(%s: %s) and was shared BY REFERENCE with the clone. "
                    "If it is a live analysis handle (a pyCAPS Problem, a "
                    "solver session), declare that on the class: "
                    "reference_attributes = (%r,). If it is mutable state, "
                    "sharing it can silently corrupt both copies."
                    % (key, cls.__name__, type(exc).__name__, exc, key),
                    RuntimeWarning, stacklevel=2)
                new.__dict__[key] = val
        return new

    def setOptimizationVariables(
        self, inputVariables_optimization, outputVariables_optimization
    ):
        """Record the model variables this box is wired to, matched to
        inputs/outputs by position. Called by Formulation.RuntimeConstraint,
        not by a model author; the pairing drives the unit conversion at
        each evaluation."""
        self.inputVariables_optimization = inputVariables_optimization
        self.outputVariables_optimization = outputVariables_optimization

    def setOptimizationConstants(self, constantParams_optimization):
        """Record which formulation Constants feed this box's declared
        `constants`, matched by position exactly as inputs are. Called by
        `Formulation.RuntimeConstraint`, not by a model author."""
        self.constantParams_optimization = list(constantParams_optimization
                                                or [])

    # ---------------------------------------------------------------------------------------------------------------------
    # pyomo things
    # ---------------------------------------------------------------------------------------------------------------------
    def input_names(self):
        """Model-side input names, one per scalar, for pynumero. An indexed
        variable unwraps to one name per element (x[0], x[1], ...) since the
        solver deals in a flat vector."""
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
        """The solver's flat vector for the next iterate; drops the cache.

        Values and jacobian are asked for in separate calls, so recompute
        once and only once per point. A re-set to the bit-identical point
        keeps the cache -- SIA routinely re-hands the same iterate (260 of
        510 BlackBox calls on the capsPhase AVL demo were exact repeats).
        Anything not exactly equal is a new point.
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
        Computed with the values in one BlackBox call -- an analysis code is
        usually too expensive to evaluate twice per iterate."""
        self.fillCache()
        jac = self._cache['pyomo_jacobian']
        return jac

    def post_init_setup(self, defaultVal=1.0):
        """Give the box a starting input vector, once its length is known.
        Called by Formulation.RuntimeConstraint after attachment -- the first
        moment the number of scalar inputs exists (declared inputs may be of
        flexible length)."""
        # _NunwrappedInputs is None until the box is attached to a
        # Formulation; numpy >= 2.0 rejects None as a shape, so fall back to
        # a scalar placeholder (the numpy 1.x behavior)
        n = self._NunwrappedInputs
        self._input_values = np.ones(() if n is None else n) * defaultVal

    def attachUnits(self, val, unts):
        """Attach unts to a bare number, understood to already be a magnitude
        in those units; anything already united is returned untouched."""
        # Dimensionless boxes naturally return plain numbers: pyomo collapses
        # 'float * dimensionless' to a float, and numpy operations on
        # dimensionless arrays return plain ndarrays
        if isinstance(val, np.ndarray):
            if isinstance(val, pyomo.core.expr.ndarray.NumericNDArray):
                return val
            return val * unts
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float, np.integer, np.floating)):
            return val * unts
        return val

    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    def _output_magnitudes(self, values):
        """Each output as a bare float or ndarray, in its declared units"""
        vals = values if isinstance(values, (list, tuple)) else [values]
        out = []
        for k in range(0, len(vals)):
            u = self.outputs[k].units
            converted = self.convert(self.attachUnits(vals[k], u), u)
            out.append(np.asarray(self.pyomo_value(converted), dtype=float))
        return out

    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    def _finite_difference_jacobian(self, bb_inputs, base_values):
        """Central-difference jacobian for a values-only box.

        Refused unless solve(f, allow_blackbox_finite_difference=True) granted
        permission.  Step is fd_relative_step of each input's magnitude
        (floored at fd_absolute_step), in the declared input units; blocks
        come back shaped exactly as packOutputs expects, so the rest of
        fillCache cannot tell them from author-supplied derivatives.
        """
        if not self.allow_finite_difference:
            raise ValueError(
                "black box %r declares availableDerivative=0: it returns "
                "values only, and the optimizer needs d(output)/d(input). "
                "Either return jacobians from BlackBox, or pass "
                "allow_blackbox_finite_difference=True to solve() to "
                "approximate them by central differences (2 extra BlackBox "
                "calls per scalar input per iterate)."
                % type(self).__name__)

        base_mags = self._output_magnitudes(base_values)
        multi_out = isinstance(base_values, (list, tuple))
        n_out = len(base_mags)

        # bb_inputs carries declared constants at the end; differentiate them
        # like inputs (their columns feed the constant-sensitivity report)
        decls = list(self.inputs) + list(self.constants)
        in_mags, in_units = [], []
        for j, iv in enumerate(bb_inputs):
            in_units.append(decls[j].units)
            in_mags.append(np.asarray(self.pyomo_value(iv), dtype=float))

        # block[k][j] : (output k dims) + (input j dims)
        blocks = [[np.zeros(base_mags[k].shape + in_mags[j].shape)
                   for j in range(len(bb_inputs))]
                  for k in range(n_out)]

        for j in range(len(bb_inputs)):
            flat_idx = ([()] if in_mags[j].shape == ()
                        else list(np.ndindex(*in_mags[j].shape)))
            for idx in flat_idx:
                x = float(in_mags[j][idx]) if idx != () or in_mags[j].shape \
                    else float(in_mags[j])
                h = max(self.fd_relative_step * abs(x),
                        self.fd_absolute_step)

                def _call(sign):
                    pert = []
                    for jj, iv in enumerate(bb_inputs):
                        if jj != j:
                            pert.append(iv)
                            continue
                        mag = np.array(in_mags[jj], dtype=float, copy=True)
                        if mag.shape == ():
                            mag = mag + sign * h
                            pert.append(float(mag) * in_units[jj])
                        else:
                            mag[idx] += sign * h
                            pert.append(mag * in_units[jj])
                    return self._output_magnitudes(self.BlackBox(*pert))

                plus, minus = _call(+1.0), _call(-1.0)
                for k in range(n_out):
                    col = (plus[k] - minus[k]) / (2.0 * h)
                    if in_mags[j].shape == ():
                        blocks[k][j][...] = col
                    else:
                        blocks[k][j][(Ellipsis,) + idx] = col

        # Collapse 0-d blocks to plain floats and match packOutputs's
        # single-output convention (one row, not a list of rows).
        for k in range(n_out):
            for j in range(len(bb_inputs)):
                if blocks[k][j].shape == ():
                    blocks[k][j] = float(blocks[k][j])
        return blocks if multi_out else blocks[0]

    def fillCache(self):
        """Call `BlackBox` once and cache values + jacobian at this iterate.

        Cuts the solver's flat vector into the declared shapes, converts
        model units -> box units going in (size-checked), and converts values
        and every jacobian entry back coming out -- each d(output)/d(input)
        in output units / input units -- into the sparse block the solver
        reads. Cached because the solver asks for values and jacobian
        separately; set_input_values drops the cache, so exactly one BlackBox
        call per iterate.
        """
        if self._cache is None:
            # Build locally, publish only on success: filling self._cache in
            # place left a partial non-None cache behind on error, and the
            # next call skipped the rebuild and raised a KeyError that hid
            # the original error
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

            # Declared constants ride along as trailing arguments, converted to
            # the units the box declared just like the variable inputs
            bb_consts = []
            for k, cdecl in enumerate(self.constants):
                cparam = self.constantParams_optimization[k]
                cval = pyo.value(cparam) * pyomo_units.get_units(cparam)
                bb_consts.append(pyomo_units.convert(cval, cdecl.units))
            bb_inputs = bb_inputs + bb_consts

            bbo = self.BlackBox(*bb_inputs)

            # availableDerivative=0 means values only -- no jacobian tuple to
            # unpack.  Either raise (default) or, with permission granted
            # through solve(), build one by central differences
            if not self.availableDerivative:
                bbo = (bbo, self._finite_difference_jacobian(bb_inputs, bbo))

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
            constJacobian = np.zeros([self._NunwrappedOutputs,
                                      len(self.constants)])
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

                # Trailing row entries are d(output)/d(constant) columns.  They
                # never enter the optimizer's jacobian (a Constant is not an NLP
                # column); the sensitivity pass chain-rules them into the
                # reported d(objective)/d(constant)
                for k2, cdecl in enumerate(self.constants):
                    cparam = self.constantParams_optimization[k2]
                    raw = self.attachUnits(
                        jacobianList[i][len(self.inputs) + k2],
                        lounits / cdecl.units)
                    ocunits = pyomo_units.get_units(cparam)
                    if isinstance(oopt, pyomo.core.base.var.ScalarVar):
                        vals = [raw]
                    else:
                        vals = [raw[vi]
                                for vi in list(oopt.index_set().data())]
                    for rr, v in enumerate(vals):
                        constJacobian[ptr_row + rr, k2] = pyo.value(
                            pyomo_units.convert(v, oounits / ocunits))

                ptr_row += ptr_row_step

            cache['pyomo_jacobian'] = sps.coo_matrix(outputJacobian)
            cache['constant_jacobian'] = {
                self.constantParams_optimization[k2].name: constJacobian[:, k2]
                for k2 in range(len(self.constants))}
            self._cache = cache

    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    # These models must be defined in each individual model, just placeholders here
    def BlackBox(*args, **kwargs):
        """The analysis itself; every subclass defines this, the base raises.

        Called with one argument per declared input, in declaration order, in
        the declared units -- so open with sanitizeInputs(..., strip_units=True)
        and return via packOutputs, which derives the jacobian units. By hand:
        a single-output box returns (value, [dv_din0, ...]), a multi-output
        box ([v0, ...], [[...], ...]) -- one jacobian row per output, one
        entry per input; drop the second element for values only. Front a
        (*args, **kwargs) signature with parseInputs to also accept batches,
        keyword inputs, or options of the box's own.
        """
        raise AttributeError(errorString)

    def convert(self, val, unts):
        """Convert a scalar or array to unts. pyomo_units.convert only handles
        a united scalar; arrays go element by element. A bare number is
        treated as dimensionless first, so it raises on a dimensioned target
        rather than being assumed to be in whatever was wanted."""
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
        """The magnitude behind a united value, elementwise for an array.
        The magnitude is in whatever units the value carried, so only safe
        after conversion -- why sanitizeInputs converts before it strips."""
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
        """Normalize any BlackBox call form into (runCases, returnMode, extras).

        Accepts positional/keyword args, one case as a dict, a dict of
        vectors (one case per element), or a list of cases as dicts or
        sequences; bb([x, y]) is ambiguous and refused. runCases is always a
        list of name-keyed dicts, sanitized. returnMode is
        availableDerivative for a batch and -availableDerivative - 1
        (negative: one fewer level of indexing out) for a single case.
        extras holds undeclared kwargs plus extras['remainingArgs'] -- the
        box's own options; only the positional/keyword forms populate it.
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
        """Raise if a value disagrees with its declared size (decoded form:
        0 scalar, int, or list per dimension; FLEXIBLE_LENGTH entries are
        skipped). Rank must match as well as lengths -- a right-sized,
        wrong-shaped value is a different quantity and numpy would take it
        without comment."""
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
        """Validate and unit-convert the values handed to BlackBox.

        Takes the declared inputs positionally or by name, converts each to
        its declared units, and size-checks it. strip_units=True returns
        plain magnitudes IN THE DECLARED INPUT UNITS (convert first, then
        strip). One declared input comes back bare; several as a list.
        """
        # Declared constants arrive as trailing arguments; sanitize them
        # identically via one combined declaration list
        _decls = list(self.inputs) + list(self.constants)
        nameList = [_decls[i].name for i in range(0, len(_decls))]

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
            nameCheck = _decls[i].name
            unts = _decls[i].units
            size = _decls[i].size

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
    def constant_jacobian(self):
        """{constant name: d(outputs)/d(constant)} at the cached point, one
        flat column per declared constant over the unwrapped outputs, in
        (output optimization units)/(constant units).  Empty when the box
        declares no constants; evaluates the box if the cache is cold."""
        self.fillCache()
        return dict(self._cache.get('constant_jacobian') or {})

    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _jacobianBlockShape(output, input_):
        """Numpy shape a d(output)/d(input) block must have: (output dims) +
        (input dims), scalars contributing none.  None means do-not-check (a
        flexible-length dimension is only known at run time)."""
        dims = []
        for declared in (output.size, input_.size):
            if declared in (0, None):
                continue
            seq = declared if isinstance(declared, (list, tuple)) else [declared]
            for d in seq:
                if d == FLEXIBLE_LENGTH:
                    return None
                dims.append(int(d))
        return tuple(dims)

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
        n_const = len(self.constants)
        n_cols = n_in + n_const
        if len(jac_rows) != n_out or any(len(r) != n_cols for r in jac_rows):
            raise ValueError(
                'packOutputs expected a jacobian of %d row(s) with %d '
                'entries each (d output / d input%s)'
                % (n_out, n_cols,
                   ', then d output / d constant for the %d declared '
                   'constant(s)' % n_const if n_const else ''))

        # Check every block's shape HERE, in the modeller's own stack frame --
        # a wrong shape used to surface as an AttributeError from deep inside
        # pyomo's unit converter.  The classic slip is np.diag() on a
        # vector-output/scalar-input block: 3x3 where (3,) belongs
        in_decls = list(self.inputs) + list(self.constants)
        for k in range(n_out):
            for j in range(n_cols):
                expected = self._jacobianBlockShape(self.outputs[k],
                                                    in_decls[j])
                if expected is None:
                    continue                     # a flexible-length dimension
                got = np.shape(getattr(jac_rows[k][j], 'magnitude',
                                       jac_rows[k][j]))
                if got != expected:
                    hint = ''
                    if (len(expected) == 1 and len(got) == 2
                            and got[0] == got[1] == expected[0]):
                        hint = (" Input '%s' is a scalar, so this block is a "
                                'vector over the output elements; np.diag() '
                                'belongs only on vector-input blocks.'
                                % in_decls[j].name)
                    raise ValueError(
                        'packOutputs: the jacobian block d(%s)/d(%s) must '
                        'have shape %r (output dims + input dims), but got '
                        '%r.%s' % (self.outputs[k].name, in_decls[j].name,
                                   expected, got, hint))

        packed_jac = []
        for k in range(n_out):
            row = []
            for j in range(n_cols):
                dunits = self.outputs[k].units / in_decls[j].units
                row.append(self.convert(
                    self.attachUnits(jac_rows[k][j], dunits), dunits))
            packed_jac.append(row)

        if multi:
            return packed_vals, packed_jac
        return packed_vals[0], packed_jac[0]

    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    def checkOutputs(self, *args, **kwargs):
        """Not implemented. The output-side counterpart of sanitizeInputs;
        packOutputs covers what a box needs and fillCache checks on the way
        out. Body kept, commented, for whoever finishes it."""
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
        """The declarations as a printable string: description, then inputs
        and outputs with units, sizes, and descriptions. whitespace pads the
        columns; `summary` is the property form."""
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
