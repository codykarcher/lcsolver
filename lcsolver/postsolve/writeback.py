#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Write a solver's solution back onto the Pyomo model.

The cvxopt backends solve a transformed problem and return a raw dict whose
``x`` vector follows the ordering ``structure_detector`` publishes as
``structures['variables']``. Without this module that vector is never applied,
so after a successful solve ``pyo.value(f.x)`` still returns the initial guess.
"""

from pyomo.common.dependencies import numpy as np
from pyomo.core.base.componentuid import ComponentUID


def _resolve_on(model, v):
    """Find the counterpart of variable ``v`` on ``model``, or None.

    The CUID is built from the component object, not ``v.name``:
    ``find_component`` round-trips through a string, and a collected parent
    reports '[Unattached VarData]', which the CUID parser raises TypeError
    on. The object keeps real index values intact and never parses.
    """
    try:
        cuid = ComponentUID(v)
    except Exception:
        # v's parent component is gone; it can no longer be located by name
        return None
    return cuid.find_component_on(model)


def _name_of(v):
    """A usable name for ``v``, even if its parent component was collected."""
    try:
        return v.name
    except Exception:                                # pragma: no cover - defensive
        return str(v)


def write_solution(structures, res, model=None):
    """Set each Pyomo variable to its solved value; returns ``{name: value}``.

    ``structures`` must carry 'variables' (the detector's ordering), ``res``
    an 'x'. Pass ``model`` to resolve each variable onto the caller's model
    by ComponentUID: unit_corrector clones, so ``structures['variables']``
    belong to a copy and writing only them leaves the caller's model
    untouched. The clone is written too -- the post-solve checks need the
    DETECTED form at the SOLVED point, and re-detecting costs ~half the wall
    clock of the whole solve. Writes use ``set_value(skip_validation=True)``
    so a point marginally outside a declared bound (normal for interior
    point) is still recorded rather than raising.
    """
    if 'variables' not in structures:
        raise KeyError(
            "structures['variables'] is missing; structure_detector must "
            "publish the variable ordering for write-back to be possible")
    if res is None or 'x' not in res:
        raise KeyError("solver result has no 'x' entry to write back")

    variables = structures['variables']
    x = np.asarray(res['x'], dtype=float).ravel()

    if len(x) < len(variables):
        raise ValueError(
            f"solution vector has {len(x)} entries but the model has "
            f"{len(variables)} variables; cannot write back unambiguously")

    def _set(target, val):
        try:
            target.set_value(val, skip_validation=True)
        except TypeError:            # older Pyomo without skip_validation
            target.set_value(val)

    written = {}
    unresolved = []
    for i, v in enumerate(variables):
        val = float(x[i])
        target = v
        if model is not None:
            # resolve onto the caller's model; `variables` may belong to
            # the unit_corrector clone
            found = _resolve_on(model, v)
            if found is None:
                unresolved.append(_name_of(v))
                continue
            target = found
            _set(v, val)             # and leave the clone at the solution too
        _set(target, val)
        written[_name_of(target)] = val

    if unresolved:
        raise KeyError(
            "could not resolve these variables on the target model: "
            + ", ".join(unresolved))
    # Mark the detected form as carrying a solution, so the post-solve checks
    # will accept it instead of re-detecting.
    if model is not None:
        clone = structures.get('model') if hasattr(structures, 'get') else None
        if clone is not None:
            clone._edi_solved = True
    return written


def solution_dict(structures, res):
    """Return ``{variable_name: value}`` without modifying the model."""
    variables = structures['variables']
    x = np.asarray(res['x'], dtype=float).ravel()
    return {_name_of(v): float(x[i]) for i, v in enumerate(variables)}
