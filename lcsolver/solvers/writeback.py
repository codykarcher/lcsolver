#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Write a solver's solution back onto the Pyomo model.

The cvxopt backends solve a transformed problem (log-space for a geometric
program, matrix form for LP/QP) and return a raw solver dictionary whose ``x``
vector is indexed by the variable ordering established in
``structure_detector``. Without this module that vector is never applied to the
model, so after a successful solve ``pyo.value(f.x)`` still returns the initial
guess -- which is almost never what the caller wants.

``structure_detector`` publishes the ordering as ``structures['variables']``;
this module maps the solution vector onto those variables.
"""

from pyomo.common.dependencies import numpy as np
from pyomo.core.base.componentuid import ComponentUID


def _resolve_on(model, v):
    """Find the counterpart of variable ``v`` on ``model``, or None.

    A ``ComponentUID`` is built from the component *object* rather than from
    ``v.name``. That matters for indexed variables: ``model.find_component(name)``
    round-trips through a string, so 'sK[0]' has to be re-parsed, and a variable
    whose parent component has been collected reports its name as
    '[Unattached VarData]' -- which the CUID parser turns into a bare index with
    no component name and then raises ``TypeError`` on. Going through the object
    keeps the real index values (ints, strings, tuples) intact and never parses.
    """
    try:
        cuid = ComponentUID(v)
    except Exception:
        # v's parent component is gone, so it can no longer be located by name.
        return None
    return cuid.find_component_on(model)


def _name_of(v):
    """A usable name for ``v``, even if its parent component was collected."""
    try:
        return v.name
    except Exception:                                # pragma: no cover - defensive
        return str(v)


def write_solution(structures, res, model=None):
    """Set each Pyomo variable to its solved value.

    Parameters
    ----------
    structures : dict
        Output of ``structure_detector``; must contain ``'variables'``.
    res : dict
        Solver result dictionary; must contain ``'x'``.
    model : optional
        The model to write onto. This matters: ``unit_corrector`` calls
        ``.clone()``, so ``structures['variables']`` belong to a *copy* of the
        user's model. Writing to them leaves the caller's model untouched --
        exactly the bug this module exists to fix. When ``model`` is supplied,
        each variable is resolved onto it by ``ComponentUID`` (see
        ``_resolve_on``), which handles indexed variables correctly.

    Returns
    -------
    dict
        ``{variable_name: value}`` for every variable written.

    Notes
    -----
    Values are written with ``.set_value(..., skip_validation=True)`` so that a
    solution which sits marginally outside a declared bound (a normal outcome of
    an interior-point solve) is still recorded rather than raising.
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

    written = {}
    unresolved = []
    for i, v in enumerate(variables):
        val = float(x[i])
        target = v
        if model is not None:
            # Resolve onto the caller's model, since `variables` may belong to
            # the clone produced by unit_corrector.
            found = _resolve_on(model, v)
            if found is None:
                unresolved.append(_name_of(v))
                continue
            target = found
        try:
            target.set_value(val, skip_validation=True)
        except TypeError:            # older Pyomo without skip_validation
            target.set_value(val)
        written[_name_of(target)] = val

    if unresolved:
        raise KeyError(
            "could not resolve these variables on the target model: "
            + ", ".join(unresolved))
    return written


def solution_dict(structures, res):
    """Return ``{variable_name: value}`` without modifying the model."""
    variables = structures['variables']
    x = np.asarray(res['x'], dtype=float).ravel()
    return {_name_of(v): float(x[i]) for i, v in enumerate(variables)}
