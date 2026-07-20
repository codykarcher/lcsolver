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
        each variable is resolved by name onto it via ``find_component``.

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
            # Resolve onto the caller's model by name, since `variables` may
            # belong to the clone produced by unit_corrector.
            found = model.find_component(v.name)
            if found is None:
                unresolved.append(v.name)
                continue
            target = found
        try:
            target.set_value(val, skip_validation=True)
        except TypeError:            # older Pyomo without skip_validation
            target.set_value(val)
        written[v.name] = val

    if unresolved:
        raise KeyError(
            "could not resolve these variables on the target model: "
            + ", ".join(unresolved))
    return written


def solution_dict(structures, res):
    """Return ``{variable_name: value}`` without modifying the model."""
    variables = structures['variables']
    x = np.asarray(res['x'], dtype=float).ravel()
    return {v.name: float(x[i]) for i, v in enumerate(variables)}
