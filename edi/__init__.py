#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  A Pyomo-based framework for engineering design optimization.
#
#  Originally developed as pyomo.contrib.edi (Pyomo PR #2937) at
#  National Technology and Engineering Solutions of Sandia, LLC.
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#
#  Development of this module was conducted as part of the Institute for
#  the Design of Advanced Energy Systems (IDAES) with support through the
#  Simulation-Based Engineering, Crosscutting Research Program within the
#  U.S. Department of Energy's Office of Fossil Energy and Carbon Management.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""EDI --- the Engineering Design Interface.

A lightweight wrapper on Pyomo targeted at composing engineering design
optimization problems, with first-class support for units, black-box analysis
codes, and structure detection (LP/QP/GP/SP).

Note on history: this package began life as ``pyomo.contrib.edi``. When it was
split into a standalone distribution the modules were reorganized into
``edi.objects`` / ``edi.solvers`` / ``edi.structure`` / ``edi.units``, but the
package ``__init__`` continued to import from ``pyomo.contrib.edi`` inside a
bare ``try/except: pass``. Because that module no longer ships with Pyomo, every
import failed silently and ``import edi`` exposed none of its own API. The
imports below are local and are deliberately NOT wrapped in a bare except, so
that a broken install fails loudly instead of producing an empty namespace.
"""

# Build all of the appropriate Pyomo machinery.
import pyomo.environ  # noqa: F401

from edi.objects.formulation import Formulation

from edi.objects.blackBoxFunctionModel import (
    BlackBoxFunctionModel,
    BlackBoxFunctionModel_Variable,
    BlackBoxFunctionModel_Variable as BlackBoxVariable,
    BlackBoxFunctionModel_Variable as BBVariable,
    BlackBoxFunctionModel_Variable as BBV,
    BBList,
    TypeCheckedList,
)

from edi.solvers.sensitivity import (
    sensitivities,
    constraint_duals,
    format_sensitivities,
)
from edi.presolve import diagnose, structure_report
from edi.solvers.feasibility import FeasibilityResult, feasibility

# `from edi import units` gives Pyomo's units container, so a model needs one
# import rather than two. Declaring a Variable already takes units as a string;
# this is for the places that need the object -- `1.0 * units.m` on the right
# of a constraint, `units.convert(...)` inside a black box.
#
# `edi.units` is ALSO a subpackage (unitCorrector, unitWalker), and importing a
# submodule binds it onto its parent package -- which would overwrite this
# name. So load the subpackage first and rebind after: once `edi.units` is in
# sys.modules, a later `from edi.units.unitCorrector import ...` resolves
# through sys.modules and never touches this attribute again.
#
# Every reference in this repository is that fully-qualified form and is
# unaffected. `import edi.units` followed by attribute access would break, and
# nothing does it.
from edi.units import unitCorrector as _unitCorrector  # noqa: F401
from pyomo.environ import units


__all__ = [
    "Formulation",
    "units",
    "diagnose",
    "structure_report",
    "feasibility",
    "FeasibilityResult",
    "sensitivities",
    "constraint_duals",
    "format_sensitivities",
    "BlackBoxFunctionModel",
    "BlackBoxFunctionModel_Variable",
    "BlackBoxVariable",
    "BBVariable",
    "BBV",
    "BBList",
    "TypeCheckedList",
]

__version__ = "0.1.0"  # keep in sync with pyproject.toml
