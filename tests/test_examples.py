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

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.dependencies import attempt_import

testIndex = 0
import importlib

from pyomo.core.base.units_container import pint_available

from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.dependencies import scipy, scipy_available

egb, egb_available = attempt_import(
    "pyomo.contrib.pynumero.interfaces.external_grey_box"
)

formulation_available = False
try:
    from lcsolver import Formulation

    formulation_available = True
except ImportError:
    # Narrow on purpose. A bare `except` here turns any breakage inside the
    # package -- a SyntaxError, a NameError -- into `not available`, and the
    # skipIf below then quietly skips this whole file instead of failing.
    pass

blackbox_available = False
try:
    from lcsolver import BlackBoxFunctionModel

    blackbox_available = True
except ImportError:
    pass

if numpy_available:
    import numpy as np

# Some examples solve with SolverFactory('ipopt'), which needs the ipopt
# executable on PATH (not installed in CI or on most reviewer machines).
try:
    ipopt_available = pyo.SolverFactory('ipopt').available(exception_flag=False)
except:
    ipopt_available = False


import os
import sys
import traceback

# `examples/` lives at the top level of the standalone repository rather than
# inside the package (it did live inside it when this was pyomo.contrib.edi),
# so it is found by path and imported by name.
EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples'
)


def discover_examples():
    """Every example in the directory, rather than a list kept by hand.

    A hand-maintained registry is a registry that goes stale: an example lands,
    nobody adds the line, and it is never run again until it breaks in front of
    somebody reading the docs. The directory listing cannot go stale.
    """
    if not os.path.isdir(EXAMPLES_DIR):
        return []
    return sorted(
        fn
        for fn in os.listdir(EXAMPLES_DIR)
        if fn.endswith('.py') and not fn.startswith('_')
    )


def example_test_name(filename):
    return 'test_example_%s' % (filename[:-3],)


@unittest.skipIf(
    not egb_available, 'Testing lcsolver requires pynumero external grey boxes'
)
@unittest.skipIf(not formulation_available, 'Formulation import failed')
@unittest.skipIf(not blackbox_available, 'Blackbox import failed')
@unittest.skipIf(not numpy_available, 'Testing lcsolver requires numpy')
@unittest.skipIf(not scipy_available, 'Testing lcsolver requires scipy')
@unittest.skipIf(not pint_available, 'Testing units requires pint')
class EDIExamples(unittest.TestCase):
    def test_every_example_has_a_test(self):
        """Discovery is the whole registry, so guard discovery itself.

        If the glob ever came back empty -- a renamed directory, a packaging
        change that moves examples/ -- every example test would silently stop
        existing and the suite would still be green. That failure mode is worse
        than a broken example, so it gets its own assertion.
        """
        found = discover_examples()
        self.assertTrue(found, 'no examples discovered in %s' % (EXAMPLES_DIR,))
        for filename in found:
            self.assertTrue(
                hasattr(EDIExamples, example_test_name(filename)),
                'example %s was discovered but has no generated test' % (filename,),
            )


#: How far a written-back solution may miss a constraint, relative to the size
#: of the body. These are geometric programs solved in log space, so a converged
#: answer lands within about 1e-4 of its equalities in natural space --
#: aircraft_gp sits at 7e-5 and cooling_loop_gp at 3e-4. The number is chosen to
#: sit above that and far below a real failure: the violations this is here to
#: catch are of order 1, a constraint missed by its whole magnitude.
FEASIBLE_RTOL = 1e-3


def worst_violation(f):
    """How badly the model's current point breaks its own constraints.

    Relative to the size of the body, because these models span many orders of
    magnitude and an absolute slack means nothing next to a weight in newtons.
    Only algebraic constraints are walked: a black box is an
    ExternalGreyBoxBlock rather than a Constraint, so it is not visited here.

    Evaluated on the unit-corrected clone, never on the model as written.
    ``pyo.value`` multiplies magnitudes and ignores units, so a model that
    declares TSFC in 1/hr against a time in seconds evaluates 3600x off and
    reads as violated by its whole magnitude -- kirschen_ozturk scores 1.0 as
    written and 1.8e-08 corrected. Cloning also keeps the solved values, so
    the corrected model is at the same point.
    """
    from lcsolver.presolve.unitCorrector import unit_corrector

    corrected = unit_corrector(f)
    worst = 0.0
    for con in corrected.component_data_objects(pyo.Constraint, active=True):
        body = pyo.value(con.body)
        scale = max(1.0, abs(body))
        if con.lower is not None:
            worst = max(worst, (pyo.value(con.lower) - body) / scale)
        if con.upper is not None:
            worst = max(worst, (body - pyo.value(con.upper)) / scale)
    return worst


def create_new(filename):
    def t_function(self):
        # Importing the example runs it: they declare a model and solve it at
        # module level, which is the behaviour under test.
        if EXAMPLES_DIR not in sys.path:
            sys.path.insert(0, EXAMPLES_DIR)
        try:
            module = importlib.import_module(filename[:-3])
        except Exception:
            self.fail(
                'This example is failing: %s\n%s' % (filename, traceback.format_exc())
            )

        # Running without raising is not the same as working. Every example
        # solves and writes the answer back, so the model is left holding a
        # point that must satisfy its own constraints -- a solver that returns
        # a plausible number for an infeasible point is the failure that
        # "it imported fine" cannot see.
        f = getattr(module, 'f', None)
        if f is None:
            return              # not every example need expose its Formulation
        self.assertLessEqual(
            worst_violation(f),
            FEASIBLE_RTOL,
            '%s: the solution written back violates the model' % (filename,),
        )

    t_function.__name__ = example_test_name(filename)
    return t_function


# Every example solves, and every solve route in this repository ends at ipopt
# or cyipopt, so the whole set is gated on the executable being present.
for filename in discover_examples():
    t_Function = unittest.skipIf(
        not ipopt_available, 'ipopt executable is not available'
    )(create_new(filename))
    if pint_available:
        setattr(EDIExamples, example_test_name(filename), t_Function)


if __name__ == '__main__':
    unittest.main()
