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
    from edi import Formulation

    formulation_available = True
except:
    pass
    # formulation_available = False

blackbox_available = False
try:
    from edi import BlackBoxFunctionModel

    blackbox_available = True
except:
    pass
    # blackbox_available = False

if numpy_available:
    import numpy as np


@unittest.skipIf(
    not egb_available, 'Testing edi requires pynumero external grey boxes'
)
@unittest.skipIf(not formulation_available, 'Formulation import failed')
@unittest.skipIf(not blackbox_available, 'Blackbox import failed')
@unittest.skipIf(not numpy_available, 'Testing edi requires numpy')
@unittest.skipIf(not scipy_available, 'Testing edi requires scipy')
@unittest.skipIf(not pint_available, 'Testing units requires pint')
class EDIExamples(unittest.TestCase):
    def test_edi_example_placeholder(self):
        "A placeholder"
        pass


def create_new(filename):
    def t_function(self):
        importName = filename[0:-3]
        # filename = ".."+filename
        try:
            # `examples/` lives at the top level of the standalone repository
            # rather than inside the package (it did live inside it when this was
            # pyomo.contrib.edi). Import it by file path so the test works
            # regardless of whether examples/ is installed with the wheel.
            import os, sys
            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _ex = os.path.join(_root, "examples")
            if _ex not in sys.path:
                sys.path.insert(0, _ex)
            importlib.import_module(importName)
        except:
            self.fail("This example is failing: %s" % (filename))

    return t_function


pythonFileList = ["readme_example.py", "aircraft_gp.py"]

for filename in pythonFileList:
    testName = 'test_DocumentationExample_%d' % (testIndex)
    testIndex += 1
    t_Function = create_new(filename)
    if pint_available:
        setattr(EDIExamples, testName, t_Function)


if __name__ == '__main__':
    unittest.main()
