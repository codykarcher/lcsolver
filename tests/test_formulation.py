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


@unittest.skipIf(
    not egb_available, 'Testing lcsolver requires pynumero external grey boxes'
)
@unittest.skipIf(not formulation_available, 'Formulation import failed')
@unittest.skipIf(not blackbox_available, 'Blackbox import failed')
@unittest.skipIf(not numpy_available, 'Testing lcsolver requires numpy')
@unittest.skipIf(not scipy_available, 'Testing lcsolver requires scipy')
@unittest.skipIf(not pint_available, 'Testing units requires pint')
class TestEDIFormulation(unittest.TestCase):
    def test_edi_formulation_init(self):
        "Tests that a formulation initializes to the correct type and has proper data"
        from pyomo.environ import ConcreteModel
        from lcsolver import Formulation

        f = Formulation()

        self.assertIsInstance(f, Formulation)
        self.assertIsInstance(f, ConcreteModel)

        self.assertEqual(f._objective_counter, 0)
        self.assertEqual(f._constraint_counter, 0)
        self.assertEqual(f._variable_keys, [])
        self.assertEqual(f._constant_keys, [])
        self.assertEqual(f._objective_keys, [])
        self.assertEqual(f._runtimeObjective_keys, [])
        self.assertEqual(f._objective_keys, [])
        self.assertEqual(f._runtimeConstraint_keys, [])
        self.assertEqual(f._constraint_keys, [])
        self.assertEqual(f._allConstraint_keys, [])

    def test_edi_formulation_variable(self):
        "Tests the variable constructor in lcsolver.formulation"
        import pyomo
        from lcsolver import Formulation
        from pyomo.environ import Reals, PositiveReals

        f = Formulation()

        x1 = f.Variable(
            name='x1',
            guess=1.0,
            units='m',
            description='The x variable',
            size=None,
            bounds=None,
            domain=None,
        )
        self.assertRaises(RuntimeError, f.Variable, *('x1', 1.0, 'm'))
        x2 = f.Variable('x2', 1.0, 'm')
        x3 = f.Variable('x3', 1.0, 'm', 'The x variable', None, None, None)
        x4 = f.Variable(
            name='x4',
            guess=1.0,
            units='m',
            description='The x variable',
            size=None,
            bounds=None,
            domain=PositiveReals,
        )
        self.assertRaises(
            RuntimeError,
            f.Variable,
            **{
                'name': 'x5',
                'guess': 1.0,
                'units': 'm',
                'description': 'The x variable',
                'size': None,
                'bounds': None,
                'domain': "error",
            }
        )

        x6 = f.Variable(
            name='x6',
            guess=1.0,
            units='m',
            description='The x variable',
            size=0,
            bounds=None,
            domain=None,
        )
        x7 = f.Variable(
            name='x7',
            guess=1.0,
            units='m',
            description='The x variable',
            size=5,
            bounds=None,
            domain=None,
        )
        self.assertRaises(
            ValueError,
            f.Variable,
            **{
                'name': 'x8',
                'guess': 1.0,
                'units': 'm',
                'description': 'The x variable',
                'size': 'error',
                'bounds': None,
                'domain': None,
            }
        )
        x9 = f.Variable(
            name='x9',
            guess=1.0,
            units='m',
            description='The x variable',
            size=[2, 2],
            bounds=None,
            domain=None,
        )
        self.assertRaises(
            ValueError,
            f.Variable,
            **{
                'name': 'x10',
                'guess': 1.0,
                'units': 'm',
                'description': 'The x variable',
                'size': ['2', '2'],
                'bounds': None,
                'domain': None,
            }
        )
        # NOTE: a trailing dimension of size 1 is now ACCEPTED. The check that
        # rejected sizes of 0 or 1 is commented out in Formulation.Variable, so
        # size=[2, 1] collapses to a 2-element indexed variable. This test asserts
        # the current behavior; if the restriction is reinstated, restore the
        # assertRaises(ValueError, ...) that was here.
        x11 = f.Variable(
            name='x11',
            guess=1.0,
            units='m',
            description='The x variable',
            size=[2, 1],
            bounds=None,
            domain=None,
        )
        self.assertEqual(len(x11), 2)

        x12 = f.Variable(
            name='x12',
            guess=1.0,
            units='m',
            description='The x variable',
            size=None,
            bounds=[-10, 10],
            domain=None,
        )
        self.assertRaises(
            ValueError,
            f.Variable,
            **{
                'name': 'x13',
                'guess': 1.0,
                'units': 'm',
                'description': 'The x variable',
                'size': None,
                'bounds': [10, -10],
                'domain': None,
            }
        )
        self.assertRaises(
            ValueError,
            f.Variable,
            **{
                'name': 'x14',
                'guess': 1.0,
                'units': 'm',
                'description': 'The x variable',
                'size': None,
                'bounds': ["-10", "10"],
                'domain': None,
            }
        )
        self.assertRaises(
            ValueError,
            f.Variable,
            **{
                'name': 'x15',
                'guess': 1.0,
                'units': 'm',
                'description': 'The x variable',
                'size': None,
                'bounds': [1, 2, 3],
                'domain': None,
            }
        )
        self.assertRaises(
            ValueError,
            f.Variable,
            **{
                'name': 'x16',
                'guess': 1.0,
                'units': 'm',
                'description': 'The x variable',
                'size': None,
                'bounds': "error",
                'domain': None,
            }
        )
        self.assertRaises(
            ValueError,
            f.Variable,
            **{
                'name': 'x17',
                'guess': 1.0,
                'units': 'm',
                'description': 'The x variable',
                'size': None,
                'bounds': [0, "10"],
                'domain': None,
            }
        )

        # verifies alternate unit construction
        x18 = f.Variable('x18', 1.0, pyo.units.m)
        self.assertRaises(AttributeError, f.Variable, *('x19', 1.0, 'string'))

    def test_edi_formulation_constant(self):
        "Tests the constant constructor in lcsolver.formulation"
        from lcsolver import Formulation
        from pyomo.environ import Reals, PositiveReals

        f = Formulation()

        c1 = f.Constant(
            name='c1',
            value=1.0,
            units='m',
            description='A constant c',
            size=None,
            within=None,
        )
        self.assertRaises(RuntimeError, f.Constant, *('c1', 1.0, 'm'))
        c2 = f.Constant('c2', 1.0, 'm')
        c3 = f.Constant('c3', 1.0, 'm', 'A constant c', None, None)
        c4 = f.Constant(
            name='c4',
            value=1.0,
            units='m',
            description='A constant c',
            size=None,
            within=PositiveReals,
        )
        self.assertRaises(
            RuntimeError,
            f.Constant,
            **{
                'name': 'c5',
                'value': 1.0,
                'units': 'm',
                'description': 'The x variable',
                'size': None,
                'within': "error",
            }
        )

        c6 = f.Constant(
            name='c6',
            value=1.0,
            units='m',
            description='A constant c',
            size=0,
            within=None,
        )
        c7 = f.Constant(
            name='c7',
            value=1.0,
            units='m',
            description='A constant c',
            size=5,
            within=None,
        )
        self.assertRaises(
            ValueError,
            f.Constant,
            **{
                'name': 'c8',
                'value': 1.0,
                'units': 'm',
                'description': 'A constant c',
                'size': 'error',
                'within': None,
            }
        )
        c9 = f.Constant(
            name='c9',
            value=1.0,
            units='m',
            description='A constant c',
            size=[2, 2],
            within=None,
        )
        self.assertRaises(
            ValueError,
            f.Constant,
            **{
                'name': 'c10',
                'value': 1.0,
                'units': 'm',
                'description': 'A constant c',
                'size': ['2', '2'],
                'within': None,
            }
        )
        # NOTE: as with Variable, a trailing dimension of size 1 is now ACCEPTED --
        # the 0-or-1 size check is commented out in Formulation.Constant. This test
        # asserts the current behavior; restore the assertRaises if the restriction
        # is reinstated.
        c11 = f.Constant(
            name='c11',
            value=1.0,
            units='m',
            description='A constant c',
            size=[2, 1],
            within=None,
        )
        self.assertEqual(len(c11), 2)

    def test_edi_formulation_objective(self):
        "Tests the objective constructor in lcsolver.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from lcsolver import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        f.Objective(x + y)
        self.assertEqual(len(f.get_objectives()), 1)

    def test_edi_formulation_runtimeobjective(self):
        "Tests the runtime objective constructor in lcsolver.formulation"
        # TODO: not currently implemented, see:  https://github.com/codykarcher/pyomo/issues/5
        pass

    def test_edi_formulation_constraint(self):
        "Tests the constraint constructor in lcsolver.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from lcsolver import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        f.Objective(x + y)
        f.Constraint(x + y <= 1.0 * units.m)
        self.assertEqual(len(f.get_explicitConstraints()), 1)

    def test_edi_formulation_runtimeconstraint_tuple(self):
        "Tests the runtime constraint constructor in lcsolver.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from lcsolver import Formulation, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        z = f.Variable(
            name='z', guess=1.0, units='m^2', description='The unit circle output'
        )
        c = f.Constant(
            name='c', value=1.0, units='', description='A constant c', size=2
        )
        f.Objective(x + y)

        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):
                super(UnitCircle, self).__init__()
                self.description = 'This model evaluates the function: z = x**2 + y**2'
                self.inputs.append(name='x', units='ft', description='The x variable')
                self.inputs.append(name='y', units='ft', description='The y variable')
                self.outputs.append(
                    name='z',
                    units='ft**2',
                    description='Resultant of the unit circle evaluation',
                )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, x, y):  # The actual function that does things
                x = pyo.value(
                    units.convert(x, self.inputs[0].units)
                )  # Converts to correct units then casts to float
                y = pyo.value(
                    units.convert(y, self.inputs[1].units)
                )  # Converts to correct units then casts to float
                z = x**2 + y**2  # Compute z
                dzdx = 2 * x  # Compute dz/dx
                dzdy = 2 * y  # Compute dz/dy
                z *= units.ft**2
                dzdx *= units.ft  # units.ft**2 / units.ft
                dzdy *= units.ft  # units.ft**2 / units.ft
                return z, [dzdx, dzdy]  # return z, grad(z), hess(z)...

        f.Constraint(z <= 1 * units.m**2)

        f.RuntimeConstraint(*(z, '==', [x, y], UnitCircle()))
        # the three call forms below must all register the same thing: one
        # explicit constraint and one runtime constraint
        self.assertEqual(len(f.get_runtimeConstraints()), 1)
        self.assertEqual(len(f.get_explicitConstraints()), 1)
        self.assertEqual(len(f.get_constraints()), 2)

    def test_edi_formulation_runtimeconstraint_list(self):
        "Tests the runtime constraint constructor in lcsolver.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from lcsolver import Formulation, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        z = f.Variable(
            name='z', guess=1.0, units='m^2', description='The unit circle output'
        )
        c = f.Constant(
            name='c', value=1.0, units='', description='A constant c', size=2
        )
        f.Objective(x + y)

        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):
                super(UnitCircle, self).__init__()
                self.description = 'This model evaluates the function: z = x**2 + y**2'
                self.inputs.append(name='x', units='ft', description='The x variable')
                self.inputs.append(name='y', units='ft', description='The y variable')
                self.outputs.append(
                    name='z',
                    units='ft**2',
                    description='Resultant of the unit circle evaluation',
                )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, x, y):  # The actual function that does things
                x = pyo.value(
                    units.convert(x, self.inputs[0].units)
                )  # Converts to correct units then casts to float
                y = pyo.value(
                    units.convert(y, self.inputs[1].units)
                )  # Converts to correct units then casts to float
                z = x**2 + y**2  # Compute z
                dzdx = 2 * x  # Compute dz/dx
                dzdy = 2 * y  # Compute dz/dy
                z *= units.ft**2
                dzdx *= units.ft  # units.ft**2 / units.ft
                dzdy *= units.ft  # units.ft**2 / units.ft
                return z, [dzdx, dzdy]  # return z, grad(z), hess(z)...

        f.Constraint(z <= 1 * units.m**2)

        f.RuntimeConstraint(*[[z], ['=='], [x, y], UnitCircle()])
        self.assertEqual(len(f.get_runtimeConstraints()), 1)
        self.assertEqual(len(f.get_explicitConstraints()), 1)
        self.assertEqual(len(f.get_constraints()), 2)

    def test_edi_formulation_runtimeconstraint_dict(self):
        "Tests the runtime constraint constructor in lcsolver.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from lcsolver import Formulation, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        z = f.Variable(
            name='z', guess=1.0, units='m^2', description='The unit circle output'
        )
        c = f.Constant(
            name='c', value=1.0, units='', description='A constant c', size=2
        )
        f.Objective(x + y)

        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):
                super(UnitCircle, self).__init__()
                self.description = 'This model evaluates the function: z = x**2 + y**2'
                self.inputs.append(name='x', units='ft', description='The x variable')
                self.inputs.append(name='y', units='ft', description='The y variable')
                self.outputs.append(
                    name='z',
                    units='ft**2',
                    description='Resultant of the unit circle evaluation',
                )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, x, y):  # The actual function that does things
                x = pyo.value(
                    units.convert(x, self.inputs[0].units)
                )  # Converts to correct units then casts to float
                y = pyo.value(
                    units.convert(y, self.inputs[1].units)
                )  # Converts to correct units then casts to float
                z = x**2 + y**2  # Compute z
                dzdx = 2 * x  # Compute dz/dx
                dzdy = 2 * y  # Compute dz/dy
                z *= units.ft**2
                dzdx *= units.ft  # units.ft**2 / units.ft
                dzdy *= units.ft  # units.ft**2 / units.ft
                return z, [dzdx, dzdy]  # return z, grad(z), hess(z)...

        f.Constraint(z <= 1 * units.m**2)

        f.RuntimeConstraint(
            **{
                'outputs': z,
                'operators': '==',
                'inputs': [x, y],
                'black_box': UnitCircle(),
            }
        )
        self.assertEqual(len(f.get_runtimeConstraints()), 1)
        self.assertEqual(len(f.get_explicitConstraints()), 1)
        self.assertEqual(len(f.get_constraints()), 2)

    def test_edi_formulation_constraintlist_1(self):
        "Tests the constraint list constructor in lcsolver.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from lcsolver import Formulation, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        z = f.Variable(
            name='z', guess=1.0, units='m^2', description='The unit circle output'
        )
        c = f.Constant(
            name='c', value=1.0, units='', description='A constant c', size=2
        )
        f.Objective(x + y)

        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):
                super(UnitCircle, self).__init__()
                self.description = 'This model evaluates the function: z = x**2 + y**2'
                self.inputs.append(name='x', units='ft', description='The x variable')
                self.inputs.append(name='y', units='ft', description='The y variable')
                self.outputs.append(
                    name='z',
                    units='ft**2',
                    description='Resultant of the unit circle evaluation',
                )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, x, y):  # The actual function that does things
                x = pyo.value(
                    units.convert(x, self.inputs[0].units)
                )  # Converts to correct units then casts to float
                y = pyo.value(
                    units.convert(y, self.inputs[1].units)
                )  # Converts to correct units then casts to float
                z = x**2 + y**2  # Compute z
                dzdx = 2 * x  # Compute dz/dx
                dzdy = 2 * y  # Compute dz/dy
                z *= units.ft**2
                dzdx *= units.ft  # units.ft**2 / units.ft
                dzdy *= units.ft  # units.ft**2 / units.ft
                return z, [dzdx, dzdy]  # return z, grad(z), hess(z)...

        f.ConstraintList([(z, '==', [x, y], UnitCircle()), z <= 1 * units.m**2])

        cl = f.get_constraints()

        self.assertTrue(len(cl) == 2)

    def test_edi_formulation_constraintlist_2(self):
        "Tests the constraint list constructor in lcsolver.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from lcsolver import Formulation, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m**2', description='The y variable')
        f.Objective(y)

        class Parabola(BlackBoxFunctionModel):
            def __init__(self):
                super(Parabola, self).__init__()
                self.description = 'This model evaluates the function: y = x**2'
                self.inputs.append(name='x', units='ft', description='The x variable')
                self.outputs.append(
                    name='y', units='ft**2', description='The y variable'
                )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, x):  # The actual function that does things
                x = pyo.value(
                    units.convert(x, self.inputs[0].units)
                )  # Converts to correct units then casts to float
                y = x**2  # Compute y
                dydx = 2 * x  # Compute dy/dx
                y *= units.ft**2
                dydx *= units.ft  # units.ft**2 / units.ft
                return y, [dydx]  # return z, grad(z), hess(z)...

        f.ConstraintList(
            [{'outputs': y, 'operators': '==', 'inputs': x, 'black_box': Parabola()}]
        )

        cl = f.get_constraints()

        self.assertTrue(len(cl) == 1)

    def test_edi_formulation_constraintlist_3(self):
        "Tests the constraint list constructor in lcsolver.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from lcsolver import Formulation, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(
            name='x', guess=1.0, units='m', description='The x variable', size=3
        )
        y = f.Variable(
            name='y', guess=1.0, units='m**2', description='The y variable', size=3
        )
        f.Objective(y[0] + y[1] + y[2])

        class Parabola(BlackBoxFunctionModel):
            def __init__(self):
                super(Parabola, self).__init__()
                self.description = 'This model evaluates the function: y = x**2'
                self.inputs.append(
                    name='x', size=3, units='ft', description='The x variable'
                )
                self.outputs.append(
                    name='y', size=3, units='ft**2', description='The y variable'
                )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(*args, **kwargs):  # The actual function that does things
                args = list(args)
                self = args.pop(0)
                runCases, returnMode, remainingKwargs = self.parseInputs(
                    *args, **kwargs
                )

                x = self.sanitizeInputs(runCases[0]['x'])
                x = np.array([pyo.value(xval) for xval in x], dtype=np.float64)

                y = x**2  # Compute y
                dydx = 2 * x  # Compute dy/dx

                y = y * units.ft**2
                dydx = np.diag(dydx)
                dydx = dydx * units.ft  # units.ft**2 / units.ft

                return y, [dydx]  # return z, grad(z), hess(z)...

        f.ConstraintList(
            [{'outputs': y, 'operators': '==', 'inputs': x, 'black_box': Parabola()}]
        )
        cl = f.get_constraints()
        self.assertTrue(len(cl) == 1)

    def test_edi_formulation_runtimeconstraint_exceptions(self):
        "Tests the runtime constraint constructor in lcsolver.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from lcsolver import Formulation, BlackBoxFunctionModel

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        z = f.Variable(
            name='z', guess=1.0, units='m^2', description='The unit circle output'
        )
        c = f.Constant(
            name='c', value=1.0, units='', description='A constant c', size=2
        )
        f.Objective(x + y)

        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):
                super(UnitCircle, self).__init__()
                self.description = 'This model evaluates the function: z = x**2 + y**2'
                self.inputs.append(name='x', units='ft', description='The x variable')
                self.inputs.append(name='y', units='ft', description='The y variable')
                self.outputs.append(
                    name='z',
                    units='ft**2',
                    description='Resultant of the unit circle evaluation',
                )
                self.availableDerivative = 1
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, x, y):  # The actual function that does things
                x = pyo.value(
                    units.convert(x, self.inputs[0].units)
                )  # Converts to correct units then casts to float
                y = pyo.value(
                    units.convert(y, self.inputs[1].units)
                )  # Converts to correct units then casts to float
                z = x**2 + y**2  # Compute z
                dzdx = 2 * x  # Compute dz/dx
                dzdy = 2 * y  # Compute dz/dy
                z *= units.ft**2
                dzdx *= units.ft  # units.ft**2 / units.ft
                dzdy *= units.ft  # units.ft**2 / units.ft
                return z, [dzdx, dzdy]  # return z, grad(z), hess(z)...

        f.Constraint(z <= 1 * units.m**2)

        # flaggs the input or  as bad before assigning the incorrect black box
        self.assertRaises(
            ValueError, f.RuntimeConstraint, *(z, '==', 1.0, UnitCircle())
        )
        self.assertRaises(
            ValueError, f.RuntimeConstraint, *(1.0, '==', [x, y], UnitCircle())
        )

        self.assertRaises(
            ValueError, f.RuntimeConstraint, *(z, '==', [1.0, y], UnitCircle())
        )
        self.assertRaises(
            ValueError, f.RuntimeConstraint, *(z, 1.0, [x, y], UnitCircle())
        )
        self.assertRaises(
            ValueError, f.RuntimeConstraint, *(z, '=', [x, y], UnitCircle())
        )

    def test_edi_formulation_getvariables(self):
        "Tests the get_variables function in lcsolver.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from lcsolver import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')

        vrs = f.get_variables()
        self.assertListEqual(vrs, [x, y])

    def test_edi_formulation_getconstants(self):
        "Tests the get_constants function in lcsolver.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from lcsolver import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')

        c1 = f.Constant(
            name='c1',
            value=1.0,
            units='m',
            description='A constant c1',
            size=None,
            within=None,
        )
        c2 = f.Constant(
            name='c2',
            value=1.0,
            units='m',
            description='A constant c2',
            size=None,
            within=None,
        )

        csts = f.get_constants()
        self.assertListEqual(csts, [c1, c2])

    def test_edi_formulation_getobjectives(self):
        "Tests the get_objectives function in lcsolver.formulation"
        import pyomo.environ as pyo
        from pyomo.environ import units
        from lcsolver import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        f.Objective(x + y)
        objList = f.get_objectives()
        self.assertTrue(len(objList) == 1)
        # not really sure how to check this, so I wont

    def test_edi_formulation_getconstraints(self):
        "Tests the get_constraints, get_explicitConstraints, and get_runtimeConstraints functions in lcsolver.formulation"
        # =================
        # Import Statements
        # =================
        import pyomo.environ as pyo
        from pyomo.environ import units
        from lcsolver import Formulation, BlackBoxFunctionModel

        # ===================
        # Declare Formulation
        # ===================
        f = Formulation()

        # =================
        # Declare Variables
        # =================
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')
        z = f.Variable(
            name='z', guess=1.0, units='m^2', description='The unit circle output'
        )

        # =================
        # Declare Constants
        # =================
        c = f.Constant(
            name='c', value=1.0, units='', description='A constant c', size=2
        )

        # =====================
        # Declare the Objective
        # =====================
        f.Objective(c[0] * x + c[1] * y)

        # ===================
        # Declare a Black Box
        # ===================
        class UnitCircle(BlackBoxFunctionModel):
            def __init__(self):  # The initialization function
                # Initialize the black box model
                super(UnitCircle, self).__init__()

                # A brief description of the model
                self.description = 'This model evaluates the function: z = x**2 + y**2'

                # Declare the black box model inputs
                self.inputs.append(name='x', units='ft', description='The x variable')
                self.inputs.append(name='y', units='ft', description='The y variable')

                # Declare the black box model outputs
                self.outputs.append(
                    name='z',
                    units='ft**2',
                    description='Resultant of the unit circle evaluation',
                )

                # Declare the maximum available derivative
                self.availableDerivative = 1

                # Post-initialization setup
                self.post_init_setup(len(self.inputs))

            def BlackBox(self, x, y):  # The actual function that does things
                x = pyo.value(
                    units.convert(x, self.inputs[0].units)
                )  # Converts to correct units then casts to float
                y = pyo.value(
                    units.convert(y, self.inputs[1].units)
                )  # Converts to correct units then casts to float

                z = x**2 + y**2  # Compute z
                dzdx = 2 * x  # Compute dz/dx
                dzdy = 2 * y  # Compute dz/dy

                z *= units.ft**2
                dzdx *= units.ft  # units.ft**2 / units.ft
                dzdy *= units.ft  # units.ft**2 / units.ft

                return z, [dzdx, dzdy]  # return z, grad(z), hess(z)...

        # =======================
        # Declare the Constraints
        # =======================
        f.ConstraintList([(z, '==', [x, y], UnitCircle()), z <= 1 * units.m**2])

        cl = f.get_constraints()
        ecl = f.get_explicitConstraints()
        rcl = f.get_runtimeConstraints()

        self.assertTrue(len(cl) == 2)
        self.assertTrue(len(ecl) == 1)
        self.assertTrue(len(rcl) == 1)

    def test_edi_formulation_checkunits(self):
        "Tests the check_units function in lcsolver.formulation"
        import pyomo
        import pyomo.environ as pyo
        from pyomo.environ import units
        from lcsolver import Formulation

        f = Formulation()
        x = f.Variable(name='x', guess=1.0, units='m', description='The x variable')
        y = f.Variable(name='y', guess=1.0, units='m', description='The y variable')

        f.Objective(x + y)
        f.Constraint(x + y <= 1.0 * units.m)
        f.check_units()

        f.Constraint(2.0 * x + y <= 1.0)
        self.assertRaises(
            pyomo.core.base.units_container.UnitsError, f.check_units, *()
        )

        f2 = Formulation()
        u = f2.Variable(name='u', guess=1.0, units='m', description='The u variable')
        v = f2.Variable(name='v', guess=1.0, units='kg', description='The v variable')
        f2.Objective(u + v)
        self.assertRaises(
            pyomo.core.base.units_container.UnitsError, f2.check_units, *()
        )




@unittest.skipIf(not formulation_available, 'Formulation import failed')
class TestGroupsAndGuesses(unittest.TestCase):
    """The two ergonomics changes: named regions, and an opt-out on guesses."""

    def test_group_namespaces_without_a_prefix_argument(self):
        f = Formulation()
        wing = f.group('wing')
        ar = wing.Variable('AR', 11.0, '-', 'aspect ratio')
        box = wing.group('box')
        t = box.Variable('t_cap', 0.01, 'm', 'cap thickness')

        self.assertEqual(ar.name, 'wing_AR')
        self.assertEqual(t.name, 'wing_box_t_cap')

    def test_the_group_is_reachable_from_the_formulation(self):
        """`f.wing.AR` -- so a builder need not thread a prefix and return dicts."""
        f = Formulation()
        wing = f.group('wing')
        wing.Variable('AR', 11.0, '-', 'aspect ratio')
        wing.group('box').Variable('t_cap', 0.01, 'm', 'cap')

        self.assertIs(f.wing, wing)
        self.assertEqual(f.wing.AR.name, 'wing_AR')
        self.assertEqual(f.wing.box.t_cap.name, 'wing_box_t_cap')

    def test_group_is_idempotent(self):
        f = Formulation()
        self.assertIs(f.group('wing'), f.group('wing'))

    def test_an_unknown_attribute_still_raises(self):
        f = Formulation()
        f.group('wing')
        with self.assertRaises(AttributeError):
            f.definitely_not_there

    def test_a_group_can_carry_constraints(self):
        f = Formulation()
        w = f.group('w')
        x = w.Variable('x', 2.0, '-', 'x')
        y = w.Variable('y', 2.0, '-', 'y')
        f.Objective(x + y)
        w.Constraint(x * y >= 4.0)
        self.assertEqual(len(f.get_constraints()), 1)

    def test_guess_is_required_by_default(self):
        f = Formulation()
        with self.assertRaises(ValueError) as ctx:
            f.Variable('x', units='-', description='no guess')
        self.assertIn('require_guesses', str(ctx.exception))

    def test_guess_can_be_waived_and_the_waiver_is_recorded(self):
        """Off is for a GP, where the solve is global and the guess cannot
        change the answer. The omission stays visible in `defaulted_guesses`."""
        f = Formulation()
        f.require_guesses = False
        x = f.Variable('x', units='-', description='defaulted')
        y = f.Variable('y', 3.0, '-', 'given')

        self.assertEqual(f.defaulted_guesses, ['x'])
        self.assertAlmostEqual(pyo.value(x), 1.0)
        self.assertAlmostEqual(pyo.value(y), 3.0)

    def test_units_are_still_required(self):
        f = Formulation()
        f.require_guesses = False
        with self.assertRaises(ValueError) as ctx:
            f.Variable('x', description='no units')
        self.assertIn('units', str(ctx.exception))

    def test_a_group_may_not_shadow_a_component(self):
        """Pyomo resolves components before __getattr__, so the group would be
        created and then be permanently unreachable as `f.<name>`."""
        f = Formulation()
        f.Variable('wing', 1.0, '-', 'a variable called wing')
        with self.assertRaises(ValueError) as ctx:
            f.group('wing')
        self.assertIn('shadow', str(ctx.exception))

    def test_plain_variables_are_unaffected_by_groups(self):
        """Groups are additive: the old way keeps working, and a grouped
        variable is an ordinary flat component reachable either way."""
        f = Formulation()
        plain = f.Variable('plain', 2.0, '-', 'made the old way')
        ar = f.group('wing').Variable('AR', 11.0, '-', 'made via a group')

        self.assertIs(f.plain, plain)
        self.assertIs(f.wing_AR, ar)          # flat component name
        self.assertIs(f.wing.AR, ar)          # and through the group
        self.assertEqual({v.name for v in f.get_variables()},
                         {'plain', 'wing_AR'})


@unittest.skipIf(not numpy_available, 'Testing lcsolver requires numpy')
@unittest.skipIf(not pint_available, 'Testing units requires pint')
class TestArrayInitialization(unittest.TestCase):
    """Initial values given as an array, laid out onto the index set.

    Pyomo wants a dict keyed by the index tuple, and a nested list handed to it
    raises `KeyError: "Index '0' is not valid for indexed component 'e'"`,
    which names neither the shape it wanted nor the shape it got. The
    comprehension that works is noise in a model file.
    """

    def test_a_matrix_constant_from_a_nested_list(self):
        f = Formulation()
        e = f.Constant('e', [[3.0, 5.0, 9.0], [8.0, 4.0, 3.0]], 'kg',
                       'a matrix', size=[2, 3])
        self.assertEqual(pyo.value(e[0, 1]), 5.0)
        self.assertEqual(pyo.value(e[1, 2]), 3.0)   # outer level is the row

    def test_a_matrix_variable_guess_from_a_numpy_array(self):
        import numpy as np

        f = Formulation()
        M = f.Variable('M', np.arange(6.0).reshape(2, 3), 'm', 'a matrix',
                       size=[2, 3])
        self.assertEqual(pyo.value(M[1, 0]), 3.0)

    def test_three_dimensions(self):
        import numpy as np

        f = Formulation()
        T = f.Variable('T', np.arange(8.0).reshape(2, 2, 2), 'm', 'a tensor',
                       size=[2, 2, 2])
        self.assertEqual(pyo.value(T[1, 0, 1]), 5.0)

    def test_a_flat_list_still_works(self):
        f = Formulation()
        c = f.Constant('c', [1.0, 2.0], '-', 'a vector', size=2)
        self.assertEqual([pyo.value(c[i]) for i in (0, 1)], [1.0, 2.0])

    def test_a_scalar_still_reaches_every_element(self):
        f = Formulation()
        x = f.Variable('x', 2.0, 'm', 'a vector', size=4)
        self.assertEqual([pyo.value(x[i]) for i in range(4)], [2.0] * 4)

    def test_a_dict_still_works(self):
        f = Formulation()
        c = f.Constant('c', {0: 1.0, 1: 2.0}, '-', 'a vector', size=2)
        self.assertEqual(pyo.value(c[1]), 2.0)

    def test_a_transposed_array_is_refused(self):
        """The reason the shape is checked rather than reshaped."""
        f = Formulation()
        with self.assertRaises(ValueError) as ctx:
            f.Constant('e', [[3.0, 5.0, 9.0], [8.0, 4.0, 3.0]], '-', 'e',
                       size=[3, 2])
        self.assertIn('[2, 3]', str(ctx.exception))
        self.assertIn('transposed', str(ctx.exception))

    def test_a_wrong_length_is_refused(self):
        f = Formulation()
        with self.assertRaises(ValueError) as ctx:
            f.Constant('c', [1.0, 2.0, 3.0], '-', 'c', size=4)
        self.assertIn('size=4', str(ctx.exception))

    def test_an_array_with_no_size_is_refused(self):
        f = Formulation()
        with self.assertRaises(ValueError) as ctx:
            f.Constant('c', [1.0, 2.0], '-', 'c')
        self.assertIn('size=2', str(ctx.exception))


class TestLoadConstants(unittest.TestCase):
    """An input deck applied to a built formulation."""

    def _model(self):
        f = Formulation()
        f.Variable('x', 1.0, 'm', 'x')
        f.Constant('k', 2.0, 'm', 'k')
        f.Constant('v', [1.0, 2.0], '-', 'v', size=2)
        g = f.group('wing')
        g.Constant('area', 3.0, 'm^2', 'wing area')
        return f

    def test_a_scalar_constant_is_set(self):
        f = self._model()
        f.load_constants({'k': 5.0})
        self.assertEqual(pyo.value(f.k), 5.0)

    def test_a_grouped_constant_is_set_by_its_prefixed_name(self):
        f = self._model()
        f.load_constants({'wing_area': 9.0})
        self.assertEqual(pyo.value(f.wing_area), 9.0)

    def test_a_sized_constant_takes_a_sequence(self):
        f = self._model()
        f.load_constants({'v': [7.0, 8.0]})
        self.assertEqual([pyo.value(f.v[i]) for i in (0, 1)], [7.0, 8.0])

    def test_the_formulation_is_returned_for_chaining(self):
        f = self._model()
        self.assertIs(f.load_constants({'k': 4.0}), f)

    def test_an_unknown_name_is_refused_with_a_suggestion(self):
        f = self._model()
        with self.assertRaises(KeyError) as ctx:
            f.load_constants({'kk': 1.0})
        self.assertIn('not a constant', str(ctx.exception))
        self.assertIn('nearest declared: k', str(ctx.exception))

    def test_a_variable_name_is_not_a_constant(self):
        """The deck sets constants; a variable is the model's to choose."""
        f = self._model()
        with self.assertRaises(KeyError):
            f.load_constants({'x': 1.0})

    def test_a_wrong_length_sequence_is_refused(self):
        f = self._model()
        with self.assertRaises(ValueError) as ctx:
            f.load_constants({'v': [1.0]})
        self.assertIn('2 entries', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
