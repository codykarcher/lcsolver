#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Rows without a namespace.

A `ConstraintGenerator` relates quantities its caller already declared and owns
nothing itself -- which is what separates it from a `SubModel`, and what lets a
posynomial fit and a black box sit behind the same call.

The load-bearing part is that its return goes into `ConstraintList` WHOLE. A
generator that needs five rows must read at the call site exactly like one that
needs one, or the caller is coupled to the very thing the type exists to let it
ignore.
"""
import pyomo.environ as pyo
import pytest

from lcsolver import ConstraintGenerator, Formulation, units
from lcsolver.objects.blackBoxFunctionModel import BlackBoxFunctionModel


def _model():
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='m', description='x')
    f.Objective(x)
    return f, x


def test_a_generator_declares_nothing_and_posts_through_its_caller():
    class Floor(ConstraintGenerator):
        def generate_rows(self, x, at):
            return [x >= at * units.m]

    f, x = _model()
    f.floor = Floor()
    f.ConstraintList([f.floor.generate_rows(x, 2.0)])

    assert len(f._allConstraint_keys) == 1
    # it is attached, and it is NOT a block: no name, no group, no deck keys
    assert isinstance(f.floor, ConstraintGenerator)
    assert [v.name for v in f.get_variables()] == ['x']


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
def test_however_many_rows_it_returns_read_the_same(n):
    """Four is the case that used to break.

    A bare list was dispatched to RuntimeConstraint, so a four-row generator
    was read as [outputs, operators, inputs, box] and every other count raised
    on arity. Rows are told from a black box by what the entry IS now, so no
    count is special.
    """
    class NRows(ConstraintGenerator):
        def generate_rows(self, x):
            return [x >= float(i + 1) * units.m for i in range(n)]

    f, x = _model()
    f.ConstraintList([NRows().generate_rows(x)])
    assert len(f._allConstraint_keys) == n


def test_a_black_box_is_still_told_apart_from_a_list_of_rows():
    class UnitCircle(BlackBoxFunctionModel):
        def __init__(self):
            super().__init__()
            self.description = 'unit circle'
            self.inputs.append(name='x', units='m', description='x')
            self.inputs.append(name='y', units='m', description='y')
            self.outputs.append(name='z', units='m**2', description='z')
            self.availableDerivative = 1

        def BlackBox(self, x, y):
            x = pyo.value(units.convert(x, self.inputs[0].units))
            y = pyo.value(units.convert(y, self.inputs[1].units))
            return x**2 + y**2, [2 * x, 2 * y]

    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='m', description='x')
    y = f.Variable(name='y', guess=1.0, units='m', description='y')
    z = f.Variable(name='z', guess=1.0, units='m**2', description='z')
    f.Objective(z)
    f.ConstraintList([
        x + y <= 1.0 * units.m,
        [z, '==', [x, y], UnitCircle()],      # four parts, last one the box
        ])
    assert len(f._runtimeConstraint_keys) == 1
    assert len(f._allConstraint_keys) == 2


def test_generate_holographic_rows_defaults_to_none():
    class Bare(ConstraintGenerator):
        def generate_rows(self, x):
            return [x >= 1.0 * units.m]

    assert Bare().generate_holographic_rows() == []


def test_generate_rows_must_be_implemented():
    class Empty(ConstraintGenerator):
        pass

    with pytest.raises(NotImplementedError) as e:
        Empty().generate_rows()
    assert 'Empty' in str(e.value)
