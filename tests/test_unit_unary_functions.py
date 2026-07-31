"""Unary functions must pass the unit checker.

``handle_unary_node`` opened with ``units.get_units(arg1)``, where ``arg1`` is
the walker's own ``unitsPack`` namedtuple rather than a Pyomo expression. That
raised ``AttributeError: 'unitsPack' object has no attribute
'is_expression_type'`` before any branch ran, so *every* unary function failed
-- including ``sqrt``, which the one branch below it existed to support.

The AttributeError reached the user through the unit reporter, which dutifully
printed both sides' units and observed that they agreed:

    Error in units for objective 'objective_1':
        sin(x) + x**2
        [dimensionless]  =/=  [dimensionless]

-- a contradiction on its face, and a bad way to learn that ``sqrt`` was not
supported. ``x ** 0.5`` was unaffected, being a power node, which is why models
that spell their roots that way never noticed.

The rules mirror ``pyomo.core.base.units_container`` so the two agree.
"""
import pyomo.environ as pyo
import pytest

from edi import Formulation
from edi.presolve.unitCorrector import UnitMismatch, unit_corrector

#: (name, callable, rule). 'dimensionless' takes a dimensionless argument and
#: returns one; 'same' preserves units; 'sqrt' halves the exponents.
FUNCTIONS = [
    ('sqrt', pyo.sqrt, 'sqrt'),
    ('sin', pyo.sin, 'dimensionless'),
    ('cos', pyo.cos, 'dimensionless'),
    ('tan', pyo.tan, 'dimensionless'),
    ('sinh', pyo.sinh, 'dimensionless'),
    ('cosh', pyo.cosh, 'dimensionless'),
    ('tanh', pyo.tanh, 'dimensionless'),
    ('asin', pyo.asin, 'dimensionless'),
    ('acos', pyo.acos, 'dimensionless'),
    ('atan', pyo.atan, 'dimensionless'),
    ('exp', pyo.exp, 'dimensionless'),
    ('log', pyo.log, 'dimensionless'),
    ('log10', pyo.log10, 'dimensionless'),
    ('ceil', pyo.ceil, 'same'),
    ('floor', pyo.floor, 'same'),
]


@pytest.mark.parametrize('name,fn,rule', FUNCTIONS,
                         ids=[f[0] for f in FUNCTIONS])
def test_dimensionless_argument_passes(name, fn, rule):
    """Every one of them, applied to a dimensionless variable."""
    f = Formulation()
    x = f.Variable(name='x', guess=0.5, units='-', description='x')
    f.Objective(fn(x) + x)
    f.ConstraintList([x >= 0.1])
    unit_corrector(f)              # must not raise


def test_sqrt_halves_the_units():
    """``sqrt(S*AR)`` is a length when S is an area. The point of the fix."""
    f = Formulation()
    S = f.Variable(name='S', guess=30.0, units='m^2', description='area')
    AR = f.Variable(name='AR', guess=20.0, units='-', description='AR')
    b = f.Variable(name='b', guess=25.0, units='m', description='span')
    f.Objective(b)
    f.ConstraintList([b >= pyo.sqrt(S * AR)])
    unit_corrector(f)              # m vs sqrt(m^2) -- must not raise


def test_sqrt_still_catches_a_real_mismatch():
    """The fix must not turn sqrt into a hole in the checker."""
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='m', description='x')
    y = f.Variable(name='y', guess=1.0, units='m', description='y')
    f.Objective(y)
    f.ConstraintList([y >= pyo.sqrt(x)])          # m vs m**0.5
    with pytest.raises(UnitMismatch):
        unit_corrector(f)


@pytest.mark.parametrize('name,fn', [('sin', pyo.sin), ('exp', pyo.exp),
                                     ('log', pyo.log)],
                         ids=['sin', 'exp', 'log'])
def test_dimensional_argument_is_rejected_by_name(name, fn):
    """sin(metres) is meaningless, and the message should say which function."""
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='m', description='x')
    y = f.Variable(name='y', guess=1.0, units='-', description='y')
    f.Objective(y)
    f.ConstraintList([y >= fn(x)])
    with pytest.raises(UnitMismatch) as excinfo:
        unit_corrector(f)
    assert f'{name}() requires a dimensionless argument' in str(excinfo.value)


def test_ceil_preserves_units():
    """Pyomo's rule, mirrored: ceil of a length is a length."""
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='m', description='x')
    y = f.Variable(name='y', guess=1.0, units='m', description='y')
    f.Objective(y)
    f.ConstraintList([y >= pyo.ceil(x)])
    unit_corrector(f)              # must not raise


def test_unknown_function_says_so():
    """An unhandled function should name itself, not fail as a unit mismatch."""
    from edi.presolve.unitWalker import _UNARY_UNITS
    assert 'sqrt' in _UNARY_UNITS and 'sin' in _UNARY_UNITS
    # The table is the contract with pyomo's own; if pyomo grows a function we
    # do not know, the error names it rather than reporting a bogus mismatch.
    from pyomo.core.base.units_container import PintUnitExtractionVisitor
    pyomo_known = set(PintUnitExtractionVisitor.unary_function_method_map)
    assert pyomo_known == set(_UNARY_UNITS), (
        'EDI and pyomo disagree about which unary functions exist: '
        f'{pyomo_known ^ set(_UNARY_UNITS)}')


def test_solves_end_to_end_with_sqrt():
    """The whole path, not just the walker."""
    from edi.solvers.solver import solve
    f = Formulation()
    S = f.Variable(name='S', guess=30.0, units='m^2', description='area')
    AR = f.Variable(name='AR', guess=20.0, units='-', description='AR')
    b = f.Variable(name='b', guess=25.0, units='m', description='span')
    S_req = f.Constant(name='S_req', value=27.84, units='m^2', description='S')
    AR_req = f.Constant(name='AR_req', value=20.39, units='-', description='AR')
    f.Objective(b)
    f.ConstraintList([b >= pyo.sqrt(S * AR), S >= S_req, AR >= AR_req])
    solve(f, sensitivities=False)
    assert float(f.solution['b']) == pytest.approx((27.84 * 20.39) ** 0.5,
                                                   rel=1e-6)
