"""Clone semantics of a black box holding a live analysis handle.

A pyCAPS Problem, a CFD session, a ctypes wrapper -- none of these can be
deep-copied, and every formulation clone (unit correction clones the model)
must therefore share them by reference. That sharing used to happen through
a bare ``except: share``, which also silently aliased any MUTABLE state that
failed to copy. Now sharing is a declaration (``reference_attributes``), an
undeclared non-copyable attribute warns [LC-W311], and internal caches are
reset on the clone rather than copied or shared.
"""
import copy
import warnings

import pytest

try:
    from lcsolver import BlackBoxFunctionModel, Formulation
    from lcsolver.presolve.unitCorrector import unit_corrector
    available = True
except Exception:                                    # pragma: no cover
    available = False


class _LiveHandle:
    """Refuses deepcopy the way a ctypes/SWIG handle does."""

    def __deepcopy__(self, memo):
        raise TypeError('ctypes objects containing pointers cannot be pickled')


class _DeclaredBox(BlackBoxFunctionModel if available else object):
    reference_attributes = ('capsProblem',)

    def __init__(self):
        super().__init__()
        self.capsProblem = _LiveHandle()
        self.inputs.append('x', units='-', description='in')
        self.outputs.append(name='y', units='-')
        self.availableDerivative = 1

    def BlackBox(self, x):
        x = self.sanitizeInputs(x, strip_units=True)
        return self.packOutputs(float(x) ** 2, [2.0 * float(x)])


class _UndeclaredBox(BlackBoxFunctionModel if available else object):
    def __init__(self):
        super().__init__()
        self.capsProblem = _LiveHandle()
        self.inputs.append('x', units='-', description='in')
        self.outputs.append(name='y', units='-')
        self.availableDerivative = 1

    def BlackBox(self, x):
        x = self.sanitizeInputs(x, strip_units=True)
        return self.packOutputs(float(x) ** 2, [2.0 * float(x)])


@pytest.mark.skipif(not available, reason='LCsolver import failed')
class TestReferenceAttributes:

    def test_declared_handle_is_shared_silently(self):
        box = _DeclaredBox()
        with warnings.catch_warnings():
            warnings.simplefilter('error')       # any warning fails the test
            clone = copy.deepcopy(box)
        assert clone.capsProblem is box.capsProblem

    def test_undeclared_handle_is_shared_with_lc_w311(self):
        box = _UndeclaredBox()
        with pytest.warns(RuntimeWarning, match='LC-W311') as rec:
            clone = copy.deepcopy(box)
        assert clone.capsProblem is box.capsProblem
        msg = str(rec[0].message)
        assert "'capsProblem'" in msg            # names the attribute
        assert 'reference_attributes' in msg     # names the fix

    def test_ordinary_attributes_are_still_copied(self):
        box = _DeclaredBox()
        box.settings = {'alpha': 1.0}
        clone = copy.deepcopy(box)
        assert clone.settings == box.settings
        assert clone.settings is not box.settings
        assert clone.inputs is not box.inputs

    def test_cache_is_reset_not_shared(self):
        box = _DeclaredBox()
        box._cache = {'stale': 'state'}
        clone = copy.deepcopy(box)
        assert clone._cache is None

    def test_declarations_accumulate_over_inheritance(self):
        class _Derived(_DeclaredBox):
            reference_attributes = ('session',)

            def __init__(self):
                super().__init__()
                self.session = _LiveHandle()

        box = _Derived()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            clone = copy.deepcopy(box)
        assert clone.capsProblem is box.capsProblem   # parent's declaration
        assert clone.session is box.session           # own declaration


@pytest.mark.skipif(not available, reason='LCsolver import failed')
def test_model_clone_carries_the_shared_handle():
    """The path this exists for: unit_corrector clones the formulation, and
    the clone's box must drive the SAME live analysis."""
    f = Formulation()
    x = f.Variable('x', 2.0, '-', 'in', bounds=[0.5, 4.0])
    y = f.Variable('y', 4.0, '-', 'out', bounds=[0.1, 20.0])
    f.Objective(y)
    box = _DeclaredBox()
    f.RuntimeConstraint([f.y], ['=='], [f.x], box)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        corrected = unit_corrector(f)
    handles = {id(getattr(b, 'capsProblem', None)) for b in _find_boxes(corrected)}
    assert handles == {id(box.capsProblem)}, \
        'every cloned box must share the one live handle'


def _find_boxes(model):
    """Every BlackBoxFunctionModel reachable from a cloned pyomo model."""
    from lcsolver.objects.blackBoxFunctionModel import BlackBoxFunctionModel
    from pyomo.contrib.pynumero.interfaces.external_grey_box import (
        ExternalGreyBoxBlock,
    )
    out = []
    for blk in model.component_data_objects(ExternalGreyBoxBlock,
                                            descend_into=True):
        ex = blk.get_external_model()
        if isinstance(ex, BlackBoxFunctionModel):
            out.append(ex)
    return out


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
