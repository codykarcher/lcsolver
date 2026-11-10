#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Blocks that assemble themselves, and refuse to be half-assembled.

A `SubModel` is attached by assignment (one string, one spelling) and its
LAST input builds it. The failure worth testing: a block whose last input
never arrives posts nothing, and the formulation stays solvable answering
an easier question. So `solve` refuses while one is attached, listing every
block and every missing input at once.
"""
import pytest

from lcsolver import Formulation, SubModel, units
from lcsolver.presolve.reductions import unbuilt_blocks, unbuilt_blocks_check
from lcsolver.solvers.solver import PresolveError, solve


class Widget(SubModel):
    """A block with one setting and two inputs: a variable and a constant."""

    model_inputs = ('weight_in', 'floor')
    provides = {'headroom': 'what is left under the cap, as an expression'}

    def build(self):
        f = self.formulation
        n = f.retype_to_float(self.settings.get('n_parts', 2))
        parts = [self.Variable(f'part_{i}', 1.0, 'kg', f'part {i}')
                 for i in range(int(n))]
        total = self.Variable('total', 1.0, 'kg', 'total mass')
        self.ConstraintList(
            [total >= f.scalar_sum(parts + [self.weight_in])]
            + [p >= self.floor for p in parts])
        self.headroom = f.scalar_sum(parts)


def _attached(n_parts=2):
    f = Formulation()
    f.widget = Widget(n_parts=n_parts)
    return f


# ---------------------------------------------------------------- attachment
def test_assignment_hands_the_block_its_formulation_and_its_name():
    f = _attached()
    assert f.widget.formulation is f
    assert f.widget.name == 'widget'


def test_the_attribute_name_is_the_component_prefix():
    f = _attached()
    f.widget.weight_in = f.Variable(name='W', guess=1.0, units='kg',
                                    description='W')
    f.widget.floor = 1.0 * units.kg
    # one name, three places: attribute path, component name, deck key
    assert f.widget.total is f.widget_total
    assert f.widget_total.name == 'widget_total'


def test_a_block_may_still_be_handed_its_formulation_the_old_way():
    f = Formulation()
    w = Widget(f, 'widget', n_parts=2)
    assert w.formulation is f and w.name == 'widget'


# --------------------------------------------------------------- build timing
def test_the_last_input_is_what_builds_the_block():
    f = _attached()
    assert not f.widget.is_built()
    assert f.widget.pending_inputs() == ['floor', 'weight_in']

    f.widget.weight_in = f.Variable(name='W', guess=1.0, units='kg',
                                    description='W')
    assert not f.widget.is_built(), 'one input still missing'
    assert f.widget.pending_inputs() == ['floor']

    f.widget.floor = 1.0 * units.kg
    assert f.widget.is_built()
    assert f.widget.rows, 'building posts the rows'


def test_a_setting_changes_which_rows_exist():
    f = _attached(n_parts=5)
    f.widget.weight_in = f.Variable(name='W', guess=1.0, units='kg',
                                    description='W')
    f.widget.floor = 1.0 * units.kg
    assert len(f.widget.variables) == 6      # five parts and the total


def test_the_block_solves_to_the_sum_it_declared():
    f = _attached(n_parts=2)
    W = f.Variable(name='W', guess=1.0, units='kg', description='W')
    f.widget.weight_in = W
    f.widget.floor = 3.0 * units.kg
    f.ConstraintList([W >= 2.0 * units.kg])
    f.Objective(f.widget.total)
    solve(f, sensitivities=False)
    assert f.solution.objective == pytest.approx(8.0)   # 3 + 3 + 2


# ------------------------------------------------- the half-assembled model
def test_an_unbuilt_block_is_found_with_what_it_waits_for():
    f = _attached()
    f.widget.floor = 1.0 * units.kg
    assert unbuilt_blocks(f) == [('widget', ['weight_in'])]


def test_solve_refuses_a_formulation_with_an_unbuilt_block():
    f = _attached()
    z = f.Variable(name='z', guess=1.0, units='kg', description='z')
    f.ConstraintList([z >= 1.0 * units.kg])
    f.Objective(z)
    with pytest.raises(PresolveError) as e:
        solve(f, sensitivities=False)
    assert 'LC-E003' in str(e.value)
    assert 'weight_in' in str(e.value)


def test_every_block_and_every_missing_input_is_reported_at_once():
    f = Formulation()
    f.first = Widget(n_parts=2)
    f.second = Widget(n_parts=2)
    f.second.floor = 1.0 * units.kg
    with pytest.raises(PresolveError) as e:
        unbuilt_blocks_check(f)
    text = str(e.value)
    assert 'first' in text and 'second' in text
    assert text.count('weight_in') == 2      # not just the first failure
    assert 'floor' in text                   # only the one that is missing it


def test_a_built_block_is_not_reported():
    f = _attached()
    f.widget.weight_in = f.Variable(name='W', guess=1.0, units='kg',
                                    description='W')
    f.widget.floor = 1.0 * units.kg
    assert unbuilt_blocks(f) == []
    unbuilt_blocks_check(f)


# ------------------------------------------------------ scalar_sum semantics
def test_scalar_sum_refuses_a_vector():
    f = Formulation()
    a = f.Variable(name='a', guess=1.0, units='kg', description='a')
    v = f.Variable(name='v', guess=1.0, units='kg', size=3, description='v')
    with pytest.raises(TypeError, match='vector'):
        f.scalar_sum([a, v])


def test_scalar_sum_seeds_on_the_first_part_so_units_are_checked():
    f = Formulation()
    a = f.Variable(name='a', guess=1.0, units='kg', description='a')
    b = f.Variable(name='b', guess=1.0, units='kg', description='b')
    expr = f.scalar_sum([a, b])
    assert str(units.get_units(expr)) == str(units.get_units(a))
    assert f.scalar_sum([]) == 0


def test_scalar_sum_reaches_through_a_group():
    f = Formulation()
    g = f.group('g')
    a = g.Variable('a', 1.0, 'kg', 'a')
    b = g.Variable('b', 1.0, 'kg', 'b')
    assert g.scalar_sum([a, b]) is not None


# ------------------------------------------------- retype_to_float semantics
def test_retype_to_float_reads_a_constant_and_a_number():
    f = Formulation()
    c = f.Constant(name='c', value=2.5, units='-', description='c')
    assert f.retype_to_float(c) == 2.5
    assert f.retype_to_float(3) == 3.0
    assert isinstance(f.retype_to_float(3), float)
    assert f.group('g').retype_to_float(c) == 2.5


# ------------------------------------------------------------- the status report
def _connected(n_parts=2):
    f = _attached(n_parts=n_parts)
    f.widget.weight_in = f.Variable(name='W', guess=1.0, units='kg',
                                    description='gross weight')
    f.widget.floor = f.Constant(name='floor', value=1.0, units='kg',
                                description='minimum part mass')
    return f


def test_status_lists_the_model_inputs_in_one_section():
    text = _connected().widget.status_text()
    assert 'Model inputs (2 of 2 connected)' in text
    # each says what it is actually wired to, by the assembly's name for it
    assert 'weight_in' in text and 'W' in text
    assert 'floor' in text


def test_status_calls_out_an_input_that_is_not_connected():
    f = _attached()
    f.widget.floor = 1.0 * units.kg
    text = f.widget.status_text()
    assert 'NOT BUILT' in text and 'It has posted nothing.' in text
    assert 'weight_in' in text and 'NOT CONNECTED' in text
    assert 'Model inputs (1 of 2 connected)' in text


def test_status_lists_what_the_block_declares():
    text = _connected(n_parts=3).widget.status_text()
    assert 'Variables declared here (4)' in text     # three parts and the total
    assert 'widget_total' in text
    assert 'total mass' in text                      # the description, too


def test_status_says_nothing_is_declared_yet_before_the_block_builds():
    text = _attached().widget.status_text()
    assert 'Variables declared here: none yet, the block has not built' in text


def test_status_reports_what_the_block_provides_and_how_to_use_it():
    text = _connected().widget.status_text()
    assert 'Also provides' in text
    assert 'headroom' in text
    assert 'what is left under the cap' in text
    # an expression reports as one, not under its Pyomo operator name
    assert '<expression>' in text


def test_status_names_a_block_that_was_never_attached():
    text = Widget(n_parts=2).status_text()
    assert 'NOT ATTACHED' in text


def test_status_shows_the_settings_that_chose_the_rows():
    text = _connected(n_parts=3).widget.status_text()
    assert 'n_parts' in text and '3' in text


def test_get_status_prints_the_same_report(capsys):
    f = _connected()
    f.widget.get_status()
    assert capsys.readouterr().out.strip() == f.widget.status_text().strip()


def test_required_inputs_is_model_inputs_in_order():
    assert Widget.required_inputs() == ['weight_in', 'floor']


def test_the_old_input_lists_still_build_a_block_and_warn():
    with pytest.warns(DeprecationWarning, match='model_inputs'):
        class Old(SubModel):
            input_variables = ('a',)
            input_constants = ('b',)
            inputs = ('c',)
            # this fixture exercises the GATING, not a real model, so its
            # build() deliberately ignores the inputs the staleness check
            # wants used
            _allow_unused_inputs = ('a', 'b', 'c')

            def build(self):
                self.Variable('x', 1.0, 'kg', 'x')

    assert Old.required_inputs() == ['a', 'b', 'c']
    f = Formulation()
    f.old = Old()
    assert 'Model inputs (0 of 3 connected)' in f.old.status_text()
    f.old.a, f.old.b = 1.0, 2.0
    assert not f.old.is_built()
    f.old.c = 3.0
    assert f.old.is_built()


def test_model_inputs_alone_does_not_warn():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')

        class New(SubModel):
            model_inputs = ('a',)
            _allow_unused_inputs = ('a',)

            def build(self):
                pass


# ------------------------------------------------- all the inputs, or none
def test_every_input_may_be_passed_to_the_constructor():
    f = Formulation()
    W = f.Variable(name='W', guess=1.0, units='kg', description='W')
    f.widget = Widget(n_parts=2, weight_in=W, floor=1.0 * units.kg)
    assert f.widget.is_built()
    assert f.widget.settings == {'n_parts': 2}, 'inputs are not settings'


def test_a_partial_constructor_call_is_refused():
    f = Formulation()
    W = f.Variable(name='W', guess=1.0, units='kg', description='W')
    with pytest.raises(TypeError) as e:
        f.widget = Widget(n_parts=2, weight_in=W)
    text = str(e.value)
    assert 'was given 1 of its 2 inputs' in text
    assert 'floor' in text                   # the missing one is named


def test_a_partial_call_names_every_missing_input():
    class Big(SubModel):
        model_inputs = ('a', 'b', 'c', 'd', 'e')
        # the point of this fixture is the constructor's error message; it
        # never gets far enough to build, so nothing uses the inputs
        _allow_unused_inputs = ('a', 'b', 'c', 'd', 'e')

        def build(self):
            pass

    with pytest.raises(TypeError) as e:
        Big(a=1.0)
    text = str(e.value)
    for name in ('b', 'c', 'd', 'e'):
        assert f' {name}' in text
    assert '4 input(s) are missing' in text


def test_passing_no_inputs_still_defers():
    f = _attached()
    assert not f.widget.is_built()
    assert f.widget.pending_inputs() == ['floor', 'weight_in']


def test_the_constructor_form_solves_to_the_same_answer():
    f = Formulation()
    W = f.Variable(name='W', guess=1.0, units='kg', description='W')
    f.widget = Widget(n_parts=2, weight_in=W, floor=3.0 * units.kg)
    f.ConstraintList([W >= 2.0 * units.kg])
    f.Objective(f.widget.total)
    solve(f, sensitivities=False)
    assert f.solution.objective == pytest.approx(8.0)


# ------------------------------------------- model_inputs: one list for all
class Product(SubModel):
    """y >= a * b, where a and b are whatever the assembly hands in."""

    model_inputs = ('a', 'b')

    def build(self):
        m = self.bind_inputs()
        self.y = self.Variable('y', 1.0, 'm^2', 'bound on the product')
        self.Constraint(self.y >= m.a * m.b)


def _product(variable_slot):
    """x (a variable, held at 2 m by a row) and c (a constant, 3 m) wired
    into Product with the variable in ``variable_slot``."""
    f = Formulation()
    x = f.Variable(name='x', guess=1.0, units='m', description='x')
    c = f.Constant(name='c', value=3.0, units='m', description='c')
    f.product = Product()
    constant_slot = 'b' if variable_slot == 'a' else 'a'
    setattr(f.product, variable_slot, x)
    setattr(f.product, constant_slot, c)
    f.ConstraintList([x >= 2.0 * units.m])
    f.Objective(f.product.y)
    return f


@pytest.mark.parametrize('variable_slot', ['a', 'b'])
def test_a_model_input_takes_a_variable_or_a_constant(variable_slot):
    """The block declares names, not kinds: either slot takes either."""
    f = _product(variable_slot)
    assert f.product.is_built()
    solve(f, sensitivities=False)
    assert f.solution.objective == pytest.approx(6.0, rel=1e-6)


def test_status_lists_model_inputs_in_declared_order_by_what_is_wired():
    text = _product('b').product.status_text()          # a <- c, b <- x
    assert 'Model inputs (2 of 2 connected)' in text
    lines = text.splitlines()
    row_a = next(i for i, line in enumerate(lines) if line.startswith('   a '))
    row_b = next(i for i, line in enumerate(lines) if line.startswith('   b '))
    assert row_b == row_a + 1, 'one section, declared order'
    assert '<- c [m]' in lines[row_a]                    # the constant
    assert '<- x [m]' in lines[row_b]                    # the variable
    # the headings the split used to print are gone
    for heading in ('Input variables', 'Input constants', 'kind not declared'):
        assert heading not in text


def test_bind_inputs_hands_back_every_model_input_read_only():
    f = _product('a')
    m = f.product.bind_inputs()
    assert m.a is f.x and m.b is f.c
    assert dir(m) == ['a', 'b']
    with pytest.raises(AttributeError, match='read-only'):
        m.a = f.c
    with pytest.raises(AttributeError, match="'z' is not a declared input"):
        m.z


def test_a_model_input_no_build_uses_is_refused_when_the_class_is_defined():
    with pytest.raises(TypeError) as e:
        class Stale(SubModel):
            model_inputs = ('a', 'never_read')

            def build(self):
                self.Constraint(self.a >= 1.0)
    text = str(e.value)
    assert "['never_read']" in text
    assert 'model_inputs' in text                        # says where to fix it


def test_a_subclass_inherits_and_may_extend_model_inputs():
    class Inherits(Product):
        pass

    class Extends(Product):
        model_inputs = Product.model_inputs + ('cap',)

        def build(self):
            super().build()
            self.Constraint(self.y <= self.cap)

    assert Inherits.required_inputs() == ['a', 'b']
    assert Extends.required_inputs() == ['a', 'b', 'cap']

    f = Formulation()
    f.extends = Extends()
    f.extends.a = 2.0 * units.m
    f.extends.b = 3.0 * units.m
    assert not f.extends.is_built(), 'still waiting on the input it added'
    assert f.extends.pending_inputs() == ['cap']
    f.extends.cap = 10.0 * units.m ** 2
    assert f.extends.is_built() and len(f.extends.rows) == 2


def test_model_inputs_and_an_old_list_merge_without_repeating_a_name():
    with pytest.warns(DeprecationWarning, match='input_constants'):
        class Mixed(SubModel):
            model_inputs = ('a', 'b')
            input_constants = ('b', 'c')         # b is named in both
            _allow_unused_inputs = ('a', 'b', 'c')

            def build(self):
                pass

    assert Mixed.required_inputs() == ['a', 'b', 'c']    # model_inputs first
    f = Formulation()
    f.mixed = Mixed()
    assert 'Model inputs (0 of 3 connected)' in f.mixed.status_text()


def test_inheriting_an_old_list_does_not_warn_a_second_time():
    import warnings

    with pytest.warns(DeprecationWarning, match='model_inputs'):
        class OldParent(SubModel):
            input_variables = ('a',)
            _allow_unused_inputs = ('a',)

            def build(self):
                pass

    # the child declares nothing of its own, so it has nothing to migrate
    with warnings.catch_warnings():
        warnings.simplefilter('error')

        class Child(OldParent):
            pass

    assert Child.required_inputs() == ['a']


def test_the_old_list_warning_names_the_class_and_every_old_list():
    with pytest.warns(DeprecationWarning) as caught:
        class Legacy(SubModel):
            input_variables = ('a',)
            inputs = ('b',)
            _allow_unused_inputs = ('a', 'b')

            def build(self):
                pass

    text = str(caught[0].message)
    assert 'Legacy' in text
    assert 'input_variables' in text and 'inputs' in text
    assert 'input_constants' not in text             # not declared, not named
    assert caught[0].filename == __file__, 'points at the class, not lcsolver'
