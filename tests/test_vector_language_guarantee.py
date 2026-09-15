#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""The Python behaviour LCsolver's vector guard depends on.

The guard yields index keys as an int subclass whose ``__radd__`` refuses,
so ``sum(x)`` (a reflected add) fails while ``i + 1`` (forward) works. That
split relies on Python trying a proper subclass's reflected method FIRST;
if the rule ever changed, ``sum(x)`` would quietly return the index sum
again. This file pins the rule itself, so the failure would say why.
"""

import pyomo.common.unittest as unittest


class _Refuses(int):
    """An int subclass that refuses reflected addition."""

    def __radd__(self, other):
        raise TypeError('reflected addition refused')


class _Plain(int):
    """An int subclass that overrides nothing."""


class TestReflectedOperatorPriority(unittest.TestCase):

    def test_reflected_add_wins_for_a_subclass(self):
        """`0 + subclass_instance` must call the subclass's __radd__.

        This is the whole mechanism. Without it the guard never fires.
        """
        with self.assertRaises(TypeError):
            0 + _Refuses(1)

    def test_forward_add_is_untouched(self):
        """`instance + 1` must use int's __add__, so index arithmetic works."""
        self.assertEqual(_Refuses(1) + 1, 2)
        self.assertEqual(_Refuses(3) - 1, 2)
        self.assertEqual(_Refuses(2) * 2, 4)

    def test_sum_goes_through_the_reflected_path(self):
        """`sum()` starts from 0, so the first addition is reflected."""
        with self.assertRaises(TypeError):
            sum([_Refuses(0), _Refuses(1), _Refuses(2)])

    def test_augmented_addition_is_caught_too(self):
        """`total += i` against an int accumulator takes the same path."""
        with self.assertRaises(TypeError):
            total = 0
            for i in (_Refuses(0), _Refuses(1)):
                total += i

    def test_a_subclass_that_overrides_nothing_still_sums(self):
        """The priority rule needs the override; inheritance alone is not it."""
        self.assertEqual(sum([_Plain(1), _Plain(2)]), 3)

    def test_the_subclass_is_still_an_int_everywhere_else(self):
        """It has to remain usable as a dictionary key and an index."""
        i = _Refuses(2)
        self.assertEqual(i, 2)
        self.assertEqual(hash(i), hash(2))
        self.assertEqual({0: 'a', 1: 'b', 2: 'c'}[i], 'c')
        self.assertEqual(['a', 'b', 'c'][i], 'c')
        self.assertEqual(sorted([_Refuses(2), _Refuses(0)]), [0, 2])

    def test_the_rule_does_not_apply_between_unrelated_types(self):
        """Stated for contrast: priority comes from the subclass
        relationship; a plain object gets the reflected call only after
        int's __add__ returns NotImplemented."""

        class Unrelated:
            def __radd__(self, other):
                return 'reflected'

        self.assertEqual(0 + Unrelated(), 'reflected')


if __name__ == '__main__':
    unittest.main()
