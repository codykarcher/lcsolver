#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Rows without a namespace: the other kind of reusable piece.

A `SubModel` owns things.  It has a name, a group, a prefix on every deck key,
a list of inputs it is waiting on, and a build that fires when the last one
lands.  That is the right shape for a piece of the problem that declares its
own quantities -- a rotor, a weight statement, a mission.

Some reusable pieces declare nothing.  A section drag polar is the example:
handed a lift coefficient, a thickness, a Reynolds number and a drag
coefficient the CALLER already declared, it states the relation between them
and has no quantities of its own to name.  Wrapping that in a SubModel gives
it a namespace it never uses and an input list it never pends on, and then
every walk of the formulation has to special-case it.

So it is its own type.  The rule that separates the two is exactly one line:

    A ConstraintGenerator declares no variables and no constants.
    If it needs to declare something, it is a SubModel.

which is what makes the distinction hold up when something walks a
formulation: a generator has no name to report, no deck keys to collect, and
nothing to be un-built.

A generator is attached like any other choice and reads as one::

    f.polar = Hoburg24xxPolar()

and callers drop its rows into their own list::

    self.ConstraintList([
        cl * m.solidity == 6. * CT,
        f.polar.generate_rows(cl, m.tau_blade, m.Re_reference, cd),
        ])

The return goes in WHOLE, not spliced: `ConstraintList` flattens a list of
rows in place, so a generator returning one row and a generator returning five
read identically at the call site.  That is deliberate -- how many rows a
generator needs is its business, and a caller that had to know would be
coupled to the very thing this type exists to let it ignore.

WHAT A GENERATOR MAY RETURN is anything `ConstraintList` accepts, which is
what makes the type worth having: the same call site takes an algebraic fit
and a black box without knowing which it got.  A posynomial polar returns one
comparison; a polar wrapping an external analysis returns
``[cd, '==', [cl, tau, Re], TheBox()]``.  Neither the block nor the assembly
changes.

VALIDITY ENVELOPES go in `generate_holographic_rows`, because ``holographic``
is a per-call flag on `ConstraintList` and a single flat list cannot say which
of its entries is a guard.  A fit that is only checked over part of the space
can state that range there, and every solve then reports whether the answer
leaned on it -- see `Formulation.HolographicConstraint`.
"""


class ConstraintGenerator:
    """Rows relating quantities the caller owns.  Declares nothing itself."""

    def generate_rows(self, *args, **kwargs):
        """The rows, as a list, to be dropped into the caller's list.

        Entries may be anything `Formulation.ConstraintList` accepts:
        algebraic comparisons, numpy arrays of them, or a black box given as
        ``[outputs, operators, inputs, box]``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement generate_rows()")

    def generate_holographic_rows(self, *args, **kwargs):
        """Rows that must hold but must not bind -- the range this generator
        is good over.  Empty unless the generator has one to state."""
        return []

    def __repr__(self):
        return f'<{type(self).__name__}, a ConstraintGenerator>'
