#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Rows without a namespace: the other kind of reusable piece.

The rule vs. SubModel: a ConstraintGenerator declares no variables and no
constants. If it needs to declare something, it is a SubModel. A generator
has no name, no deck keys, and nothing to un-build.

Attach like any other choice; drop the return in WHOLE (ConstraintList
flattens, so row count is the generator's business):

    f.polar = Hoburg24xxPolar()
    self.ConstraintList([
        cl * m.solidity == 6. * CT,
        f.polar.generate_rows(cl, m.tau_blade, m.Re_reference, cd),
        ])

May return anything ConstraintList accepts: an algebraic fit or a black box
like [cd, '==', [cl, tau, Re], TheBox()]. Validity envelopes go in
generate_holographic_rows; see Formulation.HolographicConstraint.
"""


class ConstraintGenerator:
    """Rows relating quantities the caller owns.  Declares nothing itself."""

    def generate_rows(self, *args, **kwargs):
        """The rows, as a list, to be dropped into the caller's list.
        Entries may be anything ConstraintList accepts."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement generate_rows()")

    def generate_holographic_rows(self, *args, **kwargs):
        """Rows that must hold but must not bind: the range this generator is good over."""
        return []

    def __repr__(self):
        return f'<{type(self).__name__}, a ConstraintGenerator>'
