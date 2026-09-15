#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Vector and matrix quantities that behave the way the maths is written.

Pyomo's indexed components are dicts keyed by index, so x >= y, x[-1] and
x * 2 all raised, and sum(x) silently summed the KEYS (0 + 1 + 2 ...) --
a real model asserted L_dist_sum == 10 that way. Three things live here:
Index/IndexTuple are the keys iteration yields, ints/tuples that refuse
REFLECTED addition (sum()'s first step) while forward i + 1 / i - 1 stay
untouched (pinned by tests/test_vector_language_guarantee.py);
VectorArray is a numpy array of the component's own VarData/ParamData
objects with comparisons building constraints elementwise; VectorComponent
is mixed into the Pyomo component so slicing, negative indexing and
comparisons work directly. Shapes never broadcast silently -- a mismatch
raises and names broadcast_rows / broadcast_cols, because numpy expanding
a length-3 vector across the rows of a 2x3 matrix is a wrong answer
whenever the author meant columns.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Index", "IndexTuple", "VectorArray", "VectorComponent",
           "ShapeMismatch", "as_array", "broadcast_rows", "broadcast_cols"]


_SUM_MESSAGE = (
    "sum(x) over an LCsolver vector would add its index keys (0 + 1 + 2 ...), not "
    "its elements, because iterating an indexed component yields keys. Use "
    "f.sum(x), or sum(x.values()), or sum(x[i] for i in x)."
)


class Index(int):
    """An index key that refuses to be summed. sum() starts from 0, so
    0 + key is a reflected add and __radd__ fires; key + 1 / key - 1 are
    forward ops on int's own implementations, so index arithmetic is
    unaffected."""

    __slots__ = ()

    def __radd__(self, other):
        raise TypeError(_SUM_MESSAGE)


class IndexTuple(tuple):
    """Same guard for a multi-dimensional key. sum() over these already
    failed, but with an unhelpful 'int' + 'tuple' TypeError."""

    __slots__ = ()

    def __radd__(self, other):
        raise TypeError(_SUM_MESSAGE)


def _wrap_key(key):
    if isinstance(key, tuple):
        return IndexTuple(key)
    if isinstance(key, int):
        return Index(key)
    return key


class ShapeMismatch(ValueError):
    """Two quantities in one expression do not have compatible shapes."""


def _shape_of(obj):
    if isinstance(obj, VectorComponent):
        return obj.shape
    if isinstance(obj, np.ndarray):
        return obj.shape
    return ()


def _is_scalarish(obj):
    """A single quantity, which may be combined with any shape."""
    return _shape_of(obj) == ()


class VectorArray(np.ndarray):
    """A numpy array of Pyomo objects that compares elementwise. Numpy's
    comparison ufuncs coerce to bool, which a Pyomo relational expression
    refuses (its truth isn't known until solve), so the operators collect
    the expressions instead."""

    def __array_finalize__(self, obj):
        pass

    # -- comparisons ------------------------------------------------------
    def _compare(self, other, op, symbol):
        left = np.asarray(self, dtype=object)
        if isinstance(other, VectorComponent):
            other = other.as_array()
        if _is_scalarish(other):
            right = np.empty(left.shape, dtype=object)
            right[...] = other
        else:
            right = np.asarray(other, dtype=object)
            if right.shape != left.shape:
                raise ShapeMismatch(
                    f"cannot compare shapes {left.shape} and {right.shape} "
                    f"with '{symbol}'. LCsolver does not broadcast silently: a "
                    "length-n vector is a column, and numpy would expand it "
                    "across rows just as readily, which is a different model. "
                    "Say which you mean with f.broadcast_rows(v, n) or "
                    "f.broadcast_cols(v, n).")
        out = np.empty(left.shape, dtype=object)
        for idx in np.ndindex(left.shape):
            out[idx] = op(left[idx], right[idx])
        return out.view(VectorArray)

    def __ge__(self, other):
        return self._compare(other, lambda a, b: a >= b, '>=')

    def __le__(self, other):
        return self._compare(other, lambda a, b: a <= b, '<=')

    def __eq__(self, other):
        return self._compare(other, lambda a, b: a == b, '==')

    def __ne__(self, other):
        raise TypeError(
            "'!=' does not describe a constraint. Use '==', '<=' or '>='.")

    def __lt__(self, other):
        raise TypeError(
            "'<' is a strict inequality, which an optimizer cannot enforce "
            "(there is no smallest number below a bound). Use '<='.")

    def __gt__(self, other):
        raise TypeError(
            "'>' is a strict inequality, which an optimizer cannot enforce "
            "(there is no largest number below a bound). Use '>='.")

    # Overriding __eq__ removes the inherited hash; these are containers of
    # model objects and are never used as keys, but leaving them unhashable
    # produces a confusing failure far from here.
    __hash__ = object.__hash__


def as_array(obj):
    """``obj`` as a :class:`VectorArray`, whatever kind of vector it is."""
    if isinstance(obj, VectorComponent):
        return obj.as_array()
    if isinstance(obj, np.ndarray):
        return obj.view(VectorArray)
    return np.asarray(obj, dtype=object).view(VectorArray)


def broadcast_rows(vector, n):
    """``vector`` repeated as each of ``n`` rows, giving ``(n, len(vector))``.

    For comparing a per-column quantity against a matrix::

        M  is (n, m)      cap is (m,)
        M <= f.broadcast_rows(cap, n)     # every row obeys the same caps

    A scalar is accepted as a length-1 vector: "the same value everywhere"
    is unambiguous, unlike the vector-shape guessing refused below.
    """
    arr = np.atleast_1d(as_array(vector)).view(VectorArray)
    if arr.ndim != 1:
        raise ShapeMismatch(
            f"broadcast_rows expects a scalar or 1-D quantity, got shape {arr.shape}")
    out = np.empty((int(n), arr.shape[0]), dtype=object)
    for i in range(int(n)):
        out[i, :] = arr
    return out.view(VectorArray)


def broadcast_cols(vector, n):
    """``vector`` repeated as each of ``n`` columns, giving ``(len(vector), n)``.

    For comparing a per-row quantity against a matrix::

        M  is (n, m)      cap is (n,)
        M <= f.broadcast_cols(cap, m)     # every column obeys the same caps

    A scalar is accepted as a length-1 vector: "the same value everywhere"
    is unambiguous, unlike the vector-shape guessing refused below.
    """
    arr = np.atleast_1d(as_array(vector)).view(VectorArray)
    if arr.ndim != 1:
        raise ShapeMismatch(
            f"broadcast_cols expects a scalar or 1-D quantity, got shape {arr.shape}")
    out = np.empty((arr.shape[0], int(n)), dtype=object)
    for j in range(int(n)):
        out[:, j] = arr
    return out.view(VectorArray)


def _checked(left, right, op):
    """Apply op elementwise, refusing to broadcast between shapes. numpy's
    error says nothing about the model, and when lengths do line up it
    quietly answers a different question -- same reason the comparisons
    refuse it."""
    ls = left.shape if isinstance(left, np.ndarray) else ()
    rs = right.shape if isinstance(right, np.ndarray) else ()
    if ls and rs and ls != rs:
        raise ShapeMismatch(
            f"cannot combine shapes {ls} and {rs}. A vector of one length is "
            "not a vector of another: slice one of them to match "
            f"(x[:{min(ls[0], rs[0])}]) if that is what was meant, or use "
            "f.broadcast_rows / f.broadcast_cols to say which axis lines up.")
    return op(left, right)


def _rhs(other):
    """The other operand as numpy can use it: a vector becomes its array,
    a single quantity is left alone so numpy applies it everywhere."""
    if isinstance(other, VectorComponent):
        return other.as_array()
    return other


class VectorComponent:
    """Mixed into an indexed Pyomo component to make it read like a vector.
    Only __iter__, __getitem__ and the comparisons are touched; everything
    Pyomo does goes through keys/values/items and is untouched, which is
    why a model carrying these still solves identically."""

    # set when the component is declared; (n,) or (n, m) and so on
    _edi_shape = None

    @property
    def shape(self):
        if self._edi_shape is not None:
            return tuple(self._edi_shape)
        return (len(self),)

    def __iter__(self):
        return iter(_wrap_key(k) for k in self.keys())

    def as_array(self):
        """The elements, in index order, as a VectorArray. They are the
        component's own data objects, so a constraint built from the array
        is a constraint on the declared variables."""
        shape = self.shape
        out = np.empty(shape, dtype=object)
        if len(shape) == 1:
            for n, k in enumerate(sorted(self.keys())):
                out[n] = self[k]
        else:
            for k in self.keys():
                out[tuple(k)] = self[k]
        return out.view(VectorArray)

    # -- indexing ---------------------------------------------------------
    def _is_array_style(self, key):
        if isinstance(key, (slice, np.ndarray, list)):
            return True
        if isinstance(key, int) and key < 0:
            return True
        if isinstance(key, tuple):
            return any(isinstance(k, (slice, np.ndarray, list)) for k in key)
        return False

    def __getitem__(self, key):
        # slices, lists and negative ints go through the array; plain keys
        # go to Pyomo as always
        if self._is_array_style(key):
            return self.as_array()[key]
        return super().__getitem__(key)

    # -- comparisons ------------------------------------------------------
    def __ge__(self, other):
        return self.as_array() >= other

    def __le__(self, other):
        return self.as_array() <= other

    def __eq__(self, other):
        return self.as_array() == other

    def __ne__(self, other):
        raise TypeError(
            "'!=' does not describe a constraint. Use '==', '<=' or '>='.")

    def __lt__(self, other):
        raise TypeError(
            "'<' is a strict inequality, which an optimizer cannot enforce. "
            "Use '<='.")

    def __gt__(self, other):
        raise TypeError(
            "'>' is a strict inequality, which an optimizer cannot enforce. "
            "Use '>='.")

    # -- arithmetic -------------------------------------------------------
    # Elementwise, via the array. Without these a vector could be compared
    # but not used: `V == M * a` failed, forcing `for i in range(N)` loops.
    def _arith(self, other, op):
        return _checked(self.as_array(), _rhs(other), op)

    def __mul__(self, other):
        return self._arith(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return _rhs(other) * self.as_array()

    def __add__(self, other):
        return self._arith(other, lambda a, b: a + b)

    def __radd__(self, other):
        return _rhs(other) + self.as_array()

    def __sub__(self, other):
        return self._arith(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return _rhs(other) - self.as_array()

    def __truediv__(self, other):
        return self._arith(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return _rhs(other) / self.as_array()

    def __pow__(self, other):
        return self._arith(other, lambda a, b: a ** b)

    def __rpow__(self, other):
        return _rhs(other) ** self.as_array()

    def __neg__(self):
        return -self.as_array()

    # Pyomo keeps components in ComponentMap/ComponentSet, which hash by
    # identity. Defining __eq__ above would otherwise drop __hash__ to None and
    # break them in places nowhere near this file.
    __hash__ = object.__hash__
