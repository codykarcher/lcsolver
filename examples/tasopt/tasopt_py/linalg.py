"""``gaussn.f`` -- Gaussian elimination with partial pivoting.

A literal port. TASOPT solves every dense system it has with this routine, and
this port uses it wherever the arithmetic path matters: :mod:`tasopt_py.aero`'s
``blax`` runs a *limited* Newton (twenty passes, step-limited, stopping on step
size), so the sequence of iterates -- and therefore the answer it stops at --
depends on the elimination, not only on the matrix. Substituting a library
solve with different pivoting perturbs the iterates, and a capped iteration
does not iron that out the way a converged one would.

The pivot swap is written in an unusual way. It stores ``Z(NX,NP) = Z(NP,NP)``
and never writes ``Z(NP,NP)``: the elimination below reads ``Z(K,NP)`` as the
multiplier for row ``K``, and row ``NX`` needs the *old* diagonal entry there,
so the single assignment is both halves of the swap. Reproduced as written.

Indices are 1-based, as everywhere in this port -- ``z`` is an ``(nn+1)``
square list of lists with row and column 0 unused, ``r`` an ``(nn+1)`` list.
Both are overwritten; ``r`` holds the solution on return.
"""
from __future__ import annotations

__all__ = ["gaussn"]


def gaussn(nn: int, z: list, r: list) -> None:
    """Solve ``z x = r`` in place. Assumes the system is invertible; if it is
    not, a division by zero results, exactly as the original warns."""
    for np in range(1, nn):
        np1 = np + 1

        # Find the max pivot index nx.
        nx = np
        for n in range(np1, nn + 1):
            if abs(z[n][np]) > abs(z[nx][np]):
                nx = n

        pivot = 1.0 / z[nx][np]

        # Switch pivots -- see the module docstring for why this is enough.
        z[nx][np] = z[np][np]

        # Switch rows and normalise the pivot row.
        row_np, row_nx = z[np], z[nx]
        for l in range(np1, nn + 1):
            temp = row_nx[l] * pivot
            row_nx[l] = row_np[l]
            row_np[l] = temp

        temp = r[nx] * pivot
        r[nx] = r[np]
        r[np] = temp

        # Forward eliminate everything.
        for k in range(np1, nn + 1):
            row_k = z[k]
            ztmp = row_k[np]
            if ztmp == 0.0:
                continue
            for l in range(np1, nn + 1):
                row_k[l] -= ztmp * row_np[l]
            r[k] -= ztmp * r[np]

    r[nn] = r[nn] / z[nn][nn]

    # Back substitute everything.
    for np in range(nn - 1, 0, -1):
        row_np = z[np]
        acc = r[np]
        for k in range(np + 1, nn + 1):
            acc -= row_np[k] * r[k]
        r[np] = acc
