#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Everything that happens after a solve has produced a point.

Write-back of the solution onto the model, sensitivity recovery from the
duals, the holographic-constraint check, and (re-exported from
``lcsolver.presolve.reductions``, where the machinery lives) the post-solve
quality checks.
"""

from lcsolver.postsolve.writeback import write_solution  # noqa: F401
from lcsolver.postsolve.sensitivity import sensitivities  # noqa: F401
from lcsolver.postsolve.holographic import (holographic_report,  # noqa: F401
                                            format_holographic)


def postsolve_check(*args, **kwargs):
    """See :func:`lcsolver.presolve.reductions.postsolve_check`."""
    from lcsolver.presolve.reductions import postsolve_check as _impl
    return _impl(*args, **kwargs)
