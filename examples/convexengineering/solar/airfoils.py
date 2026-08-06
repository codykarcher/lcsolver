"""Airfoil drag polar fits used by the solar aircraft.

Transcribed from the CSVs in the source trees. Both are gpfit output; see
``environment.fit_constraints`` for how the ``ftype`` field maps to
constraints.

``DAI1336A`` — the solar aircraft's wing section, fitted in three variables
(CL, Re, tau). This is what makes profile drag respond to Reynolds number
and thickness, and therefore what ties flight speed to power draw. Without
it the drag/power chain is broken and flight speed floats free.

``NACA0008`` — the tail section, fitted in two variables (Re, tau).
Max-affine with K=5, so it contributes five separate constraints.
"""
from __future__ import annotations

# solar/dai1336a.csv — SMA, K=3, d=3, variables (CL, Re, tau)
DAI1336A = dict(
    ftype="SMA", K=3, d=3, a1=6.915002404481357,
    c=[1.2593774746863988e-25, 0.018880654420562477, 1.3297088748648847e+29],
    e=[[92.28892127816388, -1.6834034485029763, -7.874453097710683],
       [8.27187422140475, -2.147236193543017, 0.7109890431440831],
       [-3.9654883145180073, -7.057351525640577, 4.481052921876957]],
    rms_err=0.05630048611723882,
    bounds=dict(CL=(0.9465, 1.5012), Re=(1.25e5, 6.0e5), tau=(0.1094, 0.146)),
)

# gplibrary tail/tail_dragfit.csv — MA, K=5, d=2, variables (Re, tau)
NACA0008 = dict(
    ftype="MA", K=5, d=2, a1=1.0,
    c=[0.3399377561914537, 5.446864658024941, 16.259467895292175,
       9.509193806494268, 218.73655005090146],
    e=[[-0.18199062784771394, 0.7746039331652241],
       [-0.4848666193196949, 0.2463415216530887],
       [-0.5399432024831557, 0.4663844545784079],
       [-0.4823469575592283, 0.4674663891252669],
       [-0.6038708953785882, 1.312443752168466]],
    rms_err=0.040701846058073914,
)

__all__ = ["DAI1336A", "NACA0008"]
