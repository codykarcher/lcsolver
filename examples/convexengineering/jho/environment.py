"""Atmosphere and wind data for the Jungle Hawk Owl mission.

Extracted from ``gassolar/environment`` so the EDI model is self-contained.
Latitude 38, 90th-percentile winds, day 355 (winter solstice), which is what
``gas.py`` uses by default.

The climb profile is ``np.linspace(0, 15000, 11)[1:]`` — ten segments from
1500 ft to 15000 ft — and the loiter sits at 15000 ft. Wind speed rises with
altitude (19 -> 33 m/s), which is what makes the high-altitude loiter
expensive: the aircraft must fly at least as fast as the wind to hold
station.
"""
from __future__ import annotations

LATITUDE, PERCENT, DAY = 38, 90, 355

CLIMB = dict(
    altitude_ft=[1500.0, 3000.0, 4500.0, 6000.0, 7500.0,
                 9000.0, 10500.0, 12000.0, 13500.0, 15000.0],
    Vwind_ms=[18.969, 20.572, 21.297, 22.036, 23.177,
              24.703, 26.479, 28.393, 30.426, 32.555],
    rho=[1.1721, 1.1210, 1.0716, 1.0239, 0.9778,
         0.9333, 0.8904, 0.8490, 0.8091, 0.7707],
    mu=[1.7861e-05, 1.7743e-05, 1.7625e-05, 1.7506e-05, 1.7387e-05,
        1.7267e-05, 1.7147e-05, 1.7026e-05, 1.6905e-05, 1.6420e-05],
)

LOITER = dict(altitude_ft=15000.0, Vwind_ms=32.5553,
              rho=0.77071, mu=1.642e-05)

# --- engine fits (gplibrary GP/aircraft/engine) ---------------------------
# power_lawfit.csv — engine weight vs sea-level max shaft power
POWER_LAW = dict(ftype="MA", K=1, d=1, a1=1.0,
                 c=[1.2784683664089935], e=[[0.7723915754684576]],
                 rms_err=0.34279129123926194)

# powerBSFCfit.csv — BSFC penalty at part power
BSFC_FIT = dict(ftype="SMA", K=2, d=1, a1=18.556942322046762,
                c=[0.008663209784767449, 1.3862822190984452],
                e=[[-7.701873956939406], [1.1292113161811266]],
                rms_err=0.006999823062023476)

__all__ = ["LATITUDE", "PERCENT", "DAY", "CLIMB", "LOITER",
           "POWER_LAW", "BSFC_FIT"]

# --- airfoil / component drag fits ----------------------------------------
# gplibrary wing/jho_fitdata.csv — the JHO wing section, in (CL, Re)
JHO_POLAR = dict(ftype="SMA", K=4, d=2, a1=3.0904265782626554,
                 c=[1.0237722536072613e-07, 0.0016819467416519377,
                    2.2831557182654378e-06, 175843713.87100798],
                 e=[[18.8561449416522, -0.1833195640493263],
                    [2.963071814236627, -0.667476893160359],
                    [-1.653325906904553, -0.3496409845113647],
                    [-0.17866429593832706, -2.713007500787293]],
                 rms_err=0.006577663039320368)

# gplibrary tail/tail_dragfit.csv — NACA 0008, in (Re, tau)
NACA0008 = dict(ftype="MA", K=5, d=2, a1=1.0,
                c=[0.3399377561914537, 5.446864658024941, 16.259467895292175,
                   9.509193806494268, 218.73655005090146],
                e=[[-0.18199062784771394, 0.7746039331652241],
                   [-0.4848666193196949, 0.2463415216530887],
                   [-0.5399432024831557, 0.4663844545784079],
                   [-0.4823469575592283, 0.4674663891252669],
                   [-0.6038708953785882, 1.312443752168466]],
                rms_err=0.040701846058073914)

__all__ += ["JHO_POLAR", "NACA0008"]
