# pycycle truth harness

Offline pycycle truth harness for the SP-native rubber engine.

This package plays the role for the engine that TASOPT 2.16 played for the
airframe: an independent, higher-fidelity implementation of the same physics,
run OFFLINE to produce station-by-station anchor data that the SP cycle rows
are validated against. Nothing in the aircraft model imports from here, and
nothing here imports from the aircraft model -- the two sides meet only in
the JSON files under ``truth/data/`` and in the validation scripts that
compare them.

Contents
--------
``hbtf.py``        The pycycle high-bypass turbofan cycle (two-spool,
                   separate-flow), adapted from pycycle's own
                   ``example_cycles/high_bypass_turbofan.py`` and
                   parameterized by an ``EngineSpec``.
``engines.py``     The two anchor engines: a CFM56-class (737-800) and a
                   GEnx-class (787), each with design values, ratings,
                   off-design points and public reference figures with
                   provenance.
``run_anchors.py`` Builds and runs each anchor at its design point plus four
                   off-design points, prints the comparison against the
                   reference figures, and dumps full station tables to
                   ``data/<engine>.json``.

Requires the pycycle install at ~/Dropbox/research/_reference/pycycle
(om-pycycle 4.4.1-dev, openmdao 3.45, CEA thermo).
