"""Make ``tasopt_py`` importable however pytest is invoked.

The port is a self-contained package under ``examples/tasopt`` rather than an
installed distribution, so running the suite from the repository root would
otherwise fail to collect these tests with ``ModuleNotFoundError``. Adding
this directory to ``sys.path`` makes both entry points work:

    pytest                          # from the edi repo root
    cd examples/tasopt && pytest    # from here
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
