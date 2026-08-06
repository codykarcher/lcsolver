#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""The README's worked example must be examples/readme_example.py, verbatim.

GitHub markdown cannot include a file, so the block is spliced in by
utilities/sync_readme.py between marker comments. This test is what makes
that a single source of truth rather than a copy that drifts.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_readme_example_is_in_sync():
    sys.path.insert(0, os.path.join(ROOT, 'utilities'))
    try:
        from sync_readme import sync
    finally:
        sys.path.pop(0)
    assert sync(check_only=True), (
        'README.md has drifted from examples/readme_example.py; run '
        '`python utilities/sync_readme.py` to regenerate the block')
