"""Inject examples/boyd.py into README.md.

GitHub markdown has no include mechanism, so the README's worked example is
spliced in between marker comments by this script, and a test
(tests/test_readme_sync.py) fails whenever the two drift. Run after editing
the example:

    python utilities/sync_readme.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, 'README.md')
EXAMPLE = os.path.join(ROOT, 'examples', 'boyd.py')

BEGIN = '<!-- BEGIN readme_example -->'
END = '<!-- END readme_example -->'


def build_block():
    code = open(EXAMPLE).read().rstrip('\n')
    return (f'{BEGIN}\n'
            '<!-- generated from examples/boyd.py -- edit that '
            'file and run `python utilities/sync_readme.py` -->\n'
            f'```python\n{code}\n```\n'
            f'{END}')


def sync(check_only=False):
    readme = open(README).read()
    pattern = re.compile(re.escape(BEGIN) + '.*?' + re.escape(END), re.S)
    if not pattern.search(readme):
        raise SystemExit(f'README.md has no {BEGIN} ... {END} markers')
    updated = pattern.sub(lambda _m: build_block(), readme)
    if check_only:
        return updated == readme
    if updated != readme:
        open(README, 'w').write(updated)
        print('README.md updated from examples/boyd.py')
    else:
        print('README.md already in sync')
    return True


if __name__ == '__main__':
    if '--check' in sys.argv:
        sys.exit(0 if sync(check_only=True) else 1)
    sync()
