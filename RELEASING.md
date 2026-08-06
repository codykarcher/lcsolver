# Releasing LCsolver to PyPI

Everything mechanical is already wired: `.github/workflows/release.yml`
builds the sdist and wheel and publishes via PyPI Trusted Publishing (OIDC --
no API tokens exist anywhere). What remains is one-time account setup and,
per release, one tag.

## One-time setup (browser, ~10 minutes)

1. Create an account at https://pypi.org/account/register/ and enable 2FA
   (mandatory). Do the same at https://test.pypi.org (separate account).
2. On **pypi.org**: *Account settings -> Publishing -> Add a pending
   publisher*:
   - PyPI project name: `lcsolver`
   - Owner: `codykarcher`   Repository: `lcsolver`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. On **test.pypi.org**, add the same pending publisher with environment
   name `testpypi`.
4. The repository must be **public** when a publish runs (Trusted Publishing
   verifies the repo, and pip users need the page anyway).

## Rehearsal (recommended before the first real release)

Actions -> release -> *Run workflow* (leave target = testpypi). When it goes
green, check https://test.pypi.org/project/lcsolver/ renders correctly and

    pip install --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/ lcsolver

installs and imports.

## Each release

1. Bump `version` in `pyproject.toml` (uploads are immutable -- every fix is
   a new version). Commit and push.
2. Tag and publish a GitHub Release:

       git tag v0.1.0
       git push origin v0.1.0
       gh release create v0.1.0 --title "v0.1.0" --generate-notes

   Publishing the Release is what triggers the PyPI upload.
3. Watch the `release` workflow; when green, verify
   https://pypi.org/project/lcsolver/.

## After the first release

- Update the install instructions in `README.md` and `docs/index.rst` from
  the git URL to `pip install lcsolver`.
- The README is the PyPI project page. Its images already use absolute
  `raw.githubusercontent.com` URLs so they render there; keep any future
  images absolute too.
