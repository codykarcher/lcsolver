# Changes made to the machine, not just the repo

Everything this branch needed that lives outside git, so it can be reviewed
or undone.

## Packages installed into the base conda env

| package | why |
|---|---|
| `edi` (editable) | you asked for it directly |
| `ad` | dependency of the `solar` reference models |
| `pypdf` | reading the papers |
| `openmdao` | required by pycycle |
| `pycycle` (editable) | third engine reference, at your suggestion |

`openmdao` pulled its own dependencies but did not change `numpy` (2.4.6),
`scipy` (1.17.1) or `pint` (0.25.3) — those are as they were.

## gpkit was broken and repaired

**This one is worth knowing about.** Partway through, the base env's `gpkit`
stopped working: `import gpkit` gave an empty namespace package with
`__file__ = None`, and `pip show gpkit` returned nothing. The directory was
left with subdirectories but no top-level modules and no dist-info — a
partial uninstall.

Cause: setting up the isolated reference env, I ran
`$GPKIT_REF_PYTHON -m pip install gpkit==1.1.0` **without** unsetting
`PYTHONPATH`. Your `.zshenv` puts the base env's `site-packages` on
`PYTHONPATH` explicitly, so pip saw the base installation through it and
uninstalled that copy while installing into the isolated env.

It has been repaired: the stale tree was removed and gpkit 1.1 reinstalled
from a source checkout. Verified working (`gpkit.__version__ == '1.1'`, solves
correctly), and the maidas suite is back to its previous **729 passed, 15
skipped**.

The lesson, which is now in `convexengineering/README.md`: with that
`PYTHONPATH` set, **every** pip command aimed at a non-base interpreter needs
`env -u PYTHONPATH`, not just the ones that read packages.

One pre-existing gpkit self-test (`t_sub.TestModelSubs.test_vector_sweep`)
fails on numpy 2.x with the ragged-array error. That is the same numpy
incompatibility that blocks the `solar`/`gassolar` models and is unrelated to
the reinstall.

## An isolated conda env was created

`gpkit-ref` — python 3.11, `numpy<2`, gpkit 1.1 from source, cvxopt, ad,
pandas, matplotlib. It exists to run the 8-year-old convexengineering models
for reference solutions without dragging the base env backwards. Delete with
`conda env remove -n gpkit-ref`; nothing in this repo needs it, because the
reference solutions are committed as JSON.

## Files placed outside the repo

`~/Dropbox/research/_reference/pycycle` — the pycycle checkout, installed
editable from there. It started in the session scratchpad, which is
temporary, so it was moved somewhere durable to keep the editable install
from breaking.

The other reference clones (gpkit, gplibrary, gpfit, SPaircraft, gassolar,
solar, jho, turbofan) are still in the session scratchpad and will disappear.
That is intentional — nothing depends on them, since every reference solution
is committed. Re-clone with the recipe in `convexengineering/README.md`.
