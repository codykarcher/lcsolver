# Contributing to EDI

Thanks for your interest in EDI. This document describes how to report problems,
ask questions, and contribute changes.

## Reporting a bug

Open an issue on the GitHub tracker and include:

- what you expected to happen and what happened instead;
- a minimal example that reproduces the problem (ideally a short `Formulation`);
- your operating system, Python version, and the output of
  `python -c "import edi, pyomo; print(edi.__version__, pyomo.version.version)"`.

## Asking a question

Questions are welcome as GitHub issues. If your question is about how to express a
particular design problem in EDI, please include the model you have so far — it is
much easier to advise on concrete code.

## Contributing code

1. Fork the repository and create a branch for your change.
2. Add or update tests. The suite lives in `tests/` and runs with
   `python -m pytest tests/`. Pull requests that change behaviour should include a
   test that fails before the change and passes after it.
3. Keep the existing code style.
4. Make sure the full suite passes locally before opening the pull request; CI runs
   the same suite on Linux, macOS, and Windows across several Python versions.
5. Open a pull request describing what the change does and why.

## Development install

```bash
git clone <your fork>
cd edi
python -m pip install -e ".[test,docs]"
python -m pytest tests/
```

Note that `mpi4py` is an optional extra (`.[parallel]`) and requires a working MPI
installation. It is not needed to run the test suite.

## Building the documentation

```bash
python -m pip install -e ".[docs]"
cd docs && python -m sphinx . _build/html
```

## Scope

EDI aims to stay a thin, readable layer over Pyomo. Contributions that add
engineering-design conveniences — units handling, black-box interfaces, structure
detection, solver routing — are in scope. Contributions that duplicate functionality
better provided by Pyomo itself generally are not.

## Code of conduct

Participation in this project is governed by the expectation that contributors treat
one another with respect. Harassment or abusive behaviour is not tolerated.
