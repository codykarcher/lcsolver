Additional Tips
---------------

* Developers need the test and documentation extras. Both are declared in
  ``pyproject.toml``, so install them by name rather than one package at a
  time --- that way the list cannot drift from what the build actually uses:

::

   pip install -e ".[test,docs]"


* To build the documentation locally:

::

   cd docs
   python -m sphinx . _build/html

then open ``docs/_build/html/index.html``. Add ``-W`` to turn warnings into
errors, which is how the documentation is expected to build:

::

   python -m sphinx -W . _build/html


* Unit tests and coverage can be run from the repository root:

::

   pytest --cov-report term-missing --cov=lcsolver -v ./tests/

or generating html output:

::

   pytest --cov-report html --cov=lcsolver -v ./tests/

By default this skips the tests marked ``slow`` and ``veryslow``, which are the
cross-path and full-model comparisons. To run everything, as continuous
integration does:

::

   pytest -m "" ./tests/
