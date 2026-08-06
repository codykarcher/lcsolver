Additional Tips
---------------

* Developers may need to install the following additional packages:

::

   pip install pytest
   pip install pytest-cov
   pip install sphinx
   pip install sphinx_rtd_theme
   pip install sphinx_copybutton


* If you wish to build the documentation locally, use:

::

   cd <path_to_edi>/docs
   make html

then open the file ``docs/_build/html/index.html``


* Unit tests and coverage can be run locally using:

::

   cd <path_to_edi>
   pytest --cov-report term-missing --cov=lcsolver -v ./tests/

or generating html output:

::

   cd <path_to_edi>
   pytest --cov-report html --cov=lcsolver -v ./tests/
