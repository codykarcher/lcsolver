# Sphinx configuration for the LCsolver documentation.
#
# Build with:
#     python -m pip install -e ".[docs]"
#     cd docs && python -m sphinx . _build/html
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "LCsolver"
copyright = "2023-2026, Cody J. Karcher, Michael L. Bynum, and NTESS"
author = "Cody J. Karcher"

try:
    from lcsolver import __version__ as release
except Exception:
    release = "0.1.0"
version = release

extensions = [
    # The technical note (PRESOLVE.md) is Markdown. Without this it is not
    # read at all -- 25 kB of documentation that existed only as a file in
    # the repo.
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pyomo": ("https://pyomo.readthedocs.io/en/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The .rst files were written for the Pyomo documentation tree, where they sat
# under contributed_packages/lcsolver/. They are reused verbatim here.
master_doc = "index"

html_theme = "sphinx_rtd_theme"
html_static_path = []

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
