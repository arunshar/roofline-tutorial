"""Sphinx settings for "The Roofline Model".

Build from the course root with:  sphinx-build -W -b html docs docs/_build/html
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "The Roofline Model"
author_name = "Arun Sharma"
author_url = "https://arunshar.com"
author = f"{author_name} and Claude Code"
copyright = f"2026, {author_name}"
release = "0.1"

# These three values name the GitHub repository behind the site. The Colab, Binder and
# "Edit on GitHub" links read them. Each lecture's setup cell holds its own copy of the
# user and repository names, so update both places if the repository moves.
github_user = "arunshar"
github_repo = "roofline-tutorial"
github_version = "main"

extensions = [
    "myst_nb",             # Markdown pages and executed notebooks
    "sphinx_copybutton",   # copy button on code blocks
    "sphinx.ext.mathjax",  # $...$ and $$...$$ math
    "quiz_directive",      # the {quiz} directive in docs/_ext
]

# labs/build_notebooks.py executes the lectures, and the notebooks are committed with
# their outputs. The docs build therefore never executes a notebook.
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "dollarmath", "deflist", "attrs_inline"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints", ".DS_Store", "Thumbs.db"]

html_theme = "sphinx_rtd_theme"
html_title = project
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "style_external_links": True,
}
html_context = {
    "display_github": True,
    "github_user": github_user,
    "github_repo": github_repo,
    "github_version": github_version,
    "conf_py_path": "/docs/",
}
