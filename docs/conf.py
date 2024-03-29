# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'CorrHOD'
copyright = '2023, Simon Bouchard'
author = 'Simon Bouchard'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', # For automatically documenting the functions
    'sphinx.ext.napoleon', # For documenting the parameters of the functions
    'sphinx.ext.intersphinx', # For linking to other packages' documentation
    'sphinx.ext.viewcode', # For linking to the source code
    'sphinx.ext.linkcode', # For linking to external codes
    'sphinx.ext.autosectionlabel', # For automatically labelling the sections
    'myst_nb', # For including jupyter notebooks
]

myst_enable_extensions = ["dollarmath"]
source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
}

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Autodoc configuration ---------------------------------------------------
# Mock imports that can't be resolved during documentation build
autodoc_mock_imports = ['cosmoprimo', 'mockfactory', 'pycorr', 'Corrfunc', 'pyrecon']

napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_preserve_defaults = True # Keep the default values of the parameters instead of replacing them with their values
autoclass_content = 'both' # Include both the class docstring and the __init__ docstring in the documentation
autodoc_member_order = 'bysource' # Order the members by the order in the source code

nb_execution_mode = 'off' # Do not execute the notebooks when building the documentation

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = 'sphinx_rtd_theme' # For readthedocs.io
html_theme = 'sphinx_book_theme'

html_title = 'CorrHOD'
html_logo = 'images/CorrHOD_logo.svg'
html_favicon = 'images/CorrHOD_favicon.png'
html_show_sourcelink = False # Remove the "view source" link
html_theme_options = {
    "repository_url": "https://github.com/SBouchard01/CorrHOD",
    "repository_branch": "main",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "use_fullscreen_button": False,
    "logo": {
      "image_light": "images/CorrHOD_logo.svg",
      "image_dark": "images/CorrHOD_logo_dark.svg", # Change logo when dark mode is activated
    },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return f"https://github.com/SBouchard01/CorrHOD/blob/main/{filename}.py"