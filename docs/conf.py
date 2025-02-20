# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import os
import sys
import synthpops as sp
on_rtd = os.environ.get('READTHEDOCS') == 'True'

# -- Project information -----------------------------------------------------

project = 'SynthPops'
copyright = f'2020 - 2021, Institute for Disease Modeling. All rights reserved.\nThese docs were built for SynthPops version {sp.__version__}.\n'
author = 'Institute for Disease Modeling'

# The short X.Y version
version = sp.__version__

# The full version, including alpha/beta/rc tags
release = sp.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here
extensions = [
    "sphinx.ext.autodoc",  # Core Sphinx library for auto html doc generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc -- causes warnings with Napoleon however
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",  # Automatically document param types (less noise in class signature)
    "sphinx_design", # Add e.g. grid layout
    'sphinx_search.extension', # search across multiple docsets in domain
]

# Use Google docstrings
napoleon_google_docstring = True

# Configure autosummary
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autosummary_ignore_module_all = False # Respect __all__
autodoc_member_order = 'bysource' # Keep original ordering
add_module_names = False  # NB, does not work
autodoc_inherit_docstrings = False # Stops sublcasses from including docs from parent classes

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Syntax highlighting style
pygments_style = "sphinx"
modindex_common_prefix = ["synthpops."]

# List of patterns, relative to source directory, to exclude
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Suppress certain warnings
suppress_warnings = ['autosectionlabel.*']

# Add variable substitutions
rst_epilog = "\n.. include:: /variables.txt"


# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "show_prev_next": True,
    "icon_links": [
        {"name": "Web", "url": "https://synthpops.org", "icon": "fas fa-home"},
        {
            "name": "GitHub",
            "url": "https://github.com/synthpops/synthpops",
            "icon": "fab fa-github-square",
        },
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["navbar-side"],
    "header_links_before_dropdown": 5,
    "footer_start": ["copyright", "footer_start"],
    "footer_end": ["theme-version", "footer_end"],
}
html_sidebars = {
    "**": ["sidebar-nav-bs", "page-toc"],
}
html_logo = "images/IDM_white.png"
html_favicon = "images/favicon.ico"
html_static_path = ['_static']
html_context = {
    'rtd_url': 'https://docs.idmod.org/projects/synthpops/en/latest',
    "versions_dropdown": {
        "latest": "devel (latest)",
        "stable": "current (stable)",
    },
    "default_mode": "light",
}
# Add any extra paths that contain custom files
if not on_rtd:
    html_extra_path = ['robots.txt']


# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_last_updated_fmt = '%Y-%b-%d'
html_show_sourcelink = True
html_show_sphinx = False
html_copy_source = False
htmlhelp_basename = 'SynthPops'

# Add customizations
def setup(app):
    app.add_css_file("theme_overrides.css")

# OpenSearch options
html_use_opensearch = 'docs.idmod.org/projects/synthpops/en/latest'

# -- RTD Sphinx search for searching across the entire domain, default child -------------
if os.environ.get('READTHEDOCS') == 'True':

    search_project_parent = "institute-for-disease-modeling-idm"
    search_project = os.environ["READTHEDOCS_PROJECT"]
    search_version = os.environ["READTHEDOCS_VERSION"]

    rtd_sphinx_search_default_filter = f"subprojects:{search_project}/{search_version}"

    rtd_sphinx_search_filters = {
        "Search this project": f"project:{search_project}/{search_version}",
        "Search all IDM docs": f"subprojects:{search_project_parent}/{search_version}",
    }
