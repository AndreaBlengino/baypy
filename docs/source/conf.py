# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import subprocess

sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'baypy'
copyright = '2023, Andrea Blengino'
author = 'Andrea Blengino'
release = subprocess.run(['git', 'describe', '--tags'], stdout = subprocess.PIPE).stdout.decode('utf-8').split('-')[0]


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'm2r2', 'sphinx.ext.intersphinx']

templates_path = ['_templates']
exclude_patterns = []

intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'pandas': ('https://pandas.pydata.org/docs/', None)}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']
add_module_names = False
html_title = 'baypy'
