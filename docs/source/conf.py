# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import json

import numpy as np

import pyparrm
from pyparrm._utils._docs import linkcode_resolve


project = "PyPARRM"
copyright = "2023, Thomas Samuel Binns"
author = "Thomas Samuel Binns"
release = "1.1.0dev"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, os.path.abspath("../../"))

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "nbsphinx",
    "nbsphinx_link",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
    "versionwarning.extension",
]

bibtex_bibfiles = ["refs.bib"]

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Version warning ---------------------------------------------------------
versions = json.load(
    "https://pyparrm.readthedocs.io/en/main/_static/versions.json"
)
version_names = np.sort([ver["name"] for ver in versions]).tolist()
warning_messages = {}
if "dev" in version_names[-1]:
    stable_name = version_names[-2]
    warning_messages[version_names[-1]] = (
        "You are not reading the documentation for the stable version of this "
        f"project. {stable_name} is the stable version."
    )
else:
    stable_name = version_names[-1]

warning_messages = {}
for version_name in version_names:
    if (
        version_name != stable_name
        and version_name not in warning_messages.keys()
    ):
        warning_messages[version_name] = (
            "You are not reading the documentation for the latest stable "
            f"version of this project. {version_name} is the latest stable "
            "version."
        )

versionwarning_messages = warning_messages

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/neuromodulation/pyparrm",
            icon="fa-brands fa-square-github",
        )
    ],
    "icon_links_label": "External Links",  # for screen reader
    "use_edit_page_button": False,
    "navigation_with_keys": False,
    "show_toc_level": 1,
    "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "switcher": {
        "json_url": "https://pyparrm.readthedocs.io/en/main/_static/versions.json",  # noqa E501
        "version_match": pyparrm.__version__,
    },
    "pygment_light_style": "default",
    "pygment_dark_style": "github-dark",
}

autodoc_member_order = "bysource"

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}


# NumPyDoc configuration -----------------------------------------------------

# Define what extra methods numpydoc will document
numpydoc_class_members_toctree = True
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    # Python
    "bool": ":class:`python:bool`",
    # Matplotlib
    "Axes": "matplotlib.axes.Axes",
    "Figure": "matplotlib.figure.Figure",
}
numpydoc_xref_ignore = {
    # words
    "instance",
    "instances",
    "of",
    "default",
    "shape",
    "or",
    "with",
    "length",
    "pair",
    "pyplot",
    "matplotlib",
    "optional",
    "kwargs",
    "in",
    "dtype",
    "object",
    # shapes
    "epochs",
    "channels",
    "rank",
    "times",
    "components",
    "frequencies",
    "x",
    "n_vertices",
    "n_faces",
    "n_channels",
    "m",
    "n",
    "n_events",
    "n_colors",
    "n_times",
    "obj",
    "n_chan",
    "n_epochs",
    "n_picks",
    "n_ch_groups",
    "n_dipoles",
    "n_ica_components",
    "n_pos",
    "n_node_names",
    "n_tapers",
    "n_signals",
    "n_step",
    "n_freqs",
    "wsize",
    "Tx",
    "M",
    "N",
    "p",
    "q",
    "r",
    "n_observations",
    "n_regressors",
    "n_cols",
    "n_frequencies",
    "n_tests",
    "n_samples",
    "n_permutations",
    "nchan",
    "n_points",
    "n_features",
    "n_parts",
    "n_features_new",
    "n_components",
    "n_labels",
    "n_events_in",
    "n_splits",
    "n_scores",
    "n_outputs",
    "n_trials",
    "n_estimators",
    "n_tasks",
    "nd_features",
    "n_classes",
    "n_targets",
    "n_slices",
    "n_hpi",
    "n_fids",
    "n_elp",
    "n_pts",
    "n_tris",
    "n_nodes",
    "n_nonzero",
    "n_events_out",
    "n_segments",
    "n_orient_inv",
    "n_orient_fwd",
    "n_orient",
    "n_dipoles_lcmv",
    "n_dipoles_fwd",
    "n_picks_ref",
    "n_coords",
    "n_meg",
    "n_good_meg",
    "n_moments",
    "n_patterns",
    "n_new_events",
}
