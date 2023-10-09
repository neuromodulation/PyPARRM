"""Tools for fetching example data."""

import os
from pathlib import Path

DATASETS = {}
DATASETS["example_data"] = "example_data.npy"
DATASETS["example_data_artefact_free"] = "example_data_artefact_free.npy"
DATASETS["matlab_filtered"] = "matlab_filtered.npy"
DATASETS["ecog_lfp_data"] = "ecog_lfp_data.npy"


def get_example_data_paths(name: str) -> str:
    """Return the path to the requested example data.

    Parameters
    ----------
    name : str
        Name of the example data.

    Returns
    -------
    path : str
        Path to the example data.
    """
    if name not in DATASETS.keys():
        raise ValueError(f"`name` must be one of: {list(DATASETS.keys())}")

    filepath_upper = Path(os.path.abspath(__file__)).parent
    return os.path.join(filepath_upper, "example_data", DATASETS[name])
