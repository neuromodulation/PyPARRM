"""Tests for utility features of PyPARRM."""

# Author(s):
#   Thomas Samuel Binns | github.com/tsbinns

import os

import numpy as np
import pytest

from pyparrm import get_example_data_paths
from pyparrm.data import DATASETS
from pyparrm._utils._power import compute_psd

sampling_freq = 20  # Hz
artefact_freq = 10  # Hz


@pytest.mark.parametrize("n_chans", [1, 2])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_compute_psd(n_chans: int, n_jobs: int) -> None:
    """Test that PSD computation runs."""
    random = np.random.default_rng(44)
    data = random.standard_normal((n_chans, 100))

    n_freqs = 5
    freqs, psd = compute_psd(
        data=data, sampling_freq=sampling_freq, n_points=n_freqs * 2, n_jobs=n_jobs
    )

    assert psd.shape == (n_chans, n_freqs)
    assert freqs.shape[0] == psd.shape[1]
    assert isinstance(freqs, np.ndarray)
    assert isinstance(psd, np.ndarray)

    max_freq = (sampling_freq / 2) - ((sampling_freq / 2) / n_freqs)
    freqs, psd = compute_psd(
        data=data,
        sampling_freq=sampling_freq,
        n_points=n_freqs * 2,
        max_freq=max_freq,
        n_jobs=n_jobs,
    )

    assert freqs.shape[0] == psd.shape[1]
    assert freqs[-1] == max_freq


def test_get_example_data_paths() -> None:
    """Test `get_example_data_paths`."""
    # test it works with correct inputs
    for name, file in DATASETS.items():
        path = get_example_data_paths(name=name)
        assert isinstance(path, str), "`path` should be a str."
        assert path.endswith(file), "`path` should end with the name of the dataset."
        assert os.path.exists(path), "`path` should point to an existing file."

    # test it catches incorrect inputs
    with pytest.raises(ValueError, match="`name` must be one of"):
        get_example_data_paths(name="not_a_name")
