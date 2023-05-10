"""Tests for features of PyPARRM."""

from multiprocessing import cpu_count

import numpy as np
import pytest

from pyparrm import PARRM
from pyparrm._utils._power import compute_psd


random = np.random.RandomState(44)
data = random.rand((2, 100))  # 2 channels, 100 timepoints
sampling_freq = 20
artefact_freq = 10


def test_parrm(indices: tuple[int], verbose: bool, n_jobs: int):
    """Test that PARRM can run."""
    parrm = PARRM(
        data=data[:, indices],
        sampling_freq=sampling_freq,
        artefact_freq=artefact_freq,
        verbose=verbose,
    )
    parrm.find_period(n_jobs=n_jobs)
    parrm.create_filter()
    filtered_data = parrm.filter_data()

    assert filtered_data.shape == data.shape
    assert isinstance(filtered_data, np.ndarray)

    if n_jobs == -1:
        assert parrm._n_jobs == cpu_count()


def test_parrm_attrs():
    """Test that attributes returned from PARRM are correct.

    The returned attributes should simply be a copy of their private
    counterparts.
    """
    parrm = PARRM(
        data=data[0],
        sampling_freq=sampling_freq,
        artefact_freq=artefact_freq,
        verbose=False,
    )
    parrm.find_period()
    parrm.create_filter()
    filtered_data = parrm.filter_data()

    assert np.all(filtered_data == parrm._filtered_data)
    assert np.all(filtered_data == parrm.filtered_data)

    assert np.all(data[0] == parrm._data)
    assert np.all(data[0] == parrm.data)

    assert parrm._period == parrm.period
    assert np.all(parrm._filter == parrm.filter)

    settings = parrm.settings
    assert settings["data"]["sampling_freq"] == sampling_freq
    assert settings["data"]["artefact_freq"] == artefact_freq
    assert settings["period"]["search_samples"] == parrm._search_samples
    assert settings["period"]["assumed_periods"] == parrm._assumed_periods
    assert settings["period"]["outlier_boundary"] == parrm._outlier_boundary
    assert settings["period"]["random_seed"] == parrm._random_seed

    assert settings["filter"]["filter_half_width"] == parrm._filter_half_width
    assert settings["filter"]["omit_n_samples"] == parrm._omit_n_samples
    assert settings["filter"]["filter_direction"] == parrm._filter_direction
    assert settings["filter"]["period_half_width"] == parrm._period_half_width


def test_parrm_wrong_type_inputs():
    """Test that inputs of wrong types to PARRM are caught."""
    # init object
    with pytest.raises:
        PARRM(
            data=data[0].tolist(),
            sampling_freq=sampling_freq,
            artefact_freq=artefact_freq,
        )
    with pytest.raises:
        PARRM(
            data=data[0],
            sampling_freq=str(sampling_freq),
            artefact_freq=artefact_freq,
        )
    with pytest.raises:
        PARRM(
            data=data[0],
            sampling_freq=sampling_freq,
            artefact_freq=str(artefact_freq),
        )
    with pytest.raises:
        PARRM(
            data=data[0],
            sampling_freq=sampling_freq,
            artefact_freq=artefact_freq,
            verbose=str(False),
        )
    parrm = PARRM(
        data=data[0],
        sampling_freq=sampling_freq,
        artefact_freq=artefact_freq,
    )

    # find_period
    with pytest.raises:
        parrm.find_period(search_samples=0)
    with pytest.raises:
        parrm.find_period(assumed_periods=[0])
    with pytest.raises:
        parrm.find_period(outlier_boundary=[0])
    with pytest.raises:
        parrm.find_period(random_seed=1.5)
    with pytest.raises:
        parrm.find_period(n_jobs=1.5)
    parrm.find_period()

    # explore_filter_params
    with pytest.raises:
        parrm.explore_filter_params(freq_res=[0])
    with pytest.raises:
        parrm.explore_filter_params(n_jobs=1.5)

    # create_filter
    with pytest.raises:
        parrm.create_filter(filter_half_width=1.5)
    with pytest.raises:
        parrm.create_filter(omit_n_samples=1.5)
    with pytest.raises:
        parrm.create_filter(filter_direction=0)
    with pytest.raises:
        parrm.create_filter(period_half_width=[0])


def test_parrm_wrong_value_inputs():
    """Test that inputs of wrong values to PARRM are caught."""
    # init object
    with pytest.raises:
        PARRM(
            data=random.rand((1, 1, 1)),
            sampling_freq=sampling_freq,
            artefact_freq=artefact_freq,
        )
    with pytest.raises:
        PARRM(
            data=data[0],
            sampling_freq=0,
            artefact_freq=artefact_freq,
        )
    with pytest.raises:
        PARRM(
            data=data[0],
            sampling_freq=sampling_freq,
            artefact_freq=0,
        )
    parrm = PARRM(
        data=data[0],
        sampling_freq=sampling_freq,
        artefact_freq=artefact_freq,
    )

    with pytest.raises:
        parrm.find_period(search_samples=np.zeros((1, 1)))
    with pytest.raises:
        parrm.find_period(search_samples=np.array([-1, 1]))
    with pytest.raises:
        parrm.find_period(search_samples=np.array([0, data.shape[1] + 1]))


def test_parrm_premature_method_calls():
    """Test that errors are raised for PARRM methods called prematurely."""
    parrm = PARRM(
        data=data[0],
        sampling_freq=sampling_freq,
        artefact_freq=artefact_freq,
        verbose=False,
    )

    with pytest.raises:
        parrm.create_filter()

    with pytest.raises:
        parrm.filter_data()


def test_parrm_missing_filter_inputs():
    """Test that PARRM can compute values for missing filter inputs."""
    parrm = PARRM(
        data=data[0],
        sampling_freq=sampling_freq,
        artefact_freq=artefact_freq,
        verbose=False,
    )
    parrm.find_period()
    parrm.create_filter()

    assert parrm._filter_half_width is not None
    assert parrm._period_half_width is not None


def test_compute_psd(indices: tuple[int], n_jobs: int):
    """Test that PSD computation runs."""
    n_freqs = 5
    psd = compute_psd(
        data=data[:, indices],
        sampling_freq=sampling_freq,
        n_freqs=n_freqs,
        n_jobs=n_jobs,
    )

    assert psd.shape == (len(indices), n_freqs)
    assert isinstance(psd, np.ndarray)
