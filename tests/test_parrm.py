"""Tests for features of PyPARRM."""

# Author(s):
#   Thomas Samuel Binns | github.com/tsbinns

import os
from multiprocessing import cpu_count

import numpy as np
import pytest

from pyparrm import get_example_data_paths, PARRM
from pyparrm.data import DATASETS
from pyparrm._utils._power import compute_psd


random = np.random.RandomState(44)
sampling_freq = 20  # Hz
artefact_freq = 10  # Hz


@pytest.mark.parametrize("n_chans", [1, 2])
@pytest.mark.parametrize("n_samples", [100, 300])
@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("n_jobs", [1, -1])
def test_parrm(n_chans: int, n_samples: int, verbose: bool, n_jobs: int):
    """Test that PARRM can run."""
    data = random.rand(n_chans, n_samples)

    parrm = PARRM(
        data=data,
        sampling_freq=sampling_freq,
        artefact_freq=artefact_freq,
        verbose=verbose,
    )
    parrm.find_period(
        assumed_periods=sampling_freq / artefact_freq,
        random_seed=44,
        n_jobs=n_jobs,
    )
    for direction in ["future", "past", "both"]:
        parrm.create_filter(filter_direction=direction)
    filtered_data = parrm.filter_data()

    assert filtered_data.shape == data.shape
    assert isinstance(filtered_data, np.ndarray)

    other_data = random.rand(1, 50)
    other_filtered_data = parrm.filter_data(other_data)
    assert other_filtered_data.shape == other_data.shape

    assert repr(parrm) == (
        f"PARRM object | Data: ({n_chans} channels x {n_samples} times) | "
        f"Period: {parrm.period :.4f}"
    )

    if n_jobs == -1:
        assert parrm._n_jobs == cpu_count()


def test_parrm_attrs():
    """Test that attributes returned from PARRM are correct.

    The returned attributes should simply be a copy of their private
    counterparts.
    """
    data = random.rand(1, 100)

    parrm = PARRM(
        data=data,
        sampling_freq=sampling_freq,
        artefact_freq=artefact_freq,
        verbose=False,
    )
    parrm.find_period()
    parrm.create_filter()
    filtered_data = parrm.filter_data()

    assert np.all(filtered_data == parrm._filtered_data)
    assert np.all(filtered_data == parrm.filtered_data)

    assert np.all(data == parrm._data)
    assert np.all(data == parrm.data)

    assert parrm._period == parrm.period
    assert np.all(parrm._filter == parrm.filter)

    settings = parrm.settings
    assert settings["data"]["sampling_freq"] == sampling_freq
    assert settings["data"]["artefact_freq"] == artefact_freq
    assert np.all(
        settings["period"]["search_samples"] == parrm._search_samples
    )
    assert settings["period"]["assumed_periods"] == parrm._assumed_periods
    assert settings["period"]["outlier_boundary"] == parrm._outlier_boundary
    assert settings["period"]["random_seed"] == parrm._random_seed
    assert settings["filter"]["filter_half_width"] == parrm._filter_half_width
    assert settings["filter"]["omit_n_samples"] == parrm._omit_n_samples
    assert settings["filter"]["filter_direction"] == parrm._filter_direction
    assert settings["filter"]["period_half_width"] == parrm._period_half_width


def test_parrm_wrong_type_inputs():
    """Test that inputs of wrong types to PARRM are caught."""
    data = random.rand(1, 100)

    # init object
    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        PARRM(
            data=data.tolist(),
            sampling_freq=sampling_freq,
            artefact_freq=artefact_freq,
        )
    with pytest.raises(
        TypeError, match="`sampling_freq` must be an int or a float."
    ):
        PARRM(
            data=data,
            sampling_freq=str(sampling_freq),
            artefact_freq=artefact_freq,
        )
    with pytest.raises(
        TypeError, match="`artefact_freq` must be an int or a float."
    ):
        PARRM(
            data=data,
            sampling_freq=sampling_freq,
            artefact_freq=str(artefact_freq),
        )
    with pytest.raises(TypeError, match="`verbose` must be a bool."):
        PARRM(
            data=data,
            sampling_freq=sampling_freq,
            artefact_freq=artefact_freq,
            verbose=str(False),
        )
    parrm = PARRM(
        data=data,
        sampling_freq=sampling_freq,
        artefact_freq=artefact_freq,
    )

    # find_period
    with pytest.raises(
        TypeError, match="`search_samples` must be a NumPy array or None."
    ):
        parrm.find_period(search_samples=0)
    with pytest.raises(
        TypeError,
        match="`assumed_periods` must be an int, a float, a tuple, or None.",
    ):
        parrm.find_period(assumed_periods=[0])
    with pytest.raises(
        TypeError,
        match="If a tuple, entries of `assumed_periods` must be ints or ",
    ):
        parrm.find_period(assumed_periods=tuple(["test"]))
    with pytest.raises(
        TypeError, match="`outlier_boundary` must be an int or a float."
    ):
        parrm.find_period(outlier_boundary=[0])
    with pytest.raises(
        TypeError, match="`random_seed` must be an int or None."
    ):
        parrm.find_period(random_seed=1.5)
    with pytest.raises(TypeError, match="`n_jobs` must be an int."):
        parrm.find_period(n_jobs=1.5)
    parrm.find_period()

    # explore_filter_params
    with pytest.raises(
        TypeError, match="`time_range` must be a list of ints or floats."
    ):
        parrm.explore_filter_params(time_range=0)
    with pytest.raises(
        TypeError, match="`time_range` must be a list of ints or floats."
    ):
        parrm.explore_filter_params(time_range=[0, "end"])
    with pytest.raises(
        TypeError, match="`time_res` must be an int or a float."
    ):
        parrm.explore_filter_params(time_res="all")
    with pytest.raises(
        TypeError, match="`freq_range` must be a list of ints or floats."
    ):
        parrm.explore_filter_params(freq_range=0)
    with pytest.raises(
        TypeError, match="`freq_range` must be a list of ints or floats."
    ):
        parrm.explore_filter_params(freq_range=[0, "Nyquist"])
    with pytest.raises(
        TypeError, match="`freq_res` must be an int or a float."
    ):
        parrm.explore_filter_params(freq_res=[0])
    with pytest.raises(TypeError, match="`n_jobs` must be an int."):
        parrm.explore_filter_params(n_jobs=1.5)

    # create_filter
    with pytest.raises(TypeError, match="`filter_half_width` must be an int."):
        parrm.create_filter(filter_half_width=1.5)
    with pytest.raises(TypeError, match="`omit_n_samples` must be an int."):
        parrm.create_filter(omit_n_samples=1.5)
    with pytest.raises(TypeError, match="`filter_direction` must be a str."):
        parrm.create_filter(filter_direction=0)
    with pytest.raises(
        TypeError, match="`period_half_width` must be an int or a float."
    ):
        parrm.create_filter(period_half_width=[0])
    parrm.create_filter()

    # filter_data
    with pytest.raises(TypeError, match="`data` must be a NumPy array."):
        parrm.filter_data(data=data.tolist())


def test_parrm_wrong_value_inputs():
    """Test that inputs of wrong values to PARRM are caught."""
    data = random.rand(1, 100)

    # init object
    with pytest.raises(ValueError, match="`data` must be a 2D array."):
        PARRM(
            data=random.rand(1, 1, 1),
            sampling_freq=sampling_freq,
            artefact_freq=artefact_freq,
        )
    with pytest.raises(ValueError, match="`sampling_freq` must be > 0."):
        PARRM(
            data=data,
            sampling_freq=0,
            artefact_freq=artefact_freq,
        )
    with pytest.raises(ValueError, match="`artefact_freq` must be > 0."):
        PARRM(
            data=data,
            sampling_freq=sampling_freq,
            artefact_freq=0,
        )
    parrm = PARRM(
        data=data,
        sampling_freq=sampling_freq,
        artefact_freq=artefact_freq,
    )

    # find_period
    with pytest.raises(
        ValueError, match="`search_samples` must be a 1D array."
    ):
        parrm.find_period(search_samples=np.zeros((1, 1)))
    with pytest.raises(
        ValueError,
        match="Entries of `search_samples` must lie in the range ",
    ):
        parrm.find_period(search_samples=np.array([-1, 1]))
    with pytest.raises(
        ValueError,
        match="Entries of `search_samples` must lie in the range ",
    ):
        parrm.find_period(search_samples=np.array([0, data.shape[1]]))
    with pytest.raises(ValueError, match="`outlier_boundary` must be > 0."):
        parrm.find_period(outlier_boundary=0)
    with pytest.raises(
        ValueError, match="`n_jobs` must be <= the number of available CPUs."
    ):
        parrm.find_period(n_jobs=cpu_count() + 1)
    with pytest.raises(
        ValueError, match="If `n_jobs` is <= 0, it must be -1."
    ):
        parrm.find_period(n_jobs=-2)
    parrm.find_period()

    # explore_filter_params
    with pytest.raises(
        ValueError, match="`time_range` must have a length of 2."
    ):
        parrm.explore_filter_params(time_range=[0, 1, 2])
    with pytest.raises(
        ValueError, match="`time_range` must lie in the range "
    ):
        parrm.explore_filter_params(time_range=[-1, 1])
    with pytest.raises(
        ValueError, match="`time_range` must lie in the range "
    ):
        parrm.explore_filter_params(
            time_range=[0, (data.shape[1] / sampling_freq) + 1]
        )
    with pytest.raises(ValueError, match="`time_range"):
        parrm.explore_filter_params(time_range=[1, 0])
    with pytest.raises(ValueError, match="`time_res` must lie in the range "):
        parrm.explore_filter_params(time_res=0)
    with pytest.raises(ValueError, match="`time_res` must lie in the range "):
        parrm.explore_filter_params(time_res=data.shape[1] / sampling_freq)
    with pytest.raises(
        ValueError, match="`freq_range` must have a length of 2."
    ):
        parrm.explore_filter_params(freq_range=[0, 1, 2])
    with pytest.raises(
        ValueError, match="`freq_range` must lie in the range "
    ):
        parrm.explore_filter_params(freq_range=[-1, 1])
    with pytest.raises(
        ValueError, match="`freq_range` must lie in the range "
    ):
        parrm.explore_filter_params(freq_range=[0, (sampling_freq / 2) + 1])
    with pytest.raises(ValueError, match="`freq_range"):
        parrm.explore_filter_params(freq_range=[1, 0])
    with pytest.raises(ValueError, match="`freq_res` must lie in the range "):
        parrm.explore_filter_params(freq_res=0)
    with pytest.raises(ValueError, match="`freq_res` must lie in the range "):
        parrm.explore_filter_params(freq_res=(sampling_freq / 2) + 1)
    with pytest.raises(
        ValueError, match="`n_jobs` must be <= the number of available CPUs."
    ):
        parrm.explore_filter_params(n_jobs=cpu_count() + 1)
    with pytest.raises(
        ValueError, match="If `n_jobs` is <= 0, it must be -1."
    ):
        parrm.explore_filter_params(n_jobs=-2)

    # create_filter
    with pytest.raises(
        ValueError, match="`filter_half_width` must lie in the range"
    ):
        parrm.create_filter(filter_half_width=1, omit_n_samples=2)
    with pytest.raises(
        ValueError, match="`filter_half_width` must lie in the range"
    ):
        parrm.create_filter(filter_half_width=((data.shape[1] - 1) // 2) + 1)
    with pytest.raises(
        ValueError, match="`omit_n_samples` must lie in the range"
    ):
        parrm.create_filter(omit_n_samples=-1)
    with pytest.raises(
        ValueError, match="`omit_n_samples` must lie in the range"
    ):
        parrm.create_filter(omit_n_samples=(data.shape[1] - 1) // 2)
    with pytest.raises(
        ValueError,
        match="`period_half_width` must be lie in the range ",
    ):
        parrm.create_filter(period_half_width=0)
    with pytest.raises(
        ValueError,
        match="`period_half_width` must be lie in the range ",
    ):
        parrm.create_filter(period_half_width=parrm.period + 1)
    with pytest.raises(ValueError, match="`filter_direction` must be one of "):
        parrm.create_filter(filter_direction="not_a_direction")
    with pytest.raises(
        RuntimeError,
        match="A suitable filter cannot be created with the specified ",
    ):
        parrm.create_filter(omit_n_samples=48)
    parrm.create_filter()

    # filter_data
    with pytest.raises(ValueError, match="`data` must be a 2D array."):
        parrm.filter_data(data=random.rand(100))


def test_parrm_premature_method_attribute_calls():
    """Test that errors raised for PARRM methods/attrs. called prematurely."""
    parrm = PARRM(
        data=random.rand(1, 100),
        sampling_freq=sampling_freq,
        artefact_freq=artefact_freq,
        verbose=False,
    )
    with pytest.raises(
        ValueError, match="The period has not yet been estimated."
    ):
        parrm.explore_filter_params()
    with pytest.raises(
        ValueError, match="The period has not yet been estimated."
    ):
        parrm.create_filter()
    with pytest.raises(
        ValueError, match="The filter has not yet been created."
    ):
        parrm.filter_data()
    with pytest.raises(
        AttributeError, match="No period has been computed yet."
    ):
        parrm.period
    with pytest.raises(
        AttributeError, match="No filter has been computed yet."
    ):
        parrm.filter
    with pytest.raises(AttributeError, match="No data has been filtered yet."):
        parrm.filtered_data
    with pytest.raises(
        AttributeError,
        match="Analysis settings have not been established yet.",
    ):
        parrm.settings

    parrm.find_period()
    with pytest.raises(
        ValueError, match="The filter has not yet been created."
    ):
        parrm.filter_data()


def test_parrm_missing_filter_inputs():
    """Test that PARRM can compute values for missing filter inputs."""
    parrm = PARRM(
        data=random.rand(1, 100),
        sampling_freq=sampling_freq,
        artefact_freq=artefact_freq,
        verbose=False,
    )
    parrm.find_period()
    parrm.create_filter()

    assert parrm._filter_half_width is not None
    assert parrm._period_half_width is not None


@pytest.mark.parametrize("n_chans", [1, 2])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_compute_psd(n_chans: int, n_jobs: int):
    """Test that PSD computation runs."""
    data = random.rand(n_chans, 100)

    n_freqs = 5
    freqs, psd = compute_psd(
        data=data,
        sampling_freq=sampling_freq,
        n_points=n_freqs * 2,
        n_jobs=n_jobs,
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
        assert path.endswith(
            file
        ), "`path` should end with the name of the dataset."
        assert os.path.exists(path), "`path` should point to an existing file."

    # test it catches incorrect inputs
    with pytest.raises(ValueError, match="`name` must be one of"):
        get_example_data_paths(name="not_a_name")
