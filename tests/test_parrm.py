"""Tests for features of PyPARRM."""

from multiprocessing import cpu_count

import numpy as np
import pytest

from pyparrm import PARRM
from pyparrm._utils._power import compute_psd


random = np.random.RandomState(44)
sampling_freq = 20  # Hz
artefact_freq = 10  # Hz


@pytest.mark.parametrize("n_chans", [1, 2])
@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_parrm(n_chans: int, verbose: bool, n_jobs: int):
    """Test that PARRM can run."""
    data = random.rand(n_chans, 100)

    parrm = PARRM(
        data=data,
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
        parrm.find_period(assumed_periods=tuple([0]))
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
    with pytest.raises(ValueError, match="If `n_jobs` is < 0, it must be -1."):
        parrm.find_period(n_jobs=-2)


def test_parrm_premature_method_calls():
    """Test that errors are raised for PARRM methods called prematurely."""
    parrm = PARRM(
        data=random.rand(1, 100),
        sampling_freq=sampling_freq,
        artefact_freq=artefact_freq,
        verbose=False,
    )
    with pytest.raises(
        ValueError, match="The period has not yet been estimated."
    ):
        parrm.create_filter()
    with pytest.raises(
        ValueError, match="The filter has not yet been created."
    ):
        parrm.filter_data()

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
    psd = compute_psd(
        data=data,
        sampling_freq=sampling_freq,
        n_freqs=n_freqs,
        n_jobs=n_jobs,
    )

    assert psd.shape == (n_chans, n_freqs)
    assert isinstance(psd, np.ndarray)
