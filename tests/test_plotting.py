"""Tests for plotting of PyPARRM."""

# Only checks initialisation of plotting tools, not their interactivity (this needs to
# be done manually currently).

# Author(s):
#   Thomas Samuel Binns | github.com/tsbinns

from multiprocessing import cpu_count

import numpy as np
import pytest
import matplotlib
from matplotlib import pyplot as plt

from pyparrm import PARRM

matplotlib.use("Agg")

sampling_freq = 50  # Hz
artefact_freq = 20  # Hz

random = np.random.default_rng(44)
n_chans = 2
n_samples = 1000
data = random.standard_normal((n_chans, n_samples))

parrm = PARRM(data=data, sampling_freq=sampling_freq, artefact_freq=artefact_freq)

parrm.find_period(assumed_periods=sampling_freq / artefact_freq, random_seed=44)


@pytest.mark.parametrize("time_range", [None, [1.0, 15.0]])
@pytest.mark.parametrize("freq_range", [None, [5.0, 25.0]])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_explore_params(
    time_range: list[float] | None, freq_range: list[float] | None, n_jobs: int
) -> None:
    """Test that `explore_filter_params` runs."""
    parrm.explore_filter_params(
        time_range=time_range, freq_range=freq_range, n_jobs=n_jobs
    )

    plt.close("all")


def test_explore_params_error_catch() -> None:
    """Test that errors are caught in `explore_filter_params`."""
    # time_range
    with pytest.raises(
        TypeError, match="`time_range` must be a list of ints or floats."
    ):
        parrm.explore_filter_params(time_range=(1, 15))
    with pytest.raises(
        TypeError, match="`time_range` must be a list of ints or floats."
    ):
        parrm.explore_filter_params(time_range=["start", "end"])
    with pytest.raises(ValueError, match="`time_range` must have a length of 2."):
        parrm.explore_filter_params(time_range=[1])
    with pytest.raises(ValueError, match="`time_range` must have a length of 2."):
        parrm.explore_filter_params(time_range=[1, 2, 3])
    with pytest.raises(
        ValueError,
        match=r"Entries of `time_range` must lie in the range \[0, max. time\].",
    ):
        parrm.explore_filter_params(time_range=[-1, 15])
    with pytest.raises(
        ValueError,
        match=r"Entries of `time_range` must lie in the range \[0, max. time\].",
    ):
        parrm.explore_filter_params(time_range=[1, n_samples / sampling_freq + 1])
    with pytest.raises(
        ValueError, match=r"`time_range\[1\]` must be > `time_range\[0\]`."
    ):
        parrm.explore_filter_params(time_range=[15, 1])

    # time_res
    with pytest.raises(TypeError, match="`time_res` must be an int or a float."):
        parrm.explore_filter_params(time_res="1")
    with pytest.raises(
        ValueError, match=r"`time_res` must lie in the range \(0, max. time\)."
    ):
        parrm.explore_filter_params(time_res=0)
    with pytest.raises(
        ValueError, match=r"`time_res` must lie in the range \(0, max. time\)."
    ):
        parrm.explore_filter_params(time_res=n_samples / sampling_freq + 1)

    # freq_range
    with pytest.raises(
        TypeError, match="`freq_range` must be a list of ints or floats."
    ):
        parrm.explore_filter_params(freq_range=(5, 25))
    with pytest.raises(ValueError, match="`freq_range` must have a length of 2."):
        parrm.explore_filter_params(freq_range=[5])
    with pytest.raises(ValueError, match="`freq_range` must have a length of 2."):
        parrm.explore_filter_params(freq_range=[5, 10, 15])
    with pytest.raises(
        ValueError,
        match=r"`freq_range` must lie in the range \(0, Nyquist frequency\].",
    ):
        parrm.explore_filter_params(freq_range=[0, sampling_freq / 2])
    with pytest.raises(
        ValueError,
        match=r"`freq_range` must lie in the range \(0, Nyquist frequency\].",
    ):
        parrm.explore_filter_params(freq_range=[1, sampling_freq / 2 + 1])
    with pytest.raises(
        ValueError, match=r"`freq_range\[1\]` must be > `freq_range\[0\]`."
    ):
        parrm.explore_filter_params(freq_range=[sampling_freq / 2, 1])

    # freq_res
    with pytest.raises(TypeError, match="`freq_res` must be an int or a float."):
        parrm.explore_filter_params(freq_res="1")
    with pytest.raises(
        ValueError, match=r"`freq_res` must lie in the range \(0, Nyquist frequency\]."
    ):
        parrm.explore_filter_params(freq_res=0)
    with pytest.raises(
        ValueError, match=r"`freq_res` must lie in the range \(0, Nyquist frequency\]."
    ):
        parrm.explore_filter_params(freq_res=sampling_freq / 2 + 1)

    # n_jobs
    with pytest.raises(TypeError, match="`n_jobs` must be an int."):
        parrm.explore_filter_params(n_jobs="all")
    with pytest.raises(
        ValueError, match=r"`n_jobs` must be <= the number of available CPUs."
    ):
        parrm.explore_filter_params(n_jobs=cpu_count() + 1)
    with pytest.raises(ValueError, match=r"If `n_jobs` is <= 0, it must be -1."):
        parrm.explore_filter_params(n_jobs=0)
