"""Tools for fitting PARRM filters to data."""

from copy import deepcopy

import numpy as np
from scipy.optimize import fmin
from scipy.signal import convolve2d


class PARRM:
    """"""

    _data = None
    _data_standard = None

    samp_freq = None
    stim_freq = None
    verbose = None

    _period = None
    search_samples = None
    assumed_periods = None
    outlier_boundary = None
    random_seed = None

    _filter = None
    sample_window_size = None
    skip_n_samples = None
    filter_direction = None
    period_window_size = None

    def __init__(
        self,
        data: np.ndarray,
        samp_freq: int | float,
        stim_freq: int | float,
        verbose: bool = True,
    ) -> None:  # noqa D107
        self._check_init_inputs(data, samp_freq, stim_freq, verbose)
        (self._n_chans, self._n_samples) = self._data.shape

    def _check_init_inputs(
        self,
        data: np.ndarray,
        samp_freq: int | float,
        stim_freq: int | float,
        verbose: bool,
    ) -> None:
        """Check initialisation inputs to object."""
        if not isinstance(data, np.ndarray):
            raise TypeError("`data` must be a NumPy array.")
        if data.ndim != 2:
            raise ValueError("`data` must be a 2D array.")
        self._data = data.copy()

        if not isinstance(samp_freq, int) and not isinstance(samp_freq, float):
            raise TypeError("`samp_freq` must be an int or a float.")
        self.samp_freq = deepcopy(samp_freq)

        if not isinstance(stim_freq, int) and not isinstance(stim_freq, float):
            raise TypeError("`stim_freq` must be an int or a float.")
        self.stim_freq = deepcopy(stim_freq)

        if not isinstance(verbose, bool):
            raise TypeError("`verbose` must be a bool.")
        self.verbose = deepcopy(verbose)

    def find_period(
        self,
        search_samples: np.ndarray | None = None,
        assumed_periods: int | float | tuple[int | float] | None = None,
        outlier_boundary: int | float = 3.0,
        random_seed: int | None = None,
    ) -> None:
        """Find the period of the stimulation artefacts."""
        self._check_sort_find_stim_period_inputs(
            search_samples, assumed_periods, outlier_boundary, random_seed
        )

        self._standardise_data()
        self._optimise_period()

    def _check_sort_find_stim_period_inputs(
        self,
        search_samples: np.ndarray | None,
        assumed_periods: int | float | tuple[int | float] | None,
        outlier_boundary: int | float,
        random_seed: int | None,
    ) -> None:
        """Check and sort `find_stim_period` inputs."""
        if search_samples is not None and not isinstance(
            search_samples, np.ndarray
        ):
            raise TypeError("`search_samples` must be a NumPy array or None.")
        if search_samples is None:
            search_samples = np.arange(self._data.shape[1])
        elif search_samples.ndim != 1:
            raise ValueError("`search_samples` must be a 1D array.")
        self.search_samples = search_samples.copy()

        if assumed_periods is not None and not isinstance(
            assumed_periods, (int, float, tuple)
        ):
            raise TypeError(
                "`assumed_periods` must be an int, a float, a tuple, or None."
            )
        if assumed_periods is None:
            assumed_periods = tuple([self.samp_freq / self.stim_freq])
        elif isinstance(assumed_periods, (int, float)):
            assumed_periods = tuple([assumed_periods])
        elif not all(
            isinstance(entry, (int, float)) for entry in assumed_periods
        ):
            raise TypeError(
                "If a tuple, entries of `assumed_periods` must be ints or "
                "floats."
            )
        self.assumed_periods = deepcopy(assumed_periods)

        if not isinstance(outlier_boundary, (int, float)):
            raise TypeError("`outlier_boundary` must be an int or a float.")
        self.outlier_boundary = deepcopy(outlier_boundary)

        if random_seed is not None and not isinstance(random_seed, int):
            raise TypeError("`random_seed` must be an int or None.")
        if random_seed is not None:
            self.random_seed = deepcopy(random_seed)

    def _standardise_data(self) -> None:
        """Standardise data to have S.D. of 1 and clipped outliers."""
        outlier_boundaries = np.full((self._n_chans,), self.outlier_boundary)

        standard_data = np.diff(self._data, axis=1)  # derivatives of data
        standard_data /= np.mean(np.abs(standard_data), axis=1)  # S.D. == 1
        standard_data = np.max(
            (
                np.min((standard_data, outlier_boundaries), axis=1),
                outlier_boundaries * -1,
            ),
            axis=1,
        )  # clip outliers

        self._data_standard = standard_data

    def _optimise_period(self) -> None:
        """Optimise signal period."""
        random_state = np.random.RandomState(self.random_seed)

        opt_sample_lens = np.unique(
            [np.min(self._n_samples, length) for length in [5e3, 1e4, 2.5e4]]
        )
        ignore_sample_portions = [0.0, 0.0, 0.95]
        bandwidths = [5, 10, 20]
        lambda_ = 1.0

        run_idx = 0
        for use_n_samples, ignore_portion, bandwidth in zip(
            opt_sample_lens, ignore_sample_portions, bandwidths
        ):
            use_idcs = self._get_centre_indices(
                use_n_samples, ignore_portion, random_state
            )

            periods = []
            for assumed_period in self.assumed_periods:
                periods.extend(
                    [
                        assumed_period
                        * (1 + np.arange(-1e-2, 1e-2, 1e-4) / run_idx),
                        assumed_period
                        * (1 + np.arange(-1e-3, 1e-3, 1e-5) / run_idx),
                    ]
                )
            periods = np.unique(periods)

            v = np.zeros_like(periods)
            for period_idx, period in enumerate(periods):
                v[period_idx] = self._optimise_local(
                    period,
                    self._data_standard,
                    use_idcs,
                    np.min(bandwidth, use_idcs.shape[0] // 4),
                    lambda_,
                )
            v_idcs = v.argsort()
            v = v[v_idcs]
            periods = periods[v_idcs]

            for j in range(np.min(5, periods.shape[0])):
                periods[j], v[j] = fmin(
                    self._optimise_local,
                    periods[j],
                    (
                        self._data_standard,
                        use_idcs,
                        np.min(bandwidth, use_idcs.shape[0] // 4),
                        lambda_,
                    ),
                    disp=False,
                )

            stim_period = periods[v.argmin()]

        self._period = fmin(
            self._optimise_local,
            stim_period,
            (
                self._data_standard,
                use_idcs,
                np.min(bandwidth, use_idcs.shape[0] // 4),
                0.0,
            ),
            disp=False,
        )

    def _get_centre_indices(
        self,
        use_n_samples: int,
        ignore_portion: float,
        random_state: np.random.RandomState,
    ) -> np.ndarray:
        """Get indices for samples in the centre of the data segment."""
        start_idx = np.ceil(
            self.search_samples[0]
            + self.search_samples[1]
            - use_n_samples / 2.0
        )
        end_idx = np.floor(
            self.search_samples[0]
            + self.search_samples[1]
            + use_n_samples / 2.0
        )
        if self._n_samples * ignore_portion < end_idx - start_idx:
            return np.arange(start_idx, end_idx)
        else:
            start_idx = self.search_samples[0] + np.floor(
                (1.0 - ignore_portion) / 2.0 * self._n_samples
            )
            end_idx = self.search_samples[1] - np.ceil(
                (1.0 - ignore_portion) / 2.0 * self._n_samples
            )
            return np.unique(
                random_state.randint(
                    0,
                    end_idx - start_idx,
                    np.min(use_n_samples, end_idx - start_idx),
                )
                + start_idx
            )

    def _optimise_local(
        self,
        period: float,
        data: np.ndarray,
        indices: np.ndarray,
        bandwidth: int,
        lambda_: float,
    ) -> float:
        """

        Notes
        -----
        ``period`` must be the first input argument for scipy.optimize.fmin to
        work on this function.
        """
        f = 0

        s = np.arange(2 * bandwidth + 1)
        s = lambda_ * s / s.sum()

        for data_chan in data:
            residuals, beta = self._fit_sinusoids_to_data(
                data_chan[indices], indices, period, bandwidth
            )
            f += residuals.mean() + s @ beta

        return f.mean(axis=0)

    def _fit_sinusoids_to_data(
        self,
        data: np.ndarray,
        indices: np.ndarray,
        period: float,
        n_waves: int,
    ):
        """"""
        angles = data * (2 * np.pi / period)
        waves = np.ones((data.shape[0], 2 * n_waves))
        for wave_idx in range(n_waves):
            waves[:, 2 * wave_idx] = np.sin(wave_idx * angles)
            waves[:, 2 * wave_idx + 1] = np.cos(wave_idx * angles)

        beta = np.linalg.solve((waves.T @ waves), waves.T @ indices)
        residuals = indices - waves * beta

        return residuals**2, beta**2

    def create_filter(
        self,
        sample_window_size: int | None = None,
        skip_n_samples: int = 0,
        filter_direction: str = "both",
        period_window_size: float | None = None,
    ) -> np.ndarray:
        """Create the PARRM filter for removing the stimulation artefacts."""
        self._check_sort_create_filter_inputs(
            sample_window_size,
            skip_n_samples,
            filter_direction,
            period_window_size,
        )

        self._generate_filter()

    def _check_sort_create_filter_inputs(
        self,
        sample_window_size: int,
        skip_n_samples: int,
        filter_direction: str,
        period_window_size: float,
    ) -> None:
        """Check and sort `create_filter` inputs."""
        if not isinstance(skip_n_samples, int):
            raise TypeError("`skip_n_samples` must be an int.")
        if skip_n_samples < 0:
            raise ValueError("`skip_n_samples` must be >= 0.")
        self.skip_n_samples = deepcopy(skip_n_samples)

        if period_window_size is None:
            period_window_size = self._period / 50
        if not isinstance(period_window_size, float):
            raise TypeError("`period_window_size` must be a float.")
        if period_window_size > self._period:
            raise ValueError("`period_window_size` must be <= the period.")
        self.period_window_size = deepcopy(period_window_size)

        # Must come after `skip_n_samples` and `period_window_size` set!
        if sample_window_size is None:
            sample_window_size = self._get_sample_window_size()
        if not isinstance(sample_window_size, int):
            raise TypeError("`sample_window_size` must be an int.")
        if sample_window_size <= skip_n_samples:
            raise ValueError(
                "`sample_window_size` must be > `skip_n_samples`."
            )
        self.sample_window_size = deepcopy(sample_window_size)

        if not isinstance(filter_direction, str):
            raise TypeError("`filter_direction` must be a str.")
        valid_filter_directions = ["both", "past", "future"]
        if filter_direction not in valid_filter_directions:
            raise ValueError(
                f"`filter_direction` must be one of {valid_filter_directions}."
            )
        self.filter_direction = deepcopy(filter_direction)

    def _get_sample_window_size(self) -> int:
        """Get appropriate `sample_window_size` if None given."""
        sample_window_size = deepcopy(self.skip_n_samples)
        check = 0
        while check < 50 and sample_window_size < 10e5:
            sample_window_size += 1
            modulus = np.mod(sample_window_size, self._period)
            if (
                modulus <= self.period_window_size
                or modulus >= self._period + self.period_window_size
            ):
                check += 1

        return sample_window_size

    def _generate_filter(self) -> None:
        """Generate linear filter for removing stimulation artefacts."""
        window = np.arange(-self.sample_window_size, self.sample_window_size)
        y = np.mod(window, self._period)

        filter_ = np.zeros_like(window)
        filter_[
            (
                y <= self.period_window_size
                or y >= self.period_window_size + self._period
            )
            & (np.abs(window) > self.skip_n_samples)
        ] += 1

        if self.filter_direction == "past":
            filter_[window > 0] = 0
        elif self.filter_direction == "future":
            filter_[window <= 0] = 0

        filter_ = -filter_ / np.max(filter_.sum(), np.finfo(filter_.dtype).eps)
        filter_[window == 0] = 1

        self._filter = filter_

    def filter_data(self) -> np.ndarray:
        """Apply the PARRM filter to the data and return it."""
        return (
            convolve2d(self._data, np.rot90(self._filter), "same") - self._data
        ) / (
            1
            - (
                convolve2d(
                    self._data, np.rot90(np.ones_like(self._data)), "same"
                )
            )
            + self._data
        )

    @property
    def period(self) -> float:
        """Return a copy of the estimated stimulation period."""
        return deepcopy(self._period)

    @property
    def filter(self) -> np.ndarray:
        """Return a copy of the PARRM filter."""
        return self._filter.copy()

    @property
    def settings(self) -> dict:
        """Return the settings used to generate the PARRM filter."""
        return {
            "stim_period": {
                "search_samples": self.search_samples,
                "assumed_period": self.assumed_periods,
                "outlier_bound": self.outlier_boundary,
                "random_seed": self.random_seed,
            },
            "filter": {
                "sample_window_size": self.sample_window_size,
                "skip_n_samples": self.skip_n_samples,
                "filter_direction": self.filter_direction,
                "period_window_size": self.period_window_size,
            },
        }
