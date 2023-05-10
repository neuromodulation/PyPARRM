"""Tools for fitting PARRM filters to data."""

from copy import deepcopy
from multiprocessing import cpu_count

import numpy as np
from pqdm.threads import pqdm
from scipy.optimize import fmin
from scipy.signal import convolve2d

from pyparrm._utils._plotting import _ExploreParams


class PARRM:
    """Class for removing stimulation artefacts from data using PARRM.

    The Period-based Artefact Reconstruction and Removal Method (PARRM) is
    described in Dastin-van Rijn *et al.* (2021) :footcite:`DastinEtAl2021`.
    PARRM assumes that the artefacts are semi-regular, periodic, and linearly
    combined with the signal of interest.

    The methods should be called in the following order:
        1. :meth:`find_period`
        2. :meth:`create_filter`
        3. :meth:`filter_data`

    Parameters
    ----------
    data : numpy.ndarray, shape of (channels, times)
        Time-series from which stimulation artefacts should be identified and
        removed.

    sampling_freq : int | float
        Sampling frequency of :attr:`data`, in Hz.

    artefact_freq : int | float
        Frequency of the stimulation artefact in :attr:`data`, in Hz.

    verbose : bool (default True)
        Whether or not to print information about the status of the processing.

    Methods
    -------
    find_period
        Find the period of the artefacts.

    explore_filter_params
        Create an interactive plot to explore filter parameters.

    create_filter
        Create the PARRM filter for removing the stimulation artefacts.

    filter_data
        Apply the PARRM filter to the data and return it.

    References
    ----------
    .. footbibliography::
    """

    _data = None
    _standard_data = None
    _filtered_data = None

    _sampling_freq = None
    _artefact_freq = None
    _verbose = None

    _period = None
    _search_samples = None
    _assumed_periods = None
    _outlier_boundary = None
    _random_seed = None
    _n_jobs = None

    _filter = None
    _filter_half_width = None
    _omit_n_samples = None
    _filter_direction = None
    _period_half_width = None

    def __init__(
        self,
        data: np.ndarray,
        sampling_freq: int | float,
        artefact_freq: int | float,
        verbose: bool = True,
    ) -> None:  # noqa D107
        self._check_init_inputs(data, sampling_freq, artefact_freq, verbose)
        (self._n_chans, self._n_samples) = self._data.shape

    def _check_init_inputs(
        self,
        data: np.ndarray,
        sampling_freq: int | float,
        artefact_freq: int | float,
        verbose: bool,
    ) -> None:
        """Check initialisation inputs to object."""
        if not isinstance(data, np.ndarray):
            raise TypeError("`data` must be a NumPy array.")
        if data.ndim != 2:
            raise ValueError("`data` must be a 2D array.")
        self._data = data.copy()

        if not isinstance(sampling_freq, (int, float)):
            raise TypeError("`sampling_freq` must be an int or a float.")
        if sampling_freq <= 0:
            raise ValueError("`sampling_freq` must be > 0.")
        self._sampling_freq = deepcopy(sampling_freq)

        if not isinstance(artefact_freq, (int, float)):
            raise TypeError("`artefact_freq` must be an int or a float.")
        if artefact_freq <= 0:
            raise ValueError("`artefact_freq` must be > 0.")
        self._artefact_freq = deepcopy(artefact_freq)

        if not isinstance(verbose, bool):
            raise TypeError("`verbose` must be a bool.")
        self._verbose = deepcopy(verbose)

    def find_period(
        self,
        search_samples: np.ndarray | None = None,
        assumed_periods: int | float | tuple[int | float] | None = None,
        outlier_boundary: int | float = 3.0,
        random_seed: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        """Find the period of the artefacts.

        Parameters
        ----------
        search_samples : numpy.ndarray | None (default None), shape of (times)
            Samples of :attr:`data` to use when finding the artefact period. If
            ``None``, all samples are used.

        assumed_periods : int | float | tuple[int or float] | None (default None)
            Guess(es) of the artefact period. If ``None``, the period is
            assumed to be ``sampling_freq`` / ``artefact_freq``.

        outlier_boundary : int | float (default 3.0)
            Boundary (in standard deviation) to consider outlier values in
            :attr:`data`.

        random_seed: int | None (default None)
            Seed to use when generating indices of samples to search for the
            period. Only used if the number of available samples is less than
            the number of requested samples.

        n_jobs : int (default 1)
            Number of jobs to run in parallel when optimising the period
            estimates. Must be less than the number of available CPUs and
            greater than 0 (unless it is -1, in which case all available CPUs
            are used).
        """  # noqa E501
        if self._verbose:
            print("\nFinding the artefact period...")

        self._reset_result_attrs()

        self._check_sort_find_stim_period_inputs(
            search_samples,
            assumed_periods,
            outlier_boundary,
            random_seed,
            n_jobs,
        )

        self._standardise_data()
        self._optimise_period_estimate()

        if self._verbose:
            print("    ... Artefact period found\n")

    def _reset_result_attrs(self) -> None:
        """Reset result attributes for when period recalculated."""
        self._standard_data = None
        self._filtered_data = None

        self._period = None
        self._search_samples = None
        self._assumed_periods = None
        self._outlier_boundary = None
        self._random_seed = None

        self._filter = None
        self._filter_half_width = None
        self._omit_n_samples = None
        self._filter_direction = None
        self._period_half_width = None

    def _check_sort_find_stim_period_inputs(
        self,
        search_samples: np.ndarray | None,
        assumed_periods: int | float | tuple[int | float] | None,
        outlier_boundary: int | float,
        random_seed: int | None,
        n_jobs: int,
    ) -> None:
        """Check and sort `find_stim_period` inputs."""
        if search_samples is not None and not isinstance(
            search_samples, np.ndarray
        ):
            raise TypeError("`search_samples` must be a NumPy array or None.")
        if search_samples is None:
            search_samples = np.arange(self._n_samples - 1)
        elif search_samples.ndim != 1:
            raise ValueError("`search_samples` must be a 1D array.")
        search_samples = np.sort(search_samples)
        if search_samples[0] < 0 or search_samples[-1] >= self._n_samples:
            raise ValueError(
                "Entries of `search_samples` must lie in the range [0, "
                "n_samples)."
            )
        self._search_samples = search_samples.copy()

        if assumed_periods is not None and not isinstance(
            assumed_periods, (int, float, tuple)
        ):
            raise TypeError(
                "`assumed_periods` must be an int, a float, a tuple, or None."
            )
        if assumed_periods is None:
            assumed_periods = tuple(
                [self._sampling_freq / self._artefact_freq]
            )
        elif isinstance(assumed_periods, (int, float)):
            assumed_periods = tuple([assumed_periods])
        elif not all(
            isinstance(entry, (int, float)) for entry in assumed_periods
        ):
            raise TypeError(
                "If a tuple, entries of `assumed_periods` must be ints or "
                "floats."
            )
        self._assumed_periods = deepcopy(assumed_periods)

        if not isinstance(outlier_boundary, (int, float)):
            raise TypeError("`outlier_boundary` must be an int or a float.")
        if outlier_boundary <= 0:
            raise ValueError("`outlier_boundary` must be > 0.")
        self._outlier_boundary = deepcopy(outlier_boundary)

        if random_seed is not None and not isinstance(random_seed, int):
            raise TypeError("`random_seed` must be an int or None.")
        if random_seed is not None:
            self._random_seed = deepcopy(random_seed)

        if not isinstance(n_jobs, int):
            raise TypeError("`n_jobs` must be an int.")
        if n_jobs > cpu_count():
            raise ValueError(
                "`n_jobs` must be <= the number of available CPUs."
            )
        if n_jobs < 0 and n_jobs != -1:
            raise ValueError("If `n_jobs` is < 0, it must be -1.")
        if n_jobs == -1:
            n_jobs = cpu_count()
        self._n_jobs = deepcopy(n_jobs)

    def _standardise_data(self) -> None:
        """Standardise data to have S.D. of 1 and clipped outliers."""
        standard_data = np.diff(self._data, axis=1)  # derivatives of data
        standard_data /= np.mean(np.abs(standard_data), axis=1)[
            :, None
        ]  # S.D. == 1
        standard_data = np.clip(
            standard_data, -self._outlier_boundary, self._outlier_boundary
        )  # clip outliers

        self._standard_data = standard_data

    def _optimise_period_estimate(self) -> None:
        """Optimise artefact period estimate."""
        random_state = np.random.RandomState(self._random_seed)

        stim_period = np.nan

        opt_sample_lens = np.unique(
            [
                int(np.min((self._search_samples.shape[0], length)))
                for length in [5000, 10000, 25000]
            ]
        )
        ignore_sample_portions = [0.0, 0.0, 0.95]
        bandwidths = [5, 10, 20]
        lambda_ = 1.0

        run_idx = 1
        for use_n_samples, ignore_portion, bandwidth in zip(
            opt_sample_lens, ignore_sample_portions, bandwidths
        ):
            indices = self._get_centre_indices(
                use_n_samples, ignore_portion, random_state
            )
            bandwidth = np.min((bandwidth, indices.shape[0] // 4))

            periods = []
            for assumed_period in self._assumed_periods:
                periods.extend(
                    np.concatenate(
                        (
                            assumed_period
                            * (
                                1
                                + np.arange(-1e-2, 1e-2 + 1e-4, 1e-4) / run_idx
                            ),
                            assumed_period
                            * (
                                1
                                + np.arange(-1e-3, 1e-3 + 1e-5, 1e-5) / run_idx
                            ),
                        )
                    )
                )
            periods = np.unique(periods)

            optimise_local_args = [
                {
                    "period": period,
                    "data": self._standard_data,
                    "indices": indices,
                    "bandwidth": bandwidth,
                    "lambda_": lambda_,
                }
                for period in periods
            ]
            v = np.array(
                pqdm(
                    optimise_local_args,
                    self._optimise_local,
                    self._n_jobs,
                    argument_type="kwargs",
                    desc="Optimising period estimates",
                    disable=not self._verbose,
                )
            )

            v_idcs = v.argsort()
            v = v[v_idcs]
            periods = periods[v_idcs[v != np.inf]]  # ignore invalid `v`s
            if periods.shape == (0,):  # if no valid periods
                raise ValueError(
                    "The period cannot be estimated from the data. Check "
                    "that your data does not contain NaNs."
                )

            n_iters = np.min((5, periods.shape[0]))
            fmin_args = [
                {
                    "func": self._optimise_local,
                    "x0": period,
                    "args": (self._standard_data, indices, bandwidth, lambda_),
                    "full_output": True,
                    "disp": False,
                }
                for period in periods[:n_iters]
            ]
            output = pqdm(
                fmin_args,
                fmin,
                self._n_jobs,
                argument_type="kwargs",
                desc="Optimising period estimates",
                disable=not self._verbose,
            )
            for iter_idx in range(n_iters):
                periods[iter_idx] = output[iter_idx][0]
                v[iter_idx] = output[iter_idx][1]
            stim_period = periods[v.argmin()]

            run_idx += 1

        if np.isnan(stim_period):
            raise ValueError(
                "The period cannot be estimated from the data. Check that "
                "your data does not contain NaNs."
            )

        # final optimisation run with 0 regularisation for best fit
        self._period = fmin(
            self._optimise_local,
            stim_period,
            (
                self._standard_data,
                indices,
                bandwidth,
                0.0,
            ),
            disp=False,
        )[0]

    def _get_centre_indices(
        self,
        use_n_samples: int,
        ignore_portion: float,
        random_state: np.random.RandomState,
    ) -> np.ndarray:
        """Get indices for samples in the centre of the data segment.

        Parameters
        ----------
        use_n_samples : int
            Number of samples to use from the data segment.

        ignore_portion : float
            Portion of the data segment to ignore when getting the indices.

        random_state : numpy.random.RandomState
            Random state object to use to generate numbers if the available
            number of samples is less than that requested.

        Returns
        -------
        use_indices : numpy.ndarray, shape of (samples)
            Indices of samples in the centre of the data segment.
        """
        sample_range = self._search_samples[0] + self._search_samples[-1]
        start_idx = int(np.ceil((sample_range - use_n_samples) / 2))
        end_idx = int(np.floor((sample_range + use_n_samples) / 2))

        if self._n_samples * ignore_portion < end_idx - start_idx:
            return np.arange(start_idx, end_idx + 1)

        start_idx = self._search_samples[0] + np.floor(
            (1.0 - ignore_portion) / 2.0 * self._n_samples
        )
        end_idx = self._search_samples[1] - np.ceil(
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
        """???

        Parameters
        ----------
        period : float
            Estimate of the artefact period.

        data : numpy.ndarray, shape of (channels, times)
            Data containing artefacts whose period should be estimated.

        indices : numpy.ndarray, shape of (samples)
            Samples to use to estimate the period in `data`.

        bandwidth: int


        lambda_ : float
            Regularisation parameter.

        Returns
        -------
        f : float
            ???

        Notes
        -----
        ``period`` must be the first input argument for scipy.optimize.fmin to
        work on this function.
        """
        f = 0.0

        s = np.arange(1, 2 * bandwidth + 2)
        s = lambda_ * s / s.sum()

        for data_chan in data:
            residuals, beta = self._fit_waves_to_data(
                data_chan[indices], indices, period, bandwidth
            )
            if isinstance(residuals, float):
                return np.inf

            f += residuals.mean() + s @ beta

        return f / self._n_chans

    def _fit_waves_to_data(
        self,
        data: np.ndarray,
        indices: np.ndarray,
        period: float,
        n_waves: int,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
        """Fit sine and cosine waves to the data.

        Parameters
        ----------
        data : numpy.ndarray

        indices : numpy.ndarray

        period : float

        n_wave : int
            Number of sine and cosine waves to combine with `data`.

        Returns
        -------
        residuals : numpy.ndarray
            ??

        beta : numpy.ndarray
            ??

        error : numpy.linalg.LinAlgError
            Returned if the wave-data combination is singular.
        """
        angles = (indices + 1) * (2 * np.pi / period)
        waves = np.ones((data.shape[0], 2 * n_waves + 1))
        for wave_idx in range(1, n_waves + 1):
            waves[:, 2 * wave_idx - 1] = np.sin(wave_idx * angles)
            waves[:, 2 * wave_idx] = np.cos(wave_idx * angles)

        try:  # ignores LinAlgError for singular matrices
            beta = np.linalg.solve(waves.T @ waves, waves.T @ data)
        except np.linalg.LinAlgError:
            return np.inf, np.inf

        residuals = data - waves @ beta

        return residuals**2, beta**2

    def explore_filter_params(
        self, freq_res: int | float = 5.0, n_jobs: int = 1
    ) -> None:
        """Create an interactive plot to explore filter parameters.

        Can only be called after the artefact period has been estimated with
        :meth:`find_period`.

        Parameters
        ----------
        freq_res : int | float (default 5.0)
            Frequency resolution, in Hz, to use when computing the power
            spectra of the data. Must be > 0 and <= the Nyquist frequency.

        n_jobs : int (default 1)
            Number of jobs to run in parallel when computing the power spectra.
            Must be < the number of available CPUs and > 0 (unless it is -1, in
            which case all available CPUs are used).

        Notes
        -----
        It is recommended that you call this method from a script; behaviour in
        notebooks is not guaranteed!
        """
        if self._verbose:
            print("Opening the filter parameter explorer...")

        if self._period is None:
            raise ValueError(
                "The period has not yet been estimated. The `find_period` "
                "method must be called first."
            )
        param_explorer = _ExploreParams(self, freq_res, n_jobs)
        param_explorer.plot()

    def create_filter(
        self,
        filter_half_width: int | None = None,
        omit_n_samples: int = 0,
        filter_direction: str = "both",
        period_half_width: int | float | None = None,
    ) -> None:
        """Create the PARRM filter for removing the stimulation artefacts.

        Can only be called after the artefact period has been estimated with
        :meth:`find_period`.

        Parameters
        ----------
        filter_half_width : int | None (default None)
            Half-width of the filter to create, in samples. If ``None``, a
            filter half-width will be generated based on ``omit_n_samples``.

        omit_n_samples : int (default 0)
            Number of samples to omit from the centre of ``filter_half_width``.

        filter_direction : str (default "both")
            Direction from which samples should be taken to create the filter,
            relative to the centre of the filter window. Can be: "both" for
            backward and forward samples; "past" for backward samples only; and
            "future" for forward samples only.

        period_half_width : int | float | None (default None)
            Half-width of the window in samples of period space for which
            points at a similar location in the waveform will be averaged. If
            ``None``, :attr:`period` / 50 is used.
        """
        if self._verbose:
            print("Creating the filter...")

        if self._period is None:
            raise ValueError(
                "The period has not yet been estimated. The `find_period` "
                "method must be called first."
            )

        self._check_sort_create_filter_inputs(
            filter_half_width,
            omit_n_samples,
            filter_direction,
            period_half_width,
        )

        self._generate_filter()

        if self._verbose:
            print("    ... Filter created\n")

    def _check_sort_create_filter_inputs(
        self,
        filter_half_width: int,
        omit_n_samples: int,
        filter_direction: str,
        period_half_width: int | float,
    ) -> None:
        """Check and sort `create_filter` inputs."""
        if not isinstance(omit_n_samples, int):
            raise TypeError("`omit_n_samples` must be an int.")
        if omit_n_samples < 0 or omit_n_samples >= (self._n_samples - 1) // 2:
            raise ValueError(
                "`omit_n_samples` must lie in the range [0, (no. of samples - "
                "1) // 2]."
            )
        self._omit_n_samples = deepcopy(omit_n_samples)

        if period_half_width is None:
            period_half_width = self._period / 50
        if not isinstance(period_half_width, (int, float)):
            raise TypeError("`period_half_width` must be an int or a float.")
        if period_half_width > self._period:
            raise ValueError("`period_half_width` must be <= the period.")
        self._period_half_width = deepcopy(period_half_width)

        # Must come after `omit_n_samples` and `period_half_width` set!
        if filter_half_width is None:
            filter_half_width = self._get_filter_half_width()
        if not isinstance(filter_half_width, int):
            raise TypeError("`filter_half_width` must be an int.")
        if filter_half_width <= omit_n_samples:
            raise ValueError("`filter_half_width` must be > `omit_n_samples`.")
        if filter_half_width > ((self._n_samples - 1) // 2):
            raise ValueError(
                "`filter_half_width` must be <= ((no. of samples - 1) // 2)"
            )
        self._filter_half_width = deepcopy(filter_half_width)

        if not isinstance(filter_direction, str):
            raise TypeError("`filter_direction` must be a str.")
        valid_filter_directions = ["both", "past", "future"]
        if filter_direction not in valid_filter_directions:
            raise ValueError(
                f"`filter_direction` must be one of {valid_filter_directions}."
            )
        self._filter_direction = deepcopy(filter_direction)

    def _get_filter_half_width(self) -> int:
        """Get appropriate `filter_half_width`, if None given."""
        filter_half_width = deepcopy(self._omit_n_samples)
        check = 0
        while check < 50 and filter_half_width < (self._n_samples - 1) // 2:
            filter_half_width += 1
            modulus = np.mod(filter_half_width, self._period)
            if (
                modulus <= self._period_half_width
                or modulus >= self._period + self._period_half_width
            ):
                check += 1

        return filter_half_width

    def _generate_filter(self) -> None:
        """Generate linear filter for removing stimulation artefacts."""
        window = np.arange(
            -self._filter_half_width, self._filter_half_width + 1
        )
        modulus = np.mod(window, self._period)

        filter_ = np.zeros_like(window, dtype=np.float64)
        filter_[
            (
                (modulus <= self._period_half_width)
                | (modulus >= self._period - self._period_half_width)
            )
            & (np.abs(window) > self._omit_n_samples)
        ] += 1.0

        if self._filter_direction == "past":
            filter_[window > 0] = 0
        elif self._filter_direction == "future":
            filter_[window <= 0] = 0

        if np.all(filter_ == np.zeros_like(filter_)):
            raise RuntimeError(
                "A suitable filter cannot be created with the specified "
                "settings. Try reducing the number of omitted samples and/or "
                "increasing the filter half-width."
            )

        filter_ = -filter_ / np.max(
            (filter_.sum(), np.finfo(filter_.dtype).eps)
        )

        filter_[window == 0] = 1

        self._filter = filter_

    def filter_data(self) -> np.ndarray:
        """Apply the PARRM filter to the data and return it.

        Can only be called after the filter has been created with
        :meth:`create_filter`.

        Returns
        -------
        filtered_data : numpy.ndarray, shape of (channels, times)
            The filtered, artefact-free data.
        """
        if self._verbose:
            print("Filtering the data...")

        if self._filter is None:
            raise ValueError(
                "The filter has not yet been created. The `create_filter` "
                "method must be called first."
            )

        numerator = (
            convolve2d(self._data.T, self._filter[:, np.newaxis], "same")
            - self._data.T
        )
        denominator = 1 - convolve2d(
            np.ones_like(self._data).T,
            self._filter[:, np.newaxis],
            "same",
        )

        self._filtered_data = (numerator / denominator + self._data.T).T

        if self._verbose:
            print("    ... Data filtered\n")

        return self._filtered_data

    @property
    def data(self) -> np.ndarray:
        """Return a copy of the data."""
        if self._data is None:
            raise AttributeError("No data has been provided yet.")
        return self._data.copy()

    @property
    def period(self) -> float:
        """Return a copy of the estimated stimulation period."""
        if self._period is None:
            raise AttributeError("No period has been computed yet.")
        return deepcopy(self._period)

    @property
    def filter(self) -> np.ndarray:
        """Return a copy of the PARRM filter."""
        if self._filter is None:
            raise AttributeError("No filter has been computed yet.")
        return self._filter.copy()

    @property
    def filtered_data(self) -> np.ndarray:
        """Return a copy of the filtered data."""
        if self._filtered_data is None:
            raise AttributeError("No data has been filtered yet.")
        return deepcopy(self._filtered_data)

    @property
    def settings(self) -> dict:
        """Return the settings used to generate the PARRM filter."""
        if self._period is None or self._filter is None:
            raise AttributeError(
                "Analysis settings have not been established yet."
            )
        return {
            "data": {
                "sampling_freq": self._sampling_freq,
                "artefact_freq": self._artefact_freq,
            },
            "period": {
                "search_samples": self._search_samples,
                "assumed_periods": self._assumed_periods,
                "outlier_boundary": self._outlier_boundary,
                "random_seed": self._random_seed,
            },
            "filter": {
                "filter_half_width": self._filter_half_width,
                "omit_n_samples": self._omit_n_samples,
                "filter_direction": self._filter_direction,
                "period_half_width": self._period_half_width,
            },
        }
