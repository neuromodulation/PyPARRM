from copy import deepcopy

import numpy as np


class PARRM:
    """"""

    def __init__(self, data: np.ndarray, samp_freq: float, stim_freq: float, verbose:bool = True) -> None:
        """"""

        self._data = data.copy()
        self.samp_freq = deepcopy(samp_freq)
        self.stim_freq = deepcopy(stim_freq)
        self.verbose = deepcopy(verbose)
    
    def find_stim_period(self, search_samples: tuple[int] = (0, -1), assumed_periods: float | tuple[float] = None, outlier_boundary: float = 3.0) -> None:
        """"""


        if assumed_periods is None:
            self.assumed_periods = tuple([self.samp_freq / self.stim_freq])
        elif isinstance(assumed_periods, float):
            self.assumed_periods = tuple([assumed_periods])
        elif isinstance(assumed_periods, tuple):
            self.assumed_periods = deepcopy(assumed_periods)
        else:
            raise TypeError(
                "`assumed_periods` must be a float, tuple, or None."
            )

        outlier_bound = np.full((self.n_chans, ), outlier_boundary)

        standard_data = np.diff(self._data, axis=1)  # derivatives of data
        standard_data /= np.mean(np.abs(standard_data), axis=1)  # S.D. == 1
    	standard_data = np.max(np.min(standard_data, outlier_bound), outlier_bound * -1)  # clip outliers
    
    def _optimise_local(t, lfp, p, bandwidth: float, lambda_: float = 1.0) -> float:
        """"""
        f = 0

        s = np.arange(2*bandwidth)
        s = lambda_ * s / s.sum()
