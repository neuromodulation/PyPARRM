"""Tools for computing signal power."""

import numpy as np
from scipy.fft import fft


def compute_psd(
    data: np.ndarray, sampling_freq: int, n_freqs: int, n_jobs: int = 1
) -> np.ndarray:
    """Compute power spectral density of data.

    Parameters
    ----------
    data : numpy.ndarray, shape of (channels, times)
        Data for which power should be computed (assumed to be real-valued).

    sampling_freq : int
        Sampling frequency, in Hz, of `data`.

    n_freqs : int
        How many frequencies the power spectra should be computed for.

    n_jobs : int (default 1)
        Number of jobs to run in parallel.

    Returns
    -------
    psd : numpy.ndarray, shape of (channels, frequencies)
        Power spectral density of `data`. As `data` is assumed to be
        real-valued, only positive frequencies are returned. The zero frequency
        is also discarded.

    Notes
    -----
    No checks on the inputs are performed for speed.
    Data is converted to, and power is returned as, float32 values for speed.
    """
    fft_coeffs = fft(data.astype(np.float32), n_freqs * 2, workers=n_jobs)[
        ..., 1 : n_freqs + 1
    ]
    psd = (1.0 / (sampling_freq * n_freqs * 2)) * np.abs(fft_coeffs).astype(
        np.float32
    ) ** 2
    psd[:-1] *= 2

    return psd
