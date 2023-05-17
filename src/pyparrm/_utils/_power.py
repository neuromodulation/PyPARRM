"""Tools for computing signal power."""

# Author(s):
#   Thomas Samuel Binns | github.com/tsbinns

import numpy as np
from scipy.fft import fft, fftfreq  # faster than numpy for real-valued inputs


def compute_psd(
    data: np.ndarray,
    sampling_freq: int,
    n_points: int,
    max_freq: int | float | None = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute power spectral density of data.

    Parameters
    ----------
    data : numpy.ndarray, shape of (channels, times)
        Data for which power should be computed (assumed to be real-valued).

    sampling_freq : int
        Sampling frequency, in Hz, of `data`.

    n_points : int
        How many frequencies the power spectra should be computed for.

    max_freq : int | float | None (default None)
        The maximum frequency that should be returned. If ``None``, values for
        all computed frequencies returned.

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
    fft_coeffs = fft(data.astype(np.float32), n_points * 2, workers=n_jobs)[
        ..., 1 : n_points + 1
    ]
    psd = (1.0 / (sampling_freq * n_points * 2)) * np.abs(fft_coeffs).astype(
        np.float32
    ) ** 2
    psd[:-1] *= 2
    freqs = np.abs(fftfreq(n_points * 2, 1.0 / sampling_freq)[1 : n_points + 1])

    return psd[..., np.where(freqs <= max_freq)[0]]
