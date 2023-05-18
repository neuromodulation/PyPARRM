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
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density of data.

    Parameters
    ----------
    data : numpy.ndarray, shape of (channels, times)
        Data for which power should be computed (assumed to be real-valued).

    sampling_freq : int
        Sampling frequency, in Hz, of `data`.

    n_points : int
        Number of points to use when computing the Fourier coefficients. Should
        be double the desired number of frequencies in the power spectra.

    max_freq : int | float | None (default None)
        The maximum frequency that should be returned. If ``None``, values for
        all computed frequencies returned.

    n_jobs : int (default 1)
        Number of jobs to run in parallel.

    Returns
    -------
    freqs :numpy.ndarray, shape of (frequencies)
        Frequencies in `psd`.

    psd : numpy.ndarray, shape of (channels, frequencies)
        Power spectral density of `data`.

    Notes
    -----
    No checks on the inputs are performed for speed.

    Data is converted to, and power is returned as, float32 values for speed.

    As `data` is assumed to be real-valued, only positive frequencies are
    returned. The zero frequency is also discarded.
    """
    freqs = np.abs(
        fftfreq(n_points, 1.0 / sampling_freq)[1 : (n_points // 2) + 1]
    )
    if max_freq is None:
        max_freq = freqs[-1]
    max_freq_i = np.argwhere(freqs <= max_freq)[-1][0]

    fft_coeffs = fft(data.astype(np.float32), n_points, workers=n_jobs)[
        ..., 1 : (n_points // 2) + 1
    ]
    psd = (1.0 / (sampling_freq * n_points)) * np.abs(fft_coeffs).astype(
        np.float32
    ) ** 2
    psd[:-1] *= 2

    return freqs[: max_freq_i + 1], psd[..., : max_freq_i + 1]
