"""Tools for computing signal power."""

import numpy as np


def _compute_psd(
    data: np.ndarray, sampling_freq: int, n_freqs: int
) -> np.ndarray:
    """Compute power spectral density of data.

    Parameters
    ----------
    data : numpy.ndarray, shape of (times)
        Data of a single channel for which power should be computed (assumed to
        be real-valued).

    sampling_freq : int
        Sampling frequency, in Hz, of `data`.

    n_freqs : int
        How many frequencies the power spectra should be computed for.

    Returns
    -------
    psd : numpy.ndarray, shape of (frequencies)
        Power spectral density of `data`. As `data` is assumed to be
        real-valued, only positive frequencies are returned. The zero frequency
        is also discarded.

    Notes
    -----
    No checks on the inputs are performed for speed.
    """
    fft_coeffs = np.fft.fft(data, n_freqs * 2)[1 : n_freqs + 1]
    psd = (1 / sampling_freq * n_freqs * 2) * np.abs(fft_coeffs) ** 2
    psd[:-1] *= 2

    return psd
