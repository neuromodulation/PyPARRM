"""
===========================================================
Using PyPARRM to filter out stimulation artefacts from data
===========================================================

This example demonstrates how the PARRM algorithm :footcite:`DastinEtAl2021`
can be used to identify and remove stimulation artefacts from
electrophysiological data in the PyPARRM package.
"""

# %%

import numpy as np
from matplotlib import pyplot as plt

from pyparrm import PARRM

###############################################################################
# Background
# ----------
# When delivering electrical stimulation to bioligical tissues for research or
# clinical purposes, it is often the case that electrophysiological recordings
# collected during this time are contaminated by stimulation artefacts. This
# contamination makes it difficult to analyse the underlying physiological or
# pathophysiological electrical activity.
#
# To this end, the Period-based Artefact Reconstruction and Removal Method
# (PARRM) was developed, enabling the removal of stimulation arefacts from
# electrophysiological recordings in a robust and computationally cheap manner
# :footcite:`DastinEtAl2021`.
#
# To demonstrate how PARRM can be used to remove stimulation arfectacts from
# data, we will start by loading some example data. This is the same example
# data used in the MATLAB implementation of the method
# (https://github.com/neuromotion/PARRM), consisting of a single channel
# with 19,130 timepoints, a sampling frequency of 200 Hz, and containing
# stimulation artefacts with a frequency of 150 Hz.

# %%

data = np.load("example_data.npy")
sampling_freq = 200  # Hz
artefact_freq = 150  # Hz

print(
    f"`data` has shape: ({data.shape[0]} channel(s), {data.shape[1]} "
    "timepoints)"
)

###############################################################################
# Finding the period of the stimulation artefacts
# -----------------------------------------------
# Having loaded the example data, we can now find the period of the stimulation
# artefacts, which we require to remove them from the data. Having imported the
# PARRM object, we initialise it, providing the data, its sampling frequency,
# and the stimulation frequency.
#
# After this, we find the period of the artefact using the ``find_period``
# method. By default, all timepoints of the data will be used for this, and the
# initial guess of the period will be taken as the sampling frequency divided
# by the artefact frequency. The settings for finding the period can be
# specified in the method call, however the default settings should suffice as
# a starting point for period estimation.
#
# The period is found using a grid search, with the goal of minimising the mean
# squared error between the data and the best fitting sinusoidal harmonics of
# the period found with linear regression.

# %%

parrm = PARRM(
    data=data, sampling_freq=sampling_freq, artefact_freq=artefact_freq
)
parrm.find_period()

print(f"Estimated artefact period: {parrm.period:.4f}")

###############################################################################
# Creating the filter and removing the artefacts
# ----------------------------------------------
# Now that we have an estimate of the artefact period, we can design a filter
# to remove it from the data using the ``create_filter`` method. When creating
# the filter, there are four key parameters.
#
# First is the size of the filter window, specified with the
# ``filter_half_width`` parameter. This should be chosen based on the
# timescale of which the artefact shape varies. If no such timescale is known,
# the power spectra of the filtered data can be inspected and the size of the
# filter window tuned until artefact-related peaks are sufficiently attenuated.
#
# Next is the number of samples to omit from the centre of the filter window,
# specified with the ``omit_n_samples`` parameter. This parameter serves to
# control for overfitting to features of interest in the underlying data. For
# instance, if there is a physiological signal of interest known to occur on a
# particular timescale, an appropriate number of samples should be omitted
# according to this range of time.
#
# Another parameter that can be tweaked is the direction considered when
# building the filter, determined by the ``filter_direction`` parameter. This
# can be used to control whether the filter window takes only previous samples,
# future samples, or both previous and future samples into account, based on
# their position relative to the centre of the filter window.
#
# Finally, the period window size can also be specified using the
# ``period_half_width`` parameter. The size of this window should be based on
# changes in the waveform of the artefact, which can be estimated by plotting
# the data on the timescale of the period and identifying the timescale over
# which features remain fairly constant. This parameter controls which samples
# are combined on the timescale of the period.
#
# Here, we specify that the filter should have a half-width of 2,000 samples,
# ignoring those 20 samples adjacent to the centre of the filter window,
# considering samples both before and after the centre of the filter window,
# and finally using a half-width of 0.01 samples in the period space.
#
# Once the filter has been created, it can be applied using the ``filter_data``
# method, which returns the artefact-free data. The filter itself, as well as a
# copy of the filtered data, can be accessed from the ``filter`` and
# ``filtered_data`` attributes, respectfully.

# %%

parrm.create_filter(
    filter_half_width=2000,
    omit_n_samples=20,
    filter_direction="both",
    period_half_width=0.01,
)
filtered_data = parrm.filter_data()

# comparison to true artefact free data
artefact_free = np.load("example_data_artefact_free.npy")
start = 598  # same start time as MATLAB example
end = 1011  # same end time as MATLAB example
times = np.arange(start, end) / sampling_freq

fig, axis = plt.subplots(1, 1)
inset_axis = axis.inset_axes((0.12, 0.6, 0.5, 0.35))

# main plot
axis.plot(
    times, data[0, start:end], color="black", alpha=0.2, label="Unfiltered"
)
axis.plot(
    times, artefact_free[0, start:end], linewidth=3, label="Artefact Free"
)
axis.plot(times, filtered_data[0, start:end], label="Filtered (PyPARRM)")
axis.legend()
axis.set_xlabel("Time (s)")
axis.set_ylabel("Amplitude (mV)")
axis.set_title("PARRM Filtered vs. Artefact Free")

# inset plot
inset_axis.plot(times[:50], artefact_free[0, start : start + 50], linewidth=3)
inset_axis.plot(times[:50], filtered_data[0, start : start + 50])
axis.indicate_inset_zoom(inset_axis, edgecolor="black", alpha=0.4)
inset_axis.patch.set_alpha(0.7)

fig.tight_layout()
fig.show()

###############################################################################
# Python vs. MATLAB implementation differences
# --------------------------------------------
# Whilst the Python and MATLAB implementations of PARRM give extremely similar
# results, rounding errors in floating point numbers mean that the results are
# not perfectly identical. Across the entire ~100 second duration of this
# recording, however, the total deviation between the implementations is only
# around 0.7 mV. In practice, both implementations are suitable for identifying
# and removing stimulation artefacts from electrophysiological recordings.

# %%

# filtered data computed in MATLAB
matlab_filtered = np.load("matlab_filtered.npy")

fig, axis = plt.subplots(1, 1)
inset_axis = axis.inset_axes((0.12, 0.6, 0.5, 0.35))

# main plot
axis.plot(
    times,
    matlab_filtered[0, start:end],
    linewidth=3,
    label="Filtered (MATLAB PARRM)",
)
axis.plot(times, filtered_data[0, start:end], label="Filtered (PyPARRM)")
axis.legend()
axis.set_xlabel("Time (s)")
axis.set_ylabel("Amplitude (mV)")
axis.set_title("PyPARRM vs. MATLAB PARRM")
ylim = axis.get_ylim()
axis.set_ylim(ylim[0], ylim[1] * 3)

# inset plot
inset_axis.plot(
    times[:50], matlab_filtered[0, start : start + 50], linewidth=3
)
inset_axis.plot(times[:50], filtered_data[0, start : start + 50])
axis.indicate_inset_zoom(inset_axis, edgecolor="black", alpha=0.4)

fig.tight_layout()
fig.show()

print(
    "Deviation between Python and MATLAB implementations: "
    f"{np.linalg.norm(filtered_data - matlab_filtered):.1f} mV"
)  # Euclidean distance across whole recording

###############################################################################
# References
# -----------------------------------------------------------------------------
# .. footbibliography::

# %%
