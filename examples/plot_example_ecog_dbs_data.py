"""
===========================================================
ECoG Stimulation Artifac removal using PyPARRM 
===========================================================

In this example PARRM is demonstrated to remove a deep brain stimulation
artifact from a cortical recording.
"""

# %%

import numpy as np
from matplotlib import pyplot as plt

from pyparrm import PARRM

###############################################################################
# Background
# ----------
# Deep Brain Stimulation induces a strong artifact at the stimulation frequency
# and additional harmonics that deteriorate the signal quality and lower signal
# to noise ratio. In this example invasive recordings from a Parkinson's 
# patient is analyzed and the effect of PARRM artifact removal demonstrated.

# %%
data_ecog = np.load("ecog_stim_on_data.npy")

sampling_freq = 1376  # Hz
artefact_freq = 130 # Hz

###############################################################################
# Parametrizing PARRM for cortical signals
# ----------------------------------------
# The optimal filter parameters are estimated using the 
# :meth:`explore_filter_params
# <pyparrm.parrm.PARRM.explore_filter_params>` method.

# %%

parrm_ecog = PARRM(
    data=data_ecog,
    sampling_freq=sampling_freq,
    artefact_freq=artefact_freq,
    verbose=False,
)
parrm_ecog.find_period()

parrm_ecog.explore_filter_params(freq_res=5)

parrm_ecog.create_filter(
    filter_direction="both",
)

filtered_data_ecog = parrm_ecog.filter_data()

###############################################################################
# Investigating the artifact removal effect
# ----------------------------------------
# We can then visualize the cleaned signal, and see on different time scales
# that the correct ECOG artifact representation is removed.  

# %%
time_ = np.arange(0, data_ecog.shape[1]/sampling_freq, 1/sampling_freq)

plt.figure(figsize=(12,4))

plt.subplot(121)
plt.plot(time_[sampling_freq*10:], data_ecog[0, sampling_freq*10:], label="Unfiltered")
plt.plot(time_[sampling_freq*10:], filtered_data_ecog[0,sampling_freq*10:], label="Filtered (PyPARRM)")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title("Cortical Filtered Signal")

plt.subplot(122)
plt.plot(time_[sampling_freq*10:sampling_freq*10+sampling_freq], data_ecog[0,sampling_freq*10:sampling_freq*10+sampling_freq], label="Unfiltered")
plt.plot(time_[sampling_freq*10:sampling_freq*10+sampling_freq], filtered_data_ecog[0,sampling_freq*10:sampling_freq*10+sampling_freq], label="Filtered (PyPARRM)")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title("Cortical Filtered Signal")

plt.tight_layout()

plt.show()

# %%
