import numpy as np

import mne

from matplotlib import pyplot as plt

PATH_STIM_ON = r"C:\Users\ICN_admin\Documents\Datasets\Berlin\sub-002\ses-EcogLfpMedOff03\ieeg\sub-002_ses-EcogLfpMedOff03_task-SelfpacedRotationR_acq-StimOn_run-01_ieeg.vhdr"


raw = mne.io.read_raw_brainvision(PATH_STIM_ON, preload=True)
ch_names = [c for c in raw.ch_names if "ECOG" in c or "SQUARED_ROTATION" in c]
raw_p = raw.pick(picks=ch_names)
data = np.expand_dims(raw_p.get_data()[1,:20*int(raw_p.info["sfreq"])], axis=0)

raw = mne.io.read_raw_brainvision(PATH_STIM_ON, preload=True)
ch_names = [c for c in raw.ch_names if "LFP" in c]
raw_p = raw.pick(picks=ch_names)
raw.pick(picks="LFP_L_3_STN_BS")
raw.plot_psd()
data = np.expand_dims(raw.get_data()[0,:20*int(raw.info["sfreq"])], axis=0)

np.save("examples\\lfp_stn_stim_on_data.npy", data)
