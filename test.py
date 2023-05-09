from pyparrm import PARRM
import numpy as np
import time

if __name__ == "__main__":
    data = np.load("examples\\example_data.npy")

    # data = np.vstack((data, np.random.rand(data.shape[1])))

    parrm = PARRM(data, sampling_freq=200, artefact_freq=150)
    start = time.time()
    parrm.find_period(n_jobs=10)
    print(time.time() - start)
    parrm.explore_filter_params(freq_res=1, n_jobs=10)
