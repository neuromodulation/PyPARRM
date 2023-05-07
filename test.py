from pyparrm import PARRM
import numpy as np

data = np.load("examples\\example_data.npy")

data = np.vstack((data, np.random.rand(data.shape[1])))

parrm = PARRM(data, 200, 150)
parrm.find_period()
parrm.explore_filter_params(5, 10)
