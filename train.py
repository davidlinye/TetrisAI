from random_forest import RandomForest
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class Train():
    def __init__(self, model):
        self.model = model

    def reshape(self, arrays):
        arrays = np.asarray(arrays)
        nsamples, nx, ny = arrays.shape
        reshaped_arrays = arrays.reshape((nsamples,nx*ny))
        return reshaped_arrays

    def train(self, arrays, scores):
        if self.model == "rf":
            reg = RandomForest()
            reshaped_arrays = self.reshape(arrays)
            # reg.fit(reshaped_arrays, scores)
            accuracy = reg.test(reshaped_arrays, scores)
            print(accuracy)