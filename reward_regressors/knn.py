from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from reward_regressors.base_regressor import BaseRegressor

class KNN(BaseRegressor):
    def __init__(self, n_neighbors=5, weights='uniform', leaf_size=30, p=2, n_jobs=None):
        super().__init__(KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, leaf_size=leaf_size, p=p, n_jobs=n_jobs))