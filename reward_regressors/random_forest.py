from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from reward_regressors.base_regressor import BaseRegressor

class RandomForest(BaseRegressor):
    def __init__(self, max_depth = 20, random_state = 0):
        super().__init__(RandomForestRegressor(max_depth=max_depth, random_state=random_state))

    def test():
        RandomForestRegressor.score()