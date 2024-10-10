from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from reward_regressors.base_regressor import BaseRegressor

class LinearRegressor(BaseRegressor):
    def __init__(self):
        super().__init__(LinearRegression())