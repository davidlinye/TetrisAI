from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from reward_regressors.base_regressor import BaseRegressor

class CNN(BaseRegressor):
    def __init__(self, 
                 hidden_layer_sizes=(100,), 
                 activation='relu', 
                 solver='adam', 
                 alpha=0.0001, 
                 batch_size='auto', 
                 learning_rate='constant', 
                 learning_rate_init=0.001, 
                 power_t=0.5, 
                 max_iter=200, 
                 shuffle=True, 
                 random_state=None, 
                 tol=0.0001, 
                 verbose=False, 
                 warm_start=False, 
                 momentum=0.9, 
                 nesterovs_momentum=True, 
                 early_stopping=False, 
                 validation_fraction=0.1, 
                 beta_1=0.9, 
                 beta_2=0.999, 
                 epsilon=1e-08, 
                 n_iter_no_change=10, 
                 max_fun=15000):
        super().__init__(MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol, verbose=verbose, warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping, validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change, max_fun=max_fun))