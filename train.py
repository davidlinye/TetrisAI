from reward_regressors.random_forest import RandomForest
from reward_regressors.linear_regressor import LinearRegressor
from reward_regressors.knn import KNN
from reward_regressors.cnn import CNN
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import math
import joblib

class Train():
    def __init__(self, model):
        self.model = model

    def reshape(self, arrays):
        arrays = np.asarray(arrays)
        nsamples, nx, ny = arrays.shape
        reshaped_arrays = arrays.reshape((nsamples,nx*ny))
        return reshaped_arrays

    def nameformat(self, name):
        if name == "linear":
            return "Linear regressor predictions"
        elif name == "knn":
            return "K-nearest neighbours predictions"
        elif name == "cnn":
            return "Convolutional neural network predictions"
        else:
            return "Random forest predictions"
        
    def cnn_initialisation(self):
        hidden_layer_sizes=(100,)
        activation='relu'
        solver='adam'
        alpha=0.0001
        batch_size='auto'
        learning_rate='constant'
        learning_rate_init=0.001
        power_t=0.5
        max_iter=200
        shuffle=True
        random_state=None
        tol=0.0001
        verbose=False
        warm_start=False
        momentum=0.9
        nesterovs_momentum=True
        early_stopping=False
        validation_fraction=0.1
        beta_1=0.9
        beta_2=0.999
        epsilon=1e-08
        n_iter_no_change=10
        max_fun=15000
        return CNN(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol, verbose=verbose, warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping, validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change, max_fun=max_fun)

    def train(self, arrays, scores, testset_percentage = 0):
        boxplot = True
        if self.model:
            if self.model == "linear":
                reg = LinearRegressor()
            elif self.model == "knn":
                reg = KNN()
            elif self.model == "cnn":
                reg = self.cnn_initialisation()
            else:
                reg = RandomForest()
            reshaped_arrays = self.reshape(arrays)
            if testset_percentage <= 0:
                accuracy = reg.fit_and_score(reshaped_arrays, scores)
                print(accuracy)
            else:
                print("Generating data points")
                true_scores, pred_scores = reg.fit_and_generate_datapoints(arrays, scores, testset_percentage)
                
                # Plot the data and the regression line
                # x = [i for i in range(0, 21, 2)]
                # print(x)
                # xint = range(min(x), math.ceil(max(x))+1)
                name = str(self.nameformat(self.model))
                if boxplot:
                    name += " boxplot"
                fig = plt.figure(figsize=(8, 6))
                if boxplot:
                    plt.scatter(true_scores, pred_scores, color="aqua", label="Data points")
                else:
                    plt.scatter(true_scores, pred_scores, color="blue", label="Data points")
                plt.plot(true_scores, true_scores, color="red", linewidth=2, label="Regression line")
                plt.xticks(range(1,21))
                plt.title(f"{name} accuracy plot")
                plt.xlabel("truth score")
                plt.ylabel("predicted score")
                plt.legend()
                plt.grid(True)

                #statistics
                # print(true_scores)
                # print(pred_scores)
                # range(22) because 0 is not used, and 1 - 20 are, which are 21 total arrays
                x_values = [[] for _ in range(22)]
                for i in range(len(pred_scores)):
                    # for every entry, add value to list
                    x_values[true_scores[i] - 1].append(pred_scores[i])
                
                # ax = fig.add_axes([])

                if boxplot:
                    plt.boxplot(x_values)

                #for every list of an x value, calculate average
                averages = []
                # print(len(x))
                for i, x in enumerate(x_values):
                    if len(x) != 0:
                        average = sum(x) / len(x)
                        averages.append((i+1, average))
                print("averages:")
                print(np.array(averages))

                if not boxplot:
                    name = "non-boxplots/" + name

                # plt.savefig(f"graphs/{name}.png")
                # print(f"Figure saved to graphs/{name}.png")

                #export model to save time
                # joblib.dump(reg.reg, f"models/{self.model}.pkl")