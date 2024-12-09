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
import bisect

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
        hidden_layer_sizes=(20,)
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

    def train(self, arrays, scores, folder_location, testset_percentage = 0, forwards = 0):
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
                
                #split into two graphs
                true_scores_start = [t for t, p in zip(true_scores, pred_scores) if t < 40]
                pred_scores_start = [p for t, p in zip(true_scores, pred_scores) if t < 40]
                true_scores_end = [t for t, p in zip(true_scores, pred_scores) if t >= 70]
                pred_scores_end = [p for t, p in zip(true_scores, pred_scores) if t >= 70]

                name = str(self.nameformat(self.model))
                if boxplot:
                    name += " boxplot"
                name += " start"
                fig, ax1 = plt.subplots(figsize=(8, 6))
                if boxplot:
                    ax1.scatter(true_scores_start, pred_scores_start, color="aqua", label="Data points")
                else:
                    ax1.scatter(true_scores_start, pred_scores_start, color="blue", label="Data points")
                ax1.plot(true_scores_start, true_scores_start, color="red", linewidth=2, label="Regression line")
                ax1.set_title(f"{name} accuracy plot")
                ax1.set_xlabel("truth score")
                ax1.set_ylabel("predicted score")
                ax1.legend()
                ax1.grid(True)

                #statistics
                # print(true_scores)
                # print(pred_scores)
                # use sorted array to get the maximum value below 50 for graph x-ticks
                sorted_true_scores = sorted(true_scores)
                x_range = sorted_true_scores[bisect.bisect_left(sorted_true_scores, 50) - 1] + 1
                # print(sorted_true_scores)
                # print(bisect.bisect_left(sorted_true_scores, 50))
                # print(x_range)
                x_values = [[] for _ in range(x_range)]
                for i in range(len(pred_scores_start)):
                    # for every entry, add value to list
                    x_values[true_scores_start[i] - 1].append(pred_scores_start[i])
                
                ax1.set_xlim(0, x_range)
                # ax1.set_xticks([x for x in range(1,40)])
                # ax = fig.add_axes([])

                if boxplot:
                    # ax2 = ax1.twinx()
                    ax1.boxplot(x_values)
                    # ax2.set_xticks(ax1.get_xticks())
                    # ax2.set_xticklabels(ax1.get_xticks())

                # plt.tight_layout()

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

                plt.savefig(f"graphs/{name}_{folder_location}.png")
                print(f"Figure saved to graphs/{name}_{folder_location}.png")


                if forwards == 1:

                    name = str(self.nameformat(self.model))
                    if boxplot:
                        name += " boxplot"
                    name += " end"
                    fig, ax3 = plt.subplots(clear=True, figsize=(8, 6))
                    if boxplot:
                        ax3.scatter(true_scores_end, pred_scores_end, color="aqua", label="Data points")
                    else:
                        ax3.scatter(true_scores_end, pred_scores_end, color="blue", label="Data points")
                    ax3.plot(true_scores_end, true_scores_end, color="red", linewidth=2, label="Regression line")
                    ax3.set_title(f"{name} accuracy plot end")
                    ax3.set_xlabel("truth score")
                    ax3.set_ylabel("predicted score")
                    ax3.legend()
                    ax3.grid(True)

                    #statistics
                    # print(true_scores)
                    # print(pred_scores)
                    # use sorted array to get the minimum value above 50 for graph x-ticks
                    x_range = sorted_true_scores[bisect.bisect_right(sorted_true_scores, 50)] - 1

                    # temporary hard-coded fix
                    if x_range < 69:
                        x_range = 69

                    x_values = [[] for _ in range(x_range, 101)]
                    for i in range(len(pred_scores_end)):
                        # for every entry, add value to list
                        # print(true_scores_end[i])
                        x_values[true_scores_end[i] - x_range - 1].append(pred_scores_end[i])
                    # ax = fig.add_axes([])

                    ax3.set_xlim(x_range, 100)
                    # ax3.set_xticks([x for x in range(70, 101)])

                    if boxplot:
                        # ax4 = ax3.twinx()
                        print(len(x_values))
                        print(range(x_range, len(x_values) + x_range))
                        ax3.boxplot(x_values, positions=[x for x in range(x_range, len(x_values) + x_range) ])
                        # ax4.set_xticks(ax3.get_xticks())
                        # ax4.set_xticklabels(ax3.get_xticks())

                    # plt.tight_layout()
                    

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

                    plt.savefig(f"graphs/{name}_{folder_location}.png")
                    print(f"Figure saved to graphs/{name}_{folder_location}.png")

                # export model to save time
                joblib.dump(reg.reg, f"models/{self.model}.pkl")