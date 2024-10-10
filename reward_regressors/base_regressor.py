from sklearn.model_selection import train_test_split
import numpy as np
import random

class BaseRegressor:
    def __init__(self, model):
        self.reg = model

    def fit(self, arrays, scores):
        print("fitting")
        self.est = self.reg.fit(arrays, scores)

    def fit_and_generate_datapoints(self, arrays, scores, testset_percentage):
        train_X, train_Y, test_X, test_Y = self.split_dataset(arrays, scores, testset_percentage)
        # print(train_X.shape)
        # print(test_X.shape)
        self.reg.fit(train_X, train_Y)
        
        true_scores = test_Y
        # print(test_X.shape)
        pred_scores = self.predict(test_X)
        # for i in range(int(len(test_X))):
        #     if i % 1000 == 0:
        #         print("Data point ", i)
        #     true_scores.append(test_Y[i])
        #     pred_score = self.predict(test_X[i])
        #     pred_scores.append(pred_score)
        return true_scores, pred_scores
    
    def predict(self, array):
        return self.reg.predict(array)

    def split_dataset(self, arrays, scores, testset_percentage = 25):
        arrays = list(arrays)
        for i in range(len(arrays)):
            # if i == 0:
            #     print(arrays[i].shape)
            #     print(arrays[i])
            arrays[i] = np.append(arrays[i], scores[i])
            # if i == 0:
            #     print(arrays[i].shape)
            #     print(arrays[i])
        
        arrays = np.asarray(arrays)
        train_test = train_test_split(arrays, test_size=float(testset_percentage)/100)
        train_Y = []
        test_Y = []

        # print(train_test[0].shape)
        train_test[0] = list(train_test[0])
        # print(len(train_test[0]))
        # print(train_test[0][0])
        train_test[1] = list(train_test[1])

        for i in range(len(train_test[0])):
            train_Y.append(train_test[0][i][-1])
            train_test[0][i] = train_test[0][i][:-1]

        for i in range(len(train_test[1])):
            test_Y.append(train_test[1][i][-1])
            train_test[1][i] = train_test[1][i][:-1]

        train_test[0] = np.asarray(train_test[0])
        train_test[1] = np.asarray(train_test[1])

        train_X = train_test[0]
        test_X = train_test[1]
        return train_X, train_Y, test_X, test_Y

    def fit_and_score(self, arrays, scores):
        train_X, train_Y, test_X, test_Y = self.split_dataset(arrays, scores)
        self.reg.fit(train_X, train_Y)
        accuracy = self.reg.score(test_X, test_Y)
        return accuracy