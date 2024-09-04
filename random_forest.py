from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

class RandomForest:
    def __init__(self):
        self.reg = RandomForestRegressor(max_depth=20, random_state=0)

    def fit(self, arrays, scores):
        self.est = self.reg.fit(arrays, scores)

    def split_dataset(self, arrays, scores):
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
        train_test = train_test_split(arrays)
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

    def test(self, arrays, scores):
        train_X, train_Y, test_X, test_Y = self.split_dataset(arrays, scores)
        self.reg.fit(train_X, train_Y)
        accuracy = self.reg.score(test_X, test_Y)
        return accuracy