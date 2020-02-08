# -*- coding: utf-8 -*-

# import pandas as pd
import numpy as np
# from sklearn.preprocessing import LabelEncoder, MaxAbsScaler


class ICDClassifier(object):
    def __init__(self, phi=1.618):
        self.phi = phi

    def fit(self, X, y):
        self.intervals = {c: tuple([[] for x in range(X.shape[1])])
                          for c in np.unique(y)}
        print(self.intervals)

        for c in np.unique(y):
            p = [x for x in range(X.shape[0]) if y[x] == c]
            for j in range(X.shape[1]):
                p = sorted(p, key=lambda x: X[x, j])
                print(f'p:{p}')
                a, b = X[p[0], j], X[p[1], j]
                for i in range(len(p)-1):
                    print(f'j:{j} i:{p[i]}, c:{c}, y:{y[p[i]]}, \
X[{p[i]}, {j}]:{X[p[i], j]}')
                    for d in range(i+2, len(p)):
                        pass

                self.intervals[c][j].append((a, b))
        return self.intervals

    def predict(self):
        pass


# data = pd.read_csv('iris.csv').values

# X = data[:, :-1]
# X = data[:, 2].reshape(-1, 1)
# X = MaxAbsScaler().fit(X).transform(X)
# y = data[:, -1]
# y = LabelEncoder().fit(y).transform(y)

X = np.array([[5.1, 3.5, 1.4, 0.2],
              [4.9, 3.0, 1.4, 0.2],
              [7.0, 3.2, 4.7, 1.4],
              [6.4, 3.2, 4.5, 1.5],
              [6.9, 3.1, 4.9, 1.5],
              [5.5, 2.3, 4.0, 1.3],
              [6.5, 2.8, 4.6, 1.5]])

y = np.array([0, 0, 1, 1, 1, 1, 1]).reshape(-1, 1)

clf = ICDClassifier()

intervals = clf.fit(X, y)
