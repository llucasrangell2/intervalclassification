# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BASE_COLORS
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler


data = pd.read_csv('iris.csv').values

X = data[:, :-1]
# X = data[:, 2].reshape(-1, 1)
y = data[:, -1]
X = MaxAbsScaler().fit(X).transform(X)
y = LabelEncoder().fit(y).transform(y)

colormap = np.array(list(BASE_COLORS.keys()))

plt.xlabel('x')
plt.ylabel('y')
for i in range(X.shape[1]):
    for j in range(i+1, X.shape[1]):
        plt.scatter(X[:, i],
                    X[:, j],
                    c=colormap[y])
        plt.show()
