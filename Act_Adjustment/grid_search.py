import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split, GridSearchCV

def regressor_from_data(path):
    MLP = MLPRegressor(max_iter=10000,activation='identity', hidden_layer_sizes=(100, 100, 100, 100, 100, 100), alpha=0.01, random_state=69, early_stopping=False)
    data = np.loadtxt(path)
    x = data[:,1:-1]
    y = data[:,0]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    MLP.fit(X_train, y_train)