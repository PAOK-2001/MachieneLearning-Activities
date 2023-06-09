import pandas as pd
import numpy as np
from sklearn.utils.fixes import loguniform
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, RepeatedKFold


if __name__ == "__main__":
    regressor = MLPRegressor(max_iter=10000,activation='identity', hidden_layer_sizes=(100, 100, 100, 100, 100, 100), alpha=0.01, random_state=69, early_stopping=False)
    data = np.loadtxt('./data/misterious_data_5.txt')
    x = data[:,1:-1]
    y = data[:,0]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    # Baseline
    regressor.fit(X_train,y_train)
    print("--- BASELINE ---")
    print("R2: {:.2f}".format(regressor.score(X_test, y_test)))
    pred = regressor.predict(X_test)
    print("M.S.E: {:.2f}".format(mean_squared_error(y_test, pred)))
    param_grid = [
        {'activation': ['identity', 'logistic', 'tanh', 'relu']},
        {'alpha': [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]},
        {'hidden_layer_sizes': [(2,2,2),(100,100,100),(10,10),(5,5,5,5)]},
        {'learning_rate': ['constant', 'invscaling', 'adaptive']}
    ]
    cross_validator = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    optimizer = GridSearchCV(estimator=regressor, param_grid= param_grid, n_jobs=-1, cv=cross_validator, scoring="neg_mean_squared_error")
    hyper_parameters = optimizer.fit(X_train, y_train)
    best_regressor = hyper_parameters.best_estimator_
    print("--- HYPER-PARAMETER TUNNED ---")
    print("R2: {:.2f}".format(best_regressor.score(X_test, y_test)))
    pred = best_regressor.predict(X_test)
    print("M.S.E: {:.2f}".format(mean_squared_error(y_test, pred)))


