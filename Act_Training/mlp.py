import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold

def kfold_accuracy(x, y, mlp):
    kf = KFold()
    accuracy = 0
    for i, (train_index, test_index) in enumerate(kf.split(x)):
        X_test = [x[index] for index in test_index]
        y_test = [y[index] for index in test_index]
        y_pred = mlp.predict(X_test)
        accuracy += accuracy_score(y_test, y_pred)
    accuracy /= kf.get_n_splits()
    return accuracy

def train_MLP_classifier_from_dataset(path):
    MLP = MLPClassifier(hidden_layer_sizes=(100,100),max_iter = 100,activation = "relu", solver = 'adam')
    data = np.loadtxt(path)
    x = data[:,1:-1]
    y = data[:,0]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    MLP.fit(X_train, y_train)
    accuracy = kfold_accuracy(x, y, MLP)
    print('Accuracy: {:.2f}'.format(accuracy))

def train_MLP_regression_diabetes():
    MLP = MLPRegressor(max_iter=10000,activation='identity', hidden_layer_sizes=(100, 100, 100, 100, 100, 100), alpha=0.01, random_state=69, early_stopping=False)
    data, target = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    MLP.fit(X_train, y_train)
    pred = MLP.predict(X_test)
    r_squared = MLP.score(X_test, y_test)
    mse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"R2: {r_squared}")
    print(f"MSE: {mse}")

#ex1
print("Part 1 results:")
res_ex1 = train_MLP_classifier_from_dataset('./data/misterious_data_1.txt')
#ex2
print("Part 2 results:")
res_ex2 = train_MLP_classifier_from_dataset('./data/misterious_data_4.txt')
#ex3
print("Part 3 results:")
res_ex3 = train_MLP_regression_diabetes()
