import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import load_wine
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

def train_MLP_classifier(neurons_per_layer,layers):
    size = (neurons_per_layer,)*layers
    MLP = MLPClassifier(hidden_layer_sizes=size,max_iter = 300,activation = "relu", solver = 'adam')
    data, target = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    MLP.fit(X_train, y_train)
    accuracy = kfold_accuracy(data, target, MLP)
    print(f"MLP with {layers} layers/ {neurons_per_layer} neurons each")
    print('Accuracy: {:.2f}'.format(accuracy))
    return accuracy

if __name__ == "__main__":
    neurons = 5
    test_sizes = [4,6,8,10,30,40,50,80,90,100,120]
    acc = []
    for size in test_sizes:
        accuracy = train_MLP_classifier(neurons,size)
        acc.append(accuracy)
    plt.figure(1)
    plt.plot(test_sizes, acc)
    plt.xlabel('Layers')
    plt.ylabel("Acurracy")
    plt.title(f"Layers vs Accuracy ({neurons} neurons per layer)")
    plt.savefig(f"./out/{neurons}_neurons.png")

        



