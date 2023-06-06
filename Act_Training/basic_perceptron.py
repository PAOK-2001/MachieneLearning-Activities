# Equipo 3
# Abraham De Alba Franco
# Pablo Agustín Ortega-Kral
# Francisco José Ramírez Aldaco
# Alejandro Flores Madriz

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

class Perceptron:
    def __init__(self, epochs, use_svm = False):
        self.weights = None
        self.epochs = epochs
        self.use_svm = use_svm
        
    def predict(self,input):
        weighted_input = np.dot(input, self.weights)
        return np.sign(weighted_input)
    
    def kfold_accuracy(self, x, y):
        kf = KFold()
        accuracy = 0
        for i, (train_index, test_index) in enumerate(kf.split(x)):
            X_test = [x[index] for index in test_index]
            y_test = [y[index] for index in test_index]
            y_pred = self.predict(X_test)
            accuracy += accuracy_score(y_test, y_pred)
        accuracy = accuracy/kf.get_n_splits()
        return accuracy
    
    def display_metrics(self,err,accur,type):
        accuracy = np.array(accur)
        plt.figure(1)
        plt.plot(np.arange(self.epochs), accuracy)
        plt.xlabel("Epochs")
        plt.ylabel("Acurracy")
        if(self.use_svm):
            plt.title("Accuracy vs Epochs (SVM)")
            plt.savefig(f"./out/single_neuron_{type}_svm.png")
        else: 
            plt.title("Accuracy vs Epochs")
            plt.savefig(f"./out/single_neuron_{type}.png")
        
        plt.figure(2)
        err = np.array(err)
        plt.plot(np.arange(self.epochs), err)
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        if(self.use_svm):
            plt.title("Error vs Epochs (SVM)")
            plt.savefig(f"./out/error_{type}_svm.png")
        else: 
            plt.title("Error vs Epochs")
            plt.savefig(f"./out/error_{type}.png")
        plt.show()

    def stochastic_gradient_descent(self, train, validate, x_test, y_test, learning_rate):
        print(f"Starting training with {self.epochs} epochs and stochastic_gradient_descent")
        alpha = learning_rate
        number_indexes = train.shape[0]
        number_features = train.shape[1]
        self.weights = 2*np.random.rand(number_features)-1
        accuracy = []
        error = []
        for epoch in range(self.epochs):
            indexes = np.random.permutation(number_indexes)
            for index in indexes:
                curr_pred = self.predict(train[index, :])
                if(self.use_svm):
                    if validate[index]*curr_pred < 1:
                        self.weights = self.weights + alpha*validate[index]*train[index, :]
                elif(not self.use_svm):
                    self.weights = self.weights + alpha*(validate[index]-curr_pred)*train[index, :]
            if x_test is not None and y_test is not None:
                test_prediction = self.predict(x_test)
                acc = self.kfold_accuracy(x_test,y_test)
                accuracy.append(acc)
                error.append(1-acc)
        if x_test is not None and y_test is not None:
            self.display_metrics(error,accuracy,'stochastic')
        return self.weights
    
    def batch_gradient_descent(self, train, validate, x_test, y_test, learning_rate):
        print(f"Starting training with {self.epochs} epochs and batch_gradient_descent")
        alpha = learning_rate
        number_indexes = train.shape[0]
        number_features = train.shape[1]
        self.weights = np.ones(shape=(number_features))
        accuracy = []
        error = []
        for epoch in range(self.epochs):
            w = 0
            for index in range(number_indexes):
                pred = self.predict(train[index, :])
                if(self.use_svm):
                    if validate[index]*pred < 1:
                        w += (validate[index]*train[index, :])
                else:
                    w += ((validate[index]-pred)*train[index, :])
            self.weights += alpha*w            
            if x_test is not None and y_test is not None:
                test_prediction = self.predict(x_test)
                acc = self.kfold_accuracy(x_test,y_test)
                accuracy.append(acc)
                error.append(1-acc)
        if x_test is not None and y_test is not None:
            self.display_metrics(error,accuracy,'batch')
        return self.weights
    
    def mini_gradient_descent(self, train, validate, x_test, y_test, learning_rate, batch_size):
        alpha = learning_rate
        number_indexes = train.shape[0]
        number_features = train.shape[1]
        self.weights = np.ones(shape=(number_features))
        accuracy = []
        error = []
        n = 0
        for epoch in range(self.epochs):
            w = 0
            for index in range(n,n+batch_size):
                pred = self.predict(train[index, :])
                if (self.use_svm):
                    if validate[index]*pred < 1:
                        w += (validate[index]*train[index, :])
                else:
                    w += ((validate[index]-pred)*train[index, :])
            self.weights += alpha*w
            if x_test is not None and y_test is not None:
                test_prediction = self.predict(x_test)
                acc = self.kfold_accuracy(x_test,y_test)
                accuracy.append(acc)
                error.append(1-acc)
            n = n + batch_size if n + batch_size > number_indexes else 0
        if x_test is not None and y_test is not None:
            self.display_metrics(error,accuracy,'mini_batch')
        return self.weights
            
    def train(self,training_type ,train, validate,x_test = None,y_test = None ,learning_rate = 0.001):
        if(training_type == 'stochastic'): return self.stochastic_gradient_descent(train, validate, x_test, y_test, learning_rate)
        if(training_type == 'batch'): return self.batch_gradient_descent(train, validate, x_test, y_test, learning_rate)
        if(training_type == 'mini_batch'): return self.mini_gradient_descent(train, validate, x_test, y_test, learning_rate, batch_size=111)

if __name__ == "__main__":
    path = './data/misterious_data_1.txt'
    data = np.loadtxt(path)
    x = data[:,1:-1]
    y = data[:,0]
    for i in range(len(y)):
        y[i] = (y[i] - 1.5)*2
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = Perceptron(400,False)
    model.train('stochastic',X_train,y_train,X_test,y_test,0.001) 