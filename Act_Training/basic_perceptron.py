# Equipo 3
# Abraham De Alba Franco
# Pablo Agustín Ortega-Kral
# Francisco José Ramírez Aldaco
# Alejandro Flores Madriz

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron:
    def __init__(self, epochs, use_svm = False):
        self.bias = 0
        self.weights = None
        self.epochs = epochs
        self.use_svm = use_svm

    def activation_funtion(self, input):
        return np.heaviside(input,0)
        
    def predict(self,input):
        weighted_input = np.dot(input, self.weights) + self.bias
        return self.activation_funtion(weighted_input)
    
    def mse_cost(self,predition,truth):
        return np.mean((np.square(truth-predition)))
    
    def display_metrics(self,cost,accur,type):
        accuracy = np.array(accur)
        costs = np.array(cost)
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
        plt.plot(np.arange(self.epochs), costs)
        plt.xlabel("Epochs")
        plt.ylabel("Cost")
        if(self.use_svm):
            plt.title("Cost vs Epochs (SVM)")
            plt.savefig(f"./out/cost_{type}_svm.png")
        else: 
            plt.title("Cost vs Epochs")
            plt.savefig(f"./out/cost_{type}.png")
        plt.show()

    def stochastic_gradient_descent(self, train, validate, x_test, y_test, learning_rate):
        print(f"Starting training with {self.epochs} epochs and stochastic_gradient_descent")
        alpha = learning_rate
        number_indexes = train.shape[0]
        number_features = train.shape[1]
        self.weights = 2*np.random.rand(number_features)-1
        accuracy = []
        costs    = []
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
                acc = accuracy_score(y_test,test_prediction)
                accuracy.append(acc)
                costs.append(self.mse_cost(test_prediction,y_test))
        if x_test is not None and y_test is not None:
            self.display_metrics(costs,accuracy,'stochastic')
        return self.weights, self.bias
    
    def batch_gradient_descent(self, train, validate, x_test, y_test, learning_rate):
        alpha = learning_rate
        number_features = train.shape[1]
        self.weights = np.ones(shape=(number_features))
        accuracy = []
        costs    = []
        for epoch in range(self.epochs):
            pred = self.predict(train)
            if(self.use_svm):
                if validate*pred < 1:
                    self.weights = self.weights + alpha*validate*train
            elif(not self.use_svm):
                self.weights = self.weights + alpha* np.dot((validate-pred),train)
            if x_test is not None and y_test is not None:
                test_prediction = self.predict(x_test)
                acc = accuracy_score(y_test,test_prediction)
                accuracy.append(acc)
                costs.append(self.mse_cost(test_prediction,y_test))
        if x_test is not None and y_test is not None:
            self.display_metrics(costs,accuracy,'batch')
        return self.weights, self.bias
    
    def mini_gradient_descent(self, train, validate, x_test, y_test, learning_rate):
        alpha = learning_rate
        number_indexes = train.shape[0]
        number_features = train.shape[1]
        self.weights = 2*np.random.rand(number_features)-1
        accuracy = []
        for epoch in range(self.epochs):
            indexes = np.random.permutation(number_indexes)
            for index in indexes:
                curr_pred = self.predict(train[index, :])
                self.weights = self.weights + alpha*(validate[index]-curr_pred)*train[index, :]
            if x_test is not None and y_test is not None:
                test_prediction = self.predict(x_test)
                acc = accuracy_score(y_test,test_prediction)
                accuracy.append(acc)
        if x_test is not None and y_test is not None:
            #Display accuracy
            accuracy = np.array(accuracy)
            plt.plot(np.arange(self.epochs), accuracy)
            plt.title("Accuracy vs Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Acurracy")
            plt.savefig("./out/single_neuron_mini_batch.png")
            plt.show()
        return self.weights, self.bias
            
    def train(self,training_type ,train, validate,x_test = None,y_test = None ,learning_rate = 0.001):
        if(training_type == 'stochastic'): return self.stochastic_gradient_descent(train, validate, x_test, y_test, learning_rate)
        if(training_type == 'batch'): return self.batch_gradient_descent(train, validate, x_test, y_test, learning_rate)
        if(training_type == 'mini_batch'): return self.mini_gradient_descent(train, validate, x_test, y_test, learning_rate)

if __name__ == "__main__":
    path = './data/misterious_data_1.txt'
    data = np.loadtxt(path)
    x = data[:,1:-1]
    y = data[:,0]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    model = Perceptron(300,True)
    model.train('batch',X_train,y_train,X_test,y_test,0.001)