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
    def __init__(self, epochs):
        self.bias = 0
        self.weights = None
        self.epochs = epochs

    def activation_funtion(self, input):
        return np.heaviside(input,0)
        
    def predict(self,input):
        weighted_input = np.dot(input, self.weights) + self.bias
        return self.activation_funtion(weighted_input)
        
    def stochastic_gradient_descent(self, train, validate, x_test, y_test, learning_rate):
        lr = learning_rate
        number_indexes = train.shape[0]
        number_features = train.shape[1]
        self.weights = 2*np.random.rand(number_features)-1
        accuracy = []
        for epoch in range(self.epochs):
            indexes = np.random.permutation(number_indexes)
            for index in indexes:
                curr_pred = self.predict(train[index, :])
                self.weights = self.weights + lr*(validate[index]-curr_pred)*train[index, :]
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
            plt.savefig("./out/single_neuro_stochastic.png")
            plt.show()
        return self.weights, self.bias
    
    def batch_gradient_descent(self, train, validate):
        number_features = train.shape[1]
        self.weights = np.zeros(number_features)
        for i in range(self.epochs):
            for data in range(len(train)):
                weighted_input = np.dot(train,self.weights) + self.bias
                prediction = self.activation_funtion(weighted_input)
        return self.weights, self.bias
    
    def mini_gradient_descent(self, train, validate):
        number_features = train.shape[1]
        self.weights = np.zeros(number_features)
        for i in range(self.epochs):
            for data in range(len(train)):
                weighted_input = np.dot(train,self.weights) + self.bias
                prediction = self.activation_funtion(weighted_input)
        return self.weights, self.bias
            
    def train(self,training_type ,train, validate,x_test = None,y_test = None ,learning_rate = 0.001):
        if(training_type == 'stochastic'): return self.stochastic_gradient_descent(train, validate, x_test, y_test, learning_rate)
        if(training_type == 'batch'): return self.batch_gradient_descent(train, validate)
        if(training_type == 'mini_batch'): return self.mini_gradient_descent(train, validate)

if __name__ == "__main__":
    path = './data/misterious_data_1.txt'
    data = np.loadtxt(path)
    x = data[:,1:-1]
    y = data[:,0]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    model = Perceptron(300)
    model.train('stochastic',X_train,y_train,X_test,y_test,0.001)