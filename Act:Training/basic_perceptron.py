# Equipo 3
# Abraham De Alba Franco
# Pablo Agustín Ortega-Kral
# Francisco José Ramírez Aldaco
# Alejandro Flores Madriz

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class Perceptron:
    def __init__(self, epochs, activation_funtion):
        self.bias = 0
        self.weights = None
        self.epochs = epochs
        self.function = activation_funtion

    def activation_funtion(self, input):
        if(self.function == 'relu'):
            return np.max(0.0, input)
        if(self.function == 'heaviside'):
            return np.heaviside(input,0)
        if(self.function == 'sigmoid'):
            return 1.0/(1.0 + np.exp(-input))
        
    def predict(self,input):
        weighted_input = np.dot(input, self.weights) + self.bias
        return self.activation_funtion(weighted_input)
        
    def stochastic_gradient_descent(self, train, validate, x_test, y_test, learning_rate):
        lr = learning_rate
        number_indexes = train.shape[0]
        number_features = train.shape[1]
        self.weights = 2*np.random(number_features)-1
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
                plt.plot( np.array(len(self.epochs)), accuracy)
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
            
    def train(self,training_type ,train, validate,x_test,y_test,learning_rate):
        if(training_type == 'stochastic'): return self.stochastic_gradient_descent(train, validate, x_test, y_test, learning_rate)
        if(training_type == 'batch'): return self.batch_gradient_descent(train, validate)
        if(training_type == 'mini_batch'): return self.mini_gradient_descent(train, validate)
        



if __name__ == "__main__":
    model = Perceptron(100,'sigmoid')
    model.train('stochastic')
    