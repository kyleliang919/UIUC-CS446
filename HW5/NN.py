'''
    A 3 layer neural network (input layer, hidden layer, output layer) for MNIST dataset
'''
import random
import numpy as np
import NN_functions as nf
from numpy import argmax
"""
Initializes and returns an instance of the NN class.
Parameters:
    -domain: one of ['mnist', 'circles']
    -batch_size: the number of examples to consider for each batch update
    -learning rate: the gradient descent learning rate
    -activation function: one of ['tanh', 'relu']
    -hidden_layer_width: the number of nodes to include in the hidden layer
"""
def create_NN(domain, batch_size, learning_rate, activation_function, hidden_layer_width):
    if domain == 'mnist':
        input_size = 784
    elif domain == 'circles':
        input_size = 2
    else:
        raise Exception('Domain must be one of [mnist, circles]')
    layer_sizes = [input_size, hidden_layer_width, 2] #input, hidden, output layer sizes
    iterations = 100
    if activation_function == 'tanh':
        input_activation_func = nf.tanh
        output_func = nf.tanh
        activation_func_derivative= nf.tanh_derivative
        output_func_derivative = nf.tanh_derivative
    elif activation_function == 'relu':
        input_activation_func = nf.relu
        output_func = nf.relu
        activation_func_derivative= nf.relu_derivative
        output_func_derivative = nf.relu_derivative
    else:
        raise Exception('Activation function not recognized: %s'%activation_function)
    return NN(batch_size, iterations, learning_rate, layer_sizes, activation_func=input_activation_func, 
                output_func=output_func, activation_func_derivative=activation_func_derivative,
                output_func_derivative=output_func_derivative)

class NN ():

    def __init__ (self, batch_size, iterations, learning_rate, sizes = [], activation_func = None, 
                  output_func = None, activation_func_derivative = None, output_func_derivative = None,
                  loss_func_derivative = None):
        self.layers = len(sizes)
        self.sizes = sizes
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        
        if activation_func is None:
            self.activation_func = nf.sigmoid
        else:
            self.activation_func = activation_func
        
        if output_func is None:
            self.output_func = nf.sigmoid
        else:
            self.output_func = output_func
        
        if activation_func_derivative is None:
            self.activation_func_derivative = nf.sigmoid_derivative
        else:
            self.activation_func_derivative = activation_func_derivative
        
        if output_func_derivative is None:
            self.output_func_derivative = nf.sigmoid_derivative
        else:
            self.output_func_derivative = output_func_derivative
            
        if loss_func_derivative is None:
            self.loss_func_derivative = nf.squared_loss_gradient
        else:
            self.loss_func_derivative = loss_func_derivative      
        
        
        #### Initialization of Parameters #####
        
        '''
            List of bias arrays per layer (starts from the first hidden layer)
        '''
        self.biases = [0.01*np.ones((y, 1)) for y in sizes[1:]]
        
        '''
            List of weight matrices per layer (starts from the (input - first hidden layer) combo)
            Each row of the matrix corresponds to input weights for 1 target hidden unit
        '''
        self.weights = [np.sqrt(2/x)*np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
    #endDef
    
    def transform_data(self, data, is_train):
        new_training_data = []
    
        for x,y in data:
            new_x = x.reshape((len(x),1))
            if is_train:
                new_y = np.zeros((2,1))
                new_y[y] = 1
            else:
                new_y = y
            new_training_data.append((new_x, new_y))
        return new_training_data
        
    def train (self, training_data):
        num_train = len(training_data)
        
        training_data = self.transform_data(training_data, True)
        print "Training NN:" 
        for j in xrange(self.iterations):
            if (j%10 == 0):
                print "\tTraining step %d"%j
            #random shuffling of the training dataset at the start of the iteration
            random.shuffle(training_data)
            
            mini_batches = [training_data[k : k + self.batch_size]
                for k in xrange(0, num_train, self.batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, self.learning_rate)
            #endFor
            
            ###Done making 1 pass over the entire dataset###
            
        #endFor
        
        transformed_training_data = []
        
        for (x,y) in training_data:
            true_label = argmax(y)
            transformed_training_data.append((x, true_label))
        
        return self.evaluate(transformed_training_data)
        
    #endDef

    def train_with_learning_curve(self, training_data):
        num_train = len(training_data)
        test_training_data = training_data
        training_data = self.transform_data(training_data, True)

        learning_curve_data = []        
        for j in xrange(self.iterations):
            if j%10 == 0:
                print "Training step %d"%j
            #random shuffling of the training dataset at the start of the iteration
            random.shuffle(training_data)
            
            mini_batches = [training_data[k : k + self.batch_size]
                for k in xrange(0, num_train, self.batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, self.learning_rate)
            #endFor
            
            ###Done making 1 pass over the entire dataset###
            learning_curve_data.append((j+1, self.evaluate(test_training_data)))
            print learning_curve_data[-1]
            
        #endFor
        
        transformed_training_data = []
        
        
        return learning_curve_data
        


    def update_mini_batch (self, mini_batch, l_rate):
        grad_b_accumulated = [np.zeros(b.shape) for b in self.biases]
        grad_w_accumulated = [np.zeros(w.shape) for w in self.weights]
        
        #Accumulate the gradients over the mini batches
        for x, y in mini_batch:
            #use backpropagation to calculate the gradients
            grad_b, grad_w = self.backprop(x, y)
            
            grad_b_accumulated = [nb + dnb for nb, dnb in zip(grad_b_accumulated, grad_b)]
            grad_w_accumulated = [nw + dnw for nw, dnw in zip(grad_w_accumulated, grad_w)]
        #endFor
        
        #Update the parameters
        self.weights = [w - (l_rate/len(mini_batch)) * grad_w
                        for w, grad_w in zip(self.weights, grad_w_accumulated)]
        
        self.biases = [b - (l_rate/len(mini_batch)) * grad_b
                       for b, grad_b in zip(self.biases, grad_b_accumulated)]
        
    #endDef
    

    def feedforward (self, x):
        a = x
        
        for b, w in zip(self.biases[: -1], self.weights[: -1]):
            a = self.activation_func(np.dot(w, a) + b)
        #endFor
        
        b_last = self.biases[-1]
        w_last = self.weights[-1]
            
        y = self.output_func(np.dot(w_last, a) + b_last)
        
        return y

    #endDef


    def backprop (self, x, y):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        
        ####### Forward Pass #######
        a = x
        As = [a] # list to store all the activations, layer by layer
        Zs = [] # list to store all the z vectors, layer by layer
        
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, a) + b
            Zs.append(z)
            a = self.activation_func(z)
            As.append(a)
        #endfor
        
        b_last = self.biases[-1]
        w_last = self.weights[-1]
        z = np.dot(w_last, a) + b_last
        Zs.append(z)
        a = self.output_func(z)
        As.append(a)
            
    ####### Backward Pass #######
        
    #Last layer
        delta = self.loss_func_derivative(As[-1], y) * self.output_func_derivative(Zs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, As[-2].transpose())
    
    #Back Passes -- Remember, 1st layer is the input layer
        for l in xrange(2, self.layers):
            z = Zs[-l]
            sp = self.activation_func_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, As[-l-1].transpose())
        #endFor
            
        return (grad_b, grad_w)
    
    #endDef

    def evaluate (self, test_data):
        n_test = len(test_data)
        test_data = self.transform_data(test_data, False)
        
        #O-1 error with the argmax value
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        
        correct = sum(int(x == y) for (x, y) in test_results)
        
        return correct*100/float(n_test) 
    
    #endDef
#endClass

