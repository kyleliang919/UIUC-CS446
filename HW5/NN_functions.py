'''
Library of functions for Neural Networks
'''
import numpy as np 
 

"""
Implements the gradient of the squared loss function
Parameters:
    -output_activations: a numpy ndarray of shape (2,1) containing the values of the output layer
    -y: a numpy ndarray of shape (2,1) containing the correct values for the output layer

Returns:
    a float value representing the gradient of the error with respect to the value of the output
    activations. This should be a numpy ndarray with the same shape as the inputs
"""
def squared_loss_gradient (output_activations, y):
    #IMPLEMENT THIS!
    
    return -(y-output_activations)
#endDef

def sigmoid (z):
    return 1.0/(1.0 + np.exp(-z))
#endDef

def sigmoid_derivative (z):
    return sigmoid(z) * (1-sigmoid(z))
#endDef

def relu (z):
    return np.maximum(0, z)
#endDef

"""
Implements the derivative of the relu function, evaluated element-wise over the input vector
Parameters:
    -z: a numpy nd-array containing the output values of a layer of the neural network

Returns:
    a numpy nd-array having the same shape as the input where each index represents the derivative
    of the relu function applied to the corresponding coordinate of the input
"""
def relu_derivative (z):
    #IMPLEMENT THIS!
    output=np.zeros(z.shape)
    output[z>0]=1
    return output
#endDef    

def tanh (z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
#endDef

def tanh_derivative (z):
    return np.ones(z.shape) - np.square(tanh(z))
#endDef
