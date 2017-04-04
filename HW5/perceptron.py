from __future__ import print_function

import sys
import numpy as np

from abc import ABCMeta, abstractmethod
import data_loader


ETA = "eta"
GAMMA = "gamma"
ALPHA = "alpha"


class BaseLearner(object):
    __metaclass__ = ABCMeta

    def __init__(self, dimensions, parameters):
        self.parameters = parameters
        self.weight_vector = np.zeros(dimensions)
        self.theta = 0

    def predictionScore(self, x, y):
        y = 2*y - 1 #This converts {0,1} to {-1,1}
        return y * (np.dot(self.weight_vector, x) + self.theta)

    @abstractmethod
    def runIterationOnOneExample(self, x, y): pass

    def evaluate(self, data):
        num_instances = len(data)

        mistakes = 0.0
        for i in range(num_instances):
            if self.predictionScore(data[i][0], data[i][1]) <= 0:
                mistakes += 1

        return (num_instances-mistakes)/num_instances

    def train(self, data, num_rounds=100):
        num_instances = len(data)

        for iteration in range(num_rounds):
            update_count = 0
            for i in range(num_instances):
                update_count += self.runIterationOnOneExample(data[i][0], data[i][1])
            
        return self.evaluate(data)

    def train_with_learning_curve(self, data, num_rounds=100):
        num_instances = len(data)
        learning_curve_data = []
        for iteration in range(num_rounds):
            update_count = 0
            for i in range(num_instances):
                update_count += self.runIterationOnOneExample(data[i][0], data[i][1])
            learning_curve_data.append((iteration+1, self.evaluate(data)))
            
        return learning_curve_data




class Perceptron(BaseLearner):
    """
    Perceptron implementation:

    Usage:

    learner = Perceptron(weight_vector_dimension)
    """


    def __init__(self, dimensions):
        BaseLearner.__init__(self, dimensions, {})

    def runIterationOnOneExample(self, x, y):
        y = 2*y - 1 #This converts {0,1} to {-1, 1}
        if self.predictionScore(x, y) <= 0:
            self.weight_vector = self.weight_vector + (y * x)
            self.theta = self.theta + y
            return 1

        return 0

