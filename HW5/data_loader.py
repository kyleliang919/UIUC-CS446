import numpy as np

"""
This method returns a list of tuples, representing the data points.
The first entry of each tuple is a numpy ndarray containing the pixel values
The second entry is an integer representing the label.
"""
def load_circle_data():
    train_data = []
    test_data = []
    with open('data/circles.train.txt', 'r') as fin:
        for line in fin:
            data = line.split()
            label = int(data[2])
            #label = np.zeros((2,1))
            #label[int(data[2])] = 1
            point = np.zeros(2)
            point[0] = float(data[0])
            point[1] = float(data[1])
            train_data.append((point, label))
    with open('data/circles.test.txt', 'r') as fin:
        for line in fin:
            data = line.split()
            label = int(data[2])
            point = np.zeros(2)
            point[0] = float(data[0])
            point[1] = float(data[1])
            test_data.append((point, label))

    return (train_data, test_data)

"""
This method returns a list of tuples, representing the data points.
The first entry of each tuple is a numpy ndarray containing the pixel values
The second entry of each tuple  is the label - it equals 0 if the digit is a 3, 
    and equals 1 if the digit is a 8
"""
def load_mnist_data():
    training_data, test_data = get_transformed_data(False)
    return convert_to_binary(3,8, training_data, test_data)
    



#The rest of this file contains the methods necessary to support loading the mnist data

"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np

def load_original_data ():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def get_transformed_data (validation = False):
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    training_data, validation_data, test_data = load_original_data()
    training_results = [vectorized_result(y) for y in training_data[1]]
    training_data = zip(training_data[0], training_results)
    
    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = zip(test_inputs, test_data[1])
    
    
    if validation is True:
        validation_data = zip(validation_inputs, validation_data[1])
        return (training_data, validation_data, test_data)
    else:
        validation_results = [vectorized_result(y) for y in validation_data[1]]
        validation_data = zip(validation_data[0], validation_results)
        training_data.extend(validation_data)
        return (training_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
def convert_to_binary (digit1, digit2, training_data, test_data, validation_data = None):
    binary_training_data = []
    
    for (x,y) in training_data:
        if int(y[digit1]) is 1:
            binary_training_data.append((x, 0))
        elif int(y[digit2]) is 1:
            binary_training_data.append((x, 1))
    
    binary_test_data = []
    
    for (x,y) in test_data:
        if int(y) is digit1:
            binary_test_data.append((x, 0))
        elif int(y) is digit2:
            binary_test_data.append((x, 1))
    
    if validation_data is None:
        return (binary_training_data, binary_test_data)
    else:
        binary_validation_data = []
    
        for (x,y) in validation_data:
            if int(y) is digit1:
                binary_validation_data.append((x, 0))
            elif int(y) is digit2:
                    binary_validation_data.append((x, 1))
                    
        return (binary_training_data, binary_validation_data, binary_test_data)
