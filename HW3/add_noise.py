import numpy as np

""" Function to add noise to `y` and `x`.

Assume that `n` is the number of features and `k` is the number of instances.

-----------
Parameters:
    y: k-by-1 numpy array, each element can be 1 or -1

    x: k-by-n numpy array

    noise_y_rate: a number between [0, 1]

    noise_x_rate: a number between [0, 1]

--------------
Example usage:

    from add_noise import add_noise
    (y_data, x_data) = add_noise(y, x, 0.01, 0.005)

-----
Note:
Please do not modify this file.
"""


def add_noise(y, x, noise_y_rate, noise_x_rate):
    assert(noise_y_rate >= 0 and noise_y_rate <= 1)
    assert(noise_x_rate >= 0 and noise_x_rate <= 1)

    tmp_y = (y + np.ones_like(y)) / 2
    noise_y = np.random.random(y.shape) < noise_y_rate
    tmp_y=tmp_y.astype(int)
    noise_y=noise_y.astype(int)
    #np.set_printoptions(threshold=np.nan)
    #print(tmp_y)
    new_y = np.bitwise_xor(tmp_y, noise_y)
    new_y = (2 * new_y) - np.ones_like(new_y)

    noise_x = np.random.random(x.shape) < noise_x_rate
    x=x.astype(int)
    noise_x=noise_x.astype(int)
    new_x = np.bitwise_xor(x, noise_x)

    return (new_y, new_x)
