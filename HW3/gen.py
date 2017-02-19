import numpy as np

from add_noise import add_noise

""" Function that generates the dataset.

-----------
Parameters:
    l: integer

    m: integer

    n: integer

    number_of_instances: integer
    Integer that indicates the size of the dataset. Represented by
    `k` in the problem statement.

    noise: boolean
    Boolean that indicates if we should add noise to the dataset.

--------------
Example usage:

    from gen import gen
    (y, x) = gen(10, 100, 500, 50000, False)

-----
Note:
Please do not modify this file.
Precondition: l <= m <= n
"""


def gen(l, m, n, number_of_instances, noise):
    assert(l <= m and m <= n)  # precondition

    # balanced dataset
    p_number_of_instances = int(number_of_instances / 2)
    n_number_of_instances = number_of_instances - p_number_of_instances

    # positive example
    p_y = np.ones((p_number_of_instances, 1), dtype=int)
    p_x_first_part = np.zeros((p_number_of_instances, m), dtype=int)
    p_x_second_part = (np.random.random((p_number_of_instances, n-m)) < 0.5)
    p_x_second_part = p_x_second_part.astype(int)

    # columnwise append
    p_x = np.append(p_x_first_part, p_x_second_part, axis=1)

    for i in range(p_number_of_instances):
        candidates = np.random.permutation(m)
        n_nonzeros = l
        active_features = candidates[:n_nonzeros]
        p_x[i][active_features] = 1  # set non_zeros to 1

    # negative example
    n_y = -1 * np.ones((n_number_of_instances, 1), dtype=int)
    n_x_first_part = np.zeros((n_number_of_instances, m), dtype=int)
    n_x_second_part = (np.random.random((n_number_of_instances, n-m)) < 0.5)
    n_x_second_part = n_x_second_part.astype(int)

    # columnwise append
    n_x = np.append(n_x_first_part, n_x_second_part, axis=1)

    for i in range(n_number_of_instances):
        candidates = np.random.permutation(m)
        n_nonzeros = l - 2
        active_features = candidates[:n_nonzeros]
        n_x[i][active_features] = 1  # set non_zero to 1

    y = np.append(p_y, n_y)
    x = np.append(p_x, n_x, axis=0)

    shuffle_indices = np.random.permutation(number_of_instances)
    y = y[shuffle_indices]
    x = x[shuffle_indices][:]

    # sanity check
    validate_dataset(y, x, l, m, n, number_of_instances)

    if noise is True:
        noise_y_rate = 0.05
        noise_x_rate = 0.001
        return add_noise(y, x, noise_y_rate, noise_x_rate)

    return (y, x)


""" Validate a generated dataset to check for the l-of-m-of-n concept class.

Note: This function will pass silently for data without any noise.
"""


def validate_dataset(y, x, l, m, n, number_of_instances):
    for idx in range(number_of_instances):
        current_y = y[idx]
        current_x = x[idx]

        if current_y > 0 and np.sum(current_x[:m]) < l:
            print("Invalid positive example found.")
        elif current_y <= 0 and np.sum(current_x[:m]) >= l:
            print("Invalid negative example found.")
