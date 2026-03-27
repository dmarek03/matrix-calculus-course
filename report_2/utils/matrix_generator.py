import numpy as np


def generate_matrix(n: int, is_vector: bool = False):
    np.random.seed(42)
    if is_vector:
        return np.random.randint(1, 10, size=n).astype(float)
    return np.random.randint(1, 10, size=(n, n)).astype(float)
