import numpy as np
from numpy.typing import NDArray


def generate_matrix(n: int, seed: int | None = None) -> NDArray:
    """
    Generates a random square matrix n×n

    Args:
        n: matrix size
        seed: random generator seed (optional)

    Returns:
        n×n matrix with random values
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(n, n)


def generate_well_conditioned_matrix(n: int, seed: int | None = None) -> NDArray:
    """
    Generates a well-conditioned square matrix n×n

    Args:
        n: matrix size
        seed: random generator seed (optional)

    Returns:
        well-conditioned n×n matrix
    """
    if seed is not None:
        np.random.seed(seed)
    A = np.random.rand(n, n)
    return A @ A.T + n * np.eye(n)


def generate_ill_conditioned_matrix(n: int) -> NDArray:
    """
    Generates an ill-conditioned square matrix n×n (Hilbert matrix)

    Args:
        n: matrix size

    Returns:
        ill-conditioned n×n matrix
    """
    # Hilbert matrix is a classic example of an ill-conditioned matrix
    return np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
