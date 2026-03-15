import numpy as np


def generate_matrix(size: int, seed: int | None = None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(size, size)


def check_multiplication_correctness(matrix_1, matrix_2, result_matrix) -> bool:
    return np.allclose(result_matrix, matrix_1 @ matrix_2)


def print_matrix(matrix) -> None:
    for row in matrix:
        print(" ".join(f"{x:.6f}" for x in row))
