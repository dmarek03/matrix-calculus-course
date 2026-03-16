from .binet import binet_multiply
from .classical import classical_multiply


def hybrid_multiply(matrix_1, matrix_2, l_size=2):
    n = len(matrix_1)
    
    if (n & (n - 1) != 0):
        raise ValueError("Matrix size must be a power of 2.")

    if n <= 2 ** l_size:
        return classical_multiply(matrix_1, matrix_2)

    return binet_multiply(matrix_1, matrix_2)

