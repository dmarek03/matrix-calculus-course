import numpy as np


def lu_decomposition(A):
    A = A.copy()
    n = len(A)
    L = np.eye(n)

    for col in range(n):
        if abs(A[col, col]) < 1e-12:
            raise ValueError(f"Zero na przekątnej w kolumnie {col}!")
        for row in range(col + 1, n):
            l_ij = A[row, col] / A[col, col]
            L[row, col] = l_ij
            A[row, col:] -= l_ij * A[col, col:]

    U = A
    P = np.arange(n)  # brak pivotingu = permutacja tożsamościowa

    return L, U, P