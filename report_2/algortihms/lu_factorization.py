import numpy as np


def lu_decomposition(A, b):
    A = A.copy()
    b= b.copy()
    n = len(A)
    L= np.eye(n)
    U = np.zeros((n, n))

    for col in range(n):
        if abs(A[col, col]) < 1e-12:
            raise ValueError(f"Zero na przekątnej w kolumnie {col}!")
        for row in range(col + 1, n):
            l_ij = A[row, col] / A[col, col]
            L[row, col] = l_ij
            A[row, col:] -= l_ij * A[col, col:]
            
    U = A 
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-12:
             raise ValueError(f"Macierz U jest osobliwa w wierszu {i}")
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return L, U, x
    