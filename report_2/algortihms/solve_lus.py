from .lu_factorization_with_pivot import lu_with_pivot
from .lu_factorization import lu_decomposition
import numpy as np


def solve_lu(A, b, algorithm):
    if algorithm == "lu_with_pivot":
        L, U, P = lu_with_pivot(A)
    else:
        L, U, P = lu_decomposition(A)

    b_perm = b[P]

    n = len(b)

    # forward substitution (Ly = Pb)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b_perm[i] - np.dot(L[i, :i], y[:i])

    # backward substitution (Ux = y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-12:
            raise ValueError("Dzielenie przez zero w U")
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x
