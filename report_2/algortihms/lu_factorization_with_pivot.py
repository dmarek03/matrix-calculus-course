import numpy as np


def lu_with_pivot(A):
    A = A.copy()
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n, n))
    P = np.arange(n)

    for k in range(n):
        pivot_index = np.argmax(np.abs(A[k:, k])) + k
        if pivot_index != k:
            A[[k, pivot_index], :] = A[[pivot_index, k], :]
            P[[k, pivot_index]] = P[[pivot_index, k]]
        U[k, k:] = A[k, k:]
        for i in range(k+1, n):
            L[i, k] = A[i, k] / U[k, k]
            A[i, k+1:] -= L[i, k] * U[k, k+1:]
    return L, U, P