import numpy as np


def gauss_no_pivot(A, b):
    A = A.copy()
    b = b.copy()
    n = len(A)

    for k in range(n):
        pivot = A[k, k]
        if pivot == 0:
            raise ValueError("Zero values at diagonal")
        A[k, k:] /= pivot
        b[k] /= pivot
        for i in range(k + 1, n):
            factor = A[i, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i] - np.dot(A[i, i + 1 :], x[i + 1 :])
    return x
