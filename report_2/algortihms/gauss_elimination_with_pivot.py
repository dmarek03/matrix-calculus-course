import numpy as np


def gauss_with_pivot(A_, b_):
    A = A_.copy()
    b = b_.copy()
    n = len(A)

    for col in range(n):
        max_row_index = np.argmax(np.abs(A[col:, col])) + col
        if max_row_index != col:
            A[[col, max_row_index]] = A[[max_row_index, col]]
            b[[col, max_row_index]] = b[[max_row_index, col]]
        pivot = A[col, col]
        A[col, col:] /= pivot
        b[col] /= pivot
        if abs(pivot) < 1e-12:
            raise ValueError(
                f"Macierz jest osobliwa lub bliska osobliwości (kolumna {col})"
            )
        for i in range(col + 1, n):
            factor = A[i, col]
            A[i, col:] -= factor * A[col, col:]
            b[i] -= factor * b[col]

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = b[i] - np.dot(A[i, i + 1 :], x[i + 1 :])
    return x
