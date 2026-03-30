from algortihms.lu_factorization_with_pivot import lu_with_pivot
from algortihms.lu_factorization import lu_decomposition
import numpy as np


def solve_lu(A, b, algorithm):
    if algorithm == 'lu_with_pivot':
        L, U, P = lu_with_pivot(A)
        b_perm = b[P]  # permutacja potrzebna tylko przy pivotingu
    else:        
        L, U, P = lu_decomposition(A)  # czy lu_decomposition przyjmuje tylko A?
        b_perm = b[P] if P is not None else b.copy()
    
    n = len(b)

    y = np.zeros(n)
    for i in range(n):
        y[i] = b_perm[i] - np.dot(L[i, :i], y[:i])
    
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    return x