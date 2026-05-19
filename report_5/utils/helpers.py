import numpy as np


def print_matrix(M: np.ndarray, name: str, precision: int = 6) -> None:
    print(f"\n{name}:")
    with np.printoptions(precision=precision, suppress=True):
        print(M)


def matrix_norm(M: np.ndarray, p) -> float:
    """Entry-wise p-norm — works for all p including 3, 4, and infinity."""
    return float(np.linalg.norm(M.ravel(), ord=p))
