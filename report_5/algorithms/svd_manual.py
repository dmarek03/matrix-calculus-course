import numpy as np


def compute_svd_manual(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SVD of A = U D V^T via eigendecomposition of B = AA^T.

    Algorithm:
      1. B = AA^T
      2. Eigenvalues lambda_i and eigenvectors u_i of B  (np.linalg.eigh, symmetric)
      3. U = [u_1 | u_2 | ... ] (columns), D = diag(sqrt(lambda_i))
      4. V = A^T U inv(D), where inv(D)_ii = 1 / D_ii

    Returns:
        U — n×n matrix of left singular vectors (columns)
        D — n×n diagonal matrix of singular values
        V — m×n matrix of right singular vectors (columns)
    """
    B = A @ A.T
    eigenvalues, U = np.linalg.eigh(B)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    U = U[:, idx]

    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))
    D = np.diag(singular_values)

    D_inv = np.zeros_like(D)
    for i, sv in enumerate(singular_values):
        if sv > 1e-10:
            D_inv[i, i] = 1.0 / sv

    V = A.T @ U @ D_inv

    return U, D, V
