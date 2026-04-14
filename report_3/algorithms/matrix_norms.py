import numpy as np
from numpy.typing import NDArray


def norm_1(M: NDArray) -> float:
    """
    Calculates matrix norm ||M||₁ (column norm)
    ||M||₁ = max_j (sum_i |M[i,j]|)

    Args:
        M: input matrix

    Returns:
        norm ||M||₁
    """
    return float(np.max(np.sum(np.abs(M), axis=0)))


def norm_2(M: NDArray) -> float:
    """
    Calculates matrix norm ||M||₂ (spectral norm)
    ||M||₂ = sqrt(lambda_max(M^T * M))
    where lambda_max is the largest eigenvalue

    Args:
        M: input matrix

    Returns:
        norm ||M||₂
    """
    eigenvalues = np.linalg.eigvalsh(M.T @ M)
    return float(np.sqrt(np.max(eigenvalues)))


def condition_number_1(M: NDArray) -> float:
    """
    Calculates matrix condition number in norm ||·||₁
    κ₁(M) = ||M||₁ * ||M⁻¹||₁

    Args:
        M: input matrix (must be square and invertible)

    Returns:
        condition number κ₁(M)
    """
    M_inv = np.linalg.inv(M)
    return norm_1(M) * norm_1(M_inv)


def condition_number_2(M: NDArray) -> float:
    """
    Calculates matrix condition number in norm ||·||₂
    κ₂(M) = ||M||₂ * ||M⁻¹||₂ = sigma_max / sigma_min
    where sigma is singular values

    Args:
        M: input matrix (must be square and invertible)

    Returns:
        condition number κ₂(M)
    """
    singular_values = np.linalg.svd(M, compute_uv=False)
    return float(singular_values[0] / singular_values[-1])
