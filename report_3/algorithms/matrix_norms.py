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


def norm_p(M: NDArray, p: float= np.inf) -> float:
    """
    Calculates matrix norm ||M||_p
    ||M||_p = max_{x != 0} (||Mx||_p / ||x||_p)
 
    For matrices, this is computed as the largest singular value
    of the scaled matrix, using the definition:
    ||M||_p = sup_{||x||_p = 1} ||Mx||_p
 
    For p=1: max column sum (use norm_1)
    For p=2: spectral norm (use norm_2)
    For p=inf: max row sum (use norm_inf)
 
    For general p, this uses scipy or iterative power method approximation.
    Here we use numpy's built-in matrix norm with Hölder exponent p.
 
    Args:
        M: input matrix
        p: norm order (p >= 1)
 
    Returns:
        norm ||M||_p
    """
    return float(np.linalg.norm(M, ord=p))
 
 
def condition_number_p(M: NDArray, p: float=np.inf) -> float:
    """
    Calculates matrix condition number in norm ||·||_p
    κ_p(M) = ||M||_p * ||M⁻¹||_p
 
    Args:
        M: input matrix (must be square and invertible)
        p: norm order (p >= 1)
 
    Returns:
        condition number κ_p(M)
    """
    M_inv = np.linalg.inv(M)
    return norm_p(M, p) * norm_p(M_inv, p)
 
 
def norm_inf(M: NDArray) -> float:
    """
    Calculates matrix norm ||M||_∞ (row norm)
    ||M||_∞ = max_i (sum_j |M[i,j]|)
 
    Args:
        M: input matrix
 
    Returns:
        norm ||M||_∞
    """
    return float(np.max(np.sum(np.abs(M), axis=1)))
 
 
def condition_number_inf(M: NDArray) -> float:
    """
    Calculates matrix condition number in norm ||·||_∞
    κ_∞(M) = ||M||_∞ * ||M⁻¹||_∞
 
    Args:
        M: input matrix (must be square and invertible)
 
    Returns:
        condition number κ_∞(M)
    """
    M_inv = np.linalg.inv(M)
    return norm_inf(M) * norm_inf(M_inv)
 