import numpy as np
from numpy.typing import NDArray

from report_3.algorithms.matrix_norms import (
    norm_1,
    norm_2,
    condition_number_1,
    condition_number_2,
)
from report_3.utils.matrix_helpers import (
    generate_matrix,
    generate_well_conditioned_matrix,
    generate_ill_conditioned_matrix,
)


def demonstrate_calculations(M: NDArray) -> None:

    n = M.shape[0]
    print("\n" + "=" * 80)
    print(f"Matrix M ({n}×{n}):")
    print("=" * 80)
    print(M)

    # ========================================================================
    # 1. MATRIX NORM ||M||₁
    # ========================================================================
    print("\n" + "-" * 80)
    print("1. Norm ||M||₁")
    print("-" * 80)
    column_sums = np.sum(np.abs(M), axis=0)
    for j in range(n):
        print(f"Column {j}: Σ|M[i,{j}]| = {column_sums[j]:.6f}")

    result_norm_1 = norm_1(M)
    print(f"\n||M||₁ = {result_norm_1:.6f}")

    # ========================================================================
    # 2. CONDITION NUMBER κ₁(M)
    # ========================================================================
    print("\n" + "-" * 80)
    print("2. Condition number κ₁(M)")
    print("-" * 80)
    try:
        M_inv = np.linalg.inv(M)
        result_norm_1_inv = norm_1(M_inv)
        result_cond_1 = condition_number_1(M)

        print(f"||M||₁ = {result_norm_1:.6f}")
        print(f"||M⁻¹||₁ = {result_norm_1_inv:.6f}")
        print(f"\nκ₁(M) = {result_cond_1:.6f}")
    except np.linalg.LinAlgError:
        print("Error: Matrix is singular")

    # ========================================================================
    # 3. MATRIX NORM ||M||₂
    # ========================================================================
    print("\n" + "-" * 80)
    print("3. Norm ||M||₂")
    print("-" * 80)
    eigenvalues = np.linalg.eigvalsh(M.T @ M)
    print("Eigenvalues λ(M^T × M):")
    for i, eigval in enumerate(eigenvalues):
        print(f"  λ_{i} = {eigval:.6f}")

    max_eigenvalue = np.max(eigenvalues)
    result_norm_2 = norm_2(M)
    print(f"\nλ_max = {max_eigenvalue:.6f}")
    print(f"||M||₂ = √λ_max = {result_norm_2:.6f}")

    # ========================================================================
    # 4. CONDITION NUMBER κ₂(M)
    # ========================================================================
    print("\n" + "-" * 80)
    print("4. Condition number κ₂(M)")
    print("-" * 80)
    try:
        singular_values = np.linalg.svd(M, compute_uv=False)
        print("Singular values σ(M):")
        for i, sv in enumerate(singular_values):
            print(f"  σ_{i} = {sv:.6f}")

        sigma_max = singular_values[0]
        sigma_min = singular_values[-1]
        result_cond_2 = condition_number_2(M)

        print(f"\nσ_max = {sigma_max:.6f}")
        print(f"σ_min = {sigma_min:.6f}")
        print(f"κ₂(M) = {result_cond_2:.6f}")
    except np.linalg.LinAlgError:
        print("Error: Cannot compute SVD")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"  ||M||₁  = {result_norm_1:.6f}")
    print(f"  κ₁(M)   = {result_cond_1:.6f}")
    print(f"  ||M||₂  = {result_norm_2:.6f}")
    print(f"  κ₂(M)   = {result_cond_2:.6f}")
    print("=" * 80 + "\n")




if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("\nExample 1: Random matrix 4×4")
    M1 = generate_matrix(4, seed=42)
    demonstrate_calculations(M1)

    print("\n\n" + "="*80)
    print("\nExample 2: Well-conditioned matrix 4×4")
    M2 = generate_well_conditioned_matrix(4, seed=42)
    demonstrate_calculations(M2)

    print("\n\n" + "="*80)
    print("\nExample 3: Ill-conditioned matrix (Hilbert) 4×4")
    M3 = generate_ill_conditioned_matrix(4)
    demonstrate_calculations(M3)
