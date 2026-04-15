from py_compile import main

import numpy as np
from numpy.typing import NDArray

from algorithms.matrix_norms import (
    norm_1,
    norm_2,
    condition_number_1,
    condition_number_2,
    norm_p,
    condition_number_p, 
    norm_inf,
    condition_number_inf,
)
from utils.matrix_helpers import (
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


def demonstrate_calculations_summary(M: NDArray, p: float = None) -> None:
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
        result_cond_1 = float("nan")

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
        result_cond_2 = float("nan")

    # ========================================================================
    # 5. MATRIX NORM ||M||_p  (tylko jeśli p podane)
    # ========================================================================
    result_norm_p = result_cond_p = result_norm_inf = result_cond_inf = None

    if p is not None:
        print("\n" + "-" * 80)
        print(f"5. Norm ||M||_p  (p = {p})")
        print("-" * 80)
        col_p_sums = np.sum(np.abs(M) ** p, axis=0)
        for j in range(n):
            print(f"Column {j}: (Σ|M[i,{j}]|^{p:.0f})^(1/{p:.0f}) = "
                  f"({col_p_sums[j]:.6f})^(1/{p:.0f}) = "
                  f"{col_p_sums[j] ** (1.0 / p):.6f}")

        result_norm_p = norm_p(M, p)
        print(f"\n||M||_p = max over columns = {result_norm_p:.6f}")

        # ====================================================================
        # 6. CONDITION NUMBER κ_p(M)
        # ====================================================================
        print("\n" + "-" * 80)
        print(f"6. Condition number κ_p(M)  (p = {p})")
        print("-" * 80)
        try:
            M_inv = np.linalg.inv(M)
            result_norm_p_inv = norm_p(M_inv, p)
            result_cond_p = condition_number_p(M, p)
            print(f"||M||_p     = {result_norm_p:.6f}")
            print(f"||M⁻¹||_p   = {result_norm_p_inv:.6f}")
            print(f"\nκ_p(M) = ||M||_p × ||M⁻¹||_p = {result_cond_p:.6f}")
        except np.linalg.LinAlgError:
            print("Error: Matrix is singular")
            result_cond_p = float("nan")

        # ====================================================================
        # 7. MATRIX NORM ||M||_∞
        # ====================================================================
        print("\n" + "-" * 80)
        print("7. Norm ||M||_∞")
        print("-" * 80)
        row_sums = np.sum(np.abs(M), axis=1)
        for i in range(n):
            print(f"Row {i}: Σ|M[{i},j]| = {row_sums[i]:.6f}")

        result_norm_inf = norm_inf(M)
        print(f"\n||M||_∞ = max over rows = {result_norm_inf:.6f}")

        # ====================================================================
        # 8. CONDITION NUMBER κ_∞(M)
        # ====================================================================
        print("\n" + "-" * 80)
        print("8. Condition number κ_∞(M)")
        print("-" * 80)
        try:
            M_inv = np.linalg.inv(M)
            result_norm_inf_inv = norm_inf(M_inv)
            result_cond_inf = condition_number_inf(M)
            print(f"||M||_∞     = {result_norm_inf:.6f}")
            print(f"||M⁻¹||_∞   = {result_norm_inf_inv:.6f}")
            print(f"\nκ_∞(M) = ||M||_∞ × ||M⁻¹||_∞ = {result_cond_inf:.6f}")
        except np.linalg.LinAlgError:
            print("Error: Matrix is singular")
            result_cond_inf = float("nan")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"  ||M||₁  = {result_norm_1:.6f}")
    print(f"  κ₁(M)   = {result_cond_1:.6f}")
    print(f"  ||M||₂  = {result_norm_2:.6f}")
    print(f"  κ₂(M)   = {result_cond_2:.6f}")
    if p is not None:
        print(f"  ||M||_p  (p={p:.0f})  = {result_norm_p:.6f}")
        print(f"  κ_p(M)   (p={p:.0f})  = {result_cond_p:.6f}")
        print(f"  ||M||_∞              = {result_norm_inf:.6f}")
        print(f"  κ_∞(M)               = {result_cond_inf:.6f}")
    print("=" * 80 + "\n")
 


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("\nExample 1: Random matrix 4×4")
    M1 = generate_matrix(4, seed=42)
    demonstrate_calculations_summary(M1)

    print("\n\n" + "="*80)
    print("\nExample 2: Well-conditioned matrix 4×4")
    M2 = generate_well_conditioned_matrix(4, seed=42)
    demonstrate_calculations_summary(M2)
    print("\n\n" + "="*80)
    print("\nExample 3: Ill-conditioned matrix (Hilbert) 4×4")
    M3 = generate_ill_conditioned_matrix(4)
    demonstrate_calculations_summary(M3)

main()