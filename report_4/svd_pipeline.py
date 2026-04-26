import numpy as np
from pathlib import Path
from utils import visualize_matrix, print_matrix, print_vector, save_results_to_file


def run_svd_pipeline(A):

    n, m = A.shape
    print("="*80)
    print("SVD PIPELINE: Method via AA^T")
    print("="*80)
    print(f"Matrix A size: {n}x{m}\n")

    results_dir = Path("results")
    viz_dir = Path("visualizations")
    results_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    ### STEP 1: Display/visualize matrix A
    print("\n" + "="*80)
    print("STEP 1: Input matrix A")
    print("="*80)
    print_matrix(A, "Matrix A")
    visualize_matrix(A, "Matrix A", viz_dir / "step1_matrix_A.png")
    results['A'] = A

    ### STEP 2: Compute and display/visualize matrix AA^T (nxn)
    print("\n" + "="*80)
    print("STEP 2: Computing AA^T")
    print("="*80)
    AAT = A @ A.T
    print_matrix(AAT, "Matrix AA^T")
    visualize_matrix(AAT, "Matrix AA^T (nxn)", viz_dir / "step2_AAT.png")
    results['AAT'] = AAT

    ### STEP 3: Compute eigenvalues lambda_i and eigenvectors U_i of AA^T
    print("\n" + "="*80)
    print("STEP 3: Eigenvalues and eigenvectors of AA^T")
    print("="*80)
    eigenvalues_AAT, eigenvectors_AAT = np.linalg.eigh(AAT)

    idx = eigenvalues_AAT.argsort()[::-1]
    eigenvalues_AAT = eigenvalues_AAT[idx]
    eigenvectors_AAT = eigenvectors_AAT[:, idx]

    print_vector(eigenvalues_AAT, "Eigenvalues lambda_i of AA^T")
    print_matrix(eigenvectors_AAT, "Matrix U (eigenvectors as columns)")
    results['eigenvalues_AAT'] = eigenvalues_AAT
    results['eigenvectors_AAT'] = eigenvectors_AAT

    ### STEP 4: Display/visualize matrix U and diagonal S (S_ii = sqrt(lambda_i))
    print("\n" + "="*80)
    print("STEP 4: Matrix U and diagonal S")
    print("="*80)
    U = eigenvectors_AAT

    singular_values = np.sqrt(np.maximum(eigenvalues_AAT, 0))
    S_square = np.diag(singular_values)

    print_matrix(U, "Matrix U = [U_1, U_2, ..., U_n]")
    print_vector(singular_values, "Singular values (S_ii = sqrt(lambda_i))")
    print_matrix(S_square, "Matrix S (nxn, square)")

    visualize_matrix(U, "Matrix U (eigenvectors of AA^T)", viz_dir / "step4_matrix_U.png")
    visualize_matrix(S_square, "Matrix S (diagonal)", viz_dir / "step4_matrix_S.png")

    results['U'] = U
    results['S'] = S_square
    results['singular_values'] = singular_values

    ### STEP 5: Compute matrix V using V = A^T U S^(-1)
    print("\n" + "="*80)
    print("STEP 5: Computing V = A^T U S^(-1)")
    print("="*80)

    S_inv = np.zeros_like(S_square)
    for i in range(len(singular_values)):
        if singular_values[i] > 1e-10:
            S_inv[i, i] = 1.0 / singular_values[i]

    V_full = A.T @ U @ S_inv
    V = V_full[:, :min(n, m)]

    print_matrix(V, "Matrix V computed from V = A^T U S^(-1)")
    results['V'] = V

    ### STEP 6: Display/visualize matrix V^T (as rows)
    print("\n" + "="*80)
    print("STEP 6: Matrix V^T (vectors as rows)")
    print("="*80)
    VT = V.T
    print_matrix(VT, "Matrix V^T = [V_1, V_2, ..., V_m]^T (as rows)")
    visualize_matrix(VT, "Matrix V^T", viz_dir / "step6_matrix_VT.png")
    results['VT'] = VT

    ### STEP 7: Compute and display/visualize matrix A^T A (mxm)
    print("\n" + "="*80)
    print("STEP 7: Computing A^T A")
    print("="*80)
    ATA = A.T @ A
    print_matrix(ATA, "Matrix A^T A (mxm)")
    visualize_matrix(ATA, "Matrix A^T A (mxm)", viz_dir / "step7_ATA.png")
    results['ATA'] = ATA


    ### STEP 8: compute eigenvalues lambda_i and eigenvectors V_i of A^T A (mxm).
    print("\n" + "="*80)
    print("STEP 8: Eigenvalues and eigenvectors of A^T A")
    print("="*80)

    eigenvalues_ATA, eigenvectors_ATA = np.linalg.eigh(ATA)

    idx = eigenvalues_ATA.argsort()[::-1]
    eigenvalues_ATA = eigenvalues_ATA[idx]
    eigenvectors_ATA = eigenvectors_ATA[:, idx]

    print_vector(eigenvalues_ATA, "Eigenvalues lambda_i of A^T A")
    print_matrix(eigenvectors_ATA, "Matrix V (eigenvectors as columns)")
    results['eigenvalues_ATA'] = eigenvalues_ATA
    results['eigenvectors_ATA'] = eigenvectors_ATA

    visualize_matrix(
        eigenvectors_ATA,
        "Matrix V (eigenvectors of A^T A)",
        viz_dir / "step8_matrix_V.png"
    )

    ### STEP 9: Build matrix V^T (eigenvectors as rows) and diagonal matrix S
    print("\n" + "="*80)
    print("STEP 9: Matrix V^T and diagonal S from eigenvalues of A^T A")
    print("="*80)

    VT_from_ATA = eigenvectors_ATA.T
    singular_values_ATA = np.sqrt(np.maximum(eigenvalues_ATA, 0))
    S_from_ATA = np.diag(singular_values_ATA)

    print_matrix(VT_from_ATA, "Matrix V^T = [V_1, V_2, ..., V_m]^T (as rows)")
    print_vector(singular_values_ATA, "Singular values from A^T A (S_ii = sqrt(lambda_i))")
    print_matrix(S_from_ATA, "Matrix S (mxm, diagonal)")

    visualize_matrix(VT_from_ATA, "Matrix V^T from A^T A eigenvectors", viz_dir / "step9_matrix_VT_from_ATA.png")
    visualize_matrix(S_from_ATA, "Matrix S from A^T A eigenvalues", viz_dir / "step9_matrix_S_from_ATA.png")

    results['VT_from_ATA'] = VT_from_ATA
    results['S_from_ATA'] = S_from_ATA
    results['singular_values_ATA'] = singular_values_ATA

    
    # STEP 10:  STEP 10: Compute U = A V S^{-1}.
    print("\n" + "="*80)
    print("STEP 10: Computing U = A V S^(-1)")
    print("="*80)

    S_from_ATA_inv = np.zeros_like(S_from_ATA)
    for i in range(len(singular_values_ATA)):
        if singular_values_ATA[i] > 1e-10:
            S_from_ATA_inv[i, i] = 1.0 / singular_values_ATA[i]

    U_from_AVS = A @ eigenvectors_ATA @ S_from_ATA_inv

    print_matrix(S_from_ATA_inv, "Matrix S^(-1) (mxm, diagonal)")
    print_matrix(U_from_AVS, "Matrix U computed from U = A V S^(-1)")
    visualize_matrix(U_from_AVS, "Matrix U from A V S^(-1)", viz_dir / "step10_matrix_U_from_AVS.png")

    results['S_from_ATA_inv'] = S_from_ATA_inv
    results['U_from_AVS'] = U_from_AVS


    #STEP 11: Display/visualize matrix [U_1 U_2 ... U_m] (columns = left singular vectors).
    print("\n" + "="*80)
    print("STEP 11: Matrix [U_1 U_2 ... U_m] (left singular vectors as columns)")
    print("="*80)

    U_left_singular = U_from_AVS[:, :m]
    print_matrix(U_left_singular, "Matrix [U_1 U_2 ... U_m] (columns)")
    visualize_matrix(
        U_left_singular,
        "Matrix [U_1 U_2 ... U_m]",
        viz_dir / "step11_matrix_U_left_singular_vectors.png"
    )

    results['U_left_singular_vectors'] = U_left_singular

    # STEP 12: Compare the two SVD decompositions:
    # Method 1 (via AA^T): U1, S1, V1 from previous steps
    # Method 2 (via A^T A): U2, S2, V2 computed here
    print("\n" + "="*80)
    print("STEP 12: Compare Method 1 vs Method 2")
    print("="*80)

    r = min(n, m)

    U1 = U[:, :r]
    S1 = np.diag(singular_values[:r])
    V1 = V[:, :r]

    U2 = U_from_AVS[:, :r]
    S2 = np.diag(singular_values_ATA[:r])
    V2 = eigenvectors_ATA[:, :r]

    print_matrix(U1, "Method 1 - U1")
    print_matrix(S1, "Method 1 - S1")
    print_matrix(V1, "Method 1 - V1")

    print_matrix(U2, "Method 2 - U2")
    print_matrix(S2, "Method 2 - S2")
    print_matrix(V2, "Method 2 - V2")


    comparison_error_method1 = np.linalg.norm(A - (U1 @ S1 @ V1.T), ord='fro')
    comparison_error_method2 = np.linalg.norm(A - (U2 @ S2 @ V2.T), ord='fro')

    print(f"Method 1 reconstruction error ||A - U1S1V1^T||_F: {comparison_error_method1:.8e}")
    print(f"Method 2 reconstruction error ||A - U2S2V2^T||_F: {comparison_error_method2:.8e}")

    # STEP 13: Compute dim R(A) (rank / range) and dim N(A) (null space / kernel).
    print("\n" + "="*80)
    print("STEP 13: Rank and null space dimension")
    print("="*80)

    tolerance = 1e-10
    rank_A = np.sum(singular_values_ATA > tolerance)
    dim_range = rank_A
    dim_null = m - rank_A

    print(f"Matrix A dimensions: {n} x {m}")
    print(f"Singular values: {singular_values_ATA}")
    print(f"Threshold for non-zero singular value: {tolerance}")
    print(f"Number of singular values > threshold: {rank_A}")
    print(f"dim R(A) (rank / dimension of range): {dim_range}")
    print(f"dim N(A) (dimension of null space): {dim_null}")
    print(f"Rank-nullity check: rank + null_dim = {rank_A} + {dim_null} = {rank_A + dim_null}, m = {m}")

    # Basis of null space: right singular vectors corresponding to zero singular values
    if dim_null > 0:
        null_space_basis = eigenvectors_ATA[:, rank_A:m]
        print_matrix(null_space_basis, f"Basis of N(A) (null space) - {dim_null} vector(s)")
    else:
        null_space_basis = np.array([])
        print("N(A) is trivial (null space is zero-dimensional)")

    results['rank_A'] = rank_A
    results['dim_range_A'] = dim_range
    results['dim_null_A'] = dim_null
    if dim_null > 0:
        results['null_space_basis'] = null_space_basis


    results_file = results_dir / "svd_pipeline_results.txt"
    save_results_to_file(results_file, {
        'Matrix A size': f"{n}x{m}",
        'Eigenvalues of AA^T (lambda_i)': eigenvalues_AAT,
        'Singular values (sigma_i = sqrt(lambda_i))': singular_values,
        'Matrix U (eigenvectors of AA^T)': U,
        'Matrix V (from V = A^T U S^(-1))': V,
        'Eigenvalues of A^T A (lambda_i)': eigenvalues_ATA,
        'Matrix V (eigenvectors of A^T A)': eigenvectors_ATA,
        'Matrix V^T (eigenvectors of A^T A as rows)': VT_from_ATA,
        'Diagonal S from A^T A (S_ii = sqrt(lambda_i))': S_from_ATA,
        'Inverse diagonal S^(-1) from A^T A': S_from_ATA_inv,
        'Matrix U from U = A V S^(-1)': U_from_AVS,
        'Matrix [U_1 U_2 ... U_m] (left singular vectors as columns)': U_left_singular,
        'Rank of A': rank_A,
        'dim R(A) (range)': dim_range,
        'dim N(A) (null space)': dim_null,
    })

    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)

    return results


def main():

    np.random.seed(42)
    n, m = 5, 3
    A = np.random.randn(n, m)
    results = run_svd_pipeline(A)

    return results


if __name__ == "__main__":
    main()
