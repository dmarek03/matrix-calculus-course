import numpy as np
from pathlib import Path
from report_4.utils import visualize_matrix, print_matrix, print_vector, save_results_to_file


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

    results_file = results_dir / "svd_pipeline_results.txt"
    save_results_to_file(results_file, {
        'Matrix A size': f"{n}x{m}",
        'Eigenvalues of AA^T (lambda_i)': eigenvalues_AAT,
        'Singular values (sigma_i = sqrt(lambda_i))': singular_values,
        'Matrix U (eigenvectors of AA^T)': U,
        'Matrix V (from V = A^T U S^(-1))': V,
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
