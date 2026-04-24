import numpy as np


def print_matrix(matrix, name, precision=4):

    print(f"\n{name}:")
    print(f"Size: {matrix.shape}")
    np.set_printoptions(precision=precision, suppress=True, linewidth=200)
    print(matrix)
    print()


def print_vector(vector, name, precision=4):
    print(f"\n{name}:")
    np.set_printoptions(precision=precision, suppress=True, linewidth=200)
    print(vector)
    print()


def save_results_to_file(filepath, results_dict):

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SVD RESULTS (Steps 1-7): Method via AA^T\n")
        f.write("="*80 + "\n\n")

        for key, value in results_dict.items():
            f.write(f"{key}:\n")
            f.write(str(value) + "\n\n")

    print(f"Saved results to: {filepath}")
