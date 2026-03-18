def add_matrices(matrix_1, matrix_2) -> list[list[float]]:
    n = len(matrix_1)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = matrix_1[i][j] + matrix_2[i][j]
    return result


def binet_multiply(matrix_1, matrix_2) -> [list[list[float]], int]:
    n = len(matrix_1)
    if n & (n - 1) != 0:
        raise ValueError("Matrix size must be a power of 2.")
    if n < 1:
        raise ValueError("Matrix size must be greater than 0.")
    if n == 1:
        return [[matrix_1[0][0] * matrix_2[0][0]]], 1
    if n > 1:
        mid = n // 2
        A11 = [row[:mid] for row in matrix_1[:mid]]
        A12 = [row[mid:] for row in matrix_1[:mid]]
        A21 = [row[:mid] for row in matrix_1[mid:]]
        A22 = [row[mid:] for row in matrix_1[mid:]]

        B11 = [row[:mid] for row in matrix_2[:mid]]
        B12 = [row[mid:] for row in matrix_2[:mid]]
        B21 = [row[:mid] for row in matrix_2[mid:]]
        B22 = [row[mid:] for row in matrix_2[mid:]]

        M1, f1 = binet_multiply(A11, B11)
        M2, f2 = binet_multiply(A12, B21)
        M3, f3 = binet_multiply(A11, B12)
        M4, f4 = binet_multiply(A12, B22)
        M5, f5 = binet_multiply(A21, B11)
        M6, f6 = binet_multiply(A22, B21)
        M7, f7 = binet_multiply(A21, B12)
        M8, f8 = binet_multiply(A22, B22)

        C11 = add_matrices(M1, M2)
        C12 = add_matrices(M3, M4)
        C21 = add_matrices(M5, M6)
        C22 = add_matrices(M7, M8)

        result = []
        for i in range(mid):
            result.append(C11[i] + C12[i])
        for i in range(mid):
            result.append(C21[i] + C22[i])

        total_flops = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + 4 * (mid**2)
        return result, total_flops
