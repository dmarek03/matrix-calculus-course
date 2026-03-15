def classical_multiply(matrix_1, matrix_2):
    n = len(matrix_1)
    result = [[0.0]*n for _ in range(n)]
    flops_cnt = 0
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += matrix_1[i][k] * matrix_2[k][j]
                flops_cnt += 2
            result[i][j] = s

    return result, flops_cnt
