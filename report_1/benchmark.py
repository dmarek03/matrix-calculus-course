import time

import numpy as np
from numpy import floating

from report_1.algorithms.hybrid import hybrid_multiply
from report_1.utils.matrix_helpers import (
    generate_matrix,
    check_multiplication_correctness,
)


def benchmark(
    sizes: list[int], repeats: int = 10, l_size: int = 32
) -> tuple[list[floating], list[floating]]:

    times = []
    ops_counts = []

    for n in sizes:
        t_list = []
        ops_list = []
        for _ in range(repeats):
            A = generate_matrix(n)
            B = generate_matrix(n)

            start = time.time()
            C, ops = hybrid_multiply(A, B, l_size)
            end = time.time()

            assert check_multiplication_correctness(A, B, C), f"Błąd mnożenia dla n={n}"

            t_list.append(end - start)
            ops_list.append(ops)

        times.append(np.mean(t_list))
        ops_counts.append(np.mean(ops_list))
    return times, ops_counts
