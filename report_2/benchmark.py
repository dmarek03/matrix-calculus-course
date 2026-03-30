import time
import matplotlib.pyplot as plt

from .algortihms.gauss_elimination import gauss_no_pivot
from .algortihms.gauss_elimination_with_pivot import gauss_with_pivot
from .algortihms.solve_lus import solve_lu
from .utils.matrix_generator import generate_matrix


def benchmark(sizes, special_sizes=None):
    times = {"gauss_pivot": [], "gauss_no_pivot": [], "lu_pivot": [], "lu_no_pivot": []}

    for n in sizes:
        A = generate_matrix(n)
        b = generate_matrix(n, True)

        start = time.perf_counter()
        gauss_with_pivot(A, b)
        times["gauss_pivot"].append(time.perf_counter() - start)

        start = time.perf_counter()
        gauss_no_pivot(A, b)
        times["gauss_no_pivot"].append(time.perf_counter() - start)

        start = time.perf_counter()
        solve_lu(A, b, algorithm="lu_with_pivot")
        times["lu_pivot"].append(time.perf_counter() - start)

        start = time.perf_counter()
        solve_lu(A, b, algorithm="lu_no_pivot")
        times["lu_no_pivot"].append(time.perf_counter() - start)

        print(f"n={n} done")

    plt.figure()
    plt.plot(sizes, times["gauss_pivot"], label="Gauss + pivot")
    plt.plot(sizes, times["gauss_no_pivot"], label="Gauss")
    plt.plot(sizes, times["lu_pivot"], label="LU + pivot")
    plt.plot(sizes, times["lu_no_pivot"], label="LU")

    if special_sizes is not None:
        for s in special_sizes:
            if s in sizes:
                idx = sizes.index(s)
                plt.scatter(s, times["gauss_pivot"][idx], marker="x", s=100, zorder=5)
                plt.scatter(
                    s, times["gauss_no_pivot"][idx], marker="o", s=100, zorder=5
                )
                plt.scatter(s, times["lu_pivot"][idx], marker="^", s=100, zorder=5)
                plt.scatter(s, times["lu_no_pivot"][idx], marker=">", s=100, zorder=5)

    plt.xlabel("Rozmiar macierzy n")
    plt.ylabel("Czas (s)")
    plt.title("Benchmark metod rozwiązywania Ax = b")
    plt.legend()
    plt.grid()
    plt.show()

    if special_sizes is not None:
        for s in special_sizes:
            if s in sizes:
                idx = sizes.index(s)
                values = [
                    times["gauss_pivot"][idx],
                    times["gauss_no_pivot"][idx],
                    times["lu_pivot"][idx],
                    times["lu_no_pivot"][idx],
                ]
                labels = ["Gauss + pivot", "Gauss", "LU + pivot", "LU"]

                plt.figure()
                plt.bar(
                    labels, values, color=["blue", "orange", "green", "red"], width=0.5
                )
                plt.ylabel("Czas (s)")
                plt.title(f"Czasy rozwiązywania dla n={s} (special size)")
                plt.show()
