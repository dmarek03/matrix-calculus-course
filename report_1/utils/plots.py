import matplotlib.pyplot as plt


def plot_results_multi(sizes, results, title_prefix=""):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    for label, (times, ops_counts) in results.items():
        ax[0].plot(sizes, times, marker="o", label=label)
        ax[1].plot(sizes, ops_counts, marker="o", label=label)

    ax[0].set_xlabel("Rozmiar macierzy (n)")
    ax[0].set_ylabel("Czas wykonania [s]")
    ax[0].set_title(f"{title_prefix} Czas wykonania vs rozmiar macierzy")
    ax[0].set_yscale("log")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].set_xlabel("Rozmiar macierzy (n)")
    ax[1].set_ylabel("Liczba operacji zmiennoprzecinkowych")
    ax[1].set_title(f"{title_prefix} Liczba operacji vs rozmiar macierzy")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.show()
