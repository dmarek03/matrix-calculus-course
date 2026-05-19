import os
import matplotlib.pyplot as plt
import numpy as np



def run_power_method_experiment(A_orig, epsilon=0.0001):
    norms = [1, 2, 3, 4, np.inf]
    results = {p: [] for p in norms}

    for p in norms:
        A_current = np.copy(A_orig)

        for i_eig in range(3):
            errors = []

            while True:
                z0 = np.random.uniform(0, 1, size=3)
                w0 = np.dot(A_current, z0)
                max_w0 = np.max(w0)
                diff = w0 - max_w0 * z0
                if np.linalg.norm(diff, ord=p) >= 1e-8:
                    break

            x = z0 / np.linalg.norm(z0, ord=p)
            lambda_old = 0.0

            for iteration in range(500):
                w = np.dot(A_current, x)
                max_wi = np.max(w)

                current_error = np.linalg.norm(w - max_wi * x, ord=p)
                errors.append(current_error)

                lambda_new = np.linalg.norm(w, ord=p)
                if lambda_new == 0:
                    break

                x_next = w / lambda_new

                if iteration > 0 and np.abs(lambda_new - lambda_old) < epsilon:
                    break

                x = x_next
                lambda_old = lambda_new

            results[p].append(errors)

            rayleigh_lambda = np.dot(x, np.dot(A_current, x)) / np.dot(x, x)
            v2 = x / np.linalg.norm(x, ord=2)
            A_current = A_current - rayleigh_lambda * np.outer(v2, v2)

    return results


def plot_results(results):
    output_dir = "visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    norms = [1, 2, 3, 4, np.inf]
    norm_labels = {
        1: "p=1",
        2: "p=2",
        3: "p=3",
        4: "p=4",
        np.inf: "p=infinity",
    }
    file_labels = {1: "p1", 2: "p2", 3: "p3", 4: "p4", np.inf: "p_inf"}

    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    fig.suptitle(
        "Wykresy zbieżności algorytmu metody potęgowej (Zadanie 3)",
        fontsize=16,
        y=1.01,
    )

    for i_norm, p in enumerate(norms):
        for i_eig in range(3):
            errors = results[p][i_eig]
            ax = axes[i_norm, i_eig]

            ax.plot(
                range(1, len(errors) + 1),
                errors,
                marker="o",
                color="blue",
                markersize=3,
                linewidth=1,
            )
            ax.set_yscale("log")
            ax.set_title(f"Norma: {norm_labels[p]} | Wartość własna {i_eig+1}")
            ax.set_xlabel("Iteracje")
            ax.set_ylabel("Błąd (error)")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    grid_path = os.path.join(output_dir, "wszystkie_wykresy_zbiorczo.png")
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()

    for p in norms:
        for i_eig in range(3):
            errors = results[p][i_eig]

            plt.figure(figsize=(6, 4))
            plt.plot(
                range(1, len(errors) + 1),
                errors,
                marker="o",
                color="darkblue",
                markersize=4,
                linewidth=1.5,
            )
            plt.yscale("log")
            plt.title(
                f"Zbieżność metody potęgowej\nNorma: {norm_labels[p]} | Wartość własna {i_eig+1}"
            )
            plt.xlabel("Iteracje")
            plt.ylabel("Błąd (error)")
            plt.grid(True, which="both", linestyle="--", alpha=0.5)
            plt.tight_layout()

            file_name = f"wykres_norma_{file_labels[p]}_wartosc_{i_eig+1}.png"
            single_path = os.path.join(output_dir, file_name)

            plt.savefig(single_path, dpi=100, bbox_inches="tight")
            plt.close()