import numpy as np

def generate_matrix(seed=42):
    """
    Generuje losową macierz symetryczną 3x3 o dodatnich wartościach własnych.
    Zapewnia poprawną zbieżność metody potęgowej oraz stabilność rozkładu SVD.
    """

    np.random.seed(seed)
    eigenvalues = np.sort(np.random.uniform(1, 15, size=3))[::-1]
    D = np.diag(eigenvalues)
    Q, _ = np.linalg.qr(np.random.randn(3, 3))
    A = Q.dot(D).dot(Q.T)
    return A