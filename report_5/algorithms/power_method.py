import numpy as np

def power_method(A, epsilon=1e-8, p=2, max_iter=1000):
    """
    Implementacja metody potęgowej dla macierzy 3x3 zgodnie z wytycznymi.
    
    A        - macierz kwadratowa 3x3
    epsilon  - warunek stopu dla iteracji
    p        - rodzaj normy (np. 2 dla euklidesowej, np.inf dla maksimum)
    max_iter - maksymalna liczba iteracji zabezpieczająca przed pętlą nieskończoną
    """
    A = np.array(A, dtype=float)
    if A.shape != (3, 3):
        raise ValueError("Macierz musi mieć wymiary 3x3.")
        
    while True:
        z0 = np.random.uniform(0, 1, size=3)
        
        w0 = np.dot(A, z0)
        max_w0 = np.max(w0)
        
        diff = w0 - max_w0 * z0
        error = np.linalg.norm(diff, ord=p)
        
        if error >= 1e-8:
            break
            
    x = z0 / np.linalg.norm(z0, ord=p)
    
    lambda_old = 0.0
    
    for iteration in range(max_iter):
        x_next = np.dot(A, x)
        
        lambda_new = np.linalg.norm(x_next, ord=p)
        
        x_next = x_next / lambda_new
        
        if np.abs(lambda_new - lambda_old) < epsilon:
            print(f"Zbieżność osiągnięta po {iteration} iteracjach.")
            return lambda_new, x_next
            
        x = x_next
        lambda_old = lambda_new
        
    print("Osiągnięto maksymalną liczbę iteracji.")
    return lambda_old, x