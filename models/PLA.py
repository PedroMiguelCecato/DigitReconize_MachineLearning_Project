import numpy as np

class PocketPLA:
    def __init__(self):
        self.w = None

    def get_w(self):
        return self.w

    def execute(self, X, y, max_iter=1000):
        # Adiciona bias internamente
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])

        # Inicializa w como vetor de zeros
        self.w = np.zeros(X_bias.shape[1])

        # Armazena o melhor w (pocket) e o menor erro
        pocket_w = self.w.copy()
        best_error = self.errorIN(X_bias, y)

        for _ in range(max_iter):
            for i in range(len(y)):
                if np.sign(np.dot(self.w, X_bias[i])) != y[i]:
                    # Atualiza w com erro atual
                    self.w += y[i] * X_bias[i]

                    # Calcula erro com novo w
                    current_error = self.errorIN(X_bias, y)

                    # Se for melhor, guarda no pocket
                    if current_error < best_error:
                        pocket_w = self.w.copy()
                        best_error = current_error
                    break  # volta ao início do loop principal

        # Ao final, define o melhor w encontrado
        self.w = pocket_w
    
    def getOriginalY(self, originalX):
        """Calcula Y original da fronteira de decisão (para plot 2D)."""
        return (-self.w[0] - self.w[1]*originalX) / self.w[2]

    def h(self, x):
        # Adiciona bias automaticamente ao vetor de entrada
        x_bias = np.insert(x, 0, 1)  # insere o 1 no início
        return np.sign(np.dot(self.w, x_bias))
    
    def predict(self, X):
        """Predição em lote (usada na avaliação)."""
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.sign(X_bias @ self.w)
    
    def errorIN(self, X_bias, y):
        error = 0
        for i in range(len(y)):
            if np.sign(np.dot(self.w, X_bias[i])) != y[i]:
                error += 1
        return error