import numpy as np

class RegressaoLinear:
    def __init__(self):
        self.w = None

    def execute(self, _X, _y, show=False):
        X0 = np.asarray(_X, dtype=float)
        y = np.asarray(_y, dtype=float).ravel()

        # Garante formato 2D
        if X0.ndim == 1:
            X0 = X0.reshape(-1, 1)

        # Matriz de projeto
        X = np.c_[np.ones((X0.shape[0], 1)), X0]

        if show:
            print("X =\n", X)
            print("y =\n", y)

        # Solução via pseudo-inversa (SVD internamente)
        self.w = np.linalg.pinv(X) @ y

        if show:
            print(f"W =\n{self.w}")

    def predict(self, _x, class_output=False):
        x0 = np.asarray(_x, dtype=float)
        if x0.ndim == 1:
            x0 = x0.reshape(1, -1)

        X = np.c_[np.ones((x0.shape[0], 1)), x0]
        y_pred = X @ self.w

        if class_output:
            y_class = np.sign(y_pred)
            y_class[y_class == 0] = 1
            return y_class

        return y_pred

    def get_w(self):
        return self.w