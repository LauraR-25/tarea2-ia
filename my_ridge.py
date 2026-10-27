import numpy as np


class MyRidge:

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self.X_mean_ = None
        self.y_mean_ = None

    @staticmethod
    def _as_2d(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _check_fitted(self):
        if self.coef_ is None:
            raise RuntimeError("El modelo no está entrenado. Llama primero a fit().")

    def fit(self, X, y):
        X = self._as_2d(X)
        y = np.asarray(y, dtype=float).ravel()

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X tiene {X.shape[0]} filas pero y tiene {y.shape[0]} valores.")
        if self.alpha < 0:
            raise ValueError("alpha debe ser mayor o igual que 0.")

        self.X_mean_ = X.mean(axis=0)
        self.y_mean_ = y.mean()

        Xc = X - self.X_mean_
        yc = y - self.y_mean_

        n_features = Xc.shape[1]
        A = Xc.T @ Xc + self.alpha * np.eye(n_features)
        b = Xc.T @ yc

        try:
            self.coef_ = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(A, b, rcond=None)[0]

        self.intercept_ = self.y_mean_ - self.X_mean_ @ self.coef_
        return self

    def predict(self, X):
        self._check_fitted()
        X = self._as_2d(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(f"Se esperaban {self.coef_.shape[0]} features y se recibieron {X.shape[1]}.")
        return (X - self.X_mean_) @ self.coef_ + self.y_mean_

    def score(self, X, y):
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res / ss_tot

    def rmse(self, X, y):
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        return np.sqrt(np.mean((y - y_pred) ** 2))
