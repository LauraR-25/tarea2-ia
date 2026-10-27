# =====================================================================
# PUNTO 2 — Clase MyRidge (versión comentada)
#
# Es el mismo código que my_ridge.py en la carpeta principal, pero con
# cada línea explicada. El notebook usa my_ridge.py; los scripts de esta
# carpeta usan esta copia.
#
# Regresión Ridge resuelta con la solución cerrada:
#
#     θ = (Xᵀ X + λ I)⁻¹ Xᵀ y
#
# El intercepto θ0 NO se penaliza. Para lograrlo, fit centra X e y
# (les resta su media), resuelve la ecuación sobre los datos centrados
# SIN columna de unos y luego recupera el intercepto con:
#
#     θ0 = ȳ − x̄ᵀ θ
#
# Solo usa NumPy (no se permite sklearn dentro de la clase).
# Las líneas que empiezan con "#" son comentarios: Python las ignora.
# =====================================================================

import numpy as np  # Librería de cálculo con matrices; la llamamos "np" para abreviar


class MyRidge:  # Define la clase: un "molde" que agrupa los datos y las funciones del modelo
    """Ridge regression con solución cerrada, sin penalizar el intercepto."""

    def __init__(self, alpha=1.0):          # Se ejecuta al crear el modelo, por ejemplo MyRidge(alpha=1.0)
        self.alpha = alpha                  # λ: fuerza de la penalización. 0 = regresión lineal normal
        self.coef_ = None                   # θ1..θn: un coeficiente por feature (se llena en fit)
        self.intercept_ = None              # θ0: el intercepto (se llena en fit)
        self.X_mean_ = None                 # Media de cada feature, calculada en fit
        self.y_mean_ = None                 # Media de la etiqueta y, calculada en fit

    # ------------------------------------------------------------------
    # Funciones auxiliares (empiezan con "_" porque son de uso interno)
    # ------------------------------------------------------------------
    @staticmethod                           # No usa datos guardados del modelo
    def _as_2d(X):                          # Asegura que X sea una tabla (filas = ejemplos, columnas = features)
        X = np.asarray(X, dtype=float)      # Convierte a arreglo de NumPy con números decimales
        if X.ndim == 1:                     # Si llega una sola columna como lista plana...
            X = X.reshape(-1, 1)            # ...la convierte en tabla de 1 columna
        return X                            # Devuelve X con forma (m, n)

    def _check_fitted(self):                # Verifica que el modelo ya fue entrenado
        if self.coef_ is None:              # Si coef_ sigue vacío, nunca se llamó a fit
            raise RuntimeError("El modelo no está entrenado. Llama primero a fit().")

    # ------------------------------------------------------------------
    # fit: entrenar
    # ------------------------------------------------------------------
    def fit(self, X, y):                    # Calcula θ e intercepto a partir de los datos de entrenamiento
        X = self._as_2d(X)                  # Asegura forma de tabla
        y = np.asarray(y, dtype=float).ravel()  # Convierte y en una lista plana de decimales

        if X.shape[0] != y.shape[0]:        # Debe haber un valor de y por cada fila de X
            raise ValueError(f"X tiene {X.shape[0]} filas pero y tiene {y.shape[0]} valores.")
        if self.alpha < 0:                  # λ negativo no tiene sentido (premiaría coeficientes grandes)
            raise ValueError("alpha debe ser mayor o igual que 0.")

        # 1) Medias calculadas SOLO con los datos que recibe fit
        self.X_mean_ = X.mean(axis=0)       # x̄: media de cada columna (axis=0 = recorrer las filas)
        self.y_mean_ = y.mean()             # ȳ: media de la etiqueta

        # 2) Centrar: restar las medias. Así el intercepto queda fuera del problema
        Xc = X - self.X_mean_               # Cada columna de Xc tiene media 0
        yc = y - self.y_mean_               # yc tiene media 0

        # 3) Ecuación de Ridge sobre los datos centrados (sin columna de unos)
        n_features = Xc.shape[1]            # Número de features (columnas)
        A = Xc.T @ Xc + self.alpha * np.eye(n_features)  # XᵀX + λI  (np.eye crea la matriz identidad I)
        b = Xc.T @ yc                       # Xᵀy

        try:                                # Intenta resolver el sistema A·θ = b
            self.coef_ = np.linalg.solve(A, b)   # solve: más estable que calcular la inversa con inv
        except np.linalg.LinAlgError:       # Solo puede fallar si alpha = 0 y XᵀX es singular
            self.coef_ = np.linalg.lstsq(A, b, rcond=None)[0]  # Respaldo: solución de mínimos cuadrados

        # 4) Recuperar el intercepto: θ0 = ȳ − x̄ᵀθ
        self.intercept_ = self.y_mean_ - self.X_mean_ @ self.coef_

        return self                         # Devuelve el modelo para poder encadenar: MyRidge().fit(X, y).predict(X)

    # ------------------------------------------------------------------
    # predict: predecir
    # ------------------------------------------------------------------
    def predict(self, X):                   # Estima y para datos nuevos
        self._check_fitted()                # Error claro si no se entrenó
        X = self._as_2d(X)                  # Asegura forma de tabla
        if X.shape[1] != self.coef_.shape[0]:   # Debe tener las mismas features que en el entrenamiento
            raise ValueError(f"Se esperaban {self.coef_.shape[0]} features y se recibieron {X.shape[1]}.")
        # Se centran los datos nuevos con las medias GUARDADAS en fit (nunca con las de X nuevo)
        # y se vuelve a sumar ȳ. Es equivalente a X @ coef_ + intercept_.
        return (X - self.X_mean_) @ self.coef_ + self.y_mean_

    # ------------------------------------------------------------------
    # Métricas
    # ------------------------------------------------------------------
    def score(self, X, y):                  # R² de las predicciones sobre X respecto a y (igual que sklearn)
        y = np.asarray(y, dtype=float).ravel()   # Valores reales como lista plana
        y_pred = self.predict(X)            # Predicciones del modelo
        ss_res = np.sum((y - y_pred) ** 2)  # Suma de errores al cuadrado (lo que el modelo NO explica)
        ss_tot = np.sum((y - y.mean()) ** 2)     # Variación total de y respecto de su propia media
        if ss_tot == 0:                     # y constante: R² no está definido
            return 1.0 if ss_res == 0 else 0.0   # Mismo criterio que sklearn
        return 1.0 - ss_res / ss_tot        # R² = 1 − SS_res / SS_tot

    def rmse(self, X, y):                   # Raíz del error cuadrático medio (en las mismas unidades que y)
        y = np.asarray(y, dtype=float).ravel()   # Valores reales como lista plana
        y_pred = self.predict(X)            # Predicciones del modelo
        return np.sqrt(np.mean((y - y_pred) ** 2))  # √( promedio de (y − ŷ)² )


if __name__ == "__main__":                  # Ejemplo de uso con datos pequeños (solo al ejecutar este archivo)
    X_ej = np.array([[1, 1], [2, 1], [3, 2], [4, 3]], dtype=float)   # 4 ejemplos, 2 features
    y_ej = np.array([6, 8, 11, 14], dtype=float)                     # Generados con y = 3 + 2·x1 + 1·x2
    for a in [0.0, 1.0, 100.0]:                                      # Tres valores de λ
        m = MyRidge(alpha=a).fit(X_ej, y_ej)
        print(f"λ = {a:>5}: coef_ = {m.coef_.round(3)}, intercept_ = {m.intercept_:.3f}, "
              f"R² = {m.score(X_ej, y_ej):.3f}, RMSE = {m.rmse(X_ej, y_ej):.3f}")
    # Con λ = 0 recupera exactamente (2, 1) y 3; al subir λ los coeficientes se encogen
    # y el intercepto se acerca a la media de y (9,75), porque no se penaliza.
