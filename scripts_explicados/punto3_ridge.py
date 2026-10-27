# =====================================================================
# PUNTO 3 — Corregir el sobreajuste del Punto 1 con MyRidge
#
# Qué hace este script:
#   1. Estandariza las 164 features con StandardScaler, ajustado SOLO con entrenamiento.
#   2. Elige λ por validación cruzada de 5 folds sobre np.logspace(-4, 4, 50).
#   3. Grafica el RMSE de entrenamiento y de validación en función de λ (eje logarítmico).
#   4. Entrena el modelo final y reporta el RMSE de prueba UNA sola vez.
#   5. Verifica con np.allclose que MyRidge coincide con Ridge de sklearn,
#      con features estandarizadas y sin estandarizar.
#
# Cómo ejecutarlo (desde la carpeta del proyecto):
#     python scripts_explicados/punto3_ridge.py
# =====================================================================

import numpy as np                                    # Cálculo numérico
import matplotlib.pyplot as plt                       # Gráficos
from sklearn.preprocessing import StandardScaler      # (x − media) / desviación
from sklearn.model_selection import KFold             # Divide los datos en k partes
from sklearn.linear_model import Ridge                # Solo para VERIFICAR nuestra clase
from sklearn.metrics import mean_squared_error        # Error cuadrático medio

from punto1_sobreajuste import preparar_datos, regresion_lineal, RANDOM_STATE   # Mismos datos del Punto 1
from punto2_my_ridge import MyRidge                   # Nuestra clase (Punto 2)

LAMBDAS = np.logspace(-4, 4, 50)                      # 50 valores de λ entre 0,0001 y 10.000 (escala logarítmica)
KFOLD = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)   # 5 folds, mezclados de forma reproducible


def estandarizar(X_train, X_test):
    """StandardScaler ajustado SOLO con entrenamiento; la prueba usa la misma media y desviación."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)         # Aprende media y desviación del entrenamiento y transforma
    X_test_s = scaler.transform(X_test)               # Solo transforma: no aprende nada de la prueba (evita fuga)
    return X_train_s, X_test_s


def validacion_cruzada(X_train, y_train):
    """
    Para cada λ: entrena en 4 folds y valida en el quinto, rotando las 5 veces.
    Recibe X SIN estandarizar porque en cada fold el escalador se ajusta solo con la
    parte de entrenamiento de ese fold (así la validación no filtra información).
    Devuelve el RMSE medio de entrenamiento y de validación por λ.
    """
    rmse_tr = np.zeros((len(LAMBDAS), KFOLD.get_n_splits()))    # Fila = λ, columna = fold
    rmse_val = np.zeros((len(LAMBDAS), KFOLD.get_n_splits()))

    for j, (idx_tr, idx_val) in enumerate(KFOLD.split(X_train)):    # idx_* = números de fila de cada parte
        sc = StandardScaler().fit(X_train[idx_tr])                  # Escalador del fold: solo su entrenamiento
        X_tr, X_val = sc.transform(X_train[idx_tr]), sc.transform(X_train[idx_val])
        y_tr, y_val = y_train[idx_tr], y_train[idx_val]
        for i, lam in enumerate(LAMBDAS):
            modelo = MyRidge(alpha=lam).fit(X_tr, y_tr)             # Entrena con ese λ
            rmse_tr[i, j] = modelo.rmse(X_tr, y_tr)                 # Error sobre lo que vio
            rmse_val[i, j] = modelo.rmse(X_val, y_val)              # Error sobre lo que NO vio

    return rmse_tr.mean(axis=1), rmse_val.mean(axis=1)              # Promedio de los 5 folds


def graficar(rmse_tr, rmse_val, mejor_lambda):
    """RMSE de entrenamiento y validación vs λ, con eje λ logarítmico."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(LAMBDAS, rmse_tr, marker="o", markersize=3, label="Entrenamiento (media 5 folds)")
    ax.plot(LAMBDAS, rmse_val, marker="o", markersize=3, label="Validación (media 5 folds)")
    ax.axvline(mejor_lambda, color="gray", linestyle="--", label=f"λ elegido = {mejor_lambda:.1f}")
    ax.set_xscale("log")                              # Lo pide el enunciado
    ax.set_xlabel("λ (alpha)")
    ax.set_ylabel("RMSE (rupias)")
    ax.set_title("MyRidge: RMSE según λ")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
    # Lectura: λ pequeño -> entrenamiento ≈ 0 y validación alta (sobreajuste).
    # λ óptimo -> mínimo de validación. λ grande -> ambos suben (modelo demasiado simple).


def verificar_contra_sklearn(X, y, alpha, etiqueta):
    """Compara MyRidge con sklearn.Ridge (mismos datos, mismo λ) usando np.allclose."""
    mio = MyRidge(alpha=alpha).fit(X, y)
    sk = Ridge(alpha=alpha).fit(X, y)                 # Misma función de costo y también sin penalizar el intercepto
    rmse_sk = np.sqrt(mean_squared_error(y, sk.predict(X)))
    checks = {
        "coef_": np.allclose(mio.coef_, sk.coef_),
        "intercept_": np.allclose(mio.intercept_, sk.intercept_),
        "R²": np.allclose(mio.score(X, y), sk.score(X, y)),
        "RMSE": np.allclose(mio.rmse(X, y), rmse_sk),
    }
    print(f"Verificación {etiqueta}: " + ", ".join(f"{k} = {v}" for k, v in checks.items()))
    return all(checks.values())


def ejecutar_punto3(datos, mostrar=True):
    """Corre todo el Punto 3 y devuelve lo que necesita el Punto 4."""
    X_train_s, X_test_s = estandarizar(datos["X_train_poly"], datos["X_test_poly"])

    rmse_tr, rmse_val = validacion_cruzada(datos["X_train_poly"], datos["y_train"])
    mejor_lambda = LAMBDAS[np.argmin(rmse_val)]       # λ con menor RMSE de validación

    modelo_final = MyRidge(alpha=mejor_lambda).fit(X_train_s, datos["y_train"])   # Entrena con todo el entrenamiento

    if mostrar:
        graficar(rmse_tr, rmse_val, mejor_lambda)
        rmse_train_lr, rmse_test_lr = regresion_lineal(datos)                     # Referencia del Punto 1
        rmse_test_ridge = modelo_final.rmse(X_test_s, datos["y_test"])            # ← ÚNICA evaluación en prueba
        print(f"λ elegido por validación cruzada: {mejor_lambda:.4f}")
        print(f"RMSE prueba LinearRegression (Punto 1): {rmse_test_lr:,.0f}")
        print(f"RMSE prueba MyRidge:                    {rmse_test_ridge:,.0f}")
        print(f"Reducción: {100 * (1 - rmse_test_ridge / rmse_test_lr):.2f} %\n")

        verificar_contra_sklearn(X_train_s, datos["y_train"], mejor_lambda, "164 features ESTANDARIZADAS")
        verificar_contra_sklearn(datos["X_train"], datos["y_train"], mejor_lambda, "8 features SIN estandarizar")
        # Con las 164 polinomiales SIN estandarizar no se verifica: valores de hasta 1e15 hacen que
        # XᵀX + λI tenga número de condición ≈ 1e37, imposible de resolver con precisión en float64.

    return {"X_train_s": X_train_s, "X_test_s": X_test_s, "mejor_lambda": mejor_lambda, "modelo_final": modelo_final}


if __name__ == "__main__":
    ejecutar_punto3(preparar_datos())
