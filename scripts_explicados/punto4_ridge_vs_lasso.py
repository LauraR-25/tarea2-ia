# =====================================================================
# PUNTO 4 — Ridge vs. Lasso
#
# Qué hace este script (sobre los mismos datos estandarizados del Punto 3):
#   1. Elige el α de Ridge y el de Lasso por validación cruzada con GridSearchCV.
#   2. Muestra una tabla con los coeficientes de MyRidge, Ridge y Lasso lado a lado.
#   3. Grafica los caminos de coeficientes de Ridge y Lasso en función de α (eje log).
#   4. Cuenta cuántos coeficientes de Lasso son exactamente 0, qué features elimina,
#      y compara el RMSE de prueba de ambos modelos.
#   5. Dibuja la geometría que explica por qué Lasso da ceros exactos y Ridge no.
#
# Cómo ejecutarlo (desde la carpeta del proyecto):
#     python scripts_explicados/punto4_ridge_vs_lasso.py
# =====================================================================

import warnings                                       # Para manejar los avisos de convergencia de Lasso

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV      # Prueba cada α con validación cruzada
from sklearn.pipeline import Pipeline                 # Encadena escalador + modelo
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning

from punto1_sobreajuste import preparar_datos, FEATURE_COLS
from punto3_ridge import ejecutar_punto3, KFOLD, LAMBDAS

# Lasso de sklearn minimiza (1/2m)·‖y − Xw‖² + α‖w‖₁. Por el factor 1/2m su α no es comparable con el
# de Ridge, así que tiene su propia grilla. Con precios de cientos de miles, su α óptimo es grande.
ALPHAS_LASSO = np.logspace(1, 6, 50)


def buscar_alphas(X_train_poly, y_train):
    """
    GridSearchCV con Pipeline(StandardScaler -> modelo): el escalador se ajusta dentro de cada fold,
    igual que en el Punto 3. Se usan los MISMOS folds (KFOLD), por eso Ridge elige el mismo λ que MyRidge.
    """
    ridge_cv = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("modelo", Ridge())]),
        {"modelo__alpha": LAMBDAS},                   # Misma grilla del Punto 3
        scoring="neg_root_mean_squared_error",        # sklearn maximiza, por eso usa el RMSE negativo
        cv=KFOLD,
    ).fit(X_train_poly, y_train)

    with warnings.catch_warnings():                   # Con α pequeños Lasso puede no converger: se ignora el aviso
        warnings.simplefilter("ignore", ConvergenceWarning)   # (esos α no se eligen; max_iter alto lo reduce)
        lasso_cv = GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("modelo", Lasso(max_iter=50000))]),
            {"modelo__alpha": ALPHAS_LASSO},
            scoring="neg_root_mean_squared_error",
            cv=KFOLD,
        ).fit(X_train_poly, y_train)

    return ridge_cv.best_params_["modelo__alpha"], lasso_cv.best_params_["modelo__alpha"]


def caminos(X_train_s, y_train, seleccionadas, nombres, alpha_ridge, alpha_lasso):
    """Coeficientes de Ridge y Lasso para cada α. Se resaltan las features que Lasso conserva."""
    alphas_ridge = np.logspace(-4, 6, 60)
    ridge_path = np.array([Ridge(alpha=a).fit(X_train_s, y_train).coef_ for a in alphas_ridge])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lasso_path = np.array([Lasso(alpha=a, max_iter=50000).fit(X_train_s, y_train).coef_ for a in ALPHAS_LASSO])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, path, alphas, alpha_opt, titulo in [
        (axes[0], ridge_path, alphas_ridge, alpha_ridge, "Ridge: se encogen sin llegar a 0"),
        (axes[1], lasso_path, ALPHAS_LASSO, alpha_lasso, "Lasso: cada vez más quedan exactamente en 0"),
    ]:
        ax.plot(alphas, np.delete(path, seleccionadas, axis=1), color="lightgray", linewidth=0.8)  # El resto en gris
        for k in seleccionadas:
            ax.plot(alphas, path[:, k], linewidth=2, label=nombres[k])                               # Las 4 de Lasso
        ax.axvline(alpha_opt, color="black", linestyle="--", label=f"α elegido = {alpha_opt:,.1f}")
        ax.set_xscale("log")
        ax.set_xlabel("α")
        ax.set_ylabel("Coeficiente")
        ax.set_title(titulo)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def geometria():
    """
    En 2D: las curvas de nivel del error son elipses. Ridge restringe los coeficientes a un CÍRCULO
    (sin esquinas) y Lasso a un ROMBO (esquinas sobre los ejes). La elipse toca el rombo en una esquina,
    donde un coeficiente vale exactamente 0; al círculo lo toca en un punto cualquiera, sin ceros.
    """
    centro = np.array([1.7, 0.45])                    # Solución de mínimos cuadrados (ilustrativa)
    A = np.array([[1.0, 0.5], [0.5, 1.0]])            # Forma de las elipses
    w1, w2 = np.meshgrid(np.linspace(-1.6, 2.6, 400), np.linspace(-1.6, 2.2, 400))
    D = np.stack([w1 - centro[0], w2 - centro[1]], axis=-1)
    error = np.einsum("...i,ij,...j->...", D, A, D)   # Error cuadrático en cada punto de la grilla

    t = np.linspace(0, 2 * np.pi, 4000)
    bordes = {"Ridge (L2): círculo": np.c_[np.cos(t), np.sin(t)],
              "Lasso (L1): rombo": np.c_[np.cos(t), np.sin(t)] / (np.abs(np.cos(t)) + np.abs(np.sin(t)))[:, None]}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (nombre, borde) in zip(axes, bordes.items()):
        d = borde - centro
        valores = np.einsum("ni,ij,nj->n", d, A, d)
        toque = borde[np.argmin(valores)]             # Punto del borde con menor error = solución regularizada
        ax.contour(w1, w2, error, levels=[valores.min() * f for f in (1, 1.6, 2.4, 3.5)], colors="tab:blue")
        ax.fill(borde[:, 0], borde[:, 1], color="tab:orange", alpha=0.3)
        ax.plot(*centro, "o", color="tab:blue")
        ax.plot(*toque, "o", color="red", markersize=8)
        ax.annotate(f"({toque[0]:.2f}, {toque[1]:.2f})", toque, textcoords="offset points", xytext=(8, -20), color="red")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_aspect("equal")
        ax.set_title(nombre)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    datos = preparar_datos()
    p3 = ejecutar_punto3(datos, mostrar=False)        # Reutiliza el Punto 3 (datos estandarizados y MyRidge)
    X_train_s, X_test_s = p3["X_train_s"], p3["X_test_s"]
    y_train, y_test = datos["y_train"], datos["y_test"]
    nombres = datos["feature_names_poly"]

    # 1) α de cada modelo por validación cruzada
    alpha_ridge, alpha_lasso = buscar_alphas(datos["X_train_poly"], y_train)
    print(f"Ridge: α = {alpha_ridge:.4f} (MyRidge eligió λ = {p3['mejor_lambda']:.4f})")
    print(f"Lasso: α = {alpha_lasso:,.2f}\n")

    # Modelos finales con todo el entrenamiento estandarizado
    ridge = Ridge(alpha=alpha_ridge).fit(X_train_s, y_train)
    lasso = Lasso(alpha=alpha_lasso, max_iter=50000).fit(X_train_s, y_train)

    # 2) Tabla de coeficientes lado a lado (primero las que Lasso conserva)
    tabla = pd.DataFrame({"MyRidge": p3["modelo_final"].coef_, "Ridge": ridge.coef_, "Lasso": lasso.coef_}, index=nombres)
    tabla = tabla.reindex(tabla["Lasso"].abs().sort_values(ascending=False).index)
    with pd.option_context("display.max_rows", 200, "display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:,.2f}".format):
        print(tabla)

    # 3) Caminos de coeficientes
    seleccionadas = np.flatnonzero(lasso.coef_ != 0)
    caminos(X_train_s, y_train, seleccionadas, nombres, alpha_ridge, alpha_lasso)

    # 4) Ceros de Lasso, features eliminadas y RMSE de prueba
    ceros = lasso.coef_ == 0                          # Comparación exacta con 0
    print(f"\nLasso deja {ceros.sum()} de {len(ceros)} coeficientes en 0. Conserva: {list(nombres[~ceros])}")
    usadas = {f for n in nombres[~ceros] for f in FEATURE_COLS if f in n}
    print("Features originales eliminadas por completo:", [f for f in FEATURE_COLS if f not in usadas])
    rmse = lambda modelo: np.sqrt(mean_squared_error(y_test, modelo.predict(X_test_s)))
    print(f"RMSE prueba  Ridge = {rmse(ridge):,.0f} | Lasso = {rmse(lasso):,.0f}")

    # 5) Geometría L1 vs L2
    geometria()
