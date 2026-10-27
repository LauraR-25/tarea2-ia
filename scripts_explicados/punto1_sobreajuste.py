# =====================================================================
# PUNTO 1 — Exploración, limpieza y sobreajuste con LinearRegression
#
# Qué hace este script:
#   1. Carga el dataset de autos usados (CarDekho) y lo explora brevemente.
#   2. Limpia los datos y arma 8 features cuantitativas.
#   3. Fuerza el sobreajuste: solo 35 autos de entrenamiento y
#      PolynomialFeatures de grado 3 (8 features -> 164).
#   4. Entrena LinearRegression y reporta el RMSE de entrenamiento y de prueba.
#
# Cómo ejecutarlo (desde la carpeta del proyecto):
#     python scripts_explicados/punto1_sobreajuste.py
#
# Los puntos 3 y 4 importan la función preparar_datos() de este archivo,
# así todos usan exactamente los mismos datos.
# =====================================================================

from pathlib import Path                              # Para armar rutas de archivos sin importar el sistema operativo

import numpy as np                                    # Cálculo numérico
import pandas as pd                                   # Tablas de datos
import matplotlib.pyplot as plt                       # Gráficos
from sklearn.model_selection import train_test_split  # Separa entrenamiento y prueba
from sklearn.preprocessing import PolynomialFeatures  # Crea potencias y productos de las features
from sklearn.linear_model import LinearRegression     # Regresión lineal de sklearn
from sklearn.metrics import mean_squared_error        # Error cuadrático medio

RANDOM_STATE = 42                                     # Semilla fija: el mismo resultado cada vez que se ejecuta
RUTA_DATOS = Path(__file__).resolve().parent.parent / "data" / "dataset.csv"   # data/dataset.csv del proyecto

# Las 8 features cuantitativas que usa el modelo
FEATURE_COLS = ["year", "km_driven", "mileage(km/ltr/kg)", "engine", "max_power",
                "seats", "km_per_year", "engine_power_ratio"]


def explorar(df):
    """Exploración breve: tipos, faltantes, duplicados, precio y tres gráficos."""
    print("--- Dimensiones ---")
    print(df.shape)                                   # (filas, columnas)
    print("\n--- Tipos de datos ---")
    print(df.dtypes)                                  # max_power aparece como texto: hay que convertirla
    print("\n--- Valores faltantes por columna ---")
    print(df.isnull().sum())                          # ~220 filas sin mileage, engine, max_power, seats
    print(f"\nFilas duplicadas: {df.duplicated().sum()}")
    print("\n--- Precio de venta ---")
    print(df["selling_price"].describe().round(0))    # Mediana 450 mil, máximo 10 millones: muy asimétrico

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))   # Tres gráficos lado a lado
    axes[0].hist(df["selling_price"], bins=60)        # Distribución del precio
    axes[0].set_title("Distribución de selling_price")
    axes[1].scatter(df["year"], df["selling_price"], s=5, alpha=0.4)       # Autos más nuevos -> más caros
    axes[1].set_title("Año vs. precio")
    axes[2].scatter(df["km_driven"], df["selling_price"], s=5, alpha=0.4)  # Más km -> más baratos
    axes[2].set_xscale("log")                         # Escala log porque los km varían mucho
    axes[2].set_title("Kilómetros vs. precio")
    plt.tight_layout()
    plt.show()


def limpiar(df):
    """Elimina faltantes, convierte max_power a número y crea las 2 features derivadas."""
    df = df.dropna().copy()                                            # Quita filas con valores faltantes
    df["max_power"] = pd.to_numeric(df["max_power"], errors="coerce")  # Texto -> número (lo que falle queda NaN)
    df = df.dropna().copy()                                            # Quita las filas que no se pudieron convertir
    df["km_per_year"] = df["km_driven"] / (2026 - df["year"] + 1)      # Km por año de uso (intensidad de uso)
    df["engine_power_ratio"] = df["engine"] / (df["max_power"] + 1e-5) # Cilindrada / potencia (+1e-5 evita dividir por 0)
    return df


def preparar_datos():
    """
    Carga, limpia, separa y expande las features. Devuelve un diccionario con todo
    lo que necesitan los puntos 3 y 4.
    """
    df = limpiar(pd.read_csv(RUTA_DATOS))             # Carga y limpieza
    X = df[FEATURE_COLS].values                       # Matriz de features (8 columnas)
    y = df["selling_price"].values                    # Variable objetivo

    # train_size=35: solo 35 autos para entrenar (el resto, ~7.870, es prueba). Pocos datos = fácil sobreajustar
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=35, random_state=RANDOM_STATE)

    # Grado 3: todas las potencias y productos hasta grado 3 -> 164 features (más que los 35 ejemplos)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)        # fit solo con entrenamiento
    X_test_poly = poly.transform(X_test)              # La prueba usa la misma transformación

    return {
        "df": df,
        "X_train": X_train, "X_test": X_test,                       # 8 features originales
        "X_train_poly": X_train_poly, "X_test_poly": X_test_poly,   # 164 features polinomiales
        "y_train": y_train, "y_test": y_test,
        "feature_names_poly": poly.get_feature_names_out(FEATURE_COLS),  # Nombres como "year^3" o "km_driven engine"
    }


def regresion_lineal(datos):
    """Entrena LinearRegression con las 164 features y devuelve el RMSE de entrenamiento y de prueba."""
    lr = LinearRegression().fit(datos["X_train_poly"], datos["y_train"])
    rmse_train = np.sqrt(mean_squared_error(datos["y_train"], lr.predict(datos["X_train_poly"])))
    rmse_test = np.sqrt(mean_squared_error(datos["y_test"], lr.predict(datos["X_test_poly"])))
    return rmse_train, rmse_test


if __name__ == "__main__":                            # Solo corre si se ejecuta este archivo (no al importarlo)
    explorar(pd.read_csv(RUTA_DATOS))
    datos = preparar_datos()
    print(f"\nEntrenamiento: {datos['X_train_poly'].shape[0]} ejemplos x {datos['X_train_poly'].shape[1]} features")
    rmse_train, rmse_test = regresion_lineal(datos)
    print("\n=== RESULTADOS PUNTO 1 ===")
    print(f"RMSE de entrenamiento: {rmse_train:,.0f}")
    print(f"RMSE de prueba:        {rmse_test:,.0f}")
    print(f"Prueba / entrenamiento: {rmse_test / rmse_train:,.0f} veces  ->  sobreajuste (alta varianza)")
