# Tarea 2 — Inteligencia Artificial: Regresión lineal, Ridge y Lasso

**Estudiantes:** Laura Ruidiaz y Angelo Maiale

Predicción del precio de venta de autos usados (dataset de CarDekho, Kaggle) para estudiar el sobreajuste de la regresión lineal y cómo corregirlo con regularización.

## Contenido

| Punto | Tema |
|---|---|
| 1 | Exploración, limpieza y sobreajuste forzado con `LinearRegression` (35 ejemplos y features polinomiales de grado 3) |
| 2 | Clase `MyRidge`: Ridge con solución cerrada, implementada solo con NumPy |
| 3 | Corrección del sobreajuste con `MyRidge`: estandarización, elección de λ por validación cruzada y verificación contra sklearn |
| 4 | Ridge vs. Lasso: coeficientes, caminos de regularización, selección de features y geometría L1/L2 |

## Estructura

```
├── data/dataset.csv            # Dataset de autos usados
├── notebook_principal.ipynb    # Solución completa con teoría y resultados
└── my_ridge.py                 # Clase MyRidge
```

## Cómo ejecutar

Con [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run jupyter notebook notebook_principal.ipynb
```

O con pip: `pip install numpy pandas matplotlib scikit-learn ipykernel` y abrir el notebook.

## Resultados

| Modelo | RMSE de prueba |
|---|---|
| Regresión lineal (sobreajustada) | ≈ 219 millones |
| MyRidge / Ridge (λ ≈ 720) | ≈ 727 mil |
| Lasso (α ≈ 95 mil, 4 de 164 features) | ≈ 583 mil |

`MyRidge` coincide con `sklearn.linear_model.Ridge` en coeficientes, intercepto, R² y RMSE.
