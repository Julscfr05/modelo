import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Columnas del dataset
COLUMNAS = [
    "metros_cuadrados",
    "num_habitaciones",
    "num_banos",
    "antiguedad",
    "distancia_centro",
    "estrato",
    "garaje",
    "zona"
]

st.title("Predicción de valor de viviendas")

# 📂 Leer CSV desde el repositorio
DATASET_PATH = "Dataset_viviendas.csv"
data = pd.read_csv(DATASET_PATH)

st.success("Dataset cargado desde el repositorio")
st.subheader("Vista previa del dataset")
st.dataframe(data.head())

# Separar variables
X = data[COLUMNAS]
y = data["valor_casa"]

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelos
modelos = {
    "Regresión Lineal": Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", LinearRegression())
    ]),
    "KNN Regressor": Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", KNeighborsRegressor(n_neighbors=5))
    ]),
    "Árbol de Decisión": Pipeline([
        ("modelo", DecisionTreeRegressor(max_depth=5, random_state=42))
    ])
}

resultados = []
modelos_entrenados = {}

# Entrenar y evaluar
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    resultados.append({
        "Modelo": nombre,
        "MSE": round(mean_squared_error(y_test, y_pred), 2),
        "MAE": round(mean_absolute_error(y_test, y_pred), 2),
        "R2": round(r2_score(y_test, y_pred), 4)
    })

    modelos_entrenados[nombre] = modelo

# Tabla comparativa
tabla_resultados = pd.DataFrame(resultados).sort_values(by="R2", ascending=False)

st.subheader("Tabla comparativa de modelos")
st.dataframe(tabla_resultados)

# Mejor modelo
mejor_nombre = tabla_resultados.iloc[0]["Modelo"]
mejor_modelo = modelos_entrenados[mejor_nombre]

st.success(f"El mejor modelo fue: {mejor_nombre}")

# Inputs
st.subheader("Ingrese los datos de la nueva vivienda")

m2 = st.number_input("Metros cuadrados", min_value=0.0)
nhabitaciones = st.number_input("Número de habitaciones", min_value=0, step=1)
nbanos = st.number_input("Número de baños", min_value=0, step=1)
antiguedad = st.number_input("Antigüedad", min_value=0, step=1)
distancia_centro = st.number_input("Distancia al centro", min_value=0.0)
estrato = st.number_input("Estrato", min_value=1, max_value=6, step=1)
garaje = st.selectbox("Garaje", [0, 1])
zona = st.selectbox(
    "Zona",
    [0, 1],
    format_func=lambda x: "Rural" if x == 0 else "Urbana"
)

# Predicción
if st.button("Predecir valor de la casa"):

    nuevo = pd.DataFrame(
        [[m2, nhabitaciones, nbanos, antiguedad, distancia_centro, estrato, garaje, zona]],
        columns=COLUMNAS
    )

    predicciones = []

    for nombre, modelo in modelos_entrenados.items():
        pred = modelo.predict(nuevo)[0]

        predicciones.append({
            "Modelo": nombre,
            "Predicción": round(pred, 2)
        })

    tabla_pred = pd.DataFrame(predicciones)

    st.subheader("Predicción de todos los modelos")
    st.dataframe(tabla_pred)

    # Mejor modelo destacado
    pred_mejor = mejor_modelo.predict(nuevo)[0]

    st.subheader("Mejor predicción")
    st.success(f"Predicción usando {mejor_nombre}: ${round(pred_mejor, 2):,.2f}")