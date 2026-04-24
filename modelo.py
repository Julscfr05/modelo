import joblib
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

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

archivo = st.file_uploader("Sube el Dataset_viviendas.csv", type=["csv"])

if archivo is not None:
    data = pd.read_csv(archivo)

    X = data[COLUMNAS]
    y = data["valor_casa"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        resultados.append({
            "Modelo": nombre,
            "MSE": round(mean_squared_error(y_test, y_pred), 2),
            "MAE": round(mean_absolute_error(y_test, y_pred), 2),
            "R2": round(r2_score(y_test, y_pred), 4),
            "ModeloEntrenado": modelo
        })

    tabla = pd.DataFrame([
        {
            "Modelo": r["Modelo"],
            "MSE": r["MSE"],
            "MAE": r["MAE"],
            "R2": r["R2"]
        }
        for r in resultados
    ]).sort_values(by="R2", ascending=False)

    st.subheader("Tabla comparativa de modelos")
    st.dataframe(tabla)

    mejor = max(resultados, key=lambda x: x["R2"])
    mejor_modelo = mejor["ModeloEntrenado"]
    mejor_nombre = mejor["Modelo"]

    st.success(f"El mejor modelo fue: {mejor_nombre}")

    st.subheader("Ingrese los datos de la nueva vivienda")

    m2 = st.number_input("Metros cuadrados", min_value=0.0)
    nhabitaciones = st.number_input("Número de habitaciones", min_value=0, step=1)
    nbanos = st.number_input("Número de baños", min_value=0, step=1)
    antiguedad = st.number_input("Antigüedad", min_value=0, step=1)
    distancia_centro = st.number_input("Distancia al centro", min_value=0.0)
    estrato = st.number_input("Estrato", min_value=1, max_value=6, step=1)
    garaje = st.selectbox("Garaje", [0, 1])
    zona = st.selectbox("Zona", [0, 1], format_func=lambda x: "Rural" if x == 0 else "Urbana")

    if st.button("Predecir valor de la casa"):
        nuevo = pd.DataFrame(
            [[m2, nhabitaciones, nbanos, antiguedad, distancia_centro, estrato, garaje, zona]],
            columns=COLUMNAS
        )

        pred = mejor_modelo.predict(nuevo)

        st.subheader("Resultado")
        st.write(f"Predicción usando {mejor_nombre}:")
        st.success(f"${round(pred[0], 2):,.2f}")