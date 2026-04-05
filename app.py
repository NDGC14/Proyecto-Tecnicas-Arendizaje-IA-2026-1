import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Predicción de Incidencia VIH",
    page_icon="📊",
    layout="centered"
)

# ==========================================
# CARGA DEL MODELO Y SCALER
# ==========================================
model = joblib.load("best_model_vih.pkl")

# Como en tu caso ganó Regresión Lineal, usamos scaler
scaler = joblib.load("scaler_vih.pkl")

# ==========================================
# LISTA DE MUNICIPIOS (basada en tus columnas)
# ==========================================
municipios = [
    "Bogotá, D.C.",
    "Cajicá",
    "Chía",
    "Cota",
    "El Rosal",
    "Facatativá",
    "Funza",
    "Gachancipá",
    "La calera",
    "Madrid",
    "Mosquera",
    "Sibaté",
    "Soacha",
    "Sopó",
    "Tabio",
    "Tenjo",
    "Tocancipá",
    "Zipaquirá"
]

# ==========================================
# TÍTULO Y DESCRIPCIÓN
# ==========================================
st.title("📊 Predicción de Incidencia del VIH")
st.markdown("""
Esta aplicación estima la **incidencia del VIH (casos por cada 100.000 habitantes)**  
utilizando el **mejor modelo de regresión entrenado** en el proyecto.

### Variables de entrada:
- Casos observados
- Población
- Muertes observadas
- Proporción de muertes femeninas
- Proporción de casos femeninos
- Tasa de mortalidad
- Año relativo
- Municipio
""")

st.info("""
⚠️ **Nota importante:**  
Esta herramienta es un **estimador analítico** basado en variables epidemiológicas observadas.  
No reemplaza un sistema de vigilancia prospectiva, sino que sirve como apoyo para el análisis e interpretación del comportamiento territorial del VIH.
""")

# ==========================================
# ENTRADAS DEL USUARIO
# ==========================================
st.subheader("Ingrese los valores de entrada")

casos = st.number_input(
    "Casos observados",
    min_value=0.0,
    value=10.0,
    step=1.0
)

poblacion = st.number_input(
    "Población",
    min_value=1.0,
    value=100000.0,
    step=1000.0
)

muertes = st.number_input(
    "Muertes observadas",
    min_value=0.0,
    value=1.0,
    step=1.0
)

prop_muertes_fem = st.slider(
    "Proporción de muertes femeninas",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.01
)

prop_casos_fem = st.slider(
    "Proporción de casos femeninos",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.01
)

tasa_mortalidad = st.number_input(
    "Tasa de mortalidad (por 100.000 hab.)",
    min_value=0.0,
    value=1.0,
    step=0.1
)

anio_rel = st.number_input(
    "Año relativo (ej. 0 para el primer año del dataset)",
    min_value=0,
    value=5,
    step=1
)

municipio = st.selectbox(
    "Municipio",
    municipios
)

# ==========================================
# CONSTRUIR VECTOR DE FEATURES
# ==========================================
def build_input_dataframe():
    # Transformaciones logarítmicas consistentes con el entrenamiento
    log_casos = np.log1p(casos)
    log_pob = np.log1p(poblacion)
    log_muertes = np.log1p(muertes)

    # Diccionario base con TODAS las columnas
    data = {
        'log_Casos': log_casos,
        'log_Pob': log_pob,
        'log_Muertes': log_muertes,
        'Prop_Muertes_Femenino': prop_muertes_fem,
        'Prop_Casos_Femenino': prop_casos_fem,
        'Tasa_Mortalidad': tasa_mortalidad,
        'Año_Rel': anio_rel,
        'Municipio_Bogotá, D.C.': 0,
        'Municipio_Cajicá': 0,
        'Municipio_Chía': 0,
        'Municipio_Cota': 0,
        'Municipio_El Rosal': 0,
        'Municipio_Facatativá': 0,
        'Municipio_Funza': 0,
        'Municipio_Gachancipá': 0,
        'Municipio_La calera': 0,
        'Municipio_Madrid': 0,
        'Municipio_Mosquera': 0,
        'Municipio_Sibaté': 0,
        'Municipio_Soacha': 0,
        'Municipio_Sopó': 0,
        'Municipio_Tabio': 0,
        'Municipio_Tenjo': 0,
        'Municipio_Tocancipá': 0,
        'Municipio_Zipaquirá': 0
    }

    # Activar dummy del municipio seleccionado
    municipio_col = f"Municipio_{municipio}"
    if municipio_col in data:
        data[municipio_col] = 1

    # Convertir a DataFrame con el orden exacto
    df_input = pd.DataFrame([data])

    ordered_columns = [
        'log_Casos', 'log_Pob', 'log_Muertes',
        'Prop_Muertes_Femenino', 'Prop_Casos_Femenino',
        'Tasa_Mortalidad', 'Año_Rel',
        'Municipio_Bogotá, D.C.', 'Municipio_Cajicá', 'Municipio_Chía',
        'Municipio_Cota', 'Municipio_El Rosal', 'Municipio_Facatativá',
        'Municipio_Funza', 'Municipio_Gachancipá', 'Municipio_La calera',
        'Municipio_Madrid', 'Municipio_Mosquera', 'Municipio_Sibaté',
        'Municipio_Soacha', 'Municipio_Sopó', 'Municipio_Tabio',
        'Municipio_Tenjo', 'Municipio_Tocancipá', 'Municipio_Zipaquirá'
    ]

    df_input = df_input[ordered_columns]
    return df_input

# ==========================================
# BOTÓN DE PREDICCIÓN
# ==========================================
if st.button("🔍 Predecir incidencia"):
    try:
        input_df = build_input_dataframe()

        # Escalado (porque ganó Regresión Lineal)
        input_scaled = scaler.transform(input_df)

        # Predicción
        prediction = model.predict(input_scaled)[0]

        # Evitar negativos por seguridad
        prediction = max(0, prediction)

        st.success(f"📈 Incidencia estimada: **{prediction:.2f} casos por cada 100.000 habitantes**")

        # Interpretación sencilla
        if prediction < 10:
            nivel = "Baja"
        elif prediction < 25:
            nivel = "Moderada"
        else:
            nivel = "Alta"

        st.markdown(f"### Interpretación")
        st.write(f"El nivel estimado de incidencia es: **{nivel}**")

        st.caption("Clasificación orientativa usada solo para fines interpretativos dentro de la aplicación.")

        # Mostrar resumen de entrada
        with st.expander("Ver detalle de los datos ingresados"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"Ocurrió un error al generar la predicción: {e}")