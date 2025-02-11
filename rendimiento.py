# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:33:41 2025

@author: jperezr
"""


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


# Estilo de fondo
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background:
radial-gradient(black 15%, transparent 16%) 0 0,
radial-gradient(black 15%, transparent 16%) 8px 8px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 0 1px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 8px 9px;
background-color:#282828;
background-size:16px 16px;
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# Cargar datos desde el archivo .xlsx
@st.cache_data
def cargar_datos():
    df = pd.read_excel('dataset.xlsx')
    return df

df = cargar_datos()

# Modelo Predictivo
X = df[['Rendimiento_PENSIONISSSTE', 'Comision_PENSIONISSSTE']]
y = df['Afiliados_PENSIONISSSTE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de características para mejorar el rendimiento de algunos modelos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo de Gradient Boosting
modelo_gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
modelo_gb.fit(X_train_scaled, y_train)

# Predicciones
predicciones_gb = modelo_gb.predict(X_test_scaled)
mae_gb = mean_absolute_error(y_test, predicciones_gb)
r2_gb = r2_score(y_test, predicciones_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, predicciones_gb))

# Validación cruzada con MAE
cv_mae_gb = -cross_val_score(modelo_gb, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error').mean()

# Predicciones a futuro
future_fechas = pd.date_range(start='2024-07-01', periods=13, freq='M')[1:]
future_data = pd.DataFrame({
    'Fecha': future_fechas,
    'Rendimiento_PENSIONISSSTE': np.random.uniform(4.5, 6.5, 12),
    'Comision_PENSIONISSSTE': np.random.uniform(0.45, 1.1, 12)
})
future_data['Prediccion_Afiliados'] = modelo_gb.predict(scaler.transform(future_data[['Rendimiento_PENSIONISSSTE', 'Comision_PENSIONISSSTE']]))

# Interfaz en Streamlit
st.title("📊 Modelo Predictivo de Competitividad - PENSIONISSSTE")
st.write("Este modelo predice la evolución de afiliados en función de rendimientos y comisiones.")

# Sidebar de ayuda
with st.sidebar:
    st.header("Ayuda")
    st.write("""
    ### Descripción del Modelo
    Este modelo predictivo utiliza el rendimiento y las comisiones de PENSIONISSSTE para predecir la evolución de los afiliados a lo largo del tiempo.
    
    ### Características:
    - **Rendimiento PENSIONISSSTE (%)**: Representa el rendimiento de las inversiones del fondo PENSIONISSSTE.
    - **Comisión PENSIONISSSTE (%)**: Tasa de comisión aplicada al fondo.
    - **Predicción de Afiliados**: La predicción de la cantidad de afiliados a futuro basada en los valores anteriores de rendimiento y comisión.

    ### ¿Cómo utilizar la aplicación?
    1. Ajusta los valores de "Rendimiento PENSIONISSSTE" y "Comisión PENSIONISSSTE" usando los sliders.
    2. Verás la predicción de afiliados en función de esos valores ajustados.
    3. También podrás visualizar los resultados históricos y las predicciones a futuro.

    ### Métricas de Evaluación
    El modelo es evaluado usando las siguientes métricas:
    - **MAE (Error Medio Absoluto)**: Mide la precisión de las predicciones.
    - **R² (Coeficiente de Determinación)**: Indica qué tan bien las predicciones explican la variabilidad de los datos.
    - **RMSE (Raíz del Error Cuadrático Medio)**: Mide la desviación de las predicciones respecto a los valores reales.

    ### Desarrollado por:
    **Javier Horacio Pérez Ricárdez**

    ### Copyright
    © 2025 Todos los derechos reservados.
    """)

# Sliders interactivos
rendimiento_input = st.slider("Rendimiento PENSIONISSSTE (%)", 4.5, 6.5, 5.5, 0.1)
comision_input = st.slider("Comisión PENSIONISSSTE (%)", 0.45, 1.1, 1.0, 0.01)

# Aplicar cambios en el modelo
nuevo_dato = pd.DataFrame({'Rendimiento_PENSIONISSSTE': [rendimiento_input], 'Comision_PENSIONISSSTE': [comision_input]})
nuevo_dato_scaled = scaler.transform(nuevo_dato)
nueva_prediccion = modelo_gb.predict(nuevo_dato_scaled)[0]
st.write(f"### Predicción de Afiliados con valores ajustados: {int(nueva_prediccion)}")

st.subheader("Datos del Dataset")
st.dataframe(df.tail(10))

st.subheader("Visualización de Rendimientos")
fig = px.line(df, x='Fecha', y=['Rendimiento_PENSIONISSSTE', 'Rendimiento_AFORE1', 'Rendimiento_AFORE2'],
              labels={'value': "Rendimiento (%)", 'variable': "AFORE"}, title="Evolución de Rendimientos")
st.plotly_chart(fig)

st.subheader("Predicción de Afiliados")
st.write(f"Error medio absoluto: {mae_gb:.2f}")
st.write(f"Coeficiente de determinación (R²): {r2_gb:.2f}")
st.write(f"RMSE: {rmse_gb:.2f}")
st.write(f"Validación cruzada MAE: {cv_mae_gb:.2f}")

st.write("### Predicciones vs. Valores Reales")
pred_df = pd.DataFrame({"Real": y_test.values, "Predicción": predicciones_gb})
st.dataframe(pred_df)

# Visualización de predicciones futuras
st.subheader("Predicción de Afiliados a Futuro")
fig_future = px.line(future_data, x='Fecha', y='Prediccion_Afiliados', title="Predicción de Afiliados de Julio 2024 a Julio 2025")
st.plotly_chart(fig_future)

# Pie de página con "Desarrollado por" y copyright
#st.markdown("""
#    ---
#    ### Desarrollado por: Javier Horacio Pérez Ricárdez
#    © 2025 Todos los derechos reservados.
#""", unsafe_allow_html=True)
#)
