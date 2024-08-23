import streamlit as st
import pandas as pd
import math

st.title("Modelo para la deteccion de fraudes por Montos")

# Cargar archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
if uploaded_file is not None:
    # Leer el archivo CSV
    df = pd.read_csv(uploaded_file)
    
# Mostrar los primeros registros
    st.write("Vista previa del archivo subido:")
    st.write(df.head())
    
    # Implementar lógica para la detección de fraudes
    st.write("Procesando los datos para la detección de fraudes...")
    
    # Ejemplo de análisis: detectar montos que sean significativamente altos
    threshold = st.slider("Selecciona el umbral para detectar fraudes", 1000, 100000, 10000)
    
    # Filtrar los datos con montos mayores al umbral
    fraud_cases = df[df['monto'] > threshold]
    
    # Mostrar los posibles casos de fraude
    st.write("Posibles casos de fraude detectados:")
    st.write(fraud_cases)
