import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

# Título de la aplicación
st.title('Detección de Fraudes - Modelo de Regresión Logística')

# Subida del archivo Excel
uploaded_file = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])

if uploaded_file is not None:
    # Cargar los datos desde el archivo Excel
    df = pd.read_excel(uploaded_file, sheet_name='FactBusDetail')
    
    # Mostrar un resumen de los datos
    st.write("Vista previa de los datos:")
    st.write(df.head())
    
    # Preprocesamiento
    label_encoder = LabelEncoder()
    df['Fraude'] = label_encoder.fit_transform(df['Fraude'])
    
    fraud_datos = df[df['R1Monto'] > 5000]
    X = fraud_datos.drop('Fraude', axis=1)
    y = fraud_datos['Fraude']
    
    # División de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Escalado de los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenamiento del modelo
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Validación cruzada
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    st.write("Puntuaciones de validación cruzada: ", cv_scores)
    
    # Predicción y matriz de confusión
    y_pred = model.predict(X_test_scaled)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.write("Matriz de confusión:")
    st.write(conf_matrix)
    
    st.write("Precisión del modelo: ", accuracy)
    
    st.write("Informe de clasificación:")
    st.write(classification_report(y_test, y_pred))

    # Nota final
    st.write("El modelo ha sido entrenado y evaluado con éxito.")
