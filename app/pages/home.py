import streamlit as st

def show():
    st.title("Bienvenido al Sistema de Predicción de Efectividad del Discurso")
    st.write("""
        Esta aplicación permite cargar nuevos datos de texto y predecir la efectividad del discurso utilizando diferentes modelos de machine learning,
        tales como regresión logística, XGBoost, redes neuronales con Keras y PyTorch.
    """)
    st.write("### Modelos disponibles:")
    st.write("""
        1. Regresión Logística
        2. XGBoost
        3. Red Neuronal Keras
        4. Red Neuronal PyTorch
    """)
    st.write("Utilice las secciones de 'Métricas' y 'Predicciones' para explorar el desempeño y probar los modelos con nuevos datos.")
