import streamlit as st
from app.pages import home, metrics

def main():
    st.set_page_config(page_title="Modelo de Clasificación", layout="wide")

    # Barra lateral de navegación
    st.sidebar.title("Navegación")
    page = st.sidebar.radio("Ir a", ["Inicio", "Métricas", "Predicciones"])

    # Páginas
    if page == "Inicio":
        home.show()
    elif page == "Métricas":
        metrics.show()
    # elif page == "Predicciones":
    #     prediction.show()

if __name__ == "__main__":
    main()
