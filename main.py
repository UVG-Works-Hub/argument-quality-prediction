import streamlit as st
from app.pages import home, metrics, predictions

def main():
    st.set_page_config(page_title="Classification Model", layout="wide")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Metrics", "Predictions"])

    # Pages
    if page == "Home":
        home.show()
    elif page == "Metrics":
        metrics.show()
    elif page == "Predictions":
        predictions.show()

if __name__ == "__main__":
    main()
