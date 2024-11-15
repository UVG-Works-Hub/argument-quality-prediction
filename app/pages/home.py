import streamlit as st

def show():
    st.title("Welcome to the Argument Effectiveness Prediction System")
    st.write("""
        This application allows you to upload new text data and predict the effectiveness of arguments using different machine learning models,
        such as logistic regression, XGBoost, and neural networks with Keras and PyTorch.
    """)
    st.write("### Available Models:")
    st.write("""
        1. Logistic Regression
        2. XGBoost
        3. Keras Neural Network
        4. PyTorch Neural Network
    """)
    st.write("Use the 'Metrics' and 'Predictions' sections to explore model performance and test the models with new data.")
