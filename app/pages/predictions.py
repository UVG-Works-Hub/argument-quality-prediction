import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
import plotly.express as px
import pandas as pd

class NeuralNetPyTorch(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetPyTorch, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Load models and encoders
def load_models():
    log_reg = joblib.load('saved/models/logistic_regression.pkl')
    xgb = joblib.load('saved/models/xgboost.pkl')
    keras_nn = load_model('saved/models/keras_nn_model.h5')

    # Recreate the PyTorch model architecture and load the weights
    pytorch_nn = NeuralNetPyTorch(input_size=5007, num_classes=3)
    pytorch_nn.load_state_dict(torch.load('saved/models/best_pytorch_nn_model.pth'))
    pytorch_nn.eval()

    le = joblib.load('saved/encoders/label_encoder.pkl')
    tfidf = joblib.load('saved/encoders/tfidf_vectorizer.pkl')
    scaler = joblib.load('saved/encoders/standard_scaler.pkl')
    ohe = joblib.load('saved/encoders/onehot_encoder.pkl')

    return log_reg, xgb, keras_nn, pytorch_nn, le, tfidf, scaler, ohe

def predict(log_reg, xgb, keras_nn, pytorch_nn, le, tfidf, scaler, ohe, text_input, discourse_type):
    # Preprocess the data
    text_transformed = tfidf.transform([text_input])
    text_length = np.array([[len(text_input)]])
    text_length_scaled = scaler.transform(text_length)

    # Encode discourse_type using OneHotEncoder
    discourse_transformed = ohe.transform([[discourse_type]])

    # Concatenate all features
    features = np.hstack([text_transformed.toarray(), text_length_scaled, discourse_transformed.toarray()])

    # Make predictions and get probabilities
    # Logistic Regression
    log_reg_prob = log_reg.predict_proba(features)
    log_reg_pred = log_reg.predict(features)

    # XGBoost
    xgb_prob = xgb.predict_proba(features)
    xgb_pred = xgb.predict(features)

    # Keras
    keras_prob = keras_nn.predict(features)
    keras_pred = np.argmax(keras_prob, axis=1)

    # PyTorch
    pytorch_nn.eval()
    with torch.no_grad():
        inputs = torch.tensor(features, dtype=torch.float32)
        outputs = pytorch_nn(inputs)
        pytorch_prob = torch.nn.functional.softmax(outputs, dim=1).numpy()
        _, pytorch_pred = torch.max(outputs, 1)

    # Decode predictions
    log_reg_class = le.inverse_transform(log_reg_pred)[0]
    xgb_class = le.inverse_transform(xgb_pred)[0]
    keras_class = le.inverse_transform(keras_pred)[0]
    pytorch_class = le.inverse_transform(pytorch_pred.numpy())[0]

    # Get probabilities for all classes
    log_reg_probs = log_reg_prob[0]
    xgb_probs = xgb_prob[0]
    keras_probs = keras_prob[0]
    pytorch_probs = pytorch_prob[0]

    # Get probability for the predicted class
    log_reg_prob_class = log_reg_prob[0][log_reg_pred[0]]
    xgb_prob_class = xgb_prob[0][xgb_pred[0]]
    keras_prob_class = keras_prob[0][keras_pred[0]]
    pytorch_prob_class = pytorch_prob[0][pytorch_pred.numpy()[0]]

    return (log_reg_class, log_reg_prob_class, log_reg_probs), \
           (xgb_class, xgb_prob_class, xgb_probs), \
           (keras_class, keras_prob_class, keras_probs), \
           (pytorch_class, pytorch_prob_class, pytorch_probs)

def show():
    st.title("Speech Effectiveness Predictions")

    # Text input for the discourse
    text_input = st.text_area("Enter the text of the speech", "")

    # Discourse type selection
    discourse_type_options = ['Claim', 'Concluding Statement', 'Counterclaim', 'Evidence', 'Lead', 'Position', 'Rebuttal']
    discourse_type = st.selectbox("Select the type of discourse", discourse_type_options)

    if st.button("Predict"):
        if text_input and discourse_type:
            # Load models
            log_reg, xgb, keras_nn, pytorch_nn, le, tfidf, scaler, ohe = load_models()

            # Make predictions
            (log_reg_class, log_reg_prob_class, log_reg_probs), \
            (xgb_class, xgb_prob_class, xgb_probs), \
            (keras_class, keras_prob_class, keras_probs), \
            (pytorch_class, pytorch_prob_class, pytorch_probs) = predict(
                log_reg, xgb, keras_nn, pytorch_nn, le, tfidf, scaler, ohe, text_input, discourse_type
            )

            # Determine the overall predicted class based on majority vote
            model_predictions = [log_reg_class, xgb_class, keras_class, pytorch_class]
            majority_class = max(set(model_predictions), key=model_predictions.count)
            majority_count = model_predictions.count(majority_class)

            st.markdown(f"""
                <div style="
                    background-color: #262730;
                    padding: 20px;
                    border-radius: 10px;
                    border: 1px solid #ddd;
                    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1)
                    margin-top: 20px;
                    margin-bottom: 20px;
                ">
                    <h3 style="color: #fafafa;">Overall Prediction: <span style="color: #007bff;">{majority_class}</span></h3>
                    <p style="font-size: 18px; color: #fafafa;">{majority_count} out of 4 models predict this class</p>
                </div>
            """, unsafe_allow_html=True)

            # Individual model predictions
            st.write("### Prediction Results by Model:")
            predictions_df = pd.DataFrame({
                'Model': ['Logistic Regression', 'XGBoost', 'Keras Neural Network', 'PyTorch Neural Network'],
                'Predicted Class': [log_reg_class, xgb_class, keras_class, pytorch_class],
                'Probability (%)': [log_reg_prob_class * 100, xgb_prob_class * 100, keras_prob_class * 100, pytorch_prob_class * 100]
            })

            fig_individual = px.bar(predictions_df, x='Model', y='Probability (%)', color='Predicted Class',
                                    text='Predicted Class', title="Predicted Class and Probability by Model")
            fig_individual.update_traces(textposition='outside')
            st.plotly_chart(fig_individual)

            # Probabilities for all classes across models
            st.write("### Class Probabilities by Model:")
            class_names = le.classes_
            probabilities_data = {
                'Class': list(class_names) * 4,
                'Probability (%)': [
                    *[prob * 100 for prob in log_reg_probs],
                    *[prob * 100 for prob in xgb_probs],
                    *[prob * 100 for prob in keras_probs],
                    *[prob * 100 for prob in pytorch_probs]
                ],
                'Model': ['Logistic Regression'] * len(class_names) + ['XGBoost'] * len(class_names) +
                          ['Keras Neural Network'] * len(class_names) + ['PyTorch Neural Network'] * len(class_names)
            }
            probabilities_df = pd.DataFrame(probabilities_data)

            # Displaying probabilities in a grouped bar chart
            fig_probs = px.bar(probabilities_df, x='Class', y='Probability (%)', color='Model', barmode='group',
                               title="Class Probabilities by Model")
            st.plotly_chart(fig_probs)

        else:
            st.warning("Please enter a text and select a discourse type to make a prediction.")
