import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

SAVE_PATH = ''

# Load pre-saved model metrics
def load_metrics():
    with open(SAVE_PATH + 'saved/metrics/models_performance.pkl', 'rb') as f:
        models_performance = pickle.load(f)
    return models_performance

def show():
    st.title("Interactive Model Metrics Dashboard")
    models_performance = load_metrics()

    # Sidebar filters
    st.sidebar.header("Filters")
    selected_models = st.sidebar.multiselect(
        "Select Models", list(models_performance.keys()), default=list(models_performance.keys())
    )
    selected_metrics = st.sidebar.multiselect(
        "Select Metrics", ["accuracy", "f1-score", "precision", "recall"], default=["accuracy", "f1-score"]
    )
    selected_classes = st.sidebar.multiselect(
        "Select Classes", ["Adequate", "Effective", "Ineffective"], default=["Adequate", "Effective", "Ineffective"]
    )

    st.sidebar.write("Choose visualization options:")
    plot_type = st.sidebar.selectbox("Plot Type", ["Bar Chart", "Radar Chart", "Line Chart"], index=0)

   # Displaying Accuracy in a Radar or Line Chart for Comparison
    if "accuracy" in selected_metrics:
        st.subheader("Model Accuracy")
        accuracy_data = {
            'Model': [model for model in selected_models],
            'Accuracy': [models_performance[model]['accuracy'] for model in selected_models]
        }
        accuracy_df = pd.DataFrame(accuracy_data)

        if plot_type == "Radar Chart":
            # Radar chart for accuracy comparison
            fig_accuracy = go.Figure()
            fig_accuracy.add_trace(go.Scatterpolar(
                r=accuracy_df['Accuracy'],
                theta=accuracy_df['Model'],
                fill='toself',
                name="Accuracy"
            ))
            fig_accuracy.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Model Accuracy Radar Chart"
            )
            st.plotly_chart(fig_accuracy)
        else:
            # Line chart or bar chart
            fig_accuracy = px.line(accuracy_df, x='Model', y='Accuracy', title='Model Accuracy Comparison') if plot_type == "Line Chart" else px.bar(accuracy_df, x='Model', y='Accuracy', title='Model Accuracy Comparison')
            fig_accuracy.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig_accuracy)

    # F1-Score by Class and Model
    if "f1-score" in selected_metrics:
        st.subheader("F1-Scores by Class and Model")
        f1_data = {'Model': [], 'Class': [], 'F1-Score': []}

        for model_name in selected_models:
            for class_label in selected_classes:
                f1_data['Model'].append(model_name)
                f1_data['Class'].append(class_label)
                f1_data['F1-Score'].append(models_performance[model_name]['classification_report'][class_label]['f1-score'])

        f1_df = pd.DataFrame(f1_data)

        if plot_type == "Radar Chart":
            fig_f1 = go.Figure()
            for model in selected_models:
                fig_f1.add_trace(go.Scatterpolar(
                    r=f1_df[f1_df['Model'] == model]['F1-Score'],
                    theta=f1_df[f1_df['Model'] == model]['Class'],
                    fill='toself',
                    name=model
                ))
            fig_f1.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="F1-Score Radar Chart by Class"
            )
            st.plotly_chart(fig_f1)
        else:
            fig_f1 = px.bar(f1_df, x='Class', y='F1-Score', color='Model', barmode='group', title='F1-Scores by Class') if plot_type == "Bar Chart" else px.line(f1_df, x='Class', y='F1-Score', color='Model', title='F1-Scores by Class')
            fig_f1.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig_f1)

    # Confusion Matrix Comparison
    st.subheader("Confusion Matrix Comparison")
    if len(selected_models) > 1:
        st.write("Comparing Confusion Matrices across Models")

        # Set up subplots for each model's confusion matrix
        fig_cm_comparison = make_subplots(
            rows=1, cols=len(selected_models),
            subplot_titles=selected_models,
            shared_yaxes=True
        )

        # Add each model's normalized confusion matrix to the subplot
        for idx, model_name in enumerate(selected_models):
            cm = models_performance[model_name]['confusion_matrix']
            normalized_cm = cm / cm.sum(axis=1, keepdims=True)  # Normalize by row sums

            heatmap = go.Heatmap(
                z=normalized_cm,
                x=['Predicted Adequate', 'Predicted Effective', 'Predicted Ineffective'],
                y=['Adequate', 'Effective', 'Ineffective'],
                colorscale="Blues",
                zmin=0,
                zmax=1,
                showscale=(idx == len(selected_models) - 1),  # Show color bar only for the last subplot
                colorbar_title="Normalized" if idx == len(selected_models) - 1 else None
            )
            fig_cm_comparison.add_trace(heatmap, row=1, col=idx + 1)

        fig_cm_comparison.update_layout(
            title="Normalized Confusion Matrix Comparison",
            height=400,
            template="plotly_white"
        )

        st.plotly_chart(fig_cm_comparison)

    else:
        # Individual confusion matrices for each model if only one is selected
        for model_name in selected_models:
            st.write(f"### {model_name} Confusion Matrix")
            cm = models_performance[model_name]['confusion_matrix']
            labels = ['Adequate', 'Effective', 'Ineffective']
            fig_cm = ff.create_annotated_heatmap(
                z=cm, x=[f"Predicted {label}" for label in labels], y=labels, colorscale='Blues', showscale=True
            )
            fig_cm.update_layout(title_text=f"{model_name} Confusion Matrix")
            st.plotly_chart(fig_cm)

    # Feature Importance (if available)
    st.subheader("Feature Importance (if available)")
    for model_name in selected_models:
        if 'feature_importances' in models_performance[model_name]:
            st.write(f"### {model_name} Feature Importance")
            importance_data = {
                'Feature': list(range(len(models_performance[model_name]['feature_importances']))),
                'Importance': models_performance[model_name]['feature_importances']
            }
            importance_df = pd.DataFrame(importance_data)
            fig_importance = px.bar(importance_df, x='Feature', y='Importance', title=f"{model_name} Feature Importance")
            st.plotly_chart(fig_importance)
