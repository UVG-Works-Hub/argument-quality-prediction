import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

SAVE_PATH = ''

# Load pre-saved model metrics
def load_metrics():
    with open(SAVE_PATH + 'saved/metrics/models_performance.pkl', 'rb') as f:
        models_performance = pickle.load(f)
    return models_performance

# Load feature importance data
def load_feature_importance(model_name):
    filename_map = {
        'Logistic Regression': 'logistic_regression_feature_importances.csv',
        'XGBoost': 'xgboost_feature_importances.csv',
        'Keras NN': 'keras_nn_shap_importances.csv',
        'PyTorch NN': 'pytorch_nn_shap_importances.csv'
    }
    file_path = os.path.join(SAVE_PATH, 'saved/metrics', filename_map[model_name])
    return pd.read_csv(file_path)

def show():
    st.title("Interactive Model Metrics Dashboard")
    models_performance = load_metrics()

    # Tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Model Performance", "Confusion Matrix Comparison", "Feature Importance"])

    # Model Performance Tab (Accuracy, F1-Score, etc.)
    with tab1:
        st.subheader("Model Performance Metrics")
        # Filters relevant to performance metrics
        selected_models = st.multiselect(
            "Select Models", list(models_performance.keys()), default=list(models_performance.keys())
        )
        selected_metrics = st.multiselect(
            "Select Metrics", ["accuracy", "f1-score", "precision", "recall"], default=["accuracy", "f1-score"]
        )
        selected_classes = st.multiselect(
            "Select Classes", ["Adequate", "Effective", "Ineffective"], default=["Adequate", "Effective", "Ineffective"]
        )
        plot_type = st.selectbox("Plot Type", ["Bar Chart", "Radar Chart", "Line Chart"], index=0)

        # Displaying Accuracy and F1 Plots
        if "accuracy" in selected_metrics:
            st.subheader("Model Accuracy")
            accuracy_data = {
                'Model': [model for model in selected_models],
                'Accuracy': [models_performance[model]['accuracy'] for model in selected_models]
            }
            accuracy_df = pd.DataFrame(accuracy_data)

            if plot_type == "Radar Chart":
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
                fig_accuracy = px.line(accuracy_df, x='Model', y='Accuracy', title='Model Accuracy Comparison') if plot_type == "Line Chart" else px.bar(accuracy_df, x='Model', y='Accuracy', title='Model Accuracy Comparison')
                fig_accuracy.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig_accuracy)

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

    # Confusion Matrix Comparison Tab
    with tab2:
        st.subheader("Confusion Matrix Comparison")
        # Filters specific to confusion matrix comparison
        cm_selected_models = st.multiselect(
            "Select Models for Confusion Matrix Comparison", list(models_performance.keys()), default=list(models_performance.keys())
        )

        if len(cm_selected_models) > 1:
            st.write("Comparing Confusion Matrices across Models")
            fig_cm_comparison = make_subplots(
                rows=1, cols=len(cm_selected_models),
                subplot_titles=cm_selected_models,
                shared_yaxes=True
            )

            for idx, model_name in enumerate(cm_selected_models):
                cm = models_performance[model_name]['confusion_matrix']
                normalized_cm = cm / cm.sum(axis=1, keepdims=True)  # Normalize by row sums

                heatmap = go.Heatmap(
                    z=normalized_cm,
                    x=['Predicted Adequate', 'Predicted Effective', 'Predicted Ineffective'],
                    y=['Adequate', 'Effective', 'Ineffective'],
                    colorscale="Blues",
                    zmin=0,
                    zmax=1,
                    showscale=(idx == len(cm_selected_models) - 1),  # Show color bar only for the last subplot
                    colorbar_title="Normalized" if idx == len(cm_selected_models) - 1 else None
                )
                fig_cm_comparison.add_trace(heatmap, row=1, col=idx + 1)

            fig_cm_comparison.update_layout(
                title="Normalized Confusion Matrix Comparison",
                height=400,
                template="plotly_white"
            )

            st.plotly_chart(fig_cm_comparison)

        else:
            for model_name in cm_selected_models:
                st.write(f"### {model_name} Confusion Matrix")
                cm = models_performance[model_name]['confusion_matrix']
                labels = ['Adequate', 'Effective', 'Ineffective']
                fig_cm = ff.create_annotated_heatmap(
                    z=cm, x=[f"Predicted {label}" for label in labels], y=labels, colorscale='Blues', showscale=True
                )
                fig_cm.update_layout(title_text=f"{model_name} Confusion Matrix")
                st.plotly_chart(fig_cm)

    # Feature Importance Tab
    with tab3:
        st.subheader("Feature Importance")
        # Filter for the top-N features
        fi_selected_models = st.multiselect(
            "Select Models for Feature Importance", list(models_performance.keys()), default=list(models_performance.keys())
        )
        top_n_features = st.selectbox("Top Features to Display", [100, 50, 10], index=2)

        for model_name in fi_selected_models:
            try:
                feature_df = load_feature_importance(model_name)

                if 'importance' in feature_df.columns:
                    feature_df = feature_df.nlargest(top_n_features, 'importance')
                    fig = px.bar(
                        feature_df,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title=f"Top {top_n_features} Feature Importances for {model_name}",
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig)

                elif 'shap_importance_class_Adequate' in feature_df.columns:
                    feature_df = feature_df.nlargest(top_n_features, ['shap_importance_class_Adequate', 'shap_importance_class_Effective', 'shap_importance_class_Ineffective'])
                    feature_df = feature_df.melt(id_vars='feature', var_name='Class', value_name='Importance')

                    fig = px.bar(
                        feature_df,
                        x='Importance',
                        y='feature',
                        color='Class',
                        orientation='h',
                        title=f"Top {top_n_features} SHAP Feature Importances for {model_name}",
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig)

            except FileNotFoundError:
                st.warning(f"Feature importance data for {model_name} not found.")
