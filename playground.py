import pandas as pd
import pickle
import joblib

SAVE_PATH = ''


# Cargar las métricas de los modelos previamente guardadas
def load_metrics():
    models_performance = {}
    with open(SAVE_PATH + 'saved/metrics/models_performance.pkl', 'rb') as f:
        models_performance = pickle.load(f)
    return models_performance

models_performance = load_metrics()

print("Hello, world!")
print(models_performance)
