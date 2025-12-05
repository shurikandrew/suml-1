import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

model = joblib.load(MODEL_PATH)


def predict(features):
    data = np.array([features])
    result = model.predict(data)[0]
    return result