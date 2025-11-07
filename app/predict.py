import joblib
import numpy as np

model = joblib.load("app/model.joblib")

flower_types_list = ["Iris Setosa", "Iris Versicolour", "Iris Virginica"]

def predict(features):
    data = np.array([features])
    result = model.predict(data)[0]
    return flower_types_list[result]