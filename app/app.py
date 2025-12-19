import streamlit as st
import predict as pr
import json

flower_types_list = ["Iris Setosa", "Iris Versicolour", "Iris Virginica"]

with open("app/model_meta.json") as f:
    meta = json.load(f)

best_model_name = meta.get("best_model", "N/A")
version = meta.get("version", "N/A")
mlflow_run_id = meta.get("mlflow_run_id", "N/A")
metrics = meta.get("metrics", {})
accuracy = metrics.get("accuracy", 0.0)

mlflow_ui_url = f"http://localhost:5000/#/experiments/0/runs/{mlflow_run_id}"

with st.form("iris_prediction_form"):
    sepal_length = st.number_input("Sepal's Length", min_value=0.0)
    sepal_width = st.number_input("Sepal Width", min_value=0.0)
    petal_length = st.number_input("Petal Length", min_value=0.0)
    petal_width = st.number_input("Petal Width", min_value=0.0)

    predict_button = st.form_submit_button("Predict")

    if predict_button:
        features = [sepal_length, sepal_width, petal_length, petal_width]
        prediction = pr.predict(features)
        st.success(f"The prediction is: {flower_types_list[prediction]}")

st.markdown("---")
st.markdown(
    f"Version: {version} • Best model: {best_model_name} • "
    f"MLflow run: {mlflow_run_id} • "
    f"Accuracy: {accuracy:.2f}"
)
