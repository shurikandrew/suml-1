import streamlit as st
import predict as pr

with st.form("iris_prediction_form"):
    sepal_length = st.number_input("Sepal Length", min_value=0.0)
    sepal_width = st.number_input("Sepal Width", min_value=0.0)
    petal_length = st.number_input("Petal Length", min_value=0.0)
    petal_width = st.number_input("Petal Width", min_value=0.0)

    predict_button = st.form_submit_button("Predict")

    if predict_button:
        features = [sepal_length, sepal_width, petal_length, petal_width]
        prediction = pr.predict(features)

        st.success(f"The prediction is: {prediction}")