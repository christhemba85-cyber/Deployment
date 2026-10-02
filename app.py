import streamlit as st
import pandas as pd
import joblib

# Load trained model and feature configuration
model = joblib.load("machine_failure_model.pkl")
feature_columns = joblib.load("model_features.pkl")


def predict_machine_failure(
    product_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear
):
    input_data = pd.DataFrame({
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
        "Type_H": [1 if product_type == "H" else 0],
        "Type_L": [1 if product_type == "L" else 0],
        "Type_M": [1 if product_type == "M" else 0]
    })

    input_data = input_data[feature_columns]

    probability = model.predict_proba(input_data)[0, 1]
    prediction = int(probability >= 0.5)

    return prediction, probability


# Application title
st.title("Machine Failure Prediction System")
st.write("Enter the machine operating conditions below to predict the probability of failure.")


# Input fields
product_type = st.selectbox(
    "Product Type",
    ["L", "M", "H"]
)

air_temperature = st.number_input(
    "Air Temperature [K]",
    min_value=290.0,
    max_value=310.0,
    value=300.0
)

process_temperature = st.number_input(
    "Process Temperature [K]",
    min_value=300.0,
    max_value=320.0,
    value=310.0
)

rotational_speed = st.number_input(
    "Rotational Speed [rpm]",
    min_value=500,
    max_value=3000,
    value=1500
)

torque = st.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=80.0,
    value=40.0
)

tool_wear = st.number_input(
    "Tool Wear [min]",
    min_value=0,
    max_value=300,
    value=100
)


# Prediction button
if st.button("Predict Machine Failure"):

    prediction, probability = predict_machine_failure(
        product_type,
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear
    )

    st.write(
        f"Estimated failure probability: **{probability * 100:.2f}%**"
    )

    if prediction == 1:
        st.error("⚠️ FAILURE LIKELY")
    else:
        st.success("✅ NO FAILURE PREDICTED")
# Clear user input
if "Air temperature [K]" not in st.session_state:
    st.session_state.air_temperature = 0
if "Process temperature [K]" not in st.session_state:
    st.session_state.process_temperature = 0

# Button to reset
if st.button("Reset"):
    st.session_state.air_temperature = 0
    st.session_state.process_temperature = 0
    st.experimental_rerun()

st.write("Air temperature [K]", st.session_state.air_temperature)
st.write("Process temperature [K]", st.session_state.process_temperature)
