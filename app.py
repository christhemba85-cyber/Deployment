import streamlit as st
import pandas as pd
import joblib
import random

# Load trained model and feature configuration
model = joblib.load("machine_failure_model.pkl")
feature_columns = joblib.load("model_features.pkl")


def get_machine_sensor_data():
    """
    Simulates real-time sensor readings from a machine.
    """

    product_type = random.choice(["L", "M", "H"])

    air_temperature = round(random.uniform(298.0, 303.0), 2)

    process_temperature = round(
        air_temperature + random.uniform(8.0, 12.0), 2
    )

    rotational_speed = random.randint(1300, 1800)

    torque = round(random.uniform(30.0, 60.0), 2)

    tool_wear = random.randint(0, 250)

    return (
        product_type,
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear
    )


def predict_machine_failure(
    product_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear
):


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
        

