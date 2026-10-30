"""
AI Apparatus-preservation – Predictive Maintenance Dashboard
Production-hardening revision.

IMPORTANT:
- The built-in machine is a SIMULATION, not a real industrial controller.
- The PC-to-manufacturing feature mapping is EXPERIMENTAL and is not a valid
  measurement of industrial machine condition.
- Never use this Streamlit application or its ML model as a safety controller.
  Real machinery protection must remain with independently engineered PLC /
  safety-controller systems.
- Machine state is session-scoped in this version. MongoDB stores history, but
  does not yet coordinate one shared live machine state across multiple users.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import re
import smtplib
import ssl
import time
import uuid
from datetime import datetime, timezone
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import psutil
import random
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import PyMongoError


# ---------------------------------------------------------------------------
# PHASE 1 – APPLICATION CONFIGURATION, LOGGING AND MODEL VALIDATION
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Machine Failure Prediction System",
    page_icon="🏭",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "machine_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = BASE_DIR / "machine_failure_model.pkl"
FEATURE_PATH = BASE_DIR / "model_features.pkl"

FAILURE_MODE_DIR = BASE_DIR / "failure_mode_models"
FAILURE_MODES_TO_USE = ["HDF", "PWF", "OSF"]

MAX_SESSION_LOG_ROWS = 2000
MACHINE_REFRESH_SECONDS = 1
EMAIL_RETRY_SECONDS = 60
FAILURE_THRESHOLD = float(os.getenv("FAILURE_THRESHOLD", "0.50"))
CRITICAL_SHUTDOWN_THRESHOLD = float(
    os.getenv("CRITICAL_SHUTDOWN_THRESHOLD", "0.75")
)
FAILURE_STREAK_LIMIT = int(os.getenv("FAILURE_STREAK_LIMIT", "3"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ai_apparatus_preservation")


def get_secret(name: str, default: str = "") -> str:
    """Read a Streamlit secret first, then an environment variable."""
    try:
        value = st.secrets.get(name, default)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


def load_model_artifacts():
    """Load trusted local model artifacts once per Streamlit server process."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    if not FEATURE_PATH.is_file():
        raise FileNotFoundError(
            f"Feature configuration not found: {FEATURE_PATH.name}"
        )

    # Only load joblib files that you created or otherwise trust.
    loaded_model = joblib.load(MODEL_PATH)
    loaded_features = joblib.load(FEATURE_PATH)

    if hasattr(loaded_features, "tolist"):
        loaded_features = loaded_features.tolist()
    loaded_features = list(loaded_features)

    if not loaded_features:
        raise ValueError("The model feature list is empty.")
    if len(set(loaded_features)) != len(loaded_features):
        raise ValueError("The model feature list contains duplicate names.")
    if not hasattr(loaded_model, "predict_proba"):
        raise TypeError(
            "The loaded model does not support predict_proba(). "
            "Use a classifier trained with probability support."
        )
    if (
        hasattr(loaded_model, "n_features_in_")
        and loaded_model.n_features_in_ != len(loaded_features)
    ):
        raise ValueError(
            "Model feature count does not match model_features.pkl: "
            f"{loaded_model.n_features_in_} versus {len(loaded_features)}."
        )

    return loaded_model, loaded_features


@st.cache_resource(show_spinner="Loading predictive maintenance model...")
def cached_model_artifacts():
    return load_model_artifacts()


try:
    model, feature_columns = cached_model_artifacts()
except Exception as exc:
    logger.exception("Could not load model artifacts")
    st.error(f"Unable to start the prediction system: {exc}")
    st.info(
        "Place machine_failure_model.pkl and model_features.pkl "
        "in the same directory as app.py."
    )
    st.stop()

# ---------------------------------------------------------------------------
# FAILURE-MODE MODEL LOADING
# ---------------------------------------------------------------------------

def load_failure_mode_artifacts():
    models = {}

    thresholds_path = (
        FAILURE_MODE_DIR / "failure_mode_thresholds.pkl"
    )

    if not thresholds_path.is_file():
        raise FileNotFoundError(
            "failure_mode_thresholds.pkl was not found."
        )

    thresholds = joblib.load(thresholds_path)

    for mode in FAILURE_MODES_TO_USE:

        model_path = (
            FAILURE_MODE_DIR / f"failure_mode_{mode}.pkl"
        )

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Missing failure-mode model: {model_path.name}"
            )

        failure_model = joblib.load(model_path)

        if not hasattr(failure_model, "predict_proba"):
            raise TypeError(
                f"{mode} model does not support predict_proba()."
            )

        if (
            hasattr(failure_model, "n_features_in_")
            and failure_model.n_features_in_ != len(feature_columns)
        ):
            raise ValueError(
                f"{mode} model feature count does not match "
                "the main model feature configuration."
            )

        models[mode] = failure_model

    # Validate and retain the thresholds for the three classifiers.
    validated_thresholds = {}

    for mode in FAILURE_MODES_TO_USE:

        threshold = float(thresholds.get(mode, 0.50))

        if not 0 <= threshold <= 1:
            raise ValueError(
                f"Invalid threshold for {mode}: {threshold}"
            )

        validated_thresholds[mode] = threshold

    return models, validated_thresholds


@st.cache_resource(show_spinner="Loading failure-mode classifiers...")
def cached_failure_mode_artifacts():
    return load_failure_mode_artifacts()


try:
    failure_mode_models, failure_mode_thresholds = (
        cached_failure_mode_artifacts()
    )

    logger.info("Failure-mode classifiers loaded successfully.")

except Exception as exc:
    failure_mode_models = {}
    failure_mode_thresholds = {}

    logger.exception("Failure-mode models could not be loaded")

    st.warning(
        "Failure-mode classification is currently unavailable. "
        "The main machine failure prediction will continue working. "
        f"Details: {type(exc).__name__}"
    )
# ---------------------------------------------------------------------------
# PHASE 2 – INPUT VALIDATION AND PREDICTION
# ---------------------------------------------------------------------------

def validate_machine_readings(
    product_type: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
) -> None:
    values = {
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
    }

    if product_type not in {"L", "M", "H"}:
        raise ValueError("Product type must be L, M or H.")

    for name, value in values.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite numeric value.")

    if air_temperature < 0 or process_temperature < 0:
        raise ValueError("Temperatures cannot be negative.")
    if rotational_speed < 0 or torque < 0 or tool_wear < 0:
        raise ValueError("Speed, torque and tool wear cannot be negative.")
    if process_temperature < air_temperature:
        raise ValueError(
            "Process temperature cannot be lower than air temperature "
            "for this application's input checks."
        )


def predict_machine_failure(
    product_type: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
):
    validate_machine_readings(
        product_type,
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear,
    )

    input_row = {
        "Air temperature [K]": float(air_temperature),
        "Process temperature [K]": float(process_temperature),
        "Rotational speed [rpm]": float(rotational_speed),
        "Torque [Nm]": float(torque),
        "Tool wear [min]": float(tool_wear),
        "Type_H": int(product_type == "H"),
        "Type_L": int(product_type == "L"),
        "Type_M": int(product_type == "M"),
    }

    missing = [column for column in feature_columns if column not in input_row]
    if missing:
        raise ValueError(
            "The model feature file contains unsupported columns: "
            + ", ".join(missing)
        )

    input_data = pd.DataFrame(
        [[input_row[column] for column in feature_columns]],
        columns=feature_columns,
    )

    probabilities = model.predict_proba(input_data)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("The model did not return a binary-class probability.")

    probability = float(probabilities[0, 1])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("The model returned an invalid probability.")

    prediction = int(probability >= FAILURE_THRESHOLD)
    return prediction, probability

def predict_failure_modes(
    product_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
):
    """
    Predict possible HDF, PWF and OSF failure modes.

    TWF is deliberately excluded because its validation performance
    was insufficient for reliable automated classification.
    """

    if not failure_mode_models:
        return {}

    validate_machine_readings(
        product_type,
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear,
    )

    input_row = {
        "Air temperature [K]": float(air_temperature),
        "Process temperature [K]": float(process_temperature),
        "Rotational speed [rpm]": float(rotational_speed),
        "Torque [Nm]": float(torque),
        "Tool wear [min]": float(tool_wear),
        "Type_H": int(product_type == "H"),
        "Type_L": int(product_type == "L"),
        "Type_M": int(product_type == "M"),
    }

    input_data = pd.DataFrame(
        [[input_row[column] for column in feature_columns]],
        columns=feature_columns,
    )

    results = {}

    for mode, failure_model in failure_mode_models.items():

        probabilities = failure_model.predict_proba(input_data)

        classes = list(failure_model.classes_)

        if 1 not in classes:
            failure_probability = 0.0
        else:
            positive_index = classes.index(1)
            failure_probability = float(
                probabilities[0, positive_index]
            )

        threshold = failure_mode_thresholds[mode]

        results[mode] = {
            "probability": failure_probability,
            "threshold": threshold,
            "prediction": int(
                failure_probability >= threshold
            ),
        }

    return results

# ---------------------------------------------------------------------------
# PHASE 3 – MACHINE SIMULATION
# ---------------------------------------------------------------------------

def calculate_running_torque_speed(tool_wear: float):
    wear = max(0.0, min(300.0, float(tool_wear)))

    if wear < 150:
        base_torque = 38 + (wear / 150) * 5
        torque_noise = 2
    elif wear < 230:
        base_torque = 43 + ((wear - 150) / 80) * 8
        torque_noise = 3
    elif wear < 270:
        base_torque = 51 + ((wear - 230) / 40) * 7
        torque_noise = 4
    else:
        base_torque = 58 + ((wear - 270) / 30) * 6
        torque_noise = 6

    torque = round(max(0.0, min(75.0, random.gauss(base_torque, torque_noise))), 2)
    torque_factor = max(0.0, min(1.0, (torque - 20) / 55))

    if wear < 150:
        base_speed, speed_noise = 1500 + torque_factor * 150, 25
    elif wear < 230:
        base_speed, speed_noise = 1450 + torque_factor * 200, 40
    elif wear < 270:
        base_speed, speed_noise = 1350 + torque_factor * 250, 60
    else:
        base_speed, speed_noise = 1250 + torque_factor * 300, 80

    speed = int(max(0, min(1900, random.gauss(base_speed, speed_noise))))
    return torque, speed


def new_machine_state():
    now = time.time()
    return {
        "product_type": random.choice(["L", "M", "H"]),
        "tool_wear": 0.0,
        "cycle": 0,
        "air_temperature": 295.0,
        "last_update": now,
        "machine_running": True,
        "machine_id": f"MCH-{uuid.uuid4().hex[:8].upper()}",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "start_timestamp": datetime.now(timezone.utc),
        "log": [],
        "startup_phase": "OFF",
        "startup_start": now,
        "shutdown_start": None,
        "failure_reason": None,
        "failure_streak": 0,
        "email_status": "NOT SENT",
        "email_attempts": 0,
        "email_last_attempt": 0.0,
        "email_error": None,
    }


def get_machine_sensor_data():
    if "machine_state" not in st.session_state:
        st.session_state.machine_state = new_machine_state()

    machine = st.session_state.machine_state
    now = time.time()

    if machine.get("shutdown_start") is None:
        machine["shutdown_start"] = None

    if machine["startup_phase"] == "SHUTTING DOWN":
        elapsed_shutdown = max(0.0, now - machine["shutdown_start"])
        progress = max(0.0, min(1.0, elapsed_shutdown / 10.0))
        speed = int(1400 * (1.0 - progress))
        torque = round(42.0 * (1.0 - progress), 2)
        air = round(
            max(
                295.0,
                machine["air_temperature"]
                - progress * (machine["air_temperature"] - 295.0),
            ),
            2,
        )
        process = round(max(295.0, air + 6.0 * (1.0 - progress)), 2)

        if elapsed_shutdown >= 10.0:
            machine["machine_running"] = False
            machine["startup_phase"] = "OFF"

        machine["last_update"] = now
        return (
            machine["product_type"], air, process, speed, torque,
            round(machine["tool_wear"], 1)
        )

    if not machine["machine_running"]:
        return (
            machine["product_type"], 295.0, 295.0, 0, 0.0,
            round(machine["tool_wear"], 1)
        )

    elapsed_startup = now - machine["startup_start"]
    elapsed_since_update = max(0.0, now - machine["last_update"])
    machine["last_update"] = now

    if elapsed_startup < 3:
        machine["startup_phase"] = "OFF"
        return machine["product_type"], 0.0, 0.0, 0, 0.0, 0.0

    if elapsed_startup < 10:
        machine["startup_phase"] = "ENGAGING"
        progress = max(0.0, min(1.0, (elapsed_startup - 3) / 7))
        torque = round(progress * 30, 2)
        speed = int(progress * 200)
        air = round(295.0 + random.uniform(-0.2, 0.2), 2)
        return machine["product_type"], air, round(air + 5, 2), speed, torque, round(machine["tool_wear"], 1)

    if elapsed_startup < 20:
        machine["startup_phase"] = "RAMPING UP"
        progress = max(0.0, min(1.0, (elapsed_startup - 10) / 10))
        torque = round(30 + progress * 15, 2)
        speed = int(200 + progress * 1300)
        air = round(295 + progress * 5 + random.uniform(-0.2, 0.2), 2)
        process = round(air + 8 + random.uniform(-0.5, 0.5), 2)
        return machine["product_type"], air, process, speed, torque, round(machine["tool_wear"], 1)

    machine["startup_phase"] = "RUNNING"

    # Tool wear
    machine["tool_wear"] = min(
        300.0, machine["tool_wear"] + elapsed_since_update * 10 
    )
    wear = machine["tool_wear"]

    if wear >= 300.0:
        machine["machine_running"] = False
        machine["failure_reason"] = "Maximum simulated tool wear reached (300 minutes)"
        machine["startup_phase"] = "OFF"
        return machine["product_type"], 295.0, 295.0, 0, 0.0, 300.0

    machine["air_temperature"] = max(
        295.0,
        min(303.0, machine["air_temperature"] + random.uniform(-0.2, 0.2)),
    )
    air = round(machine["air_temperature"], 2)
    process = round(air + random.uniform(6, 9), 2)
    torque, speed = calculate_running_torque_speed(wear)

    return (
        machine["product_type"], air, process, speed, torque, round(wear, 1)
    )


# ---------------------------------------------------------------------------
# PHASE 4 – MONGODB CONNECTION AND DURABLE LOGGING
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def cached_mongo_client(uri: str):
    if not uri:
        raise ValueError("MONGO_URI is not configured.")
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=4000,
        connectTimeoutMS=4000,
        socketTimeoutMS=5000,
        retryWrites=True,
        appname="AI-Apparatus-Preservation",
    )
    client.admin.command("ping")
    return client


def get_mongo_collection():
    mongo_uri = get_secret("MONGO_URI")
    if not mongo_uri:
        try:
            mongo_uri = str(st.secrets["MONGODB"]["MONGO_URI"])
        except Exception:
            mongo_uri = ""

    if not mongo_uri:
        return None, "MONGO_URI not configured."

    try:
        client = cached_mongo_client(mongo_uri)
        return client["machine_failure_db"]["machine_logs"], "Connected"
    except Exception as exc:
        logger.exception("MongoDB connection failed")
        return None, f"MongoDB unavailable: {type(exc).__name__}"


def save_log_to_mongodb(collection, log_entry):
    if collection is None:
        return False, "MongoDB collection is unavailable."

    document = dict(log_entry)
    document["_id"] = str(uuid.uuid4())
    document["created_at"] = datetime.now(timezone.utc)

    try:
        collection.insert_one(document)
        return True, "Saved"
    except PyMongoError as exc:
        logger.exception("MongoDB log write failed")
        return False, f"Database write failed: {type(exc).__name__}"


def append_log_to_csv(log_entry):
    """Append one row instead of rewriting the entire machine history."""
    machine_id = log_entry["machine_id"]
    path = LOG_DIR / f"{machine_id}_LIVE.csv"
    write_header = not path.exists()

    try:
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(log_entry.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(log_entry)
        return True
    except OSError:
        logger.exception("Could not append machine log to CSV")
        return False


# ---------------------------------------------------------------------------
# PHASE 5 – EMAIL ALERTS WITH DELIVERY CONFIRMATION AND RETRIES
# ---------------------------------------------------------------------------

def valid_email_address(address: str) -> bool:
    if not address:
        return False
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", address.strip()))


def generate_maintenance_recommendations(
    readings,
    failure_probability,
    failure_modes=None,
):
    recommendations = []

    failure_modes = failure_modes or {}

    if failure_modes.get("HDF", {}).get("prediction") == 1:
        recommendations.append(
            "POSSIBLE HEAT DISSIPATION FAILURE: Inspect cooling "
            "systems, ventilation, airflow and heat-transfer components."
        )

    if failure_modes.get("PWF", {}).get("prediction") == 1:
        recommendations.append(
            "POSSIBLE POWER FAILURE: Inspect electrical supply, "
            "motor power connections and operating power conditions."
        )

    if failure_modes.get("OSF", {}).get("prediction") == 1:
        recommendations.append(
            "POSSIBLE OVERSTRAIN FAILURE: Inspect mechanical loading, "
            "tooling condition and components exposed to excessive stress."
        )
    air_temp = readings["air_temperature"]
    process_temp = readings["process_temperature"]
    speed = readings["rotational_speed"]
    torque = readings["torque"]
    wear = readings["tool_wear"]

    # Tool wear inspection
    if wear >= 270:
        recommendations.append(
            "CRITICAL: Replace worn tooling and inspect the tool assembly before restarting."
        )
    elif wear >= 230:
        recommendations.append(
            "HIGH: Inspect tooling for excessive wear and schedule replacement."
        )
    elif wear >= 200:
        recommendations.append(
            "WARNING: Monitor tool wear closely and prepare replacement tooling."
        )

    # Temperature inspection
    if air_temp >= 303:
        recommendations.append(
            "Inspect machine ventilation and cooling systems due to elevated air temperature."
        )

    if process_temp >= 313:
        recommendations.append(
            "Inspect process cooling and check for abnormal heat buildup."
        )

    # Torque inspection
    if torque >= 60:
        recommendations.append(
            "Inspect motor loading, bearings and mechanical resistance due to elevated torque."
        )

    # Rotational speed inspection
    if speed > 1800 or (0 < speed < 1200):
        recommendations.append(
            "Verify rotational speed against manufacturer operating specifications."
        )

    # General high-risk recommendation
    if failure_probability >= 0.75:
        recommendations.append(
            "URGENT: Arrange an immediate maintenance inspection before returning the machine to production."
        )

    # Fallback when no individual sensor threshold is exceeded
    if not recommendations:
        recommendations.append(
            "Investigate the elevated predicted failure risk and inspect the machine's critical components."
        )

    recommendations.append(
        "Record inspection findings and verify machine condition before restarting."
    )

    return recommendations



def send_failure_email(owner_email, machine_id, failure_probability,
                       current_readings, log_df, sender_email, sender_password):
    try:
        recommendations = generate_maintenance_recommendations(
    current_readings,
    failure_probability,
    current_readings.get("failure_modes", {}),
)

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = owner_email
        msg['Subject'] = (
            f"🚨 MACHINE FAILURE ALERT - {machine_id} "
            f"- {failure_probability*100:.1f}% Risk"
        )

        recommendation_text = "\n".join(
            f"{i+1}. {action}"
            for i, action in enumerate(recommendations)
        )

        failure_modes = current_readings.get("failure_modes", {})

        failure_mode_text = "\n".join(
            f"{mode}: "
            f"{result['probability'] * 100:.2f}% "
            f"({'POSSIBLE INDICATOR' if result['prediction'] else 'NOT FLAGGED'})"
            for mode, result in failure_modes.items()
        )

        if not failure_mode_text:
            failure_mode_text = "Failure-mode classification unavailable."
        body = f"""
MACHINE FAILURE RISK ALERT
AI Apparatus-preservation System

Machine ID: {machine_id}
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

FAILURE PROBABILITY: {failure_probability*100:.2f}%

POSSIBLE FAILURE-MODE INDICATORS

{failure_mode_text}

CURRENT MACHINE READINGS

Product Type: {current_readings['product_type']}
Air Temperature: {current_readings['air_temperature']} K
Process Temperature: {current_readings['process_temperature']} K
Rotational Speed: {current_readings['rotational_speed']} rpm
Torque: {current_readings['torque']} Nm
Tool Wear: {current_readings['tool_wear']} min

RECOMMENDED MAINTENANCE ACTIONS

{recommendation_text}

IMPORTANT:
These recommendations are based on predefined sensor thresholds
and predicted failure risk. They are inspection guidelines,
not confirmed component diagnoses.

Please have a qualified maintenance technician inspect the machine
and verify its condition before restarting.

The machine operating log is attached.

For a continuous full data file, contact:
Christhembabikitsha@gmail.com

AI Apparatus-preservation
Predictive Maintenance System
"""

        msg.attach(MIMEText(body, 'plain'))

        # Attach machine operating history
        csv_data = log_df.to_csv(index=False)

        part = MIMEBase('application', 'octet-stream')
        part.set_payload(csv_data.encode('utf-8'))
        encoders.encode_base64(part)

        part.add_header(
            'Content-Disposition',
            f'attachment; filename="{machine_id}_failure_log.csv"'
        )

        msg.attach(part)

        context = ssl.create_default_context()

        with smtplib.SMTP_SSL(
            "smtp.gmail.com", 465, context=context
        ) as server:

            server.login(sender_email, sender_password)
            server.sendmail(
                sender_email,
                owner_email,
                msg.as_string()
            )

        return True, "Email sent successfully"

    except Exception as e:
        return False, str(e)



def attempt_failure_email(machine, owner_email, readings, log_df):
    sender_email = get_secret("SENDER_EMAIL")
    sender_password = get_secret("SENDER_PASSWORD")

    if not valid_email_address(owner_email):
        machine["email_status"] = "WAITING FOR VALID RECIPIENT"
        return

    if not sender_email or not sender_password:
        machine["email_status"] = "SENDER CREDENTIALS NOT CONFIGURED"
        return

    if machine.get("email_status") == "SENT":
        return

    now = time.time()
    if now - machine.get("email_last_attempt", 0) < EMAIL_RETRY_SECONDS:
        return

    machine["email_last_attempt"] = now
    machine["email_attempts"] = machine.get("email_attempts", 0) + 1
    machine["email_status"] = "PENDING"

    success, message = send_failure_email(
        owner_email,
        machine["machine_id"],
        readings["failure_probability"],
        readings,
        log_df,
        sender_email,
        sender_password,
    )

    if success:
        machine["email_status"] = "SENT"
        machine["email_error"] = None
        logger.info("Failure alert accepted by SMTP for %s", machine["machine_id"])
    else:
        machine["email_status"] = "FAILED - RETRY SCHEDULED"
        machine["email_error"] = message
        logger.error("Failure alert failed for %s: %s", machine["machine_id"], message)


# ---------------------------------------------------------------------------
# PHASE 6 – DASHBOARD COMPONENTS
# ---------------------------------------------------------------------------

def show_prediction_dashboard(
    product_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
    manual=False,
    failure_modes=None,
):

    st.subheader(
        "Manual Values Prediction" if manual else "Live Machine Sensor Readings"
    )

    if not manual and "machine_state" in st.session_state:
        phase = st.session_state.machine_state.get("startup_phase", "RUNNING")
        if phase == "RUNNING":
            st.success("RUNNING")
        elif phase == "SHUTTING DOWN":
            st.warning("EMERGENCY SHUTDOWN SIMULATION")
        elif phase == "ENGAGING":
            st.info("ENGAGING")
        elif phase == "RAMPING UP":
            st.info("RAMPING UP")
        else:
            st.info("MACHINE OFF / STARTUP")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Product Type", product_type)
        st.metric("Air Temperature", f"{air_temperature:.2f} K")
        st.metric("Process Temperature", f"{process_temperature:.2f} K")
    with col2:
        st.metric("Rotational Speed", f"{rotational_speed} rpm")
        st.metric("Torque", f"{torque:.2f} Nm")
        st.metric("Tool Wear", f"{tool_wear:.1f} min")

    if tool_wear < 150:
        wear_status, wear_icon = "NORMAL", "🟢"
    elif tool_wear < 230:
        wear_status, wear_icon = "WARNING", "🟡"
    elif tool_wear < 270:
        wear_status, wear_icon = "HIGH", "🟠"
    else:
        wear_status, wear_icon = "CRITICAL", "🔴"

    st.write(f"Tool Wear Status: **{wear_status} {wear_icon}**")

    try:
        prediction, probability = predict_machine_failure(
            product_type,
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear,
        )
        st.subheader("Machine Failure Prediction")
        st.metric("Estimated failure probability", f"{probability * 100:.2f}%")
        st.caption(
            f"Classification threshold: {FAILURE_THRESHOLD:.2f}. "
            "This probability is a model estimate, not a guarantee."
        )
        if prediction == 1:
            st.error("⚠️ ELEVATED FAILURE RISK")
        else:
            st.success("No failure predicted at the configured threshold.")
    except Exception as exc:
        logger.exception("Dashboard prediction failed")
        st.error(f"Prediction unavailable: {exc}")

    # Failure-mode classification
    if failure_modes is None:
        try:
            failure_modes = predict_failure_modes(
                product_type,
                air_temperature,
                process_temperature,
                rotational_speed,
                torque,
                tool_wear,
            )
        except Exception:
            logger.exception("Failure-mode prediction failed")
            failure_modes = {}

    st.divider()
    st.subheader("Possible Failure-Mode Indicators")

    if not failure_modes:

        st.info(
            "Failure-mode classifiers are unavailable. "
            "The overall machine failure prediction remains active."
        )

    else:

        mode_names = {
            "HDF": "Heat Dissipation Failure",
            "PWF": "Power Failure",
            "OSF": "Overstrain Failure",
        }

        mode_columns = st.columns(len(failure_modes))

        for column, (mode, result) in zip(
            mode_columns, failure_modes.items()
        ):

            with column:

                probability = result["probability"]
                flagged = result["prediction"] == 1

                st.metric(
                    mode,
                    f"{probability * 100:.2f}%"
                )

                if flagged:
                    st.warning(
                        f"Possible {mode_names[mode]} indicator"
                    )
                else:
                    st.success("Not flagged by model")

                st.caption(
                    f"Classification threshold: "
                    f"{result['threshold']:.2f}"
                )

    st.caption(
        "Failure-mode results are model-based indicators, not confirmed "
        "component diagnoses. A result not flagged by a model does not "
        "guarantee that the failure mode is absent."
    )

def render_manual_dashboard():
    st.write("Enter machine readings below to test the trained prediction model.")

    defaults = {
        "product_type": "M",
        "air_temperature": 295.0,
        "process_temperature": 302.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 0.0,
    }
    values = st.session_state.get("manual_inputs", defaults)

    with st.form("manual_machine_input_form"):
        st.subheader("Manual Machine Input")
        product_type = st.selectbox(
            "Product Type",
            ["L", "M", "H"],
            index=["L", "M", "H"].index(values["product_type"]),
        )
        col1, col2 = st.columns(2)
        with col1:
            air_temperature = st.number_input(
                "Air Temperature (K)", 0.0, 304.5,
                float(values["air_temperature"]), 0.1, format="%.2f"
            )
            process_temperature = st.number_input(
                "Process Temperature (K)", 0.0, 313.8,
                float(values["process_temperature"]), 0.1, format="%.2f"
            )
            rotational_speed = st.number_input(
                "Rotational Speed (rpm)", 0, 2886,
                int(values["rotational_speed"]), 10
            )
        with col2:
            torque = st.number_input(
                "Torque (Nm)", 0.0, 76.6,
                float(values["torque"]), 0.1, format="%.2f"
            )
            tool_wear = st.number_input(
                "Tool Wear (minutes)", 0.0, 300.0,
                float(values["tool_wear"]), 1.0, format="%.1f"
            )

        submitted = st.form_submit_button(
            "Predict Machine Failure 🔍", use_container_width=True
        )

    if submitted:
        st.session_state.manual_inputs = {
            "product_type": product_type,
            "air_temperature": air_temperature,
            "process_temperature": process_temperature,
            "rotational_speed": rotational_speed,
            "torque": torque,
            "tool_wear": tool_wear,
        }
        st.rerun()

    if "manual_inputs" in st.session_state:
        v = st.session_state.manual_inputs
        if v["rotational_speed"] == 0 and v["torque"] == 0:
            st.info("The entered readings indicate a stopped machine.")
        show_prediction_dashboard(
            v["product_type"],
            v["air_temperature"],
            v["process_temperature"],
            v["rotational_speed"],
            v["torque"],
            v["tool_wear"],
            manual=True,
        )


def render_automatic_machine_dashboard():
    st.caption(
        "Demonstration mode: readings are simulated and each browser session "
        "has its own simulated machine."
    )

    machine = st.session_state.machine_state
    owner_email = st.text_input(
        "Machine Owner Email (failure notification)",
        placeholder="owner@company.com",
        key="owner_email_input",
    )

    readings_tuple = get_machine_sensor_data()
    product_type, air, process, speed, torque, wear = readings_tuple
    phase = machine["startup_phase"]

    if machine["machine_running"] and phase == "RUNNING":
        try:
            prediction, probability = predict_machine_failure(
                product_type, air, process, speed, torque, wear
            )

            if prediction == 1:
                machine["failure_streak"] = machine.get("failure_streak", 0) + 1
            else:
                machine["failure_streak"] = 0

            if (
                probability >= CRITICAL_SHUTDOWN_THRESHOLD
                or machine["failure_streak"] >= FAILURE_STREAK_LIMIT
            ):
                # This is only a simulated UI state transition. It must never
                # control real industrial machinery.
                machine["startup_phase"] = "SHUTTING DOWN"
                machine["shutdown_start"] = time.time()
                machine["failure_reason"] = (
                    "Simulated AI risk response "
                    f"({probability * 100:.1f}% estimated risk)."
                )
                logger.warning(
                    "Simulated shutdown initiated for %s",
                    machine["machine_id"],
                )
        except Exception:
            logger.exception("Automatic machine prediction failed")

    if machine["startup_phase"] == "SHUTTING DOWN":
        st.error(
            "🚨 SIMULATED CONTROLLED SHUTDOWN: elevated model risk detected. "
            "This does not control real equipment."
        )

    if not machine["machine_running"]:
        st.error("🛑 SIMULATED MACHINE STOPPED")
        st.write(f"Reason: {machine.get('failure_reason') or 'Machine stopped'}")
        st.warning("A simulated tool replacement is required to restart.")
        if st.button("🔧 Replace Tool / Restart Simulation", use_container_width=True):
            st.session_state.machine_state = new_machine_state()
            st.rerun()
        st.stop()

    # Log every update to local CSV and MongoDB. MongoDB is the durable history
    # store; session memory retains only the latest MAX_SESSION_LOG_ROWS rows.
    try:
        prediction, probability = predict_machine_failure(
            product_type, air, process, speed, torque, wear
        )

        failure_modes = predict_failure_modes(
            product_type,
            air,
            process,
            speed,
            torque,
            wear,
        )
        started = machine["start_timestamp"]
        elapsed = (
            datetime.now(timezone.utc) - started
        ).total_seconds()

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "machine_id": machine["machine_id"],
            "start_time": machine["start_time"],
            "product_type": product_type,
            "air_temperature": air,
            "process_temperature": process,
            "rotational_speed": speed,
            "torque": torque,
            "tool_wear": wear,
            "failure_probability": round(probability, 6),
            "prediction": int(prediction),
            "HDF_probability": failure_modes.get(
                "HDF", {}
            ).get("probability"),

            "HDF_prediction": failure_modes.get(
                "HDF", {}
            ).get("prediction"),

            "PWF_probability": failure_modes.get(
                "PWF", {}
            ).get("probability"),

            "PWF_prediction": failure_modes.get(
                "PWF", {}
            ).get("prediction"),

            "OSF_probability": failure_modes.get(
                "OSF", {}
            ).get("probability"),

            "OSF_prediction": failure_modes.get(
                "OSF", {}
            ).get("prediction"),
        }


        machine["log"].append(log_entry)
        machine["log"] = machine["log"][-MAX_SESSION_LOG_ROWS:]
        csv_ok = append_log_to_csv(log_entry)

        if "mongo_status" not in st.session_state or time.time() - st.session_state.get("mongo_checked_at", 0) > 30:
            collection, mongo_status = get_mongo_collection()
            st.session_state.mongo_collection = collection
            st.session_state.mongo_status = mongo_status
            st.session_state.mongo_checked_at = time.time()

        collection = st.session_state.get("mongo_collection")
        if collection is not None:
            saved, save_message = save_log_to_mongodb(collection, log_entry)
            if not saved:
                st.session_state.mongo_status = save_message

        if prediction == 1 and machine.get("email_status") != "SENT":
            attempt_failure_email(
                machine,
                owner_email,
                {
                    "product_type": product_type,
                    "air_temperature": air,
                    "process_temperature": process,
                    "rotational_speed": speed,
                    "torque": torque,
                    "tool_wear": wear,
                    "failure_probability": probability,
                    "failure_modes": failure_modes,
                },
                pd.DataFrame(machine["log"]),
            )

        if not csv_ok:
            st.warning("Local CSV write failed. Check the application logs.")

    except Exception as exc:
        logger.exception("Machine logging or prediction failed")
        st.error(f"Machine processing error: {exc}")

    show_prediction_dashboard(
        product_type,
        air,
        process,
        speed,
        torque,
        wear,
        manual=False,
        failure_modes=failure_modes,
    )

    st.divider()
    st.subheader("📋 Machine Log")
    mongo_status = st.session_state.get(
        "mongo_status", "MongoDB connection not checked yet."
    )
    st.caption(f"MongoDB status: {mongo_status}")
    st.caption(
        f"Email alert status: {machine.get('email_status', 'NOT SENT')} "
        f"| Attempts: {machine.get('email_attempts', 0)}"
    )
    if machine.get("email_error"):
        st.caption(f"Last email issue: {machine['email_error']}")

    if machine["log"]:
        log_df = pd.DataFrame(machine["log"])
        st.dataframe(log_df.tail(10), use_container_width=True)
        st.download_button(
            "⬇️ Download Retained Session Log",
            log_df.to_csv(index=False).encode("utf-8"),
            f"{machine['machine_id']}_log.csv",
            "text/csv",
            use_container_width=True,
        )
        st.caption(
            f"The dashboard retains up to {MAX_SESSION_LOG_ROWS} recent rows "
            "in session memory. MongoDB/local CSV are used for longer history."
        )

    st.caption(
        "Refreshes automatically when Streamlit fragments are available. "
        "On older Streamlit versions, use the refresh button below."
    )
    if not hasattr(st, "fragment"):
        if st.button("Refresh machine readings"):
            st.rerun()


# ---------------------------------------------------------------------------
# PHASE 7 – EXPERIMENTAL PC RESOURCE DASHBOARD
# ---------------------------------------------------------------------------

def render_pc_dashboard():
    st.header("🖥️ Real-time PC Resource Readings")
    st.warning(
        "Experimental demonstration only. PC CPU, RAM, temperature and disk "
        "activity are not industrial machine sensor measurements. The mapped "
        "model result below must not be interpreted as actual PC or machine "
        "failure probability."
    )

    try:
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_percent = psutil.virtual_memory().percent
        disk_stats = psutil.disk_io_counters()
        total_written = disk_stats.write_bytes if disk_stats else 0

        previous_written = st.session_state.get("previous_disk_written", total_written)
        interval_written = max(0, total_written - previous_written)
        st.session_state.previous_disk_written = total_written

        try:
            temps = psutil.sensors_temperatures()
            if "coretemp" in temps and temps["coretemp"]:
                cpu_temp_c = float(temps["coretemp"][0].current)
            else:
                cpu_temp_c = 40.0 + cpu_percent * 0.3
        except Exception:
            cpu_temp_c = 40.0 + cpu_percent * 0.3

        col1, col2, col3 = st.columns(3)
        col1.metric("CPU Usage", f"{cpu_percent:.1f}%")
        col2.metric("Memory Usage", f"{memory_percent:.1f}%")
        col3.metric("CPU Temperature", f"{cpu_temp_c:.1f} °C")
        st.metric(
            "Disk writes during this refresh",
            f"{interval_written / (1024 * 1024):.2f} MB",
        )

        # Keep the legacy experimental mapping visible, but explicitly label it.
        pc_air_k = round(294.5 + cpu_percent * 0.05, 2)
        pc_process_k = max(pc_air_k, round(cpu_temp_c + 273.15, 2))
        pc_speed_rpm = int(1000 + cpu_percent * 15)
        pc_torque_nm = round(12.0 + memory_percent * 0.5, 2)
        pc_wear_min = round(
            min(300.0, interval_written / (1024 * 1024 * 50)), 1
        )
        pc_product_type = "M"

        st.subheader("Experimental mapped inputs (not physical machine readings)")
        a, b, c = st.columns(3)
        a.metric("Mapped Speed", f"{pc_speed_rpm} rpm")
        a.metric("Mapped Wear", f"{pc_wear_min} min")
        b.metric("Mapped Air Temperature", f"{pc_air_k} K")
        b.metric("Mapped Process Temperature", f"{pc_process_k} K")
        c.metric("Mapped Torque", f"{pc_torque_nm} Nm")
        c.metric("Assigned Product Type", pc_product_type)

        try:
            pc_prediction, pc_probability = predict_machine_failure(
                pc_product_type,
                pc_air_k,
                pc_process_k,
                pc_speed_rpm,
                pc_torque_nm,
                pc_wear_min,
            )
            st.subheader("Experimental model inference")
            st.metric(
                "Model score for transformed PC metrics",
                f"{pc_probability * 100:.2f}%",
            )
            if pc_prediction == 1:
                st.error("The transformed inputs crossed the model threshold.")
            else:
                st.success("The transformed inputs did not cross the model threshold.")
        except Exception as exc:
            st.error(f"Experimental inference unavailable: {exc}")

    except Exception as exc:
        logger.exception("PC diagnostics failed")
        st.error(f"Unable to read PC resource metrics: {exc}")

    if not hasattr(st, "fragment"):
        if st.button("Refresh PC readings"):
            st.rerun()


# ---------------------------------------------------------------------------
# PHASE 8 – OPTIONAL APPLICATION ACCESS GATE
# ---------------------------------------------------------------------------

def verify_app_access():
    """
    Optional single shared-password gate.

    Configure APP_ACCESS_PASSWORD in Streamlit secrets or the environment.
    This is a basic access barrier, NOT enterprise authentication or RBAC.
    """
    configured_password = get_secret("APP_ACCESS_PASSWORD")
    if not configured_password:
        return True

    if st.session_state.get("authenticated", False):
        with st.sidebar:
            st.caption("Access: authenticated session")
            if st.button("Log out"):
                st.session_state.authenticated = False
                st.rerun()
        return True

    st.title("Machine Failure Prediction System")
    st.subheader("Application sign-in")
    entered_password = st.text_input("Access password", type="password")
    if st.button("Sign in", type="primary"):
        if entered_password == configured_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect access password.")
    st.info(
        "For production, replace this shared-password gate with proper "
        "identity management and role-based access control."
    )
    return False


# ---------------------------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------------------------

if not verify_app_access():
    st.stop()

st.title("Machine Failure Prediction System")
st.caption(
    "AI Apparatus-preservation | Predictive maintenance demonstration"
)

with st.sidebar:
    st.subheader("Model configuration")
    st.write(f"Failure threshold: {FAILURE_THRESHOLD:.2f}")
    st.write(f"Critical shutdown threshold: {CRITICAL_SHUTDOWN_THRESHOLD:.2f}")
    st.write(f"Failure streak limit: {FAILURE_STREAK_LIMIT}")
    st.caption("Thresholds should be selected using validation data.")

tab1, tab2 = st.tabs(
    ["🏭 Machine Dashboard", "🖥️ Real-time PC Readings"]
)

with tab1:
    if "manual_mode" not in st.session_state:
        st.session_state.manual_mode = False

    button_label = (
        "Automatic Machine Readings 🔄"
        if st.session_state.manual_mode
        else "Enter Values Manually ✍️"
    )
    if st.button(button_label, use_container_width=True):
        st.session_state.manual_mode = not st.session_state.manual_mode
        st.rerun()

    if st.session_state.manual_mode:
        render_manual_dashboard()
    else:
        if "machine_state" not in st.session_state:
            st.session_state.machine_state = new_machine_state()

        if hasattr(st, "fragment"):
            st.fragment(run_every=f"{MACHINE_REFRESH_SECONDS}s")(
                render_automatic_machine_dashboard
            )()
        else:
            render_automatic_machine_dashboard()

with tab2:
    if hasattr(st, "fragment"):
        st.fragment(run_every=f"{MACHINE_REFRESH_SECONDS}s")(
            render_pc_dashboard
        )()
    else:
        render_pc_dashboard()

st.divider()
st.caption(
    "Safety notice: this application is a decision-support demonstration. "
    "Do not connect its prediction or simulated shutdown logic directly to "
    "industrial machinery."
)
