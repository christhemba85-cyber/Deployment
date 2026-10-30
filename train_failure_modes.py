
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    classification_report,
    confusion_matrix
)

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "ai4i2020.csv")
FEATURE_PATH = os.path.join(BASE_DIR, "model_features.pkl")

OUTPUT_DIR = os.path.join(BASE_DIR, "failure_mode_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
print("\nLoading AI4I 2020 dataset...")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)

# Required columns
sensor_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

failure_modes = ["TWF", "HDF", "PWF", "OSF"]

required_columns = ["Type"] + sensor_columns + failure_modes

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing dataset columns: {missing}")

# Load existing model feature order
feature_columns = joblib.load(FEATURE_PATH)

# Prepare sensor features
X = pd.get_dummies(
    df[["Type"] + sensor_columns],
    columns=["Type"],
    dtype=int
)

# Match existing model feature names and order
X = X.reindex(columns=feature_columns, fill_value=0)

print("\nPrepared features:", list(X.columns))

# Store thresholds for application use
failure_mode_thresholds = {}

# Train each failure-mode classifier
for mode in failure_modes:

    print("\n" + "=" * 55)
    print(f"TRAINING FAILURE MODE: {mode}")
    print("=" * 55)

    y = df[mode].astype(int)

    print("Failure cases:", int(y.sum()))
    print("Normal cases:", int((y == 0).sum()))

    if y.sum() < 2 or (y == 0).sum() < 2:
        print(f"Insufficient examples for {mode}. Skipping.")
        continue

    # Stratified training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Create classifier
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )

    selected_threshold = 0.50

    # TWF threshold selection using training data only
    if mode == "TWF":

        print("\nPerforming 5-fold threshold validation...")

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        # Generate out-of-fold probabilities
        oof_probabilities = cross_val_predict(
            model,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1
        )[:, 1]

        threshold_results = []

        for threshold in [i / 100 for i in range(1, 51)]:

            oof_predictions = (
                oof_probabilities >= threshold
            ).astype(int)

            precision = precision_score(
                y_train,
                oof_predictions,
                zero_division=0
            )

            recall = recall_score(
                y_train,
                oof_predictions,
                zero_division=0
            )

            f2 = fbeta_score(
                y_train,
                oof_predictions,
                beta=2,
                zero_division=0
            )

            threshold_results.append({
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f2": f2
            })

        # Select threshold with best F2-score.
        # Break ties using precision, then higher threshold.
        best_result = max(
            threshold_results,
            key=lambda result: (
                result["f2"],
                result["precision"],
                result["threshold"]
            )
        )

        selected_threshold = best_result["threshold"]

        print("\nBEST TWF VALIDATION RESULT")
        print("Selected threshold:", selected_threshold)
        print("Validation precision:", best_result["precision"])
        print("Validation recall:", best_result["recall"])
        print("Validation F2-score:", best_result["f2"])

    # Train model on training portion
    model.fit(X_train, y_train)

    # Predict probabilities on untouched test data
    test_probabilities = model.predict_proba(X_test)[:, 1]

    # Default predictions
    default_predictions = (test_probabilities >= 0.50).astype(int)

    if mode == "TWF":

        # Predictions using selected threshold
        predictions = (
            test_probabilities >= selected_threshold
        ).astype(int)

        print("\nTWF TEST RESULTS: DEFAULT THRESHOLD 0.50")
        print(classification_report(
            y_test,
            default_predictions,
            zero_division=0
        ))

        print("\nTWF TEST RESULTS: VALIDATED THRESHOLD")
        print("Threshold:", selected_threshold)

    else:
        predictions = default_predictions

    # Evaluation
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("Accuracy:", accuracy_score(y_test, predictions))

    print("Precision:", precision_score(
        y_test, predictions, zero_division=0
    ))

    print("Recall:", recall_score(
        y_test, predictions, zero_division=0
    ))

    print("F1-score:", f1_score(
        y_test, predictions, zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Store threshold
    failure_mode_thresholds[mode] = selected_threshold

    # Train final model on full dataset
    final_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )

    final_model.fit(X, y)

    # Save model
    model_path = os.path.join(
        OUTPUT_DIR,
        f"failure_mode_{mode}.pkl"
    )

    joblib.dump(final_model, model_path)

    print(f"\nSaved model: {model_path}")

# Save all thresholds
threshold_path = os.path.join(
    OUTPUT_DIR,
    "failure_mode_thresholds.pkl"
)

joblib.dump(failure_mode_thresholds, threshold_path)

print("\nSaved thresholds:", failure_mode_thresholds)
print("Threshold file:", threshold_path)

# RNF assessment
print("\n" + "=" * 55)
print("RANDOM FAILURE (RNF) ASSESSMENT")
print("=" * 55)

if "RNF" in df.columns:

    rnf_count = int(df["RNF"].sum())

    print("Recorded RNF cases:", rnf_count)

    print(
        "RNF will not be trained as a predictive classifier "
        "because its very small sample count does not support "
        "reliable sensor-based learning."
    )

print("\nFailure-mode training process completed.")
