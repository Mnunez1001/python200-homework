import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import requests
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

LABEL_RULE = {
    "temperature_2m_max_min_inclusive": 7,
    "temperature_2m_max_max_inclusive": 26,
    "temperature_2m_min_min_inclusive": 0,
    "precipitation_sum_max_exclusive": 3,
    "wind_speed_10m_max_max_exclusive": 30,
}

# Deploy logistic regression for its compact representation,
# inexpensive predictions, and interpretable coefficients.
DEPLOY_MODEL = "logistic_regression"


def evaluate(
    name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Print classification results and return the model's test metrics."""
    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
    }

    print(f"\n--- {name} Test Results ---")
    for metric, value in metrics.items():
        print(f"{metric.capitalize():10s}: {value:.4f}")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["skip", "good"],
            zero_division=0,
        )
    )

    return metrics


def main() -> None:
    """Fetch Las Vegas weather, train classifiers, evaluate, and save models."""
    assignment_dir = Path(__file__).resolve().parent

    # --- Task 1: Fetch the Data ----------------------------------------------------------------------

    # City chosen: Las Vegas, Nevada, using the complete year 2023.
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 36.17,
        "longitude": -115.14,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "daily": FEATURES,
        "timezone": "America/Los_Angeles",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "wind_speed_unit": "kmh",
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    df = pd.DataFrame(response.json()["daily"])
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop("time", axis=1)

    print("--- Task 1: Las Vegas Weather, 2023 ---")
    print("Downloaded shape:", df.shape)
    print(df.head())

    print("\nMissing values:")
    print(df[FEATURES].isna().sum())

    # Remove incomplete observations before labeling; otherwise a missing
    # measurement could incorrectly turn a day into a skip label.
    rows_before = len(df)
    df = df.dropna(subset=FEATURES).copy()
    print("Incomplete rows removed:", rows_before - len(df))

    if df.empty:
        raise ValueError("No complete weather observations are available.")

    # --- Task 2: Engineer the Label --------------------------------------------------------------

    # Keeping the assignment thresholds unchanged.
    df["good_for_running"] = (
        df["temperature_2m_max"].between(
            LABEL_RULE["temperature_2m_max_min_inclusive"],
            LABEL_RULE["temperature_2m_max_max_inclusive"],
        )
        & (
            df["temperature_2m_min"]
            >= LABEL_RULE["temperature_2m_min_min_inclusive"]
        )
        & (
            df["precipitation_sum"]
            < LABEL_RULE["precipitation_sum_max_exclusive"]
        )
        & (
            df["wind_speed_10m_max"]
            < LABEL_RULE["wind_speed_10m_max_max_exclusive"]
        )
    ).astype(int)

    class_counts = (
        df["good_for_running"].value_counts().reindex([0, 1], fill_value=0)
    )
    good_fraction = float(df["good_for_running"].mean())

    print("\n--- Task 2: Class Balance ---")
    print("Skip days:", class_counts[0])
    print("Good days:", class_counts[1])
    print(f"Fraction good: {good_fraction:.2%}")

    # Of 365 days, 123 (33.70%) are labeled good and 242 are labeled skip.
    # This seems reasonable under these thresholds because Las Vegas's
    # hot days exceed the 26°C limit, while wind and precipitation can
    # exclude otherwise suitable days.
    # The classes are moderately imbalanced, so accuracy alone is insufficient:
    # always predicting skip would already achieve about 66.3% accuracy.
    # Precision, recall, and F1 for good days help evaluate the minority class.

    X = df[FEATURES]
    y = df["good_for_running"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    if y_train.value_counts().reindex([0, 1], fill_value=0).min() < 5:
        raise ValueError(
            "Each training class needs at least five examples for 5-fold CV."
        )

    print("Training days:", len(X_train))
    print("Test days:", len(X_test))

    # --- Task 3: KNN Classifier --------------------------------------------------------------

    k_values = [1, 3, 5, 7, 9, 11, 15, 21]
    cv_results = {}

    print("\n--- Task 3: KNN Training Cross-Validation ---")

    for k in k_values:
        candidate = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=k)),
            ]
        )

        scores = cross_val_score(
            candidate,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy",
        )

        cv_results[k] = float(scores.mean())
        print(f"k={k:2d} | Mean CV accuracy: {scores.mean():.4f}")

    best_k = max(cv_results, key=cv_results.get)
    print("Chosen k:", best_k)

    # Choosing k using training cross-validation, leaving the test set
    # out of hyperparameter selection.
    knn_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=best_k)),
        ]
    )
    knn_pipeline.fit(X_train, y_train)

    knn_metrics = evaluate("KNN", knn_pipeline, X_test, y_test)

    # --- Task 4: Logistic Regression Classifier --------------------------------------------------------------

    logistic_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=1000)),
        ]
    )
    logistic_pipeline.fit(X_train, y_train)

    logistic_metrics = evaluate("Logistic Regression", logistic_pipeline, X_test, y_test)

    print("Logistic regression coefficients:")
    coefficients = logistic_pipeline.named_steps["logistic"].coef_[0]

    for name, coefficient in zip(FEATURES, coefficients):
        print(f"{name:22s}: {coefficient:+.4f}")

    # All four coefficients are negative, pushing predictions toward skip
    # as their values increase, holding the other features fixed.
    # Daily high (-2.0236) and wind (-1.6604) have the strongest negative
    # coefficients, followed by precipitation (-1.2448) and daily low (-0.4526).
    # Inputs are standardized, so these describe log-odds changes per
    # one-training-standard-deviation increase, not per original measurement unit.

    # Higher highs, stronger wind, and more rain pushing toward skip
    # agree with the labeling rule.
    # The negative daily-low coefficient is less intuitive because the rule
    # requires lows of at least 0°C. However, this is a conditional association:
    # high and low temperatures are related, and a linear model cannot
    # fully represent the rule's lower and upper temperature boundaries.
    # It does not establish that warmer lows cause unsuitable running weather.

    # --- Task 5: Compare and Choose --------------------------------------------------------------

    comparison = pd.DataFrame(
        {
            "KNN": knn_metrics,
            "Logistic Regression": logistic_metrics,
        }
    ).T

    print("\n--- Task 5: Model Comparison ---")
    print(comparison.round(4).to_string())

    accuracy_gap = knn_metrics["accuracy"] - logistic_metrics["accuracy"]
    print(
        "KNN minus logistic regression accuracy:",
        f"{accuracy_gap * 100:.2f} percentage points",
    )

    # KNN had higher accuracy: 97.26% versus logistic regression's 91.78%.
    # KNN made 2 false positives and no false negatives.
    # Logistic regression made 3 false positives and 3 false negatives.
    #
    # For this app, I would weigh precision for good days heavily:
    # false positives recommend running on days labeled skip.
    # Recall also matters because false negatives miss good running days.
    # KNN performed better on both measures in this test split.
    #
    # I choose logistic regression for this teaching cloud pipeline,
    # accepting four additional test errors in exchange for a compact
    # model, inexpensive predictions, and interpretable coefficients.
    # Logistic regression stores coefficients rather than retaining the
    # training observations required by KNN.
    # These are architectural advantages; I have not benchmarked actual
    # prediction speed or serialized model sizes.
    #
    # Both models provide probabilities. With the selected k=1, KNN
    # returns only 0 or 1 based on its nearest neighbor's label.
    # Logistic regression provides smoother probability estimates,
    # although calibration has not been evaluated.
    # On this small dataset, the efficiency advantage may be minor,
    # so KNN remains a reasonable choice if predictive performance
    # is the application's main priority.

    chosen_pipeline = logistic_pipeline
    chosen_metrics = logistic_metrics

    print("Selected model:", DEPLOY_MODEL)

    # --- Task 6: Save the Model ------------------------------------------------------------------------------

    model_dir = assignment_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "weather_classifier.pkl"
    metadata_path = model_dir / "weather_classifier_metadata.json"

    # Saving the entire fitted pipeline, including its scaler.
    joblib.dump(chosen_pipeline, model_path)

    metadata = {
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "feature_names": FEATURES,
        "class_labels": {"0": "skip", "1": "good"},
        "label_rule": LABEL_RULE,
        "city": "Las Vegas, Nevada",
        "latitude": params["latitude"],
        "longitude": params["longitude"],
        "timezone": params["timezone"],
        "start_date": params["start_date"],
        "end_date": params["end_date"],
        "units": {
            "temperature": "celsius",
            "precipitation": "mm",
            "wind_speed": "km/h",
        },
        "model_type": DEPLOY_MODEL,
        "selection_reason": (
            "Compact linear model with inexpensive predictions, "
            "interpretable coefficients, and smooth probability estimates; "
            "accepts lower test performance than KNN."
        ),
        "random_state": 42,
        "dataset_rows": len(df),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "good_day_fraction": good_fraction,
        "test_accuracy": chosen_metrics["accuracy"],
        "test_f1": chosen_metrics["f1"],
        "test_precision": chosen_metrics["precision"],
        "test_recall": chosen_metrics["recall"],
        "comparison": {
            "knn": {
                "best_k": best_k,
                "test_metrics": knn_metrics,
            },
            "logistic_regression": {
                "test_metrics": logistic_metrics,
            },
        },
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=False)

    # Confirming that saving and loading preserves predictions.
    loaded_pipeline = joblib.load(model_path)
    assert (
        loaded_pipeline.predict(X_test)
        == chosen_pipeline.predict(X_test)
    ).all()

    print("\n--- Task 6: Saved Files ---")
    print("Model written to:", model_path)
    print("Metadata written to:", metadata_path)
    print("Reloaded predictions match.")



if __name__ == "__main__":
    main()