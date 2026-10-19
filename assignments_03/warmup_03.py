import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Preprocessing ---

# Q1
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("--- Preprocessing Q1 ---")
print("Scaled training column means:")
for name, mean_value in zip(numeric, X_train_scaled.mean(axis=0)):
    print(f"{name:22s}: {mean_value:.12f}")

# Fit the scaler only on training data so test-set information does
# not influence the means and standard deviations used for training.


# Q2
# Selecting the same training rows used in the setup block.
season_train = df.loc[X_train.index, ["season"]]

encoder = OneHotEncoder(sparse_output=False)
season_encoded = encoder.fit_transform(season_train)

print("\n--- Preprocessing Q2 ---")
print("Encoded training season shape:", season_encoded.shape)
print("Encoded column names:", encoder.get_feature_names_out())

# If all four seasons appear in training, the single season column
# becomes four columns: one binary indicator per distinct season.
# No category is dropped because the encoder uses drop=None by default.


# Q3
feature_cols = numeric + ["season"]

# Reusing the original split's row indexes to include season while
# preserving the same training/test observations and matching labels.
X_train_full = df.loc[X_train.index, feature_cols]
X_test_full = df.loc[X_test.index, feature_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric),
        (
            "season",
            OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
            ),
            ["season"],
        ),
    ]
)

preprocessing_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

preprocessing_pipeline.fit(X_train_full, y_train)
y_pred_preprocessing = preprocessing_pipeline.predict(X_test_full)

print("\n--- Preprocessing Q3 ---")
print(
    "Pipeline test accuracy:",
    f"{accuracy_score(y_test, y_pred_preprocessing):.4f}",
)

# During fit(), the pipeline learns numeric scaling and season encoding
# from the training data, combines the transformed columns, and fits
# logistic regression. During predict(), it automatically applies those
# same learned transformations before making predictions.
# Without the pipeline, I would need to coordinate these steps manually.

# --- KNN ---------------------------------------------------------------------------------------------

# Q1
knn_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ]
)

knn_pipeline.fit(X_train, y_train)
y_pred_knn = knn_pipeline.predict(X_test)

print("\n--- KNN Q1 ---")
print(f"Test accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print("Classification report:")
print(
    classification_report(
        y_test,
        y_pred_knn,
        labels=[0, 1],
        target_names=["skip", "good"],
        zero_division=0,
    )
)


# Q2
k_values = [1, 3, 5, 7, 9, 11, 15, 21]
cv_scores_by_k = {}

print("\n--- KNN Q2 ---")

for k in k_values:
    candidate_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ]
    )

    scores = cross_val_score(
        candidate_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    cv_scores_by_k[k] = scores.mean()
    print(f"k={k:2d} | Mean CV accuracy: {scores.mean():.4f}")

best_k = max(cv_scores_by_k, key=cv_scores_by_k.get)

print("Chosen k:", best_k)

# I would choose k=11 because it has the highest mean training
# cross-validation accuracy among the tested values, about 0.9417.
# The choice uses training CV results, not test-set accuracy.
# Nearby k values perform similarly, so the advantage is modest.


# Q3
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred_unscaled = knn_unscaled.predict(X_test)

# Reusing the scaled k=5 pipeline fitted in Q1.
unscaled_accuracy = accuracy_score(y_test, y_pred_unscaled)
scaled_accuracy = accuracy_score(y_test, y_pred_knn)

print("\n--- KNN Q3 ---")
print(f"Without scaling: {unscaled_accuracy:.4f}")
print(f"With scaling:    {scaled_accuracy:.4f}")

# Scaling improved test accuracy from 0.9250 to about 0.9583.
# KNN uses distances, so features with larger numerical scales can
# dominate the distance calculation even when they are not more useful.
# Standardization puts the features on comparable scales.


# --- Logistic Regression ------------------------------------------------------------------------------------

# Q1
logistic_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=1000)),
    ]
)

logistic_pipeline.fit(X_train, y_train)
y_pred_logistic = logistic_pipeline.predict(X_test)

print("\n--- Logistic Regression Q1 ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_logistic):.4f}")
print(
    "Precision (good):",
    f"{precision_score(y_test, y_pred_logistic, pos_label=1):.4f}",
)
print(
    "Recall (good):",
    f"{recall_score(y_test, y_pred_logistic, pos_label=1):.4f}",
)
print(
    "F1 (good):",
    f"{f1_score(y_test, y_pred_logistic, pos_label=1):.4f}",
)


# Q2
coefficients = logistic_pipeline.named_steps["logistic"].coef_[0]

print("\n--- Logistic Regression Q2 ---")
for name, coefficient in zip(numeric, coefficients):
    print(f"{name:22s}: {coefficient:+.4f}")

# Negative coefficients push toward skip (0); positive ones toward good (1).
# Wind speed (-2.4438) and precipitation (-1.8891) push most strongly
# toward skip, matching the intuition that strong wind and rain can
# make running less comfortable.
# Daily high has a small negative coefficient, while daily low has
# a positive coefficient, holding the other features fixed.
#
# Because inputs are standardized, each coefficient describes the
# change in log-odds of good for a one-training-standard-deviation
# increase in that feature, not a one-unit change in the original units.
# These are fitted associations, not proof of causation.


# Q3
first_five = X_test.iloc[:5]
probabilities = logistic_pipeline.predict_proba(first_five)

# Probability columns follow the order of classes_.
good_column = list(logistic_pipeline.classes_).index(1)
good_probabilities = probabilities[:, good_column]

print("\n--- Logistic Regression Q3 ---")
for position, probability in enumerate(good_probabilities, start=1):
    print(f"Test day {position}: P(good) = {probability:.2f}")

# A probability near 0.5 means the model gives similar probabilities
# to good and skip, so it has little preference between the classes.
# It does not mean the weather itself is exactly half suitable for running.

# --- Evaluation --------------------------------------------------------------------------------------

# Q1
from pathlib import Path

assignment_dir = Path(__file__).resolve().parent
output_dir = assignment_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# Explicit order: skip=0, good=1.
cm = confusion_matrix(y_test, y_pred_logistic, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["skip", "good"],
)

fig, ax = plt.subplots(figsize=(6, 5))
display.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Logistic Regression Confusion Matrix")
fig.tight_layout()

confusion_path = output_dir / "logreg_confusion_matrix.png"
fig.savefig(confusion_path, dpi=150)
plt.close(fig)

true_negatives, false_positives, false_negatives, true_positives = cm.ravel()

print("\n--- Evaluation Q1 ---")
print("Confusion matrix:")
print(cm)
print("False positives:", false_positives)
print("False negatives:", false_negatives)
print("Figure saved to:", confusion_path)

# The model made 9 false positives and 7 false negatives.
# A false positive recommends running on a day labeled skip.
# A false negative recommends skipping a day labeled good.
# For an app prioritizing suitable running conditions, false positives
# would matter more because they could encourage running in unsuitable
# weather. False negatives instead miss a suitable running opportunity.


# --- joblib ---------------------------------------------------------------------------------------------

# Q1
model_dir = assignment_dir / "models"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "warmup_model.pkl"

joblib.dump(logistic_pipeline, model_path)
loaded = joblib.load(model_path)

assert (
    loaded.predict(X_test) == logistic_pipeline.predict(X_test)
).all()

print("\n--- joblib Q1 ---")
print("Pipeline saved to:", model_path)
print("Predictions match.")

# Saving only the LogisticRegression step would omit the fitted scaler.
# The classifier learned from standardized features, so calling predict()
# on raw measurements would give it inputs on the wrong scale and could
# produce incorrect predictions.
# Saving the complete pipeline preserves both the learned scaling
# parameters and the classifier, so raw inputs are processed consistently.