from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# --- Task 1: Fetch the Data -----------------------------------------------------------------

# City chosen: Las Vegas, Nevada.
# Using the complete calendar year 2023 and the city's local timezone.
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 36.17,
    "longitude": -115.14,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
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
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())


# --- Task 2: Look at the Features ------------------------------------------------------------------

print("\n--- Task 2: Data Exploration ---")
print("\nDescriptive statistics:")
print(df.describe())

print("\nMissing values per column:")
print(df.isna().sum())

rows_before = len(df)

# For this initial teaching model, we need to remove rows with missing values.
# This avoids inventing target temperatures or filling in measurements
# without evidence. If many rows are missing, reconsider this approach
# before modeling because dropping them could distort the dataset.
df = df.dropna().copy()

print("Rows removed due to missing values:", rows_before - len(df))
print("Missing values remaining:", int(df.isna().sum().sum()))

if df.empty:
    raise ValueError("No complete weather records remain after cleaning.")

correlations = (
    df.corr(numeric_only=True)["temperature_2m_max"].sort_values()
)

print("\nNumeric correlations with the daily high:")
print(correlations)

# Excluding the target's correlation with itself, which is always 1
# when the target has nonzero variance.
feature_correlations = correlations.drop("temperature_2m_max").dropna()
strongest_feature = feature_correlations.abs().idxmax()

print("\nFeature with the strongest absolute correlation:", strongest_feature)
print(f"Correlation: {feature_correlations[strongest_feature]:.4f}")

output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    df["temperature_2m_min"],
    df["temperature_2m_max"],
    alpha=0.6,
    edgecolors="none",
)

ax.set_title("Las Vegas: Daily High vs Low, 2023")
ax.set_xlabel("Daily Low Temperature (°C)")
ax.set_ylabel("Daily High Temperature (°C)")
ax.grid(alpha=0.25)

fig.tight_layout()

plot_path = output_dir / "las_vegas_high_vs_low.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)

print("Scatter plot saved to:", plot_path)

# Daily highs generally increase as daily lows increase.
# The points follow a strong, roughly linear upward pattern, although
# days with similar lows can still have different highs.
# This agrees with the strong positive correlation of about 0.9668.
# No missing values were found, so no rows needed to be removed.


# --- Task 3: Engineer a Feature -----------------------------------------------------------------

df["is_summer"] = (
    df["date"].dt.month.isin([6, 7, 8]).astype(int)
)

summer_days = int(df["is_summer"].sum())
non_summer_days = len(df) - summer_days

print("\n--- Task 3: Summer Feature ---")
print("Summer days:", summer_days)
print("Non-summer days:", non_summer_days)

# I expect is_summer to help because daily highs in Las Vegas tend
# to be higher in summer. The flag gives the model seasonal information.
# However, daily lows already reflect seasonality, so the flag might
# add only a small amount of information beyond the existing features.
# We need to compare test-set performance to determine whether it helps.

# --- Task 4: Baseline Model -----------------------------------------------------------------

X_baseline = df[["temperature_2m_min"]].values
y = df["temperature_2m_max"].values

X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_baseline,
    y,
    test_size=0.2,
    random_state=42,
)

baseline_model = LinearRegression()
baseline_model.fit(X_train_base, y_train_base)

y_pred_base = baseline_model.predict(X_test_base)

baseline_rmse = np.sqrt(mean_squared_error(y_test_base, y_pred_base))
baseline_r2 = baseline_model.score(X_test_base, y_test_base)

print("\n--- Task 4: Baseline Model ---")
print("Training days:", len(y_train_base))
print("Test days:", len(y_test_base))
print(f"Slope: {baseline_model.coef_[0]:.4f}")
print(f"Test RMSE: {baseline_rmse:.4f}°C")
print(f"Test R²: {baseline_r2:.4f}")
print(f"Typical error scale: approximately {baseline_rmse:.2f}°C")

# The baseline RMSE is 2.3861°C, giving a typical prediction error
# scale of about 2.39°C. Some predictions are closer and others are
# farther off; RMSE gives greater weight to larger errors.
# The slope is 1.1356: each 1°C increase in the daily low raises
# the predicted daily high by approximately 1.14°C.


# --- Task 5: Full Model -----------------------------------------------------------------

feature_cols = [
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "is_summer",
]

X = df[feature_cols].values
y = df["temperature_2m_max"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
full_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Task 5: Full Model ---")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Test RMSE: {full_rmse:.4f}°C")

print(f"\nBaseline test R²: {baseline_r2:.4f}")
print(f"Full model test R²: {test_r2:.4f}")
print(f"Change in test R²: {test_r2 - baseline_r2:+.4f}")
print(f"Reduction in RMSE: {baseline_rmse - full_rmse:+.4f}°C")
print(f"Train minus test R²: {train_r2 - test_r2:+.4f}")

print("\nFeature coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:22s}: {coef:+.3f}")

# Adding precipitation, wind, and summer status increased test R²
# from 0.9491 to 0.9526, an improvement of 0.0035.
# RMSE decreased by 0.0833°C, from 2.3861°C to 2.3028°C.
# This is a small improvement because daily low alone already
# provides substantial information about the daily high.
#
# Holding the other features fixed:
# - A 1°C higher daily low raises the predicted high by about 1.134°C.
# - An additional 1 mm of precipitation lowers it by about 0.231°C.
# - An additional 1 km/h of wind lowers it by about 0.029°C.
# - Summer status raises it by about 0.359°C versus non-summer.
#
# The small summer coefficient is interesting: summer days are hotter
# overall, but daily low already captures much of that seasonal pattern.
# These coefficients describe fitted associations, not proof of causation.
#
# Train R² is 0.9362 and test R² is 0.9526, a difference of 0.0164.
# Their similar values show no obvious train-performance advantage
# suggesting overfitting on this split. Test R² can exceed train R²
# because the sets contain different observations and target variation.
#Overall, the full model is a good fit for predicting daily highs in Las Vegas


# --- Task 6: Evaluate and Summarize -----------------------------------------------------------------

lower_limit = float(min(y_pred.min(), y_test.min())) - 1
upper_limit = float(max(y_pred.max(), y_test.max())) + 1

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(
    y_pred,
    y_test,
    alpha=0.7,
    label="Test observations",
)

ax.plot(
    [lower_limit, upper_limit],
    [lower_limit, upper_limit],
    linestyle="--",
    color="black",
    label="Perfect prediction",
)

ax.set_title("Predicted vs Actual (Full Model)")
ax.set_xlabel("Predicted Daily High (°C)")
ax.set_ylabel("Actual Daily High (°C)")
ax.set_xlim(lower_limit, upper_limit)
ax.set_ylim(lower_limit, upper_limit)
ax.set_aspect("equal", adjustable="box")
ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()

prediction_plot_path = output_dir / "predicted_vs_actual_high.png"
fig.savefig(prediction_plot_path, dpi=150)
plt.close(fig)

print("\n--- Task 6: Evaluation ---")
print("Plot saved to:", prediction_plot_path)
print("Dataset days:", len(df))
print("Test days:", len(y_test))
print(f"Full model RMSE: {full_rmse:.4f}°C")
print(f"Full model test R²: {test_r2:.4f}")

# Most test observations lie near the diagonal, showing that predictions
# track actual highs well. Errors occur throughout the temperature range,
# with several larger deviations among warmer days and in the middle.
# Visually, the hotter end appears somewhat more scattered, but the plot
# alone does not establish a clear increase in error with temperature.
# Points above the diagonal are underpredictions; points below it
# are overpredictions.

# --- Project Summary ---
#
# I used 365 days of Las Vegas weather from 2023, with no missing values.
# The split included 292 training days and 73 test days.
#
# The full model achieved a test RMSE of 2.3028°C and R² of 0.9526.
# Its prediction error scale is about 2.30°C, or a temperature difference
# of about 4.15°F. Individual errors can be smaller or larger.
# The R² indicates that the model explains approximately 95.3% of
# the variation in daily highs within this test set.
#
# Daily low appears to be the dominant predictor, supported by its
# strong correlation with daily high and the baseline's strong results.
# In the full model, a 10°C increase in daily low corresponds to an
# approximately 11.34°C increase in predicted high, with other inputs fixed.
# Raw coefficient sizes alone cannot establish feature importance
# because the features have different units and ranges.
#
# A result worth reflecting on is how little the additional features
# improved performance: RMSE decreased by only about 0.08°C.
# Daily low alone captured most of the predictable temperature variation.
#
# This is a same-day prediction exercise using same-day measurements,
# rather than a forecast made before those measurements are available.