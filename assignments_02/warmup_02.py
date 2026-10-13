from pathlib import Path

# --- scikit-learn API ---

# Q1
import numpy as np
from sklearn.linear_model import LinearRegression

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

# Creating the model.
model = LinearRegression()

# Fiting the model to the known low and high temperatures.
model.fit(temp_min, temp_max)

# Predicting two highs using a 2D array: two days, one feature per day.
predictions = model.predict(np.array([[6], [18]]))

print("--- scikit-learn Q1 ---")
print(f"Slope: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Predicted high for a low of 6°C: {predictions[0]:.2f}°C")
print(f"Predicted high for a low of 18°C: {predictions[1]:.2f}°C")


# Q2
x = np.array([10, 20, 30, 40, 50])

print("\n--- scikit-learn Q2 ---")
print("Original shape:", x.shape)

x_2d = x.reshape(-1, 1)

print("Reshaped shape:", x_2d.shape)

# scikit-learn represents X as a table: rows are samples and columns
# are features. A 2D array makes those roles explicit, even when there
# is only one feature. Here, we have five samples with one feature each.
# In reshape(-1, 1), NumPy calculates the row count and uses one column.


# --- Linear Regression ---

# Generate the shared synthetic dataset once.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)
num_days  = 120
temp_min  = rng.uniform(-5, 22, num_days)
is_rainy  = rng.integers(0, 2, num_days).astype(float)
temp_max  = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)


# Q1
# Locate outputs beside this script, regardless of the working directory.
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
plot_path = output_dir / "high_vs_low.png"

plt.figure(figsize=(8, 5))

points = plt.scatter(
    temp_min,
    temp_max,
    c=is_rainy,
    cmap="coolwarm",
    alpha=0.8,
)

plt.title("Daily High vs Low")
plt.xlabel("Daily Low Temperature (°C)")
plt.ylabel("Daily High Temperature (°C)")

colorbar = plt.colorbar(points, ticks=[0, 1])
colorbar.set_label("Rain Status")
colorbar.set_ticklabels(["Dry (0)", "Rainy (1)"])

plt.tight_layout()
plt.savefig(plot_path, dpi=150)
plt.close()

print("\n--- Linear Regression Q1 ---")
print("Number of days:", num_days)
print("Scatter plot saved to:", plot_path)

# High temperatures tend to increase as low temperatures increase.
# Rainy and dry days form overlapping bands, with rainy days generally
# having lower highs for similar lows. This suggests that is_rainy
# provides useful information beyond temp_min when predicting temp_max.
# The random noise means the groups are not perfectly separated.

# Q2
X = temp_min.reshape(-1, 1)
y = temp_max

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("\n--- Linear Regression Q2 ---")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Q3
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2_single = model.score(X_test, y_test)

print("\n--- Linear Regression Q3 ---")
print(f"Slope: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Test RMSE: {rmse:.4f}°C")
print(f"Test R²: {r2_single:.4f}")

# The slope describes how much the predicted daily high changes for
# each 1°C increase in the daily low. A slope near 1 means that a
# 1°C higher low corresponds to approximately a 1°C higher predicted
# high in this model. This model does not account for rain status.


# Q4
X_full = np.column_stack([temp_min, is_rainy])

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full,
    y,
    test_size=0.2,
    random_state=42,
)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

r2_full = model_full.score(X_test_full, y_test_full)

print("\n--- Linear Regression Q4 ---")
print(f"One-feature test R²: {r2_single:.4f}")
print(f"Two-feature test R²: {r2_full:.4f}")
print(f"Change in test R²: {r2_full - r2_single:+.4f}")
print("Adding the rain flag improved test R²:", r2_full > r2_single)

print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

# The is_rainy coefficient represents the change in predicted daily
# high when rain status changes from dry (0) to rainy (1), while
# keeping the daily low fixed. A coefficient near -4 means the model
# predicts a rainy day's high to be about 4°C lower than a dry day's
# high with the same low. The fitted value need not be exactly -4
# because the synthetic data includes random noise.


# Q5
y_pred_full = model_full.predict(X_test_full)

# Using the same limits on both axes so the diagonal represents equality.
lower_limit = float(min(y_pred_full.min(), y_test_full.min())) - 1
upper_limit = float(max(y_pred_full.max(), y_test_full.max())) + 1

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(
    y_pred_full,
    y_test_full,
    alpha=0.8,
    label="Test observations",
)

ax.plot(
    [lower_limit, upper_limit],
    [lower_limit, upper_limit],
    color="black",
    linestyle="--",
    label="Perfect prediction",
)

ax.set_title("Predicted vs Actual")
ax.set_xlabel("Predicted Daily High (°C)")
ax.set_ylabel("Actual Daily High (°C)")
ax.set_xlim(lower_limit, upper_limit)
ax.set_ylim(lower_limit, upper_limit)
ax.set_aspect("equal", adjustable="box")
ax.legend()

fig.tight_layout()

prediction_plot_path = output_dir / "predicted_vs_actual_warmup.png"
fig.savefig(prediction_plot_path, dpi=150)
plt.close(fig)

print("\n--- Linear Regression Q5 ---")
print("Predicted vs actual plot saved to:", prediction_plot_path)

# Above the diagonal: the actual high exceeds the predicted high,
# so the model underestimated the temperature.
# Below the diagonal: the actual high is below the predicted high,
# so the model overestimated the temperature.
# Points closer to the diagonal have smaller prediction errors.