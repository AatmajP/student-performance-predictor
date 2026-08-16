import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("data/student.csv")

print("Dataset:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nDataset Information:")
print(data.info())


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = data[
    [
        "hours_studied",
        "attendance",
        "previous_score",
        "assignments_completed"
    ]
]

y = data["final_score"]


print("\nFeatures:")
print(X)

print("\nTarget:")
print(y)

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("Testing Data Shape:")
print(X_test.shape)


# ==========================================
# 4. LINEAR REGRESSION
# ==========================================

linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

# Learned parameters
print("\n==============================")
print("LINEAR REGRESSION")
print("==============================")

print("Weights:", linear_model.coef_)
print("Bias:", linear_model.intercept_)


# Make predictions
linear_predictions = linear_model.predict(X_test)


# ==========================================
# 5. LINEAR REGRESSION EVALUATION
# ==========================================

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_mse = mean_squared_error(
    y_test,
    linear_predictions
)

linear_rmse = linear_mse ** 0.5

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


print("\nLinear Regression Predictions:")

for actual, predicted in zip(y_test, linear_predictions):
    print(
        f"Actual: {actual:.2f}, "
        f"Predicted: {predicted:.2f}"
    )


print("\nLinear Regression Evaluation:")
print("MAE:", linear_mae)
print("MSE:", linear_mse)
print("RMSE:", linear_rmse)
print("R²:", linear_r2)


# ==========================================
# 6. RANDOM FOREST
# ==========================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train Random Forest
rf_model.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)


# ==========================================
# 7. RANDOM FOREST EVALUATION
# ==========================================

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_mse = mean_squared_error(
    y_test,
    rf_predictions
)

rf_rmse = rf_mse ** 0.5

rf_r2 = r2_score(
    y_test,
    rf_predictions
)


print("\n==============================")
print("RANDOM FOREST")
print("==============================")

print("\nRandom Forest Predictions:")

for actual, predicted in zip(y_test, rf_predictions):
    print(
        f"Actual: {actual:.2f}, "
        f"Predicted: {predicted:.2f}"
    )


print("\nRandom Forest Evaluation:")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("RMSE:", rf_rmse)
print("R²:", rf_r2)


# ==========================================
# 8. MODEL COMPARISON
# ==========================================

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print("\n                    Linear Regression    Random Forest")

print(
    f"MAE                 {linear_mae:.4f}"
    f"              {rf_mae:.4f}"
)

print(
    f"MSE                 {linear_mse:.4f}"
    f"              {rf_mse:.4f}"
)

print(
    f"RMSE                {linear_rmse:.4f}"
    f"              {rf_rmse:.4f}"
)

print(
    f"R²                  {linear_r2:.4f}"
    f"              {rf_r2:.4f}"
)


# ==========================================
# 9. VISUALIZATION
# ==========================================

# Hours studied vs final score

plt.scatter(
    data["hours_studied"],
    data["final_score"]
)

plt.xlabel("Hours Studied")
plt.ylabel("Final Score")
plt.title("Hours Studied vs Final Score")

plt.show()


# Actual vs Linear Regression predictions

plt.scatter(
    y_test,
    linear_predictions
)

plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Linear Regression: Actual vs Predicted")

plt.show()


# Actual vs Random Forest predictions

plt.scatter(
    y_test,
    rf_predictions
)

plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title("Random Forest: Actual vs Predicted")

plt.show()