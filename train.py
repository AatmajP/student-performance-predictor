import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data/student.csv")

X = data[
    [
        "hours_studied",
        "attendance",
        "previous_score",
        "assignments_completed"
    ]
]

y = data["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

model = LinearRegression()

model.fit(X_train, y_train)

print("Weights:", model.coef_)
print("Bias:", model.intercept_)

predictions = model.predict(X_test)

print("Predictions:", predictions)
print("Actual:", y_test.values)