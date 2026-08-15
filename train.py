import pandas as pd

data = pd.read_csv("data/student.csv")

print(data.head())
print(data.shape)
print(data.info())

X = data[
    [
        "hours_studied",
        "attendance",
        "previous_score",
        "assignments_completed"
    ]
]

y = data["final_score"]

print("Features:")
print(X)

print("\nTarget:")
print(y)