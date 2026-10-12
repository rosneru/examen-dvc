import json

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv").squeeze()
model = joblib.load("models/model.pkl")

y_pred = model.predict(X_test)

pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
    "data/predictions.csv", index=False
)

scores = {
    "mse": mean_squared_error(y_test, y_pred),
    "mae": mean_absolute_error(y_test, y_pred),
    "r2": r2_score(y_test, y_pred),
}
with open("metrics/scores.json", "w") as f:
    json.dump(scores, f, indent=4)
