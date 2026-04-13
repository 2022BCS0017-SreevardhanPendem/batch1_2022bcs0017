import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("data/winequality-red.csv",sep=",")

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("feature_selection", SelectKBest(score_func=f_regression, k=8)),
        ("model", LinearRegression())
    ]
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")
results = {
    "Name": "P Sreevardhan",
    "Rollno": "Batch1_2022BCS0017",
    "mse": mse,
    "r2_score": r2
}

print(results)
