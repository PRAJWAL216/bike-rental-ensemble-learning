import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Kaggle train.csv
df = pd.read_csv("data/train.csv")

print("Dataset shape:", df.shape)
print(df.head())

# Feature engineering
df["datetime"] = pd.to_datetime(df["datetime"])
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.weekday

features = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "year", "month", "day", "hour", "weekday"
]

X = df[features]
y = df["count"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

def evaluate(name, actual, predicted):
    predicted = np.maximum(predicted, 0)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    rmsle = np.sqrt(np.mean((np.log1p(actual) - np.log1p(predicted)) ** 2))
    return [name, mae, rmse, r2, rmsle]

# 1. Decision Tree
dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# 2. Bagging - Random Forest
rf = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# 3. Boosting - AdaBoost
ada = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=5),
    n_estimators=100,
    random_state=42
)
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)

# 4. Stacking
base_models = [
    ("decision_tree", DecisionTreeRegressor(max_depth=8, random_state=42)),
    ("random_forest", RandomForestRegressor(
        n_estimators=50, random_state=42, n_jobs=-1
    )),
    ("adaboost", AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4),
        n_estimators=50,
        random_state=42
    ))
]

stack = StackingRegressor(
    estimators=base_models,
    final_estimator=LinearRegression()
)
stack.fit(X_train, y_train)
stack_pred = stack.predict(X_test)

results = pd.DataFrame([
    evaluate("Decision Tree", y_test, dt_pred),
    evaluate("Random Forest - Bagging", y_test, rf_pred),
    evaluate("AdaBoost - Boosting", y_test, ada_pred),
    evaluate("Stacking", y_test, stack_pred)
], columns=["Model", "MAE", "RMSE", "R2", "RMSLE"])

print("\nMODEL COMPARISON")
print(results.to_string(index=False))

best = results.loc[results["R2"].idxmax()]
print("\nBEST MODEL:", best["Model"])
print("R2:", round(best["R2"], 4))
print("RMSE:", round(best["RMSE"], 2))
print("MAE:", round(best["MAE"], 2))
print("RMSLE:", round(best["RMSLE"], 4))

results.to_csv("results/model_comparison.csv", index=False)

# R2 graph
plt.figure(figsize=(9, 5))
plt.bar(results["Model"], results["R2"])
plt.xlabel("Model")
plt.ylabel("R2 Score")
plt.title("Bike Rental Prediction - Model Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("results/model_comparison.png")
plt.show()

# Save sample predictions
predictions = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": np.maximum(stack_pred, 0)
})
predictions.to_csv("results/predictions.csv", index=False)

print("\nResults saved in the results/ folder.")
