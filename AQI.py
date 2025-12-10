
# AIR QUALITY INDEX (AQI) PREDICTION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("Indian_AQI.csv")
print("Initial rows:")
print(data.head())

data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
data = data.dropna(subset=["Date"])
data = data.fillna(data.median(numeric_only=True))      # fill missing numeric values

data["City"] = data["City"].astype("category").cat.codes

data["AQI"] = data[["PM2.5", "PM10", "NO2", "SO2"]].mean(axis=1)

print("\nDataset details:")
print(data.info())

X = data[["PM2.5", "PM10", "NO2", "SO2"]]
y = data["AQI"]

# Train-test split
X_t, X_te, y_t, y_te = train_test_split(X, y, test_size=0.2, random_state=7)
print(f"\nTrain size: {len(X_t)} | Test size: {len(X_te)}")

m1 = LinearRegression()
m1.fit(X_t, y_t)

m2 = DecisionTreeRegressor(random_state=7)
m2.fit(X_t, y_t)

m3 = RandomForestRegressor(n_estimators=120, random_state=7)
m3.fit(X_t, y_t)


def evaluate(true, pred):
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    return rmse, r2


preds = {
    "Linear Regression": m1.predict(X_te),
    "Decision Tree": m2.predict(X_te),
    "Random Forest": m3.predict(X_te),
}

print("\nMODEL RESULTS:")
for model_name, p in preds.items():
    rmse, r2 = evaluate(y_te, p)
    print(f"{model_name:18}  RMSE: {rmse:.2f}   R²: {r2:.3f}")


plt.figure(figsize=(6,4))
plt.scatter(y_te, preds["Random Forest"], alpha=0.6)
plt.title("Actual vs Predicted AQI - Random Forest")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.grid(True)
plt.show()


future = [[85, 160, 45, 20]]   # PM2.5, PM10, NO2, SO2
forecast = m3.predict(future)
print("\nAQI prediction for input pollutant levels:", forecast[0])


plt.figure(figsize=(9,4))
sns.barplot(data=data, x="City", y="AQI")
plt.title("Average AQI per City")
plt.show()

plt.figure(figsize=(7,5))
sns.heatmap(data[["PM2.5", "PM10", "NO2", "SO2", "AQI"]].corr(), annot=True)
plt.title("Correlation Heatmap of Pollutants")
plt.show()


