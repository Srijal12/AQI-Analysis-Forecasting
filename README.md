# AQI-Analysis-Forecasting
Air Quality Index (AQI) Analysis &amp; Forecasting using Python and Machine Learning


Overview
This project predicts the Air Quality Index (AQI) of major Indian cities using pollutant data such as PM2.5, PM10, NO2, and SO2.
The goal is to analyze pollution trends, clean real-world environmental data, build ML models, and generate AQI forecasts.

Objective
Analyze historical AQI and pollutant concentration data.
Clean and preprocess incomplete or inconsistent records.
Study correlations and patterns in pollution levels.
Develop machine learning models to predict AQI.
Visualize actual vs predicted AQI and pollutant relationships.

Tools & Technologies
Python
Pandas, NumPy – Data handling
Matplotlib, Seaborn – Visualization
Scikit-learn – Regression models

Models Used: Linear Regression, Decision Tree, Random Forest

Dataset
The dataset (Indian_AQI.csv) contains:

Column	Description
Date	Recording date
City	City name
PM2.5	Fine particulate matter
PM10	Particulate matter
NO2	Nitrogen Dioxide
SO2	Sulphur Dioxide
AQI	Calculated as mean of pollutants

AQI is computed in the script using the average of pollutant values (simplified index).

Steps Performed
1. Data Loading
Loaded CSV file into pandas
Displayed initial rows for inspection

2. Data Preprocessing
Converted date column to datetime format
Removed rows with invalid dates
Replaced missing values with column medians
Encoded city names
Calculated AQI as mean of pollutants

3. Model Training
Trained three machine learning models:
Linear Regression
Decision Tree Regressor
Random Forest Regressor
Split dataset into 80% training and 20% testing.

4. Model Evaluation
Each model was evaluated using:
RMSE (Root Mean Squared Error)
R² Score

5. Visualization
Actual vs Predicted AQI scatter plot
AQI comparison across cities
Correlation heatmap of pollutants

6. Future Forecasting
Predicted AQI for pollutant input:

PM2.5 = 85
PM10  = 160
NO2   = 45
SO2   = 20

Sample Output (Example)
Train size: 16 | Test size: 4

MODEL RESULTS:
Linear Regression      RMSE: 0.00   R²: 1.000
Decision Tree          RMSE: 4.11   R²: 0.968
Random Forest          RMSE: 2.62   R²: 0.987

AQI prediction for input pollutant levels: 75.30

Project Structure
AQI-Analysis-Forecasting/
│
├── aqi_forecasting.py
├── Indian_AQI.csv
└── README.md

 
 
How to Run
Install dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn
Run the script:
python aqi_forecasting.py

Outcome
Successfully built AQI prediction models using multiple ML algorithms.
Random Forest performed best with high accuracy.

Visual analysis highlighted pollution patterns across cities.

Generated pollution forecasts to support environmental decision-making.
