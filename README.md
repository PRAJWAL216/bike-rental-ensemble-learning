# bike-rental-ensemble-learning
# Bike Rental Demand Prediction Using Ensemble Learning

## Project Description

This project predicts the number of bikes rented using machine learning.

The Kaggle **Bike Sharing Demand** dataset is used for this project.

## Dataset

Dataset: Kaggle Bike Sharing Demand

Main file used:

`train.csv`

Target column:

`count`

## Models Used

Four models are compared:

1. Decision Tree
2. Random Forest - Bagging
3. AdaBoost - Boosting
4. Stacking Regressor

## Features Used

* Season
* Holiday
* Working Day
* Weather
* Temperature
* Feeling Temperature
* Humidity
* Windspeed
* Year
* Month
* Day
* Hour
* Weekday

## Evaluation Metrics

The models are evaluated using:

* MAE
* RMSE
* R² Score
* RMSLE

## Result

The models are compared using the R² score.

In our experiment:

**Best Model: Random Forest (Bagging)**

**R² Score: 0.9546**

## Project Structure

```text
bike-rental-ensemble-learning/
│
├── data/
│   └── train.csv
│
├── src/
│   └── bike_prediction.py
│
├── notebooks/
│   └── bike_rental_prediction.ipynb
│
├── results/
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   └── predictions.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Python program:

```bash
python src/bike_prediction.py
```

## Conclusion

Random Forest using **Bagging** gave the best R² score among the four models in this experiment. Therefore, it was selected as the best model for bike rental prediction.

## Author

**Prajwal Santosh Shinde**
