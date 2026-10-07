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

### Project Structure
```text
bike-rental-ensemble-learning/
├── README.md
├── bike_prediction.ipynb
├── bike_prediction.py
└── requirements.txt
└── train.csv
```


## Conclusion

Random Forest using **Bagging** gave the best R² score among the four models in this experiment. Therefore, it was selected as the best model for bike rental prediction.

## Author

**Prajwal Santosh Shinde**
