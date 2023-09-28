# Stock Price Prediction
This project involves predicting stock prices using machine learning models, specifically Random Forest Regression and Linear Regression. The project includes data preprocessing, model training, and evaluation.

## Introduction
In this project, we aim to predict stock prices based on historical stock data. We have used two machine learning models, Random Forest Regression and Linear Regression, to make predictions. The project involves the following steps:

1. Data loading and exploration.
2. Data preprocessing and feature scaling.
3. Model training and evaluation.
4. Visualization of predictions.
   
## Prerequisites
Before running the code, you need to have the following libraries and tools installed:

  - Python 3.x
  - Libraries: pandas, numpy, matplotlib, scikit-learn
You can install these libraries using the following command:

        pip install pandas numpy matplotlib scikit-learn
## Installation
1. Clone the repository to your local machine:

        git clone https://github.com/tsameema/TradeTrendInsights.git
2. Navigate to the project directory:

        cd stock-price-prediction
3. Install the required Python libraries (see Prerequisites section).

## Usage
1. Load the dataset: Place your stock price dataset in the project directory and update the dataset path in the code 

2. Run the Jupyter Notebook or Python script to execute the code. You can run the code cells sequentially.

3. The code will perform data preprocessing, model training, and evaluation. You will see visualizations of the stock price predictions.

## Methodology
### Data Loading and Exploration
1. Load the historical stock price dataset using pandas.
2. Convert the date column to a datetime format.
3. Set the date as the index for time-series analysis.
4. Select the 'close' column for prediction.
### Data Preprocessing and Feature Scaling
1. Scale the 'close' feature to a range between 0 and 1 using Min-Max scaling.
2. Create sequences of data for training and testing by selecting a window of historical data.
### Model Training and Evaluation
1. Random Forest Regression
    - Train a Random Forest Regression model with 100 estimators.
    - Predict the stock prices on the test data.
    - Inverse transform the scaled predictions to get actual price values.
    - Calculate the Mean Squared Error (MSE) as a measure of prediction accuracy.
2. Linear Regression
    - Train a Linear Regression model.
    - Predict the stock prices on the test data.
    - Inverse transform the scaled predictions to get actual price values.
    - Calculate the MSE for Linear Regression predictions.
3. Visualization of Predictions
    - Plot the actual and predicted stock prices for both Random Forest Regression and Linear Regression.
    - Visualize the training and testing data along with predictions.
## Project Structure
The project structure is as follows:

1. README.md: This file, providing project information.
2. StockPricePrediction.ipynb: Jupyter Notebook containing the code for stock price prediction.
3. StockPred.csv: Sample dataset (replace with your own dataset).
4. images/: Directory containing images used in the README.md file.
## Results
1. Random Forest Regression
   
        Mean Squared Error: 0.3335
2. Linear Regression
   
        Mean Squared Error: 0.0456

Feel free to modify and use this template for your own projects. Make sure to update the project-specific information and add more details as needed.
