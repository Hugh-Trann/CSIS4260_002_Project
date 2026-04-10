#  Flight Arrival Delay Prediction using Flights + Weather Data

##  Project Overview

This project predicts **flight arrival delay (`arr_delay`)** by
combining: - Flight schedule data (`nycflights.csv`) - Weather data
collected from the Open-Meteo API

The datasets are merged using: - airport (origin) - date - hourly
timestamp

The goal is to build a machine learning model that predicts how late a
flight will arrive using pre-flight information.

------------------------------------------------------------------------

##  Objective

Predict: arr_delay (arrival delay in minutes)

------------------------------------------------------------------------

##  Project Structure

    project_folder/

    final_project_with_eda_plots.py   # main ML pipeline
    download_weather_and_merge.py     # API + data merging
    app.py                           # Streamlit dashboard

     nycflights.csv
    nycflights_with_weather.csv

     requirements.txt
    
    outputs/
    ├── best_model.pkl
    ├── model_results.csv
    ├── validation_results.csv
    ├── best_model_predictions.csv
    ├── *.png (plots)
    └── project_summary.txt

------------------------------------------------------------------------

##  Feature Engineering

-   Time features (day, season, cyclical encoding)
-   Weather features (rain, snow, wind, weather risk score)
-   Historical delay features (route, carrier, hour)
-   Traffic congestion features

------------------------------------------------------------------------

##  Models Used

-   Linear Regression
-   Random Forest
-   HistGradientBoosting (best)

------------------------------------------------------------------------

##  Model Evaluation

-   Chronological split (70/15/15)
-   Metrics: RMSE, MAE, R²
-   Best model selected based on RMSE

------------------------------------------------------------------------

##  How to Run

### 1. Create virtual environment

python -m venv .venv

### 2. Activate (PowerShell)

..venv`\Scripts`{=tex}`\Activate`{=tex}.ps1

### 3. Install dependencies

pip install -r requirements.txt

### 4. Build dataset

python download_weather_and_merge.py

This step creates: nycflights_with_weather.csv

### 5. Train model

python final_project_with_eda_plots.py

### 6. Run dashboard

streamlit run app.py

------------------------------------------------------------------------

##  Dashboard

-   Input flight + weather
-   Predict delay
-   Show model results

------------------------------------------------------------------------

##  Output

-   best_model.pkl
-   model_results.csv
-   predictions

------------------------------------------------------------------------

##  Author

Le Quang Thien Nguyen
Hugh Tran
Binh Nguyen
