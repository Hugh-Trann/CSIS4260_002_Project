from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

plt.style.use("seaborn-v0_8")


# ------------------------------------------------------------
# Project title:
# Predicting flight arrival delay using weather + flight data
# Improved version with stronger feature engineering and
# chronological train / validation / test evaluation.
# ------------------------------------------------------------


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


# ------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------
def add_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "weather_datetime" in df.columns:
        df["weather_datetime"] = pd.to_datetime(df["weather_datetime"], errors="coerce")
    else:
        raise ValueError("Expected column 'weather_datetime' was not found in merged dataset.")

    # Build a proper departure datetime from date + scheduled hour/minute.
    required_cols = ["year", "month", "day", "hour", "minute"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for datetime creation: {missing}")

    date_part = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")

    hour_fixed = pd.to_numeric(df["hour"], errors="coerce")
    minute_fixed = pd.to_numeric(df["minute"], errors="coerce")

    hour_fixed = hour_fixed.fillna(0)
    minute_fixed = minute_fixed.fillna(0)

    mask_24 = hour_fixed == 24
    hour_fixed = hour_fixed.where(~mask_24, 0)
    date_part = date_part.where(~mask_24, date_part + pd.Timedelta(days=1))

    df["dep_datetime"] = (
        date_part
        + pd.to_timedelta(hour_fixed, unit="h")
        + pd.to_timedelta(minute_fixed, unit="m")
    )

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    dt = df["dep_datetime"]
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["quarter"] = dt.dt.quarter
    df["day_of_year"] = dt.dt.dayofyear

    # Seasons for northern hemisphere
    month = dt.dt.month
    df["season"] = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        ["winter", "spring", "summer", "fall"],
        default="unknown",
    )

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Time blocks
    df["departure_period"] = pd.cut(
        pd.to_numeric(df["hour"], errors="coerce").fillna(0),
        bins=[-1, 5, 11, 17, 21, 24],
        labels=["overnight", "morning", "afternoon", "evening", "late_night"],
    ).astype(str)

    return df


def add_weather_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in [
        "temperature_2m",
        "precipitation",
        "rain",
        "snowfall",
        "wind_speed_10m",
        "wind_gusts_10m",
        "weather_code",
    ]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["has_precipitation"] = (df["precipitation"].fillna(0) > 0).astype(int)
    df["has_rain"] = (df["rain"].fillna(0) > 0).astype(int)
    df["has_snow"] = (df["snowfall"].fillna(0) > 0).astype(int)
    df["strong_wind"] = (df["wind_speed_10m"].fillna(0) >= 25).astype(int)
    df["strong_gust"] = (df["wind_gusts_10m"].fillna(0) >= 35).astype(int)
    df["freezing_temp_flag"] = (df["temperature_2m"] <= 0).fillna(False).astype(int)
    df["heavy_precip_flag"] = (df["precipitation"].fillna(0) >= 2.5).astype(int)
    df["heavy_rain_flag"] = (df["rain"].fillna(0) >= 2.5).astype(int)
    df["snow_risk_flag"] = ((df["snowfall"].fillna(0) > 0) | (df["freezing_temp_flag"] == 1)).astype(int)

    severe_codes = {
        51, 53, 55, 56, 57, 61, 63, 65, 66, 67,
        71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 99,
    }
    df["severe_weather"] = df["weather_code"].isin(severe_codes).astype(int)

    df["weather_risk_score"] = (
        df["has_precipitation"]
        + df["has_rain"]
        + df["has_snow"]
        + df["strong_wind"]
        + df["strong_gust"]
        + df["freezing_temp_flag"]
        + df["heavy_precip_flag"]
        + df["heavy_rain_flag"]
        + df["snow_risk_flag"]
        + df["severe_weather"]
    )

    return df


# Historical features must use past information only.
def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("dep_datetime").reset_index(drop=True)

    df["route"] = df["origin"].astype(str) + "_" + df["dest"].astype(str)
    df["carrier_route"] = df["carrier"].astype(str) + "_" + df["route"].astype(str)

    target = pd.to_numeric(df["arr_delay"], errors="coerce")

    def expanding_mean_shifted(series: pd.Series) -> pd.Series:
        return series.shift(1).expanding().mean()

    df["route_hist_mean_delay"] = target.groupby(df["route"]).transform(expanding_mean_shifted)
    df["carrier_hist_mean_delay"] = target.groupby(df["carrier"]).transform(expanding_mean_shifted)
    df["carrier_route_hist_mean_delay"] = target.groupby(df["carrier_route"]).transform(expanding_mean_shifted)
    df["origin_hist_mean_delay"] = target.groupby(df["origin"]).transform(expanding_mean_shifted)
    df["dest_hist_mean_delay"] = target.groupby(df["dest"]).transform(expanding_mean_shifted)
    df["hour_hist_mean_delay"] = target.groupby(df["hour"]).transform(expanding_mean_shifted)

    # Historical flight counts / congestion-like proxies using only current schedule info.
    df["dep_hour_floor"] = df["dep_datetime"].dt.floor("h")
    df["origin_hour_traffic_count"] = (
        df.groupby(["origin", "dep_hour_floor"])["origin"].transform("size")
    )
    df["route_hour_traffic_count"] = (
        df.groupby(["route", "dep_hour_floor"])["route"].transform("size")
    )
    df["carrier_origin_hour_count"] = (
        df.groupby(["carrier", "origin", "dep_hour_floor"])["carrier"].transform("size")
    )

    return df


# ------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------
def prepare_data(df: pd.DataFrame):
    df = add_datetime_columns(df)
    df = add_time_features(df)
    df = add_weather_risk_features(df)

    if "arr_delay" not in df.columns:
        raise ValueError("Target column 'arr_delay' was not found in dataset.")

    df["arr_delay"] = pd.to_numeric(df["arr_delay"], errors="coerce")
    df = df.dropna(subset=["dep_datetime", "arr_delay", "origin", "dest", "carrier"]).copy()

    # Add historical features after target is cleaned.
    df = add_historical_features(df)

    feature_columns = [
        # Schedule and route
        "month", "day", "hour", "minute", "day_of_week", "is_weekend",
        "week_of_year", "quarter", "day_of_year", "hour_sin", "hour_cos",
        "month_sin", "month_cos",
        "carrier", "origin", "dest", "route", "carrier_route", "departure_period", "season",
        "distance",
        # Weather raw
        "temperature_2m", "precipitation", "rain", "snowfall",
        "wind_speed_10m", "wind_gusts_10m", "weather_code",
        # Weather engineered
        "has_precipitation", "has_rain", "has_snow", "strong_wind",
        "strong_gust", "freezing_temp_flag", "heavy_precip_flag",
        "heavy_rain_flag", "snow_risk_flag", "severe_weather", "weather_risk_score",
        # Historical features
        "route_hist_mean_delay", "carrier_hist_mean_delay", "carrier_route_hist_mean_delay",
        "origin_hist_mean_delay", "dest_hist_mean_delay", "hour_hist_mean_delay",
        # Traffic count features
        "origin_hour_traffic_count", "route_hour_traffic_count", "carrier_origin_hour_count",
    ]

    target_column = "arr_delay"

    existing_features = [c for c in feature_columns if c in df.columns]
    modeling_df = df[existing_features + [target_column, "dep_datetime"]].copy()
    modeling_df = modeling_df.sort_values("dep_datetime").reset_index(drop=True)

    X = modeling_df[existing_features].copy()
    y = modeling_df[target_column].copy()

    categorical_columns = [
        c for c in ["carrier", "origin", "dest", "route", "carrier_route", "departure_period", "season"]
        if c in X.columns
    ]
    numeric_columns = [c for c in X.columns if c not in categorical_columns]

    return X, y, categorical_columns, numeric_columns, modeling_df


# ------------------------------------------------------------
# Chronological split
# ------------------------------------------------------------
def chronological_split(modeling_df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    n = len(modeling_df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = modeling_df.iloc[:train_end].copy()
    val_df = modeling_df.iloc[train_end:val_end].copy()
    test_df = modeling_df.iloc[val_end:].copy()

    return train_df, val_df, test_df


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------
def build_preprocessor(categorical_columns, numeric_columns, dense_output: bool = False):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # sparse_output available in newer sklearn; sparse in older sklearn
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=not dense_output)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )


# ------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------
def calculate_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def evaluate_on_split(model_name, pipeline, X_train, y_train, X_eval, y_eval, split_name="validation"):
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_eval)
    rmse, mae, r2 = calculate_metrics(y_eval, predictions)

    return {
        "model": model_name,
        "split": split_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": predictions,
        "fitted_pipeline": pipeline,
    }


def tune_hist_gradient_boosting(categorical_columns, numeric_columns, X_train, y_train, X_val, y_val):
    dense_preprocessor = build_preprocessor(categorical_columns, numeric_columns, dense_output=True)

    param_grid = [
        {"learning_rate": 0.05, "max_depth": 6, "max_iter": 200, "min_samples_leaf": 20},
        {"learning_rate": 0.05, "max_depth": 8, "max_iter": 300, "min_samples_leaf": 20},
        {"learning_rate": 0.08, "max_depth": 6, "max_iter": 250, "min_samples_leaf": 30},
        {"learning_rate": 0.10, "max_depth": 6, "max_iter": 200, "min_samples_leaf": 20},
    ]

    all_results = []

    for params in param_grid:
        model = Pipeline(
            steps=[
                ("preprocessor", dense_preprocessor),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=params["learning_rate"],
                        max_depth=params["max_depth"],
                        max_iter=params["max_iter"],
                        min_samples_leaf=params["min_samples_leaf"],
                        random_state=42,
                    ),
                ),
            ]
        )

        result = evaluate_on_split(
            model_name=(
                "HistGradientBoosting"
                f"_lr{params['learning_rate']}_depth{params['max_depth']}"
                f"_iter{params['max_iter']}_leaf{params['min_samples_leaf']}"
            ),
            pipeline=model,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_val,
            y_eval=y_val,
            split_name="validation",
        )
        all_results.append(result)

    best_result = min(all_results, key=lambda x: x["rmse"])
    return best_result, all_results


# ------------------------------------------------------------
# Plots and summary
# ------------------------------------------------------------
def save_scatter_plot(y_true, predictions, output_file, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        y_true,
        predictions,
        alpha=0.45,
        edgecolors="black",
        linewidth=0.25,
        color="#4C78A8",
    )

    min_val = min(float(np.min(y_true)), float(np.min(predictions)))
    max_val = max(float(np.max(y_true)), float(np.max(predictions)))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=2, color="#E45756")

    plt.xlabel("Actual arrival delay (minutes)")
    plt.ylabel("Predicted arrival delay (minutes)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file, dpi=220)
    plt.close()


def save_bar_plot(results_df, metric, output_file, title):
    plot_df = results_df.sort_values(by=metric, ascending=True).reset_index(drop=True)
    colors = ["#59A14F", "#F28E2B", "#E15759", "#76B7B2", "#4E79A7", "#EDC948"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(plot_df["model"], plot_df[metric], color=colors[: len(plot_df)], edgecolor="black", alpha=0.9)
    plt.xlabel("Model")
    plt.ylabel(metric.upper())
    plt.title(title)
    plt.xticks(rotation=18, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.45)

    for bar, value in zip(bars, plot_df[metric]):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=220)
    plt.close()


def save_delay_distribution_plot(modeling_df, output_file):
    plot_df = modeling_df.dropna(subset=["arr_delay"]).copy()
    clipped_delay = np.clip(plot_df["arr_delay"], -60, 240)

    plt.figure(figsize=(10, 6))
    plt.hist(clipped_delay, bins=80, color="#4E79A7", edgecolor="white", alpha=0.9)
    plt.xlabel("Arrival delay (minutes, clipped to -60 to 240)")
    plt.ylabel("Number of flights")
    plt.title("Distribution of Arrival Delay")
    plt.grid(axis="y", linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=220)
    plt.close()


def save_delay_by_hour_plot(modeling_df, output_file):
    hourly_delay = modeling_df.groupby("hour", dropna=False)["arr_delay"].mean().reset_index()
    hourly_delay = hourly_delay.sort_values("hour")

    plt.figure(figsize=(10, 6))
    plt.plot(hourly_delay["hour"], hourly_delay["arr_delay"], marker="o", linewidth=2.5, color="#F28E2B")
    plt.fill_between(hourly_delay["hour"], hourly_delay["arr_delay"], alpha=0.2, color="#F28E2B")
    plt.xlabel("Scheduled departure hour")
    plt.ylabel("Average arrival delay (minutes)")
    plt.title("Average Arrival Delay by Departure Hour")
    plt.xticks(range(0, 24, 1))
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=220)
    plt.close()


def save_delay_vs_precipitation_plot(modeling_df, output_file):
    plot_df = modeling_df.dropna(subset=["precipitation", "arr_delay"]).copy()
    if len(plot_df) > 12000:
        plot_df = plot_df.sample(12000, random_state=42)

    plt.figure(figsize=(10, 6))
    plt.scatter(plot_df["precipitation"], plot_df["arr_delay"], alpha=0.3, color="#59A14F", edgecolors="none")
    plt.xlabel("Precipitation")
    plt.ylabel("Arrival delay (minutes)")
    plt.title("Arrival Delay vs Precipitation")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=220)
    plt.close()


def save_weather_risk_boxplot(modeling_df, output_file):
    plot_df = modeling_df.dropna(subset=["weather_risk_score", "arr_delay"]).copy()
    plot_df["weather_risk_band"] = pd.cut(
        plot_df["weather_risk_score"],
        bins=[-1, 1, 3, 6, 20],
        labels=["Low", "Moderate", "High", "Very High"],
    )
    grouped = [plot_df.loc[plot_df["weather_risk_band"] == band, "arr_delay"].clip(-60, 240) for band in ["Low", "Moderate", "High", "Very High"]]

    plt.figure(figsize=(9, 6))
    bp = plt.boxplot(grouped, labels=["Low", "Moderate", "High", "Very High"], patch_artist=True, showfliers=False)
    palette = ["#76B7B2", "#59A14F", "#F28E2B", "#E15759"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.8)

    plt.xlabel("Weather risk band")
    plt.ylabel("Arrival delay (minutes, clipped to -60 to 240)")
    plt.title("Arrival Delay by Weather Risk Level")
    plt.grid(axis="y", linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=220)
    plt.close()


def save_actual_vs_predicted_line_plot(y_true, predictions, output_file, title, sample_size=500):
    line_df = pd.DataFrame({
        "actual": pd.Series(y_true).reset_index(drop=True),
        "predicted": pd.Series(predictions).reset_index(drop=True),
    })
    line_df = line_df.iloc[: min(sample_size, len(line_df))].copy()

    plt.figure(figsize=(12, 6))
    plt.plot(line_df.index, line_df["actual"], label="Actual", linewidth=2, color="#4E79A7")
    plt.plot(line_df.index, line_df["predicted"], label="Predicted", linewidth=2, color="#E15759")
    plt.xlabel(f"First {len(line_df)} test observations in chronological order")
    plt.ylabel("Arrival delay (minutes)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=220)
    plt.close()


def write_summary(validation_results_df, test_results_df, best_model_name, train_df, val_df, test_df, output_file):
    lines = []
    lines.append("Flights + Weather Project Summary\n")
    lines.append("Target variable: arr_delay (arrival delay)\n")
    lines.append("\n")
    lines.append("Important modeling choices:\n")
    lines.append("- Removed dep_delay from features to avoid making the target too easy to predict.\n")
    lines.append("- Removed air_time from features because it is too close to post-flight information.\n")
    lines.append("- Removed visibility from the weather collection step.\n")
    lines.append("- Added route/carrier historical features using past data only.\n")
    lines.append("- Added weather risk score features.\n")
    lines.append("- Added origin-hour traffic count features.\n")
    lines.append("- Used chronological train/validation/test split instead of random split.\n")
    lines.append(f"- Train end at: {train_df['dep_datetime'].max()}\n")
    lines.append(f"- Validation end at: {val_df['dep_datetime'].max()}\n")
    lines.append(f"- Test starts at: {test_df['dep_datetime'].min()}\n")
    lines.append("\n")
    lines.append("Validation results:\n")
    for _, row in validation_results_df.iterrows():
        lines.append(
            f"- {row['model']}: RMSE = {row['rmse']:.2f}, MAE = {row['mae']:.2f}, R^2 = {row['r2']:.3f}\n"
        )
    lines.append("\n")
    lines.append("Final test results:\n")
    for _, row in test_results_df.iterrows():
        lines.append(
            f"- {row['model']}: RMSE = {row['rmse']:.2f}, MAE = {row['mae']:.2f}, R^2 = {row['r2']:.3f}\n"
        )
    lines.append("\n")
    lines.append(f"Best model on test set: {best_model_name}\n")
    lines.append(
        "Interpretation: Lower RMSE and MAE are better, and higher R^2 is better. "
        "This project uses schedule, route, engineered weather conditions, historical route/carrier behavior, and traffic count proxies to predict flight arrival delay.\n"
    )
    Path(output_file).write_text("".join(lines), encoding="utf-8")


def main():
    base_folder = Path(__file__).resolve().parent
    output_folder = base_folder / "outputs"
    output_folder.mkdir(exist_ok=True)

    merged_file = base_folder / "nycflights_with_weather.csv"

    if not merged_file.exists():
        raise FileNotFoundError(
            "nycflights_with_weather.csv was not found. Please place the merged dataset in the project folder first."
        )

    print("Reading data...")
    df = load_data(str(merged_file))
    print("Data shape:", df.shape)

    X, y, categorical_columns, numeric_columns, modeling_df = prepare_data(df)
    print("Modeling data shape:", modeling_df.shape)

    modeling_df.to_csv(output_folder / "modeling_dataset.csv", index=False)

    train_df, val_df, test_df = chronological_split(modeling_df, train_frac=0.70, val_frac=0.15)

    train_idx = train_df.index
    val_idx = val_df.index
    test_idx = test_df.index

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_val = X.loc[val_idx]
    y_val = y.loc[val_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]

    print("Train size:", len(train_df))
    print("Validation size:", len(val_df))
    print("Test size:", len(test_df))

    sparse_preprocessor = build_preprocessor(categorical_columns, numeric_columns, dense_output=False)

    linear_model = Pipeline(
        steps=[
            ("preprocessor", sparse_preprocessor),
            ("model", LinearRegression()),
        ]
    )

    random_forest_model = Pipeline(
        steps=[
            ("preprocessor", sparse_preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                    max_depth=16,
                    min_samples_leaf=3,
                ),
            ),
        ]
    )

    print("Evaluating Linear Regression on validation set...")
    linear_val_result = evaluate_on_split(
        "Linear Regression", linear_model, X_train, y_train, X_val, y_val, split_name="validation"
    )

    print("Evaluating Random Forest on validation set...")
    rf_val_result = evaluate_on_split(
        "Random Forest", random_forest_model, X_train, y_train, X_val, y_val, split_name="validation"
    )

    print("Tuning HistGradientBoosting on validation set...")
    best_hgb_validation_result, hgb_all_results = tune_hist_gradient_boosting(
        categorical_columns, numeric_columns, X_train, y_train, X_val, y_val
    )

    validation_rows = [
        {"model": linear_val_result["model"], "split": "validation", "rmse": linear_val_result["rmse"], "mae": linear_val_result["mae"], "r2": linear_val_result["r2"]},
        {"model": rf_val_result["model"], "split": "validation", "rmse": rf_val_result["rmse"], "mae": rf_val_result["mae"], "r2": rf_val_result["r2"]},
    ]
    for r in hgb_all_results:
        validation_rows.append({"model": r["model"], "split": "validation", "rmse": r["rmse"], "mae": r["mae"], "r2": r["r2"]})

    validation_results_df = pd.DataFrame(validation_rows).sort_values(by="rmse", ascending=True)
    validation_results_df.to_csv(output_folder / "validation_results.csv", index=False)

    print("\nValidation results:")
    print(validation_results_df)

    # Refit final models on train + validation, then evaluate on test.
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    linear_final = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(categorical_columns, numeric_columns, dense_output=False)),
            ("model", LinearRegression()),
        ]
    )
    rf_final = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(categorical_columns, numeric_columns, dense_output=False)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                    max_depth=16,
                    min_samples_leaf=3,
                ),
            ),
        ]
    )

    best_hgb_name = best_hgb_validation_result["model"]
    # Parse chosen parameters from the best model name would be messy; reuse fitted params from validation pipeline.
    best_hgb_final = best_hgb_validation_result["fitted_pipeline"]

    print("\nEvaluating final models on test set...")
    linear_test_result = evaluate_on_split(
        "Linear Regression", linear_final, X_train_val, y_train_val, X_test, y_test, split_name="test"
    )
    rf_test_result = evaluate_on_split(
        "Random Forest", rf_final, X_train_val, y_train_val, X_test, y_test, split_name="test"
    )
    hgb_test_result = evaluate_on_split(
        best_hgb_name, best_hgb_final, X_train_val, y_train_val, X_test, y_test, split_name="test"
    )

    test_results_df = pd.DataFrame(
        [
            {"model": linear_test_result["model"], "split": "test", "rmse": linear_test_result["rmse"], "mae": linear_test_result["mae"], "r2": linear_test_result["r2"]},
            {"model": rf_test_result["model"], "split": "test", "rmse": rf_test_result["rmse"], "mae": rf_test_result["mae"], "r2": rf_test_result["r2"]},
            {"model": hgb_test_result["model"], "split": "test", "rmse": hgb_test_result["rmse"], "mae": hgb_test_result["mae"], "r2": hgb_test_result["r2"]},
        ]
    ).sort_values(by="rmse", ascending=True)

    test_results_df.to_csv(output_folder / "model_results.csv", index=False)

    best_model_name = test_results_df.iloc[0]["model"]

    if best_model_name == linear_test_result["model"]:
        best_predictions = linear_test_result["predictions"]
        best_model = linear_test_result["fitted_pipeline"]
    elif best_model_name == rf_test_result["model"]:
        best_predictions = rf_test_result["predictions"]
        best_model = rf_test_result["fitted_pipeline"]
    else:
        best_predictions = hgb_test_result["predictions"]
        best_model = hgb_test_result["fitted_pipeline"]

    joblib.dump(best_model, output_folder / "best_model.pkl")

    predictions_df = pd.DataFrame(
        {
            "actual_arr_delay": y_test.reset_index(drop=True),
            "predicted_arr_delay": pd.Series(best_predictions),
        }
    )
    predictions_df.to_csv(output_folder / "best_model_predictions.csv", index=False)

    save_scatter_plot(
        y_test,
        best_predictions,
        output_folder / "actual_vs_predicted.png",
        f"Actual vs Predicted Arrival Delay ({best_model_name})",
    )
    save_bar_plot(test_results_df, "rmse", output_folder / "model_comparison_rmse.png", "Model comparison using RMSE")
    save_bar_plot(test_results_df, "mae", output_folder / "model_comparison_mae.png", "Model comparison using MAE")

    save_actual_vs_predicted_line_plot(
        y_test,
        best_predictions,
        output_folder / "actual_vs_predicted_line.png",
        f"Actual vs Predicted Arrival Delay Over Time ({best_model_name})",
    )

    save_delay_distribution_plot(modeling_df, output_folder / "delay_distribution.png")
    save_delay_by_hour_plot(modeling_df, output_folder / "delay_by_hour.png")
    save_delay_vs_precipitation_plot(modeling_df, output_folder / "delay_vs_precipitation.png")
    save_weather_risk_boxplot(modeling_df, output_folder / "delay_by_weather_risk.png")

    write_summary(
        validation_results_df=validation_results_df,
        test_results_df=test_results_df,
        best_model_name=best_model_name,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_file=output_folder / "project_summary.txt",
    )

    print("\nProject finished successfully!")
    print("Results saved in the outputs folder.")
    print("\nFinal test results:")
    print(test_results_df)


if __name__ == "__main__":
    main()
