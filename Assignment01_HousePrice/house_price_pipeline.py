from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


DATASET_NAME = "House Price Prediction Dataset Vietnam - 2024"
DATASET_URL = "https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024"
TARGET = "Price"
TARGET_UNIT = "billion VND"
RANDOM_STATE = 42
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

RAW_COLUMNS = [
    "Address", "Area", "Frontage", "Access Road", "House direction",
    "Balcony direction", "Floors", "Bedrooms", "Bathrooms",
    "Legal status", "Furniture state", "Price"
]

FULL_FEATURES = [
    "Area", "Frontage", "Access Road", "House direction", "Balcony direction",
    "Floors", "Bedrooms", "Bathrooms", "Legal status", "Furniture state",
    "Province", "District"
]

SELECTED_FEATURES = ["Area", "Floors", "Bedrooms", "Bathrooms", "Province", "District"]

NUMERIC_6 = ["Area", "Floors", "Bedrooms", "Bathrooms"]
CATEGORICAL_6 = ["Province", "District"]
NUMERIC_FULL = ["Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"]
CATEGORICAL_FULL = [
    "House direction", "Balcony direction", "Legal status",
    "Furniture state", "Province", "District"
]


def normalize_text(value):
    if pd.isna(value):
        return np.nan
    text = unicodedata.normalize("NFC", str(value)).strip()
    text = " ".join(text.split())
    text = text.rstrip(".").strip()
    return text


def extract_location(address):
    if pd.isna(address):
        return pd.Series({"Province": np.nan, "District": np.nan})
    parts = [normalize_text(part) for part in str(address).split(",")]
    parts = [part for part in parts if part]
    province = parts[-1] if len(parts) >= 1 else np.nan
    district = parts[-2] if len(parts) >= 2 else np.nan
    return pd.Series({"Province": province, "District": district})


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse=False)


def make_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_ohe()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])


def make_pipeline(model, representation="six"):
    if representation == "six":
        preprocessor = make_preprocessor(NUMERIC_6, CATEGORICAL_6)
    elif representation == "full":
        preprocessor = make_preprocessor(NUMERIC_FULL, CATEGORICAL_FULL)
    else:
        raise ValueError("representation must be 'six' or 'full'")
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


def base_models():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def metric_dict(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(math.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def cv_metrics(pipeline, X, y):
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "MSE": "neg_mean_squared_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
        "MAPE": "neg_mean_absolute_percentage_error",
    }
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=CV,
        scoring=scoring,
        n_jobs=None,
        error_score="raise",
    )
    row = {}
    compact = {}
    for metric in scoring:
        values = scores[f"test_{metric}"]
        if metric in {"MAE", "MSE", "RMSE", "MAPE"}:
            values = -values
        row[f"{metric} Mean"] = float(np.mean(values))
        row[f"{metric} Std"] = float(np.std(values))
        compact[metric] = float(np.mean(values))
    return row, compact


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if pd.isna(obj) if not isinstance(obj, (str, bool, int, float, dict, list, tuple)) else False:
        return None
    return obj


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_eda(working_df, figures_dir):
    chart_paths = []

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(working_df["Price"], bins=40, color="#2F6F73", edgecolor="white")
    ax.set_title("Price Distribution")
    ax.set_xlabel("Price (billion VND)")
    ax.set_ylabel("Listing count")
    path = figures_dir / "01_price_distribution.png"
    save_figure(fig, path)
    chart_paths.append(str(path.relative_to(figures_dir.parent)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(working_df["Area"], bins=40, color="#8C6A3F", edgecolor="white")
    ax.set_title("Area Distribution")
    ax.set_xlabel("Area")
    ax.set_ylabel("Listing count")
    path = figures_dir / "02_area_distribution.png"
    save_figure(fig, path)
    chart_paths.append(str(path.relative_to(figures_dir.parent)))

    fig, ax = plt.subplots(figsize=(8, 5))
    floors = working_df["Floors"].dropna().round().astype(int).value_counts().sort_index()
    ax.bar(floors.index.astype(str), floors.values, color="#4F6D7A")
    ax.set_title("Floors Distribution")
    ax.set_xlabel("Floors")
    ax.set_ylabel("Listing count")
    path = figures_dir / "03_floors_distribution.png"
    save_figure(fig, path)
    chart_paths.append(str(path.relative_to(figures_dir.parent)))

    fig, ax = plt.subplots(figsize=(8, 5))
    bedrooms = working_df["Bedrooms"].dropna().round().astype(int).value_counts().sort_index()
    ax.bar(bedrooms.index.astype(str), bedrooms.values, color="#C1666B")
    ax.set_title("Bedrooms Distribution")
    ax.set_xlabel("Bedrooms")
    ax.set_ylabel("Listing count")
    path = figures_dir / "04_bedrooms_distribution.png"
    save_figure(fig, path)
    chart_paths.append(str(path.relative_to(figures_dir.parent)))

    fig, ax = plt.subplots(figsize=(10, 5))
    provinces = working_df["Province"].value_counts().head(10).sort_values()
    ax.barh(provinces.index, provinces.values, color="#5B8E7D")
    ax.set_title("Top Provinces by Listing Count")
    ax.set_xlabel("Listing count")
    path = figures_dir / "05_top_provinces_distribution.png"
    save_figure(fig, path)
    chart_paths.append(str(path.relative_to(figures_dir.parent)))

    return chart_paths


def model_file_name(model_name):
    return {
        "Linear Regression": "linear_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "Random Forest": "random_forest.joblib",
        "Extra Trees": "extra_trees.joblib",
        "Gradient Boosting": "gradient_boosting.joblib",
    }[model_name]


def build_model_by_name(model_name, rf_max_depth=None):
    if model_name == "Linear Regression":
        return LinearRegression()
    if model_name == "Decision Tree":
        return DecisionTreeRegressor(random_state=RANDOM_STATE)
    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=rf_max_depth,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_name == "Extra Trees":
        return ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    if model_name == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=RANDOM_STATE)
    raise ValueError(model_name)


def run_pipeline(project_root, verbose=True):
    project_root = Path(project_root)
    data_path = project_root / "data" / "vietnam_housing_dataset.csv"
    figures_dir = project_root / "figures"
    models_dir = project_root / "models"
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if list(df.columns) != RAW_COLUMNS:
        raise RuntimeError(f"Unexpected schema: {list(df.columns)}")

    working_df = df.copy()
    location_df = working_df["Address"].apply(extract_location)
    working_df = pd.concat([working_df, location_df], axis=1)
    for col in ["Province", "District", "House direction", "Balcony direction", "Legal status", "Furniture state"]:
        working_df[col] = working_df[col].apply(normalize_text)

    inspection = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": df.isna().sum().astype(int).to_dict(),
        "missing_percent": (df.isna().mean() * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "price_range": [float(df["Price"].min()), float(df["Price"].max())],
        "area_range": [float(df["Area"].min()), float(df["Area"].max())],
        "price_median": float(df["Price"].median()),
        "area_median": float(df["Area"].median()),
        "describe_numeric": df.describe().round(4).to_dict(),
        "province_count": int(working_df["Province"].nunique(dropna=True)),
        "district_count": int(working_df["District"].nunique(dropna=True)),
        "top_provinces": working_df["Province"].value_counts().head(10).astype(int).to_dict(),
    }

    eda_charts = plot_eda(working_df, figures_dir)

    X_6 = working_df[SELECTED_FEATURES].copy()
    X_full = working_df[FULL_FEATURES].copy()
    y = working_df[TARGET].copy()
    indices = np.arange(len(working_df))
    train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=RANDOM_STATE)
    X_train_6 = X_6.iloc[train_idx].copy()
    X_test_6 = X_6.iloc[test_idx].copy()
    X_train_full = X_full.iloc[train_idx].copy()
    X_test_full = X_full.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    split_info = {
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "test_size": 0.20,
        "random_state": RANDOM_STATE,
    }

    baseline_table, baseline_compact = cv_metrics(
        make_pipeline(DummyRegressor(strategy="median"), "six"),
        X_train_6,
        y_train,
    )

    experiment1_rows = []
    experiment1_compact = {}
    for name, model in base_models().items():
        metrics_full, metrics_compact = cv_metrics(make_pipeline(model, "six"), X_train_6, y_train)
        experiment1_rows.append({"Model": name, **metrics_full})
        experiment1_compact[name] = metrics_compact
        if verbose:
            print("Experiment 1 done:", name, metrics_compact)
    experiment1_table = pd.DataFrame(experiment1_rows)
    experiment1_compact_table = pd.DataFrame([
        {"Model": name, **metrics} for name, metrics in experiment1_compact.items()
    ])
    exp1_best_model = min(experiment1_compact, key=lambda name: experiment1_compact[name]["RMSE"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(experiment1_compact_table["Model"], experiment1_compact_table["RMSE"], color="#4F6D7A")
    ax.set_title("RMSE Comparison Across Five Models")
    ax.set_ylabel("Mean CV RMSE (billion VND)")
    ax.tick_params(axis="x", rotation=25)
    save_figure(fig, figures_dir / "06_rmse_model_comparison.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(experiment1_compact_table["Model"], experiment1_compact_table["R2"], color="#5B8E7D")
    ax.set_title("R2 Comparison Across Five Models")
    ax.set_ylabel("Mean CV R2")
    ax.tick_params(axis="x", rotation=25)
    save_figure(fig, figures_dir / "07_r2_model_comparison.png")

    rf_depth_values = [5, 10, 20, None]
    experiment2_rows = []
    for depth in rf_depth_values:
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=depth,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        metrics_full, metrics_compact = cv_metrics(make_pipeline(rf, "six"), X_train_6, y_train)
        label = "None" if depth is None else str(depth)
        experiment2_rows.append({"max_depth": label, "raw_max_depth": depth, **metrics_full, **metrics_compact})
        if verbose:
            print("Experiment 2 done: max_depth", label, metrics_compact)
    experiment2_compact = pd.DataFrame([
        {k: v for k, v in row.items() if k in {"max_depth", "MAE", "MSE", "RMSE", "R2", "MAPE"}}
        for row in experiment2_rows
    ])
    best_rf_row = min(experiment2_rows, key=lambda row: row["RMSE"])
    best_rf_depth = best_rf_row["raw_max_depth"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(experiment2_compact["max_depth"], experiment2_compact["RMSE"], marker="o", color="#C1666B")
    ax.set_title("Random Forest max_depth vs Mean CV RMSE")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Mean CV RMSE (billion VND)")
    save_figure(fig, figures_dir / "08_rf_max_depth_vs_rmse.png")

    candidate_name = exp1_best_model
    candidate_rf_max_depth = None
    if best_rf_row["RMSE"] < experiment1_compact[exp1_best_model]["RMSE"]:
        candidate_name = "Random Forest"
        candidate_rf_max_depth = best_rf_depth

    candidate_model = build_model_by_name(candidate_name, candidate_rf_max_depth)
    full_metrics_full, full_metrics = cv_metrics(make_pipeline(candidate_model, "full"), X_train_full, y_train)
    six_model = build_model_by_name(candidate_name, candidate_rf_max_depth)
    six_metrics_full, six_metrics = cv_metrics(make_pipeline(six_model, "six"), X_train_6, y_train)
    experiment3_table = pd.DataFrame([
        {"Representation": "Full Engineered Features", "Feature Count": 12, **full_metrics},
        {"Representation": "Six Selected Features", "Feature Count": 6, **six_metrics},
    ])
    difference = {
        metric: six_metrics[metric] - full_metrics[metric]
        for metric in ["MAE", "MSE", "RMSE", "R2", "MAPE"]
    }
    experiment3_with_difference = pd.concat([
        experiment3_table,
        pd.DataFrame([{"Representation": "Difference (6 - Full)", "Feature Count": -6, **difference}]),
    ], ignore_index=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(experiment3_table["Representation"], experiment3_table["RMSE"], color=["#4F6D7A", "#C1666B"])
    ax.set_title("RMSE: Full 12 Features vs 6 Selected Features")
    ax.set_ylabel("Mean CV RMSE (billion VND)")
    ax.tick_params(axis="x", rotation=12)
    save_figure(fig, figures_dir / "09_representation_rmse_comparison.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(experiment3_table["Representation"], experiment3_table["R2"], color=["#4F6D7A", "#5B8E7D"])
    ax.set_title("R2: Full 12 Features vs 6 Selected Features")
    ax.set_ylabel("Mean CV R2")
    ax.tick_params(axis="x", rotation=12)
    save_figure(fig, figures_dir / "10_representation_r2_comparison.png")

    final_model = build_model_by_name(candidate_name, candidate_rf_max_depth)
    final_pipeline = make_pipeline(final_model, "six")
    final_pipeline.fit(X_train_6, y_train)
    y_pred = final_pipeline.predict(X_test_6)
    final_test_metrics = metric_dict(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.35, s=16, color="#2F6F73")
    low = min(float(y_test.min()), float(np.min(y_pred)))
    high = max(float(y_test.max()), float(np.max(y_pred)))
    ax.plot([low, high], [low, high], color="#C1666B", linestyle="--", linewidth=2)
    ax.set_title("Actual Price vs Predicted Price")
    ax.set_xlabel("Actual Price (billion VND)")
    ax.set_ylabel("Predicted Price (billion VND)")
    save_figure(fig, figures_dir / "11_actual_vs_predicted_scatter.png")

    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=40, color="#8C6A3F", edgecolor="white")
    ax.set_title("Final Residual Distribution")
    ax.set_xlabel("Residual = Actual - Predicted (billion VND)")
    ax.set_ylabel("Listing count")
    save_figure(fig, figures_dir / "12_final_residual_distribution.png")

    cv_rmse = six_metrics["RMSE"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Training CV RMSE", "Held-out Test RMSE"], [cv_rmse, final_test_metrics["RMSE"]], color=["#4F6D7A", "#C1666B"])
    ax.set_title("CV vs Held-out RMSE")
    ax.set_ylabel("RMSE (billion VND)")
    save_figure(fig, figures_dir / "13_cv_vs_heldout_rmse.png")

    deployment_metrics = dict(experiment1_compact)
    rf_deploy_depth = None
    if candidate_name == "Random Forest" and candidate_rf_max_depth is not None:
        deployment_metrics["Random Forest"] = {
            metric: float(best_rf_row[metric]) for metric in ["MAE", "MSE", "RMSE", "R2", "MAPE"]
        }
        rf_deploy_depth = candidate_rf_max_depth

    model_registry = {
        "target": TARGET,
        "target_unit": TARGET_UNIT,
        "selected_features": SELECTED_FEATURES,
        "full_engineered_features": FULL_FEATURES,
        "scientific_final_model": candidate_name,
        "model_selection_metric": "RMSE",
        "models": {},
        "final_test_metrics": {"model": candidate_name, **final_test_metrics},
    }
    for name in base_models():
        rf_depth_for_file = rf_deploy_depth if name == "Random Forest" else None
        pipeline = make_pipeline(build_model_by_name(name, rf_depth_for_file), "six")
        pipeline.fit(X_train_6, y_train)
        file_name = model_file_name(name)
        joblib.dump(pipeline, models_dir / file_name)
        config = pipeline.get_params()["model"].get_params()
        model_registry["models"][name] = {
            "file": file_name,
            "scientific_final_model": name == candidate_name,
            "cv_metrics": deployment_metrics[name],
            "configuration": clean_for_json(config),
        }
    joblib.dump(final_pipeline, models_dir / "best_model.joblib")
    model_registry["best_model_file"] = "best_model.joblib"
    (models_dir / "model_registry.json").write_text(json.dumps(clean_for_json(model_registry), indent=4, ensure_ascii=False), encoding="utf-8")

    province_district = {}
    for province, districts in working_df.groupby("Province")["District"]:
        if pd.isna(province):
            continue
        province_district[str(province)] = sorted({str(d) for d in districts.dropna().unique()})
    numeric_ranges = {}
    for feature in NUMERIC_6:
        numeric_ranges[feature] = {
            "min": float(working_df[feature].min(skipna=True)),
            "max": float(working_df[feature].max(skipna=True)),
            "median": float(working_df[feature].median(skipna=True)),
        }
    ui_metadata = {
        "selected_features": SELECTED_FEATURES,
        "target_unit": TARGET_UNIT,
        "provinces": sorted(province_district.keys()),
        "province_districts": province_district,
        "numeric_ranges": numeric_ranges,
    }
    (models_dir / "ui_metadata.json").write_text(json.dumps(clean_for_json(ui_metadata), indent=4, ensure_ascii=False), encoding="utf-8")

    demo_candidates = [
        {"Province": "Hà Nội", "district_contains": "Cầu Giấy", "Area": 35.0, "Floors": 2.0, "Bedrooms": 2.0, "Bathrooms": 1.0},
        {"Province": "Hồ Chí Minh", "district_contains": "Gò Vấp", "Area": 70.0, "Floors": 4.0, "Bedrooms": 4.0, "Bathrooms": 3.0},
        {"Province": "Đà Nẵng", "district_contains": "", "Area": 150.0, "Floors": 5.0, "Bedrooms": 6.0, "Bathrooms": 5.0},
    ]
    demos = []
    for demo in demo_candidates:
        province = demo["Province"] if demo["Province"] in province_district else sorted(province_district.keys())[0]
        district_list = province_district[province]
        district = district_list[0]
        contains = demo["district_contains"]
        if contains:
            matches = [d for d in district_list if contains in d]
            if matches:
                district = matches[0]
        row = {
            "Area": demo["Area"],
            "Floors": demo["Floors"],
            "Bedrooms": demo["Bedrooms"],
            "Bathrooms": demo["Bathrooms"],
            "Province": province,
            "District": district,
        }
        pred = float(final_pipeline.predict(pd.DataFrame([row], columns=SELECTED_FEATURES))[0])
        demos.append({"input": row, "predicted_price_billion": pred})
    (models_dir / "demo_predictions.json").write_text(json.dumps(clean_for_json(demos), indent=4, ensure_ascii=False), encoding="utf-8")

    summary = {
        "dataset": inspection,
        "eda_charts": eda_charts,
        "features": {"full": FULL_FEATURES, "selected": SELECTED_FEATURES},
        "split": split_info,
        "baseline": {"full": baseline_table, "compact": baseline_compact},
        "experiment1": {
            "full_table": experiment1_table.round(6).to_dict(orient="records"),
            "compact_table": experiment1_compact_table.round(6).to_dict(orient="records"),
            "best_model": exp1_best_model,
        },
        "experiment2": {
            "table": experiment2_compact.round(6).to_dict(orient="records"),
            "best_rf_max_depth": "None" if best_rf_depth is None else best_rf_depth,
        },
        "candidate": {
            "algorithm": candidate_name,
            "rf_max_depth": "None" if candidate_rf_max_depth is None else candidate_rf_max_depth,
        },
        "experiment3": {
            "table": experiment3_with_difference.round(6).to_dict(orient="records"),
            "difference": difference,
        },
        "final_test": final_test_metrics,
        "demos": demos,
    }
    (models_dir / "analysis_summary.json").write_text(json.dumps(clean_for_json(summary), indent=4, ensure_ascii=False), encoding="utf-8")
    return summary
