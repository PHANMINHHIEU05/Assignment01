from __future__ import annotations

import json
import shutil
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = Path("/home/hiubeo/Downloads/nhadat/vietnam_housing_dataset.csv")


PIPELINE_CODE = r'''
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
'''


APP_CODE = r'''
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from neo4j_service import (
    get_knowledge_graph_summary,
    get_recent_predictions,
    save_prediction,
    verify_connection,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"
METADATA_PATH = MODELS_DIR / "ui_metadata.json"


st.set_page_config(
    page_title="Vietnam House Price Prediction",
    page_icon="house",
    layout="centered",
)


def load_json(path: Path, label: str):
    if not path.exists():
        st.error(f"{label} was not found: {path}")
        st.stop()
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_data(ttl=60)
def check_neo4j_status():
    try:
        return bool(verify_connection())
    except Exception:
        return False


def format_metric(value):
    return f"{float(value):.4f}"


registry = load_json(REGISTRY_PATH, "model_registry.json")
metadata = load_json(METADATA_PATH, "ui_metadata.json")
selected_features = registry["selected_features"]

st.title("Vietnam House Price Prediction System")
st.caption("Vietnam Housing Dataset 2024")
st.info(
    "Educational machine-learning demonstration. Predictions are model estimates "
    "and are not official property valuations."
)

model_names = list(registry["models"].keys())
selected_model_name = st.selectbox("Choose regression model", model_names)
model_info = registry["models"][selected_model_name]
model_path = MODELS_DIR / model_info["file"]
model = load_model(str(model_path))

if selected_model_name == registry["scientific_final_model"]:
    st.success("Scientific Final Model")
else:
    st.info("Deployment Comparison Model")

province = st.selectbox("Province", metadata["provinces"])
districts = metadata["province_districts"].get(province, [])
district = st.selectbox("District", districts)

ranges = metadata["numeric_ranges"]
area = st.number_input(
    "Area",
    min_value=0.0,
    max_value=max(1000.0, float(ranges["Area"]["max"])),
    value=float(ranges["Area"]["median"]),
    step=1.0,
)
floors = st.number_input("Floors", min_value=0.0, max_value=50.0, value=float(ranges["Floors"]["median"]), step=1.0)
bedrooms = st.number_input("Bedrooms", min_value=0.0, max_value=50.0, value=float(ranges["Bedrooms"]["median"]), step=1.0)
bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=50.0, value=float(ranges["Bathrooms"]["median"]), step=1.0)

sample = pd.DataFrame([{
    "Area": np.nan if area == 0 else area,
    "Floors": np.nan if floors == 0 else floors,
    "Bedrooms": np.nan if bedrooms == 0 else bedrooms,
    "Bathrooms": np.nan if bathrooms == 0 else bathrooms,
    "Province": province,
    "District": district,
}], columns=selected_features)

if st.button("Predict House Price", type="primary"):
    predicted_billion = float(model.predict(sample)[0])
    predicted_vnd = predicted_billion * 1_000_000_000
    st.subheader("Model Estimate")
    st.metric("Estimated House Price", f"{predicted_billion:,.2f} billion VND")
    st.write(f"Approximate VND: {predicted_vnd:,.0f} VND")
    try:
        save_prediction(
            model_name=selected_model_name,
            input_values=sample.iloc[0].to_dict(),
            predicted_price_billion=predicted_billion,
        )
        st.success("Anonymous prediction saved to Neo4j Knowledge Graph.")
    except Exception as error:
        st.warning(f"Prediction completed, but Neo4j save is unavailable: {error}")

st.subheader("Selected Model Metrics")
metrics = model_info["cv_metrics"]
cols = st.columns(2)
cols[0].metric("CV MAE", format_metric(metrics["MAE"]))
cols[1].metric("CV MSE", format_metric(metrics["MSE"]))
cols[0].metric("CV RMSE", format_metric(metrics["RMSE"]))
cols[1].metric("CV R2", format_metric(metrics["R2"]))
st.metric("CV MAPE", format_metric(metrics["MAPE"]))

if selected_model_name == registry["scientific_final_model"]:
    with st.expander("Scientific Held-out Test Metrics"):
        final_metrics = registry["final_test_metrics"]
        st.write(f"MAE: {format_metric(final_metrics['MAE'])}")
        st.write(f"MSE: {format_metric(final_metrics['MSE'])}")
        st.write(f"RMSE: {format_metric(final_metrics['RMSE'])}")
        st.write(f"R2: {format_metric(final_metrics['R2'])}")
        st.write(f"MAPE: {format_metric(final_metrics['MAPE'])}")

st.divider()
st.subheader("Neo4j AuraDB")
if check_neo4j_status():
    st.success("Connected")
else:
    st.warning("Unavailable")

try:
    summary = get_knowledge_graph_summary()
    st.write("Knowledge Graph Summary")
    st.json(summary)
except Exception:
    st.caption("Knowledge Graph summary is unavailable.")

try:
    recent = get_recent_predictions(limit=5)
    if recent:
        st.write("Recent Anonymous Price Predictions")
        st.dataframe(pd.DataFrame(recent), use_container_width=True)
except Exception:
    st.caption("Recent predictions are unavailable.")
'''


NEO4J_SERVICE_CODE = r'''
from __future__ import annotations

import os
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()


CONSTRAINT_QUERIES = [
    "CREATE CONSTRAINT house_model_name_unique IF NOT EXISTS FOR (m:HouseModel) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT house_feature_name_unique IF NOT EXISTS FOR (f:HouseFeature) REQUIRE f.name IS UNIQUE",
    "CREATE CONSTRAINT house_representation_name_unique IF NOT EXISTS FOR (r:HouseRepresentation) REQUIRE r.name IS UNIQUE",
    "CREATE CONSTRAINT house_target_name_unique IF NOT EXISTS FOR (t:HouseTarget) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT house_metric_name_unique IF NOT EXISTS FOR (m:HouseMetric) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT house_province_name_unique IF NOT EXISTS FOR (p:HouseProvince) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT house_district_key_unique IF NOT EXISTS FOR (d:HouseDistrict) REQUIRE d.key IS UNIQUE",
    "CREATE CONSTRAINT house_observation_id_unique IF NOT EXISTS FOR (o:HouseObservation) REQUIRE o.observation_id IS UNIQUE",
    "CREATE CONSTRAINT house_prediction_id_unique IF NOT EXISTS FOR (p:HousePrediction) REQUIRE p.prediction_id IS UNIQUE",
]


def get_neo4j_config():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    missing = [name for name, value in {
        "NEO4J_URI": uri,
        "NEO4J_USER": user,
        "NEO4J_PASSWORD": password,
    }.items() if not value]
    if missing:
        raise RuntimeError("Missing Neo4j environment variables: " + ", ".join(missing))
    return uri, user, password, database


def create_driver():
    uri, user, password, _database = get_neo4j_config()
    return GraphDatabase.driver(uri, auth=(user, password))


def verify_connection():
    _uri, _user, _password, database = get_neo4j_config()
    with create_driver() as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            return session.run("RETURN 1 AS ok").single()["ok"] == 1


def initialize_graph(registry, metadata):
    _uri, _user, _password, database = get_neo4j_config()
    model_rows = []
    for name, info in registry["models"].items():
        model_rows.append({
            "name": name,
            "artifact_file": info["file"],
            "scientific_final_model": name == registry["scientific_final_model"],
        })
    feature_rows = [{"name": feature} for feature in registry["selected_features"]]
    metric_names = ["MAE", "MSE", "RMSE", "R2", "MAPE"]
    cv_metric_rows = []
    for model_name, info in registry["models"].items():
        for metric_name, value in info["cv_metrics"].items():
            cv_metric_rows.append({"model_name": model_name, "metric_name": metric_name, "value": float(value)})
    final_metric_rows = []
    final_metrics = registry["final_test_metrics"]
    for metric_name in metric_names:
        final_metric_rows.append({
            "model_name": final_metrics["model"],
            "metric_name": metric_name,
            "value": float(final_metrics[metric_name]),
        })
    district_rows = []
    for province, districts in metadata["province_districts"].items():
        for district in districts:
            district_rows.append({"province": province, "district": district, "key": f"{province}|{district}"})

    with create_driver() as driver:
        with driver.session(database=database) as session:
            for query in CONSTRAINT_QUERIES:
                session.run(query).consume()
            session.run(
                """
                UNWIND $rows AS row
                MERGE (m:HouseModel {name: row.name})
                SET m.artifact_file = row.artifact_file,
                    m.scientific_final_model = row.scientific_final_model
                """,
                rows=model_rows,
            ).consume()
            session.run("UNWIND $rows AS row MERGE (:HouseFeature {name: row.name})", rows=feature_rows).consume()
            session.run(
                """
                MERGE (r:HouseRepresentation {name: 'Six Selected Features'})
                SET r.feature_count = 6
                MERGE (t:HouseTarget {name: 'House Price'})
                SET t.unit = 'billion VND'
                MERGE (r)-[:HOUSE_TARGETS]->(t)
                """,
            ).consume()
            session.run("UNWIND $names AS name MERGE (:HouseMetric {name: name})", names=metric_names).consume()
            session.run(
                """
                UNWIND $features AS feature
                MATCH (f:HouseFeature {name: feature})
                MATCH (r:HouseRepresentation {name: 'Six Selected Features'})
                MERGE (f)-[:HOUSE_PART_OF_REPRESENTATION]->(r)
                WITH f
                MATCH (m:HouseModel)
                MERGE (m)-[:HOUSE_USES_FEATURE]->(f)
                """,
                features=registry["selected_features"],
            ).consume()
            session.run(
                """
                UNWIND $rows AS row
                MERGE (p:HouseProvince {name: row.province})
                MERGE (d:HouseDistrict {key: row.key})
                SET d.name = row.district
                MERGE (d)-[:HOUSE_IN_PROVINCE]->(p)
                """,
                rows=district_rows,
            ).consume()
            session.run(
                """
                UNWIND $rows AS row
                MATCH (m:HouseModel {name: row.model_name})
                MATCH (metric:HouseMetric {name: row.metric_name})
                MERGE (m)-[rel:HOUSE_HAS_CV_METRIC]->(metric)
                SET rel.value = row.value,
                    rel.evaluation = '5-fold cross-validation'
                """,
                rows=cv_metric_rows,
            ).consume()
            session.run(
                """
                MATCH (m:HouseModel {name: $model_name})
                MATCH (t:HouseTarget {name: 'House Price'})
                MERGE (m)-[:HOUSE_SCIENTIFIC_FINAL_MODEL_FOR]->(t)
                """,
                model_name=registry["scientific_final_model"],
            ).consume()
            session.run(
                """
                UNWIND $rows AS row
                MATCH (m:HouseModel {name: row.model_name})
                MATCH (metric:HouseMetric {name: row.metric_name})
                MERGE (m)-[rel:HOUSE_HAS_FINAL_TEST_METRIC]->(metric)
                SET rel.value = row.value,
                    rel.evaluation = 'held-out test'
                """,
                rows=final_metric_rows,
            ).consume()


def save_prediction(model_name, input_values, predicted_price_billion):
    _uri, _user, _password, database = get_neo4j_config()
    prediction_id = str(uuid4())
    observation_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    province = input_values.get("Province")
    district = input_values.get("District")
    measurements = [
        {"feature": key, "value": None if value is None else float(value)}
        for key, value in input_values.items()
        if key in {"Area", "Floors", "Bedrooms", "Bathrooms"}
    ]
    with create_driver() as driver:
        with driver.session(database=database) as session:
            session.run(
                """
                MERGE (target:HouseTarget {name: 'House Price'})
                MATCH (model:HouseModel {name: $model_name})
                MERGE (province:HouseProvince {name: $province})
                MERGE (district:HouseDistrict {key: $district_key})
                SET district.name = $district
                MERGE (district)-[:HOUSE_IN_PROVINCE]->(province)
                CREATE (obs:HouseObservation {
                    observation_id: $observation_id,
                    created_at: $created_at,
                    province: $province,
                    district: $district
                })
                CREATE (pred:HousePrediction {
                    prediction_id: $prediction_id,
                    predicted_price_billion: $predicted_price_billion,
                    created_at: $created_at
                })
                MERGE (obs)-[:HOUSE_LOCATED_IN]->(district)
                MERGE (obs)-[:HOUSE_HAS_PREDICTION]->(pred)
                MERGE (pred)-[:HOUSE_PRODUCED_BY]->(model)
                MERGE (pred)-[:HOUSE_PREDICTS]->(target)
                WITH obs
                UNWIND $measurements AS measurement
                MATCH (feature:HouseFeature {name: measurement.feature})
                MERGE (obs)-[rel:HOUSE_HAS_MEASUREMENT]->(feature)
                SET rel.value = measurement.value
                """,
                model_name=model_name,
                province=province,
                district=district,
                district_key=f"{province}|{district}",
                observation_id=observation_id,
                prediction_id=prediction_id,
                predicted_price_billion=float(predicted_price_billion),
                created_at=created_at,
                measurements=measurements,
            ).consume()
    return prediction_id


def get_recent_predictions(limit=5):
    _uri, _user, _password, database = get_neo4j_config()
    with create_driver() as driver:
        with driver.session(database=database) as session:
            result = session.run(
                """
                MATCH (pred:HousePrediction)-[:HOUSE_PRODUCED_BY]->(model:HouseModel)
                RETURN pred.prediction_id AS prediction_id,
                       pred.predicted_price_billion AS predicted_price_billion,
                       pred.created_at AS created_at,
                       model.name AS model
                ORDER BY pred.created_at DESC
                LIMIT $limit
                """,
                limit=int(limit),
            )
            return [dict(record) for record in result]


def get_knowledge_graph_summary():
    _uri, _user, _password, database = get_neo4j_config()
    labels = [
        "HouseModel", "HouseFeature", "HouseRepresentation", "HouseTarget",
        "HouseMetric", "HouseProvince", "HouseDistrict", "HouseObservation",
        "HousePrediction",
    ]
    with create_driver() as driver:
        with driver.session(database=database) as session:
            summary = {}
            for label in labels:
                count = session.run(f"MATCH (n:{label}) RETURN count(n) AS count").single()["count"]
                summary[label] = int(count)
            return summary
'''


INIT_GRAPH_CODE = r'''
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neo4j_service import initialize_graph, verify_connection


def main():
    registry = json.loads((PROJECT_ROOT / "models" / "model_registry.json").read_text(encoding="utf-8"))
    metadata = json.loads((PROJECT_ROOT / "models" / "ui_metadata.json").read_text(encoding="utf-8"))
    if verify_connection():
        initialize_graph(registry, metadata)
        print("House Price Neo4j graph initialized successfully.")


if __name__ == "__main__":
    main()
'''


SCHEMA_CYPHER = '''CREATE CONSTRAINT house_model_name_unique IF NOT EXISTS FOR (m:HouseModel) REQUIRE m.name IS UNIQUE;
CREATE CONSTRAINT house_feature_name_unique IF NOT EXISTS FOR (f:HouseFeature) REQUIRE f.name IS UNIQUE;
CREATE CONSTRAINT house_representation_name_unique IF NOT EXISTS FOR (r:HouseRepresentation) REQUIRE r.name IS UNIQUE;
CREATE CONSTRAINT house_target_name_unique IF NOT EXISTS FOR (t:HouseTarget) REQUIRE t.name IS UNIQUE;
CREATE CONSTRAINT house_metric_name_unique IF NOT EXISTS FOR (m:HouseMetric) REQUIRE m.name IS UNIQUE;
CREATE CONSTRAINT house_province_name_unique IF NOT EXISTS FOR (p:HouseProvince) REQUIRE p.name IS UNIQUE;
CREATE CONSTRAINT house_district_key_unique IF NOT EXISTS FOR (d:HouseDistrict) REQUIRE d.key IS UNIQUE;
CREATE CONSTRAINT house_observation_id_unique IF NOT EXISTS FOR (o:HouseObservation) REQUIRE o.observation_id IS UNIQUE;
CREATE CONSTRAINT house_prediction_id_unique IF NOT EXISTS FOR (p:HousePrediction) REQUIRE p.prediction_id IS UNIQUE;
'''


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.lstrip(), encoding="utf-8")


def make_notebook():
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python (houseprice_ml)", "language": "python", "name": "houseprice_ml"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    cells = []
    md = cells.append
    md(nbf.v4.new_markdown_cell("""# Assignment 01 - Vietnam House Price Prediction Intelligent System

## 1. System and Problem Definition

This project builds a supervised regression intelligent system for estimating Vietnam house listing prices.

Input: structured property information. Output: estimated `Price` in billion VND. Formally, the system learns a function `f(x) -> predicted house price`.

This is an educational machine-learning estimate, not an official property appraisal."""))
    md(nbf.v4.new_markdown_cell("""## 2. Dataset Source

Dataset: House Price Prediction Dataset Vietnam - 2024  
Kaggle URL: https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024  
Local copied path: `data/vietnam_housing_dataset.csv`

Target: `Price`, measured in billion VND."""))
    md(nbf.v4.new_code_cell("""from pathlib import Path
import sys
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from house_price_pipeline import run_pipeline

summary = run_pipeline(PROJECT_ROOT, verbose=True)
summary["dataset"]"""))
    md(nbf.v4.new_markdown_cell("""## 3. Initial Inspection

The executed output above reports shape, columns, dtypes, missing counts, missing percentages, duplicates, target range, and numeric descriptive statistics. The raw dataset is validated against the expected 12-column schema before modeling continues."""))
    md(nbf.v4.new_markdown_cell("""## 4. Address to Location Feature Engineering

The raw `Address` column has very high cardinality, so the complete raw address is not used directly as an ML feature. Instead, `Province` is extracted from the last comma-separated segment and `District` from the second-last comma-separated segment after Unicode NFC normalization, whitespace trimming, and trailing-dot cleanup.

Example: `Duong Nguyen Van Khoi, Phuong 11, Go Vap, Ho Chi Minh` becomes `District = Go Vap` and `Province = Ho Chi Minh` after the same deterministic parsing rule."""))
    md(nbf.v4.new_markdown_cell("""## 5. Feature Representations

Full engineered representation: 12 features obtained by replacing raw `Address` with `Province` and `District`.

Final six-feature representation required for deployment: `Area`, `Floors`, `Bedrooms`, `Bathrooms`, `Province`, `District`.

The six features are chosen because they capture size, structure, property capacity, and location while avoiding several raw attributes with substantial missingness. The experiment later tests whether this simpler representation remains competitive; it is not assumed to be automatically more accurate."""))
    md(nbf.v4.new_code_cell("""summary["features"]"""))
    md(nbf.v4.new_markdown_cell("""## 6. Exactly Five EDA Distribution Charts

The five EDA distribution charts are:

1. Price Distribution histogram, unit billion VND.
2. Area Distribution histogram.
3. Floors Distribution.
4. Bedrooms Distribution.
5. Top Provinces by Listing Count.

Evaluation charts later are separate and do not count toward these five EDA charts."""))
    md(nbf.v4.new_code_cell("""summary["eda_charts"]"""))
    md(nbf.v4.new_markdown_cell("""## 7. EDA Summary

The target and feature summaries should be interpreted from the executed dataset output. The data contains missingness in several optional property attributes, while `Address`, `Area`, and `Price` are complete. Location is expected to matter because listing counts are concentrated in major provinces and districts, but this is treated as a predictive association, not a causal claim."""))
    md(nbf.v4.new_markdown_cell("""## 8. Train/Test Split and Leakage Control

One row-index based split is used for both representations: 80% training and 20% held-out test, `random_state=42`, no stratification because this is regression.

Leakage control:

- Imputers fit only inside training folds.
- Scalers fit only inside training folds.
- OneHotEncoder fits only inside training folds.
- Model selection uses training cross-validation only.
- Hyperparameter experiments use training cross-validation only.
- Representation experiments use training cross-validation only.
- Held-out test is used once after final configuration is locked."""))
    md(nbf.v4.new_code_cell("""summary["split"]"""))
    md(nbf.v4.new_markdown_cell("""## 9. Baseline

The baseline is `DummyRegressor(strategy="median")` wrapped in the same six-feature preprocessing protocol. Metrics are MAE, MSE, RMSE, R2, and MAPE."""))
    md(nbf.v4.new_code_cell("""summary["baseline"]"""))
    md(nbf.v4.new_markdown_cell("""## Traditional Machine Learning Model Understanding

### Linear Regression
A. Input Representation: numeric columns are imputed/scaled and categorical location columns are one-hot encoded into a numeric vector.  
B. Learning Idea: learns a linear weighted relationship between input features and price.  
C. What It Learns: coefficients for each encoded feature and one intercept.  
D. Strengths: simple, fast, interpretable, useful as a linear benchmark.  
E. Weaknesses: cannot naturally capture complex nonlinear interactions among area, district, and structure.  
F. Suitability: helpful baseline for house-price data, but Vietnam listings are likely nonlinear and location-sensitive.

### Decision Tree Regressor
A. Input Representation: the same preprocessed numeric vector.  
B. Learning Idea: recursively creates feature-threshold splits to reduce regression error.  
C. What It Learns: a tree of split rules and leaf predictions.  
D. Strengths: nonlinear, understandable as rules, handles interactions.  
E. Weaknesses: high overfitting risk if unrestricted.  
F. Suitability: can capture local patterns in property data, but may be unstable.

### Random Forest Regressor
A. Input Representation: the same preprocessed vector.  
B. Learning Idea: trains many trees using bagging and random feature sampling.  
C. What It Learns: an ensemble of regression trees whose predictions are averaged.  
D. Strengths: reduces variance compared with one tree and models nonlinear structured data.  
E. Weaknesses: heavier and less interpretable than a single tree.  
F. Suitability: strong candidate for noisy housing listings with nonlinear location and size effects.

### Extra Trees Regressor
A. Input Representation: the same preprocessed vector.  
B. Learning Idea: trains a highly randomized ensemble of trees, including randomized split thresholds.  
C. What It Learns: many randomized tree structures and averaged predictions.  
D. Strengths: can reduce variance and is often strong on tabular data.  
E. Weaknesses: less interpretable and may underfit or over-randomize some patterns.  
F. Suitability: useful comparison against Random Forest for high-cardinality location data.

### Gradient Boosting Regressor
A. Input Representation: the same preprocessed vector.  
B. Learning Idea: sequentially adds trees that correct residual errors from previous trees.  
C. What It Learns: an additive ensemble of weak learners focused on residual improvement.  
D. Strengths: often accurate for nonlinear structured data.  
E. Weaknesses: sensitive to hyperparameters and less parallel due to sequential learning.  
F. Suitability: a strong traditional ML model for property-price regression."""))
    md(nbf.v4.new_markdown_cell("""## Experiment 1 - Five Model Comparison

Question: Which traditional regression algorithm provides the strongest performance using the same six-feature representation and preprocessing protocol?

Fixed: `X_train_6/y_train`, six features, preprocessing, CV folds, and metrics. Changed: regression algorithm only. The held-out test is not used."""))
    md(nbf.v4.new_code_cell("""import pandas as pd
pd.DataFrame(summary["experiment1"]["full_table"])"""))
    md(nbf.v4.new_code_cell("""pd.DataFrame(summary["experiment1"]["compact_table"])"""))
    md(nbf.v4.new_code_cell("""summary["experiment1"]["best_model"]"""))
    md(nbf.v4.new_markdown_cell("""Evaluation visualizations saved:

- `figures/06_rmse_model_comparison.png`
- `figures/07_r2_model_comparison.png`"""))
    md(nbf.v4.new_markdown_cell("""## Experiment 2 - Hyperparameter Investigation

Controlled experiment on `RandomForestRegressor(max_depth)` with values `5`, `10`, `20`, and `None`. Fixed: `n_estimators=100`, random state, six features, preprocessing, KFold, and all five metrics."""))
    md(nbf.v4.new_code_cell("""pd.DataFrame(summary["experiment2"]["table"])"""))
    md(nbf.v4.new_code_cell("""summary["experiment2"]["best_rf_max_depth"]"""))
    md(nbf.v4.new_markdown_cell("""The line chart `figures/08_rf_max_depth_vs_rmse.png` shows max_depth versus mean CV RMSE. The candidate model is selected using training CV only by comparing the Experiment 1 winner with the tuned Random Forest."""))
    md(nbf.v4.new_code_cell("""summary["candidate"]"""))
    md(nbf.v4.new_markdown_cell("""## Experiment 3 - Feature Representation Investigation

This experiment compares the full 12 engineered features against the final six selected features using the same training rows, target, candidate algorithm/configuration, CV folds, preprocessing protocol, and five metrics. The changed variable is feature representation only."""))
    md(nbf.v4.new_code_cell("""pd.DataFrame(summary["experiment3"]["table"])"""))
    md(nbf.v4.new_markdown_cell("""Interpretation should be based on the actual metrics above. If six features are worse, that is a performance/usability trade-off; if they are equal or better, the reduction is empirically supported. Deployment still uses exactly six teacher-required features."""))
    md(nbf.v4.new_markdown_cell("""## Scientific Final Configuration and Held-out Test

The final system is locked before viewing the held-out test: six-feature representation, preprocessing inside a pipeline, and the selected algorithm/configuration from training CV. The held-out test is then used once for final scientific evaluation. No tuning occurs after this result."""))
    md(nbf.v4.new_code_cell("""summary["final_test"]"""))
    md(nbf.v4.new_markdown_cell("""Final evaluation visualizations saved:

- `figures/11_actual_vs_predicted_scatter.png`
- `figures/12_final_residual_distribution.png`
- `figures/13_cv_vs_heldout_rmse.png`

MAE and RMSE are in billion VND. R2 describes explained variance quality. MAPE is an average percentage-type error. The result is not a guaranteed market value."""))
    md(nbf.v4.new_markdown_cell("""## Saved Deployment Models

Five six-feature trained pipelines are saved in `models/`:

- `linear_regression.joblib`
- `decision_tree.joblib`
- `random_forest.joblib`
- `extra_trees.joblib`
- `gradient_boosting.joblib`

The scientific final pipeline is also saved as `best_model.joblib`. Streamlit loads these artifacts and never retrains."""))
    md(nbf.v4.new_code_cell("""summary["demos"]"""))
    md(nbf.v4.new_markdown_cell("""## Controlled Experiment Summary

| Experiment | Question | Changed Variable | Fixed Variables | Main Result |
|---|---|---|---|---|
| Experiment 1 | Which algorithm is strongest? | Model algorithm | Six features, preprocessing, CV folds, metrics | Lowest CV RMSE model is selected as the initial winner |
| Experiment 2 | Which RF max_depth is best? | Random Forest max_depth | n_estimators, six features, preprocessing, folds, metrics | Best depth is chosen by lowest CV RMSE |
| Experiment 3 | Do full 12 features improve over six? | Feature representation | Candidate algorithm, train rows, folds, metrics | Reported honestly using all five metrics |
"""))
    md(nbf.v4.new_markdown_cell("""## Reflection

What worked: structured preprocessing, five-model comparison, Province/District engineering, controlled experiments, Streamlit deployment design, and Neo4j Knowledge Graph integration.

Challenges: missing data, noisy high-cardinality addresses, categorical encoding, regression generalization, and deployment packaging.

Limitations: the dataset contains property listings and may not equal final transaction prices; it represents 2024 market conditions; real estate markets change over time; Province/District extraction is simplified; several raw attributes have substantial missing data; the six-feature deployed representation is a simplification; predictions are not official valuations."""))
    md(nbf.v4.new_markdown_cell("""## Conclusion

Raw Data -> Data Representation -> EDA -> Feature Engineering -> Train/Test -> Baseline -> Five ML Models -> Experiment 1 -> Experiment 2 -> Experiment 3 -> Final Six-Feature Model -> Held-out Evaluation -> Streamlit -> Neo4j -> Deployment."""))
    nb["cells"] = cells
    path = ROOT / "notebooks" / "house_price_assignment.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


def main():
    for folder in ["data", "notebooks", "models", "figures", "neo4j"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    if not SOURCE_DATA.exists():
        raise FileNotFoundError(SOURCE_DATA)
    shutil.copy2(SOURCE_DATA, ROOT / "data" / "vietnam_housing_dataset.csv")
    write_text(ROOT / "house_price_pipeline.py", PIPELINE_CODE)
    write_text(ROOT / "app.py", APP_CODE)
    write_text(ROOT / "neo4j_service.py", NEO4J_SERVICE_CODE)
    write_text(ROOT / "neo4j" / "init_graph.py", INIT_GRAPH_CODE)
    write_text(ROOT / "neo4j" / "schema.cypher", SCHEMA_CYPHER)
    write_text(ROOT / "requirements.txt", """numpy
pandas
matplotlib
scikit-learn
joblib
jupyter
ipykernel
streamlit
neo4j
python-dotenv
""")
    write_text(ROOT / ".gitignore", """.env
__pycache__/
*.pyc
.ipynb_checkpoints/
.streamlit/secrets.toml
""")
    write_text(ROOT / ".env.example", """NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j
""")
    write_text(ROOT / "render.yaml", """services:
  - type: web
    name: vietnam-house-price-prediction
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: >
      streamlit run app.py
      --server.address 0.0.0.0
      --server.port $PORT
      --server.headless true
    envVars:
      - key: NEO4J_URI
        sync: false
      - key: NEO4J_USER
        sync: false
      - key: NEO4J_PASSWORD
        sync: false
      - key: NEO4J_DATABASE
        sync: false
""")
    write_text(ROOT / "README.md", """# Assignment 01 - Vietnam House Price Prediction Intelligent System

Dataset: House Price Prediction Dataset Vietnam - 2024  
Source: https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024

This project is a supervised regression system that estimates house listing price in billion VND. It is an educational machine-learning demonstration and not an official property valuation.

## Features

Full engineered features: Area, Frontage, Access Road, House direction, Balcony direction, Floors, Bedrooms, Bathrooms, Legal status, Furniture state, Province, District.

Final deployed six features: Area, Floors, Bedrooms, Bathrooms, Province, District.

## Models

Linear Regression, Decision Tree Regressor, Random Forest Regressor, Extra Trees Regressor, Gradient Boosting Regressor.

Metrics: MAE, MSE, RMSE, R2, MAPE. Primary selection metric: RMSE.

Controlled experiments:

1. Five-model comparison using six features.
2. Random Forest max_depth investigation.
3. Full 12 engineered features vs six selected features.

## Run Locally

```bash
conda activate houseprice_ml
streamlit run app.py
```

To initialize Neo4j locally, create `.env` from `.env.example` and run:

```bash
python neo4j/init_graph.py
```

Neo4j stores only anonymous house observations and predictions. No exact street address, name, email, phone, account, or identity is stored.

## Deployment

`render.yaml` defines a Streamlit web service. Configure Neo4j secrets in Render environment variables, not in source code.

The Streamlit layout is browser based and usable from desktop and mobile browsers.

## Limitations

The dataset contains listings, not necessarily transaction prices. The data represents 2024 market conditions. Province/District extraction is simplified. Several raw attributes contain substantial missing data. The six-feature deployed representation is a simplification.
""")
    make_notebook()
    print(f"House project files created at {ROOT}")


if __name__ == "__main__":
    main()
