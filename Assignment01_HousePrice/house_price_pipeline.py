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
REPRESENTATION_VERSION = "house_11_feature_v2"
RANDOM_STATE = 42
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

RAW_COLUMNS = [
    "Address", "Area", "Frontage", "Access Road", "House direction",
    "Balcony direction", "Floors", "Bedrooms", "Bathrooms",
    "Legal status", "Furniture state", "Price",
]

HOUSE_FEATURES_11 = [
    "Area", "Frontage", "Access Road", "House direction", "Balcony direction",
    "Floors", "Bedrooms", "Bathrooms", "Legal status", "Furniture state",
    "Location",
]

HOUSE_FEATURES_10 = [
    "Area", "Frontage", "Access Road", "House direction", "Balcony direction",
    "Floors", "Bedrooms", "Bathrooms", "Legal status", "Furniture state",
]

NUMERIC_11 = ["Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"]
CATEGORICAL_11 = ["House direction", "Balcony direction", "Legal status", "Furniture state", "Location"]
NUMERIC_10 = NUMERIC_11
CATEGORICAL_10 = ["House direction", "Balcony direction", "Legal status", "Furniture state"]


def normalize_text(value):
    if pd.isna(value):
        return np.nan
    text = unicodedata.normalize("NFC", str(value)).strip()
    text = " ".join(text.split())
    return text.rstrip(".,;").strip()


def extract_location(address):
    if pd.isna(address):
        return pd.Series({"Province": np.nan, "District": np.nan, "Location": np.nan})
    parts = [normalize_text(part) for part in str(address).split(",")]
    parts = [part for part in parts if part]
    if len(parts) < 2:
        return pd.Series({"Province": np.nan, "District": np.nan, "Location": np.nan})
    province = parts[-1]
    district = parts[-2]
    return pd.Series({"Province": province, "District": district, "Location": f"{district}, {province}"})


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse=False)


def make_preprocessor(representation="11"):
    if representation == "11":
        numeric_features = NUMERIC_11
        categorical_features = CATEGORICAL_11
    elif representation == "10":
        numeric_features = NUMERIC_10
        categorical_features = CATEGORICAL_10
    else:
        raise ValueError("representation must be '10' or '11'")
    return ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_ohe())]), categorical_features),
        ]
    )


def make_pipeline(model, representation="11"):
    return Pipeline([("preprocess", make_preprocessor(representation)), ("model", model)])


def base_models():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def build_model_by_name(model_name, rf_max_depth=None):
    if model_name == "Linear Regression":
        return LinearRegression()
    if model_name == "Decision Tree":
        return DecisionTreeRegressor(random_state=RANDOM_STATE)
    if model_name == "Random Forest":
        return RandomForestRegressor(n_estimators=100, max_depth=rf_max_depth, random_state=RANDOM_STATE, n_jobs=-1)
    if model_name == "Extra Trees":
        return ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    if model_name == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=RANDOM_STATE)
    raise ValueError(model_name)


def model_file_name(model_name):
    return {
        "Linear Regression": "linear_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "Random Forest": "random_forest.joblib",
        "Extra Trees": "extra_trees.joblib",
        "Gradient Boosting": "gradient_boosting.joblib",
    }[model_name]


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
    scores = cross_validate(pipeline, X, y, cv=CV, scoring=scoring, n_jobs=None, error_score="raise")
    detailed = {}
    compact = {}
    for metric in scoring:
        values = scores[f"test_{metric}"]
        if metric in {"MAE", "MSE", "RMSE", "MAPE"}:
            values = -values
        detailed[f"{metric} Mean"] = float(np.mean(values))
        detailed[f"{metric} Std"] = float(np.std(values))
        compact[metric] = float(np.mean(values))
    return detailed, compact


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
    if obj is None:
        return None
    return obj


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_eda(working_df, figures_dir):
    chart_paths = []
    charts = [
        ("Price Distribution", "Price", "Price (billion VND)", "01_price_distribution.png", "hist"),
        ("Area Distribution", "Area", "Area", "02_area_distribution.png", "hist"),
    ]
    for title, column, xlabel, filename, _kind in charts:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(working_df[column], bins=40, color="#2F6F73", edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Listing count")
        save_figure(fig, figures_dir / filename)
        chart_paths.append(str((figures_dir / filename).relative_to(figures_dir.parent)))

    for title, column, filename, color in [
        ("Floors Distribution", "Floors", "03_floors_distribution.png", "#4F6D7A"),
        ("Bedrooms Distribution", "Bedrooms", "04_bedrooms_distribution.png", "#C1666B"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        values = working_df[column].dropna().round().astype(int).value_counts().sort_index()
        ax.bar(values.index.astype(str), values.values, color=color)
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel("Listing count")
        save_figure(fig, figures_dir / filename)
        chart_paths.append(str((figures_dir / filename).relative_to(figures_dir.parent)))

    fig, ax = plt.subplots(figsize=(10, 5))
    provinces = working_df["Province"].value_counts().head(10).sort_values()
    ax.barh(provinces.index, provinces.values, color="#5B8E7D")
    ax.set_title("Top Provinces by Listing Count")
    ax.set_xlabel("Listing count")
    save_figure(fig, figures_dir / "05_top_provinces_distribution.png")
    chart_paths.append(str((figures_dir / "05_top_provinces_distribution.png").relative_to(figures_dir.parent)))
    return chart_paths


def get_encoded_dim(fitted_pipeline):
    preprocessor = fitted_pipeline.named_steps["preprocess"]
    return int(len(preprocessor.get_feature_names_out()))


def save_pipeline(pipeline, path):
    joblib.dump(pipeline, path, compress=("xz", 3))


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

    working_df = pd.concat([df.copy(), df["Address"].apply(extract_location)], axis=1)
    for col in ["Province", "District", "Location", "House direction", "Balcony direction", "Legal status", "Furniture state"]:
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
        "location_count": int(working_df["Location"].nunique(dropna=True)),
        "top_provinces": working_df["Province"].value_counts().head(10).astype(int).to_dict(),
    }

    eda_charts = plot_eda(working_df, figures_dir)
    X_11 = working_df[HOUSE_FEATURES_11].copy()
    X_10 = working_df[HOUSE_FEATURES_10].copy()
    y = working_df[TARGET].copy()
    indices = np.arange(len(working_df))
    train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=RANDOM_STATE)
    X_train_11, X_test_11 = X_11.iloc[train_idx].copy(), X_11.iloc[test_idx].copy()
    X_train_10, X_test_10 = X_10.iloc[train_idx].copy(), X_10.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    split_info = {"train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)), "test_size": 0.20, "random_state": RANDOM_STATE}

    baseline_detailed, baseline_compact = cv_metrics(make_pipeline(DummyRegressor(strategy="median"), "11"), X_train_11, y_train)

    experiment1_rows = []
    experiment1_compact = {}
    for name, model in base_models().items():
        detailed, compact = cv_metrics(make_pipeline(model, "11"), X_train_11, y_train)
        experiment1_rows.append({"Model": name, **detailed})
        experiment1_compact[name] = compact
        if verbose:
            print("Experiment 1 done:", name, compact)
    experiment1_table = pd.DataFrame(experiment1_rows)
    experiment1_compact_table = pd.DataFrame([{"Model": name, **metrics} for name, metrics in experiment1_compact.items()])
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

    experiment2_rows = []
    for depth in [5, 10, 20, None]:
        rf = RandomForestRegressor(n_estimators=100, max_depth=depth, random_state=RANDOM_STATE, n_jobs=-1)
        detailed, compact = cv_metrics(make_pipeline(rf, "11"), X_train_11, y_train)
        label = "None" if depth is None else str(depth)
        experiment2_rows.append({"max_depth": label, "raw_max_depth": depth, **detailed, **compact})
        if verbose:
            print("Experiment 2 done: max_depth", label, compact)
    experiment2_compact = pd.DataFrame([{k: v for k, v in row.items() if k in {"max_depth", "MAE", "MSE", "RMSE", "R2", "MAPE"}} for row in experiment2_rows])
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

    candidate_model_10 = build_model_by_name(candidate_name, candidate_rf_max_depth)
    ten_detailed, ten_metrics = cv_metrics(make_pipeline(candidate_model_10, "10"), X_train_10, y_train)
    candidate_model_11 = build_model_by_name(candidate_name, candidate_rf_max_depth)
    eleven_detailed, eleven_metrics = cv_metrics(make_pipeline(candidate_model_11, "11"), X_train_11, y_train)
    difference = {metric: eleven_metrics[metric] - ten_metrics[metric] for metric in ["MAE", "MSE", "RMSE", "R2", "MAPE"]}
    experiment3_table = pd.DataFrame([
        {"Representation": "10 Without Location", "Feature Count": 10, **ten_metrics},
        {"Representation": "11 With Location", "Feature Count": 11, **eleven_metrics},
        {"Representation": "Difference (11 - 10)", "Feature Count": 1, **difference},
    ])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(experiment3_table.iloc[:2]["Representation"], experiment3_table.iloc[:2]["RMSE"], color=["#4F6D7A", "#C1666B"])
    ax.set_title("RMSE: 10 Without Location vs 11 With Location")
    ax.set_ylabel("Mean CV RMSE (billion VND)")
    ax.tick_params(axis="x", rotation=12)
    save_figure(fig, figures_dir / "09_location_rmse_comparison.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(experiment3_table.iloc[:2]["Representation"], experiment3_table.iloc[:2]["R2"], color=["#4F6D7A", "#5B8E7D"])
    ax.set_title("R2: 10 Without Location vs 11 With Location")
    ax.set_ylabel("Mean CV R2")
    ax.tick_params(axis="x", rotation=12)
    save_figure(fig, figures_dir / "10_location_r2_comparison.png")

    final_pipeline = make_pipeline(build_model_by_name(candidate_name, candidate_rf_max_depth), "11")
    final_pipeline.fit(X_train_11, y_train)
    y_pred = final_pipeline.predict(X_test_11)
    final_test_metrics = metric_dict(y_test, y_pred)
    encoded_dim = get_encoded_dim(final_pipeline)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.35, s=16, color="#2F6F73")
    low, high = min(float(y_test.min()), float(np.min(y_pred))), max(float(y_test.max()), float(np.max(y_pred)))
    ax.plot([low, high], [low, high], color="#C1666B", linestyle="--", linewidth=2)
    ax.set_title("Actual Price vs Predicted Price")
    ax.set_xlabel("Actual Price (billion VND)")
    ax.set_ylabel("Predicted Price (billion VND)")
    save_figure(fig, figures_dir / "11_actual_vs_predicted_scatter.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_test - y_pred, bins=40, color="#8C6A3F", edgecolor="white")
    ax.set_title("Final Residual Distribution")
    ax.set_xlabel("Residual = Actual - Predicted (billion VND)")
    ax.set_ylabel("Listing count")
    save_figure(fig, figures_dir / "12_final_residual_distribution.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Training CV RMSE", "Held-out Test RMSE"], [eleven_metrics["RMSE"], final_test_metrics["RMSE"]], color=["#4F6D7A", "#C1666B"])
    ax.set_title("CV vs Held-out RMSE")
    ax.set_ylabel("RMSE (billion VND)")
    save_figure(fig, figures_dir / "13_cv_vs_heldout_rmse.png")

    deployment_metrics = dict(experiment1_compact)
    rf_deploy_depth = None
    if candidate_name == "Random Forest" and candidate_rf_max_depth is not None:
        deployment_metrics["Random Forest"] = {metric: float(best_rf_row[metric]) for metric in ["MAE", "MSE", "RMSE", "R2", "MAPE"]}
        rf_deploy_depth = candidate_rf_max_depth

    model_registry = {
        "target": TARGET,
        "target_unit": TARGET_UNIT,
        "feature_count": 11,
        "selected_features": HOUSE_FEATURES_11,
        "representation_version": REPRESENTATION_VERSION,
        "legacy_note": "Previous six-feature House Price representation is superseded.",
        "scientific_final_model": candidate_name,
        "model_selection_metric": "RMSE",
        "best_model_file": "best_model.joblib",
        "models": {},
        "final_test_metrics": {"model": candidate_name, **final_test_metrics},
    }
    for name in base_models():
        rf_depth_for_file = rf_deploy_depth if name == "Random Forest" else None
        pipeline = make_pipeline(build_model_by_name(name, rf_depth_for_file), "11")
        pipeline.fit(X_train_11, y_train)
        file_name = model_file_name(name)
        save_pipeline(pipeline, models_dir / file_name)
        reloaded = joblib.load(models_dir / file_name)
        _ = reloaded.predict(X_train_11.head(1))
        config = pipeline.get_params()["model"].get_params()
        model_registry["models"][name] = {
            "file": file_name,
            "scientific_final_model": name == candidate_name,
            "cv_metrics": deployment_metrics[name],
            "configuration": clean_for_json(config),
        }
    save_pipeline(final_pipeline, models_dir / "best_model.joblib")
    (models_dir / "model_registry.json").write_text(json.dumps(clean_for_json(model_registry), indent=4, ensure_ascii=False), encoding="utf-8")

    numeric_ranges = {
        feature: {
            "min": float(working_df[feature].min(skipna=True)),
            "max": float(working_df[feature].max(skipna=True)),
            "median": float(working_df[feature].median(skipna=True)),
        }
        for feature in NUMERIC_11
    }
    categorical_choices = {
        feature: sorted([str(value) for value in working_df[feature].dropna().unique()])
        for feature in CATEGORICAL_11
    }
    location_parts = (
        working_df[["Location", "District", "Province"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["Province", "District"])
    )
    location_mapping = {
        row.Location: {"district": row.District, "province": row.Province}
        for row in location_parts.itertuples(index=False)
    }
    ui_metadata = {
        "selected_features": HOUSE_FEATURES_11,
        "feature_count": 11,
        "representation_version": REPRESENTATION_VERSION,
        "target_unit": TARGET_UNIT,
        "location_values": sorted(location_mapping.keys()),
        "location_mapping": location_mapping,
        "numeric_ranges": numeric_ranges,
        "categorical_choices": categorical_choices,
        "missing_label": "Not provided",
    }
    (models_dir / "ui_metadata.json").write_text(json.dumps(clean_for_json(ui_metadata), indent=4, ensure_ascii=False), encoding="utf-8")

    demo_rows = [
        {"Area": 35.0, "Frontage": 4.0, "Access Road": 3.0, "House direction": "Đông", "Balcony direction": "Not provided", "Floors": 2.0, "Bedrooms": 2.0, "Bathrooms": 1.0, "Legal status": "Sổ đỏ/ Sổ hồng", "Furniture state": "Cơ bản", "Location": _choose_location(location_mapping, "Cầu Giấy, Hà Nội")},
        {"Area": 70.0, "Frontage": 5.0, "Access Road": 6.0, "House direction": "Nam", "Balcony direction": "Đông Nam", "Floors": 4.0, "Bedrooms": 4.0, "Bathrooms": 3.0, "Legal status": "Sổ đỏ/ Sổ hồng", "Furniture state": "Đầy đủ", "Location": _choose_location(location_mapping, "Gò Vấp, Hồ Chí Minh")},
        {"Area": 150.0, "Frontage": 8.0, "Access Road": 10.0, "House direction": "Tây", "Balcony direction": "Not provided", "Floors": 5.0, "Bedrooms": 6.0, "Bathrooms": 5.0, "Legal status": "Sổ đỏ/ Sổ hồng", "Furniture state": "Đầy đủ", "Location": _choose_location(location_mapping, "Cẩm Lệ, Đà Nẵng")},
    ]
    demos = []
    for row in demo_rows:
        model_row = {key: (np.nan if value == "Not provided" else value) for key, value in row.items()}
        pred = float(final_pipeline.predict(pd.DataFrame([model_row], columns=HOUSE_FEATURES_11))[0])
        demos.append({"input": row, "predicted_price_billion": pred})
    (models_dir / "demo_predictions.json").write_text(json.dumps(clean_for_json(demos), indent=4, ensure_ascii=False), encoding="utf-8")

    artifact_sizes = {path.name: int(path.stat().st_size) for path in models_dir.glob("*.joblib")}
    summary = {
        "dataset": inspection,
        "eda_charts": eda_charts,
        "features": {"representation_10": HOUSE_FEATURES_10, "representation_11": HOUSE_FEATURES_11},
        "split": split_info,
        "memory_check": {
            "province_count": inspection["province_count"],
            "district_count": inspection["district_count"],
            "location_count": inspection["location_count"],
            "encoded_dimensionality": encoded_dim,
        },
        "baseline": {"full": baseline_detailed, "compact": baseline_compact},
        "experiment1": {
            "full_table": experiment1_table.round(6).to_dict(orient="records"),
            "compact_table": experiment1_compact_table.round(6).to_dict(orient="records"),
            "best_model": exp1_best_model,
        },
        "experiment2": {
            "table": experiment2_compact.round(6).to_dict(orient="records"),
            "best_rf_max_depth": "None" if best_rf_depth is None else best_rf_depth,
        },
        "candidate": {"algorithm": candidate_name, "rf_max_depth": "None" if candidate_rf_max_depth is None else candidate_rf_max_depth},
        "experiment3": {"table": experiment3_table.round(6).to_dict(orient="records"), "difference": difference},
        "final_test": final_test_metrics,
        "demos": demos,
        "artifact_sizes_bytes": artifact_sizes,
    }
    (models_dir / "analysis_summary.json").write_text(json.dumps(clean_for_json(summary), indent=4, ensure_ascii=False), encoding="utf-8")
    return summary


def _choose_location(mapping, preferred):
    if preferred in mapping:
        return preferred
    preferred_tail = preferred.split(",")[-1].strip()
    for location in mapping:
        if location.endswith(preferred_tail):
            return location
    return sorted(mapping.keys())[0]
