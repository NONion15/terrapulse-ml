"""
Global House Price Prediction — Training Pipeline

Trains Ridge, RandomForest, LightGBM, XGBoost, and CatBoost regressors
on the 200K-row global housing dataset. Produces two model variants:
  - Full model (all features including financial)
  - Property-only model (no financial features — no target leakage)

Evaluates with Optuna hyperparameter tuning and 5-fold CV, compares
all models, and serializes the best.

Usage:
    python model/train_global.py
    python model/train_global.py --no-financial   # Property-only model
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

warnings.filterwarnings("ignore")

# --- Optional imports ---
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

RANDOM_STATE = 42
N_SPLITS = 5
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
OPTUNA_TRIALS = 12

# Columns that are likely derived from the target (price) and leak information
FINANCIAL_COLS = [
    "loan_amount", "down_payment", "emi_to_income_ratio",
    "loan_tenure_years", "monthly_expenses", "customer_salary",
]

# Classification target / IDs — not relevant for price regression
DROP_COLS = ["property_id", "decision"]


# ============================================================
# Feature Engineering
# ============================================================

class GlobalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering for the global housing dataset."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        def num(column):
            if column in X.columns:
                return pd.to_numeric(X[column], errors="coerce").fillna(0)
            return 0

        # Property age
        current_year = 2024
        X["PropertyAge"] = current_year - num("constructed_year")
        X["PropertyAge"] = X["PropertyAge"].clip(lower=0)

        # Room and bathroom density (per sq ft)
        sqft = num("property_size_sqft").clip(lower=1)
        X["RoomsPerSqFt"] = num("rooms") / sqft * 1000
        X["BathsPerSqFt"] = num("bathrooms") / sqft * 1000

        # Room to bathroom ratio
        rooms = num("rooms").clip(lower=1)
        X["BathToRoomRatio"] = num("bathrooms") / rooms

        # Amenity score
        X["AmenityScore"] = num("garage") + num("garden")

        # Risk score
        X["RiskScore"] = num("crime_cases_reported") + num("legal_cases_on_property")

        # Location quality
        X["LocationScore"] = num("neighbourhood_rating") + num("connectivity_score")

        # Financial ratios (only if columns exist — dropped in no-financial mode)
        if "loan_amount" in X.columns and "customer_salary" in X.columns:
            salary = num("customer_salary").clip(lower=1)
            X["LoanToIncome"] = num("loan_amount") / salary

        return X


# ============================================================
# Pipeline construction
# ============================================================

def get_column_groups(X):
    """Classify columns into numeric, low-cardinality, and high-cardinality."""
    X_transformed = GlobalFeatureEngineer().fit_transform(X)
    categorical = X_transformed.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = [c for c in X_transformed.columns if c not in categorical]
    low_card = [c for c in categorical if X_transformed[c].nunique(dropna=True) <= 10]
    high_card = [c for c in categorical if X_transformed[c].nunique(dropna=True) > 10]
    return numeric, low_card, high_card


def build_pipeline(model, X_reference):
    """Build a full preprocessing + model pipeline."""
    numeric_cols, low_card_cols, high_card_cols = get_column_groups(X_reference)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    low_card_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    high_card_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("target_encoder", TargetEncoder(
            target_type="continuous", smooth="auto",
            cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        )),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("low_cardinality", low_card_pipe, low_card_cols),
        ("high_cardinality", high_card_pipe, high_card_cols),
    ])

    return Pipeline([
        ("feature_engineering", GlobalFeatureEngineer()),
        ("preprocessing", preprocessor),
        ("model", model),
    ])


# ============================================================
# Metrics
# ============================================================

def log_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_on_holdout(pipeline, X_holdout, y_holdout, y_holdout_raw):
    y_pred_log = pipeline.predict(X_holdout)
    y_pred_raw = np.expm1(y_pred_log)

    return {
        "log_rmse": float(log_rmse(y_holdout, y_pred_log)),
        "r2": float(r2_score(y_holdout, y_pred_log)),
        "mae": float(mean_absolute_error(y_holdout_raw, y_pred_raw)),
        "rmse": float(np.sqrt(mean_squared_error(y_holdout_raw, y_pred_raw))),
        "median_ae": float(np.median(np.abs(y_holdout_raw - y_pred_raw))),
        "mape": float(np.mean(np.abs((y_holdout_raw - y_pred_raw) / y_holdout_raw.clip(lower=1))) * 100),
    }


# ============================================================
# Optuna tuning
# ============================================================

def _optuna_objective(trial, model_name, X, y, cv):
    if model_name == "Ridge":
        params = {"alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True)}
        model = Ridge(**params)

    elif model_name == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 8, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 8),
        }
        model = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)

    elif model_name == "LightGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        model = LGBMRegressor(**params, random_state=RANDOM_STATE, n_jobs=4, verbose=-1)

    elif model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        model = XGBRegressor(
            **params, random_state=RANDOM_STATE, n_jobs=4, tree_method="hist", verbosity=0,
        )

    elif model_name == "CatBoost":
        params = {
            "iterations": trial.suggest_int("iterations", 300, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        }
        model = CatBoostRegressor(**params, random_state=RANDOM_STATE, thread_count=4, verbose=0)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X), size=min(15_000, len(X)), replace=False,
    )
    X_sample = X.iloc[sample_idx]
    y_sample = y.iloc[sample_idx]

    pipeline = build_pipeline(model, X_sample)
    scores = cross_val_score(
        pipeline, X_sample, y_sample, cv=cv,
        scoring="neg_root_mean_squared_error", n_jobs=1,
    )
    return -scores.mean()


def tune_model(model_name, X, y, cv, n_trials=OPTUNA_TRIALS):
    if not HAS_OPTUNA:
        print("    (Optuna not installed — using defaults)")
        return {}, None

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: _optuna_objective(trial, model_name, X, y, cv),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    print(f"    Best trial: log-RMSE = {study.best_value:.5f}")
    return study.best_params, study.best_value


def build_tuned_model(model_name, best_params):
    if model_name == "Ridge":
        return Ridge(**best_params)
    elif model_name == "RandomForest":
        return RandomForestRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
    elif model_name == "LightGBM":
        return LGBMRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=4, verbose=-1)
    elif model_name == "XGBoost":
        return XGBRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=4, tree_method="hist", verbosity=0)
    elif model_name == "CatBoost":
        return CatBoostRegressor(**best_params, random_state=RANDOM_STATE, thread_count=4, verbose=0)


# ============================================================
# Cross-validation
# ============================================================

def cross_validate_model(model, X, y, cv, subsample=30_000):
    """Run k-fold CV on a subsample for speed, then report scores."""
    if len(X) > subsample:
        idx = np.random.RandomState(RANDOM_STATE).choice(
            len(X), size=subsample, replace=False,
        )
        X_cv = X.iloc[idx].reset_index(drop=True)
        y_cv = y.iloc[idx].reset_index(drop=True)
        print(f"    (Using {subsample:,} row subsample for CV)")
    else:
        X_cv = X
        y_cv = y

    fold_scores = []
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_cv), start=1):
        X_train, X_valid = X_cv.iloc[train_idx], X_cv.iloc[valid_idx]
        y_train, y_valid = y_cv.iloc[train_idx], y_cv.iloc[valid_idx]

        pipeline = build_pipeline(clone(model), X_train)
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_valid)
        score = log_rmse(y_valid, preds)
        fold_scores.append(score)
        print(f"    Fold {fold}: log-RMSE = {score:.5f}")

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"    Mean: {mean_score:.5f} ± {std_score:.5f}")
    return fold_scores, mean_score


# ============================================================
# Feature importance extraction
# ============================================================

def extract_feature_importances(pipeline, X):
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessing"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return {}

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)

    if importances is None or feature_names is None:
        return {}

    paired = sorted(
        zip(feature_names, importances), key=lambda x: abs(x[1]), reverse=True,
    )
    return {name: round(float(imp), 6) for name, imp in paired[:30]}


# ============================================================
# Artifact extraction
# ============================================================

def extract_feature_config(X, model_info):
    config = {"features": {}, "categorical_values": {}, "numeric_ranges": {}}
    for col in X.columns:
        is_cat = X[col].dtype == "object" or str(X[col].dtype) == "string"
        if is_cat:
            config["features"][col] = "categorical"
            config["categorical_values"][col] = sorted(
                [str(v) for v in X[col].dropna().unique()]
            )
        else:
            vals = pd.to_numeric(X[col], errors="coerce").dropna()
            if len(vals) == 0:
                config["features"][col] = "categorical"
                config["categorical_values"][col] = sorted(
                    [str(v) for v in X[col].dropna().unique()]
                )
            else:
                config["features"][col] = "numeric"
                config["numeric_ranges"][col] = {
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "median": float(vals.median()),
                    "mean": float(round(vals.mean(), 2)),
                }
    config["model_info"] = model_info
    return config


def extract_location_stats(df):
    """Per-country and per-city price statistics."""
    stats = {"countries": {}, "cities": {}}

    for country, group in df.groupby("country"):
        prices = group["price"]
        stats["countries"][country] = {
            "count": int(len(group)),
            "median_price": int(prices.median()),
            "mean_price": int(prices.mean()),
            "min_price": int(prices.min()),
            "max_price": int(prices.max()),
        }

    for (country, city), group in df.groupby(["country", "city"]):
        prices = group["price"]
        key = f"{country}|{city}"
        stats["cities"][key] = {
            "country": country,
            "city": city,
            "count": int(len(group)),
            "median_price": int(prices.median()),
            "mean_price": int(prices.mean()),
            "min_price": int(prices.min()),
            "max_price": int(prices.max()),
        }

    return stats


def extract_location_defaults(df, feature_cols):
    """Per-country default feature values for smart defaults."""
    defaults = {}
    for country, group in df.groupby("country"):
        row = {}
        for col in feature_cols:
            if group[col].dtype == "object":
                mode = group[col].mode()
                row[col] = str(mode.iloc[0]) if len(mode) > 0 else ""
            else:
                vals = pd.to_numeric(group[col], errors="coerce").dropna()
                row[col] = round(float(vals.median()), 1) if len(vals) > 0 else 0
        defaults[country] = row
    return defaults


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train global housing price model")
    parser.add_argument(
        "--no-financial", action="store_true",
        help="Drop financial features to avoid target leakage",
    )
    args = parser.parse_args()

    include_financial = not args.no_financial
    mode_label = "Full (with financial features)" if include_financial else "Property-only (no financial features)"

    print("=" * 60)
    print("GLOBAL HOUSE PRICE PREDICTION — TRAINING PIPELINE")
    print(f"Mode: {mode_label}")
    print("=" * 60)

    # --- Load data ---
    data_path = PROJECT_ROOT / "global_house_purchase_dataset.csv"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"\nDataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # --- Drop irrelevant columns ---
    drop = [c for c in DROP_COLS if c in df.columns]
    if not include_financial:
        drop.extend([c for c in FINANCIAL_COLS if c in df.columns])
        print(f"Dropped financial columns: {FINANCIAL_COLS}")

    X = df.drop(columns=["price"] + drop, errors="ignore")
    y_raw = df["price"]
    y = np.log1p(y_raw)

    print(f"Features: {X.shape[1]} columns")
    print(f"Target range: ${y_raw.min():,.0f} – ${y_raw.max():,.0f}")
    print(f"Target median: ${y_raw.median():,.0f}")

    # --- Leakage analysis ---
    if include_financial:
        print("\n⚠ TARGET LEAKAGE ANALYSIS:")
        for col in FINANCIAL_COLS:
            if col in df.columns:
                corr = df[col].corr(df["price"])
                flag = " ← HIGH LEAKAGE" if abs(corr) > 0.7 else ""
                print(f"    {col}: corr = {corr:.3f}{flag}")
        print("  Run with --no-financial for an honest model.\n")

    # --- Train/holdout split ---
    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE,
        stratify=df["country"],
    )
    y_holdout_raw = y_raw.loc[y_holdout.index]

    print(f"Development set: {X_dev.shape[0]:,} rows")
    print(f"Holdout set: {X_holdout.shape[0]:,} rows")

    # --- Model candidates ---
    model_names = ["Ridge", "RandomForest"]
    if HAS_LGBM:
        model_names.append("LightGBM")
    if HAS_XGB:
        model_names.append("XGBoost")
    if HAS_CATBOOST:
        model_names.append("CatBoost")

    print(f"\nModels: {', '.join(model_names)}")
    print(f"Optuna: {'Yes (' + str(OPTUNA_TRIALS) + ' trials on 15K subsample)' if HAS_OPTUNA else 'No'}")

    # --- Hyperparameter tuning ---
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    tuned_models = {}
    tuning_results = {}

    print(f"\n{'=' * 60}")
    print("HYPERPARAMETER TUNING")
    print("=" * 60)

    for name in model_names:
        print(f"\n{name}:")
        t0 = time.time()

        if HAS_OPTUNA and name != "Ridge":
            best_params, best_score = tune_model(name, X_dev, y_dev, cv)
            model = build_tuned_model(name, best_params)
            tuning_results[name] = {
                "best_params": {k: round(v, 6) if isinstance(v, float) else v
                                for k, v in best_params.items()},
                "optuna_best_score": round(best_score, 5) if best_score else None,
            }
        else:
            if name == "Ridge":
                model = Ridge(alpha=10.0)
            elif name == "RandomForest":
                model = RandomForestRegressor(
                    n_estimators=200, max_depth=15, min_samples_leaf=3,
                    random_state=RANDOM_STATE, n_jobs=-1,
                )
            elif name == "LightGBM":
                model = LGBMRegressor(
                    n_estimators=800, learning_rate=0.05, max_depth=6,
                    num_leaves=31, random_state=RANDOM_STATE, n_jobs=4, verbose=-1,
                )
            elif name == "XGBoost":
                model = XGBRegressor(
                    n_estimators=800, learning_rate=0.05, max_depth=6,
                    random_state=RANDOM_STATE, n_jobs=4, tree_method="hist", verbosity=0,
                )
            elif name == "CatBoost":
                model = CatBoostRegressor(
                    iterations=800, learning_rate=0.05, depth=6,
                    random_state=RANDOM_STATE, thread_count=4, verbose=0,
                )
            tuning_results[name] = {"best_params": {}}

        tuned_models[name] = model
        tuning_time = time.time() - t0
        tuning_results[name]["tuning_time_seconds"] = round(tuning_time, 1)
        print(f"    Tuning time: {tuning_time:.1f}s")

    # --- Cross-validation ---
    cv_results = {}

    print(f"\n{'=' * 60}")
    print("CROSS-VALIDATION (5-Fold on 30K subsample)")
    print("=" * 60)

    for name in model_names:
        print(f"\n{name}:")
        t0 = time.time()
        fold_scores, mean_score = cross_validate_model(
            tuned_models[name], X_dev, y_dev, cv,
        )
        cv_time = time.time() - t0
        cv_results[name] = {
            "mean": mean_score,
            "std": float(np.std(fold_scores)),
            "folds": [round(s, 5) for s in fold_scores],
            "cv_time_seconds": round(cv_time, 1),
        }

    # --- Select best ---
    best_name = min(cv_results, key=lambda k: cv_results[k]["mean"])
    best_model = tuned_models[best_name]

    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_name} (CV log-RMSE: {cv_results[best_name]['mean']:.5f})")
    print("=" * 60)

    # --- Holdout evaluation for ALL models ---
    print(f"\n{'=' * 60}")
    print("HOLDOUT EVALUATION")
    print("=" * 60)

    holdout_results = {}
    feature_importances = {}

    for name in model_names:
        print(f"\n{name}: training on full dev set ({X_dev.shape[0]:,} rows)...")
        t0 = time.time()
        pipeline = build_pipeline(clone(tuned_models[name]), X_dev)
        pipeline.fit(X_dev, y_dev)
        train_time = time.time() - t0

        results = evaluate_on_holdout(pipeline, X_holdout, y_holdout, y_holdout_raw)
        results["train_time_seconds"] = round(train_time, 1)
        holdout_results[name] = results

        importances = extract_feature_importances(pipeline, X_dev)
        feature_importances[name] = importances

        print(f"    Train time: {train_time:.1f}s")
        print(f"    log-RMSE: {results['log_rmse']:.5f}")
        print(f"    R²:       {results['r2']:.4f}")
        print(f"    MAE:      ${results['mae']:,.0f}")
        print(f"    RMSE:     ${results['rmse']:,.0f}")
        print(f"    MAPE:     {results['mape']:.1f}%")

    # --- Production model ---
    print(f"\n{'=' * 60}")
    print(f"PRODUCTION: Retraining {best_name} on full dataset ({len(X):,} rows)")
    print("=" * 60)

    production_pipeline = build_pipeline(clone(best_model), X)
    production_pipeline.fit(X, y)

    # --- Save artifacts ---
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if include_financial else "_nofin"

    pipeline_path = ARTIFACTS_DIR / f"global_pipeline{suffix}.joblib"
    joblib.dump(production_pipeline, pipeline_path)
    print(f"\nSaved pipeline: {pipeline_path}")

    # Feature config
    model_info = {
        "name": best_name,
        "mode": "full" if include_financial else "property_only",
        "cv_log_rmse": round(cv_results[best_name]["mean"], 5),
        "holdout_log_rmse": round(holdout_results[best_name]["log_rmse"], 5),
        "holdout_r2": round(holdout_results[best_name]["r2"], 4),
        "holdout_mae": round(holdout_results[best_name]["mae"], 0),
        "holdout_rmse": round(holdout_results[best_name]["rmse"], 0),
        "holdout_mape": round(holdout_results[best_name]["mape"], 2),
        "target_transform": "log1p",
        "training_samples": len(X),
        "n_features": X.shape[1],
        "feature_columns": X.columns.tolist(),
        "financial_features_included": include_financial,
    }

    config_path = ARTIFACTS_DIR / f"global_feature_config{suffix}.json"
    feature_config = extract_feature_config(X, model_info)
    with open(config_path, "w") as f:
        json.dump(feature_config, f, indent=2)
    print(f"Saved feature config: {config_path}")

    # Location stats
    location_stats = extract_location_stats(df)
    stats_path = ARTIFACTS_DIR / "global_location_stats.json"
    with open(stats_path, "w") as f:
        json.dump(location_stats, f, indent=2)
    print(f"Saved location stats: {stats_path}")

    # Location defaults
    feature_cols = X.columns.tolist()
    location_defaults = extract_location_defaults(df.drop(columns=["price"] + drop, errors="ignore"), feature_cols)
    defaults_path = ARTIFACTS_DIR / f"global_location_defaults{suffix}.json"
    with open(defaults_path, "w") as f:
        json.dump(location_defaults, f, indent=2, default=str)
    print(f"Saved location defaults: {defaults_path}")

    # Model comparison
    comparison = {
        "dataset": "global",
        "mode": "full" if include_financial else "property_only",
        "training_samples": len(X),
        "holdout_samples": len(X_holdout),
        "n_features": X.shape[1],
        "target_transform": "log1p",
        "cv_folds": N_SPLITS,
        "cv_subsample": 30_000,
        "optuna_trials": OPTUNA_TRIALS if HAS_OPTUNA else 0,
        "optuna_subsample": 15_000,
        "best_model": best_name,
        "ranking": sorted(model_names, key=lambda k: cv_results[k]["mean"]),
        "leakage_warning": (
            "Financial features (loan_amount, down_payment) correlate >0.85 with price. "
            "Run with --no-financial for an honest baseline."
        ) if include_financial else None,
        "models": {},
    }
    for name in model_names:
        comparison["models"][name] = {
            "cv_log_rmse": round(cv_results[name]["mean"], 5),
            "cv_std": round(cv_results[name]["std"], 5),
            "holdout_log_rmse": round(holdout_results[name]["log_rmse"], 5),
            "holdout_r2": round(holdout_results[name]["r2"], 4),
            "holdout_mae": round(holdout_results[name]["mae"], 0),
            "holdout_rmse": round(holdout_results[name]["rmse"], 0),
            "holdout_mape": round(holdout_results[name]["mape"], 2),
            "train_time_seconds": holdout_results[name].get("train_time_seconds", 0),
            "best_params": tuning_results[name].get("best_params", {}),
            "top_features": feature_importances.get(name, {}),
        }

    comp_path = ARTIFACTS_DIR / f"model_comparison_global{suffix}.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved comparison: {comp_path}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"ALL MODELS — FINAL COMPARISON ({'Full' if include_financial else 'Property-only'})")
    print("=" * 60)
    print(f"\n  {'Model':<15s} {'CV log-RMSE':>12s} {'R²':>8s} {'MAE':>12s} {'MAPE':>8s} {'Time':>8s}")
    print(f"  {'-'*15} {'-'*12} {'-'*8} {'-'*12} {'-'*8} {'-'*8}")

    for name in comparison["ranking"]:
        m = comparison["models"][name]
        marker = " ★" if name == best_name else ""
        print(
            f"  {name:<15s} {m['cv_log_rmse']:>12.5f} {m['holdout_r2']:>8.4f}"
            f" ${m['holdout_mae']:>10,.0f} {m['holdout_mape']:>7.1f}%"
            f" {m['train_time_seconds']:>6.1f}s{marker}"
        )

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best:     {best_name}")
    print(f"  Pipeline: {pipeline_path}")
    print(f"  Config:   {config_path}")
    print(f"  Stats:    {stats_path}")
    print(f"  Compare:  {comp_path}")
    print()


if __name__ == "__main__":
    main()
