"""
Ames Housing Price Prediction — Training Pipeline

Trains Ridge, RandomForest, LightGBM, XGBoost, and CatBoost regressors
with Optuna hyperparameter tuning and 5-fold CV. Evaluates on a held-out
set, compares all models, and serializes the best.

Usage:
    python model/train.py
"""

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

# --- Optional imports (degrade gracefully) ---
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
OPTUNA_TRIALS = 15  # 15 trials per model provides great exploration with fast execution


# ============================================================
# Feature Engineering
# ============================================================

class AmesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain-specific feature engineering for Ames Housing data.

    Creates 12+ derived features from the raw 79-column dataset to
    capture interactions, ratios, and binary indicators that improve
    predictive power across all model types.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        def num(column):
            if column in X.columns:
                return pd.to_numeric(X[column], errors="coerce").fillna(0)
            return 0

        # --- Original features ---
        X["MSSubClass"] = X["MSSubClass"].astype(str)
        X["TotalSF"] = num("TotalBsmtSF") + num("1stFlrSF") + num("2ndFlrSF")
        X["TotalBath"] = (
            num("FullBath")
            + 0.5 * num("HalfBath")
            + num("BsmtFullBath")
            + 0.5 * num("BsmtHalfBath")
        )
        X["HouseAge"] = num("YrSold") - num("YearBuilt")
        X["YearsSinceRemodel"] = num("YrSold") - num("YearRemodAdd")
        X["TotalPorchSF"] = (
            num("OpenPorchSF")
            + num("EnclosedPorch")
            + num("3SsnPorch")
            + num("ScreenPorch")
        )

        # --- New: interaction features ---
        X["QualCondScore"] = num("OverallQual") * num("OverallCond")
        X["GarageScore"] = num("GarageCars") * num("GarageArea")

        # --- New: binary indicators ---
        X["HasPool"] = (num("PoolArea") > 0).astype(int)
        X["HasFireplace"] = (num("Fireplaces") > 0).astype(int)
        X["Has2ndFloor"] = (num("2ndFlrSF") > 0).astype(int)
        X["HasGarage"] = (num("GarageCars") > 0).astype(int)
        X["IsRemodeled"] = (num("YearRemodAdd") != num("YearBuilt")).astype(int)

        # --- New: ratio features ---
        total_rooms = num("TotRmsAbvGrd")
        total_rooms = total_rooms.clip(lower=1)
        X["LivAreaPerRoom"] = num("GrLivArea") / total_rooms

        total_bsmt = num("TotalBsmtSF")
        total_bsmt_safe = total_bsmt.clip(lower=1)
        X["BsmtFinRatio"] = np.where(
            total_bsmt > 0,
            num("BsmtFinSF1") / total_bsmt_safe,
            0,
        )

        return X


# ============================================================
# Pipeline construction
# ============================================================

def get_column_groups(X):
    """Classify columns into numeric, low-cardinality, and high-cardinality."""
    X_transformed = AmesFeatureEngineer().fit_transform(X)
    categorical = X_transformed.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = [c for c in X_transformed.columns if c not in categorical]
    low_card = [c for c in categorical if X_transformed[c].nunique(dropna=True) <= 14]
    high_card = [c for c in categorical if X_transformed[c].nunique(dropna=True) > 14]
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
        ("feature_engineering", AmesFeatureEngineer()),
        ("preprocessing", preprocessor),
        ("model", model),
    ])


# ============================================================
# Metrics
# ============================================================

def log_rmse(y_true, y_pred):
    """Root mean squared error on log-transformed targets."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_on_holdout(pipeline, X_holdout, y_holdout, y_holdout_raw):
    """Evaluate a trained pipeline on the holdout set."""
    y_pred_log = pipeline.predict(X_holdout)
    y_pred_usd = np.expm1(y_pred_log)

    return {
        "log_rmse": float(log_rmse(y_holdout, y_pred_log)),
        "r2": float(r2_score(y_holdout, y_pred_log)),
        "mae_usd": float(mean_absolute_error(y_holdout_raw, y_pred_usd)),
        "rmse_usd": float(np.sqrt(mean_squared_error(y_holdout_raw, y_pred_usd))),
        "median_ae_usd": float(np.median(np.abs(y_holdout_raw - y_pred_usd))),
    }


# ============================================================
# Optuna hyperparameter tuning
# ============================================================

def _optuna_objective(trial, model_name, X, y, cv):
    """Optuna objective function for a single trial."""
    if model_name == "Ridge":
        params = {"alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True)}
        model = Ridge(**params)

    elif model_name == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 350, step=50),
            "max_depth": trial.suggest_int("max_depth", 8, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
        }
        model = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)

    elif model_name == "LightGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        model = LGBMRegressor(**params, random_state=RANDOM_STATE, n_jobs=4, verbose=-1)

    elif model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
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
            "iterations": trial.suggest_int("iterations", 300, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        }
        model = CatBoostRegressor(**params, random_state=RANDOM_STATE, thread_count=4, verbose=0)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipeline = build_pipeline(model, X)
    scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1,
    )
    return -scores.mean()


def tune_model(model_name, X, y, cv, n_trials=OPTUNA_TRIALS):
    """Run Optuna tuning for a model, return best params and score."""
    if not HAS_OPTUNA:
        print("    (Optuna not installed — using default params)")
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
    """Instantiate a model with tuned hyperparameters."""
    if model_name == "Ridge":
        return Ridge(**best_params)

    elif model_name == "RandomForest":
        return RandomForestRegressor(
            **best_params, random_state=RANDOM_STATE, n_jobs=-1,
        )

    elif model_name == "LightGBM":
        return LGBMRegressor(
            **best_params, random_state=RANDOM_STATE, n_jobs=4, verbose=-1,
        )

    elif model_name == "XGBoost":
        return XGBRegressor(
            **best_params, random_state=RANDOM_STATE, n_jobs=4, tree_method="hist", verbosity=0,
        )

    elif model_name == "CatBoost":
        return CatBoostRegressor(
            **best_params, random_state=RANDOM_STATE, thread_count=4, verbose=0,
        )


# ============================================================
# Cross-validation
# ============================================================

def cross_validate_model(model, X, y, cv):
    """Run k-fold cross-validation and return per-fold + aggregate scores."""
    oof_predictions = np.zeros(len(X))
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        pipeline = build_pipeline(clone(model), X_train)
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_valid)
        oof_predictions[valid_idx] = preds

        score = log_rmse(y_valid, preds)
        fold_scores.append(score)
        print(f"    Fold {fold}: log-RMSE = {score:.5f}")

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"    Mean: {mean_score:.5f} ± {std_score:.5f}")
    return oof_predictions, fold_scores, mean_score


# ============================================================
# Feature importance extraction
# ============================================================

def extract_feature_importances(pipeline, X):
    """Extract feature importances from the trained pipeline model."""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessing"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = None

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)

    if importances is None or feature_names is None:
        return {}

    paired = sorted(
        zip(feature_names, importances),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return {name: round(float(imp), 6) for name, imp in paired[:30]}


# ============================================================
# Artifact extraction
# ============================================================

def extract_feature_config(X):
    """Extract feature metadata for the prediction module."""
    config = {"features": {}, "categorical_values": {}, "numeric_ranges": {}}
    for col in X.columns:
        is_cat = X[col].dtype == "object" or str(X[col].dtype) == "string"
        if is_cat:
            config["features"][col] = "categorical"
            config["categorical_values"][col] = sorted(
                [str(v) for v in X[col].dropna().unique()]
            )
        else:
            numeric_vals = pd.to_numeric(X[col], errors="coerce").dropna()
            if len(numeric_vals) == 0:
                config["features"][col] = "categorical"
                config["categorical_values"][col] = sorted(
                    [str(v) for v in X[col].dropna().unique()]
                )
            else:
                config["features"][col] = "numeric"
                config["numeric_ranges"][col] = {
                    "min": float(numeric_vals.min()),
                    "max": float(numeric_vals.max()),
                    "median": float(numeric_vals.median()),
                    "mean": float(round(numeric_vals.mean(), 2)),
                }
    return config


def extract_neighborhood_stats(train_df):
    """Compute per-neighborhood price statistics."""
    stats = {}
    for name, group in train_df.groupby("Neighborhood"):
        prices = group["SalePrice"]
        stats[name] = {
            "count": int(len(group)),
            "median_price": int(prices.median()),
            "mean_price": int(prices.mean()),
            "min_price": int(prices.min()),
            "max_price": int(prices.max()),
            "std_price": int(prices.std()) if len(group) > 1 else 0,
        }
    return stats


def extract_neighborhood_defaults(train_df):
    """Compute per-neighborhood typical feature values for smart defaults."""
    defaults = {}
    feature_cols = [c for c in train_df.columns if c not in ("Id", "SalePrice")]
    for name, group in train_df.groupby("Neighborhood"):
        row = {}
        for col in feature_cols:
            col_dtype = str(group[col].dtype)
            if col_dtype in ("object", "string", "str"):
                mode = group[col].mode()
                row[col] = str(mode.iloc[0]) if len(mode) > 0 else ""
            else:
                numeric_vals = pd.to_numeric(group[col], errors="coerce").dropna()
                if len(numeric_vals) > 0:
                    row[col] = round(float(numeric_vals.median()), 1)
                else:
                    row[col] = 0
        defaults[name] = row
    return defaults


# ============================================================
# Main training loop
# ============================================================

def main():
    print("=" * 60)
    print("AMES HOUSING PRICE PREDICTION — TRAINING PIPELINE")
    print("=" * 60)

    # --- Load data ---
    train_path = PROJECT_ROOT / "train.csv"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    print(f"\nDataset: {train_df.shape[0]} rows × {train_df.shape[1]} columns")

    X = train_df.drop(columns=["SalePrice", "Id"])
    y_raw = train_df["SalePrice"]
    y = np.log1p(y_raw)

    print(f"Target range: ${y_raw.min():,.0f} – ${y_raw.max():,.0f}")
    print(f"Target median: ${y_raw.median():,.0f}")

    # --- Train/holdout split ---
    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE,
    )
    y_holdout_raw = y_raw.loc[y_holdout.index]

    print(f"\nDevelopment set: {X_dev.shape[0]} rows")
    print(f"Holdout set: {X_holdout.shape[0]} rows")

    # --- Define model candidates ---
    model_names = ["Ridge", "RandomForest"]
    if HAS_LGBM:
        model_names.append("LightGBM")
    else:
        print("\n⚠ LightGBM not installed — skipping")
    if HAS_XGB:
        model_names.append("XGBoost")
    else:
        print("\n⚠ XGBoost not installed — skipping")
    if HAS_CATBOOST:
        model_names.append("CatBoost")
    else:
        print("\n⚠ CatBoost not installed — skipping")

    print(f"\nModels to train: {', '.join(model_names)}")
    print(f"Optuna tuning: {'Yes (' + str(OPTUNA_TRIALS) + ' trials)' if HAS_OPTUNA else 'No (using defaults)'}")

    # --- Hyperparameter tuning ---
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    tuned_models = {}
    tuning_results = {}

    print(f"\n{'=' * 60}")
    print("HYPERPARAMETER TUNING" if HAS_OPTUNA else "TRAINING WITH DEFAULT PARAMS")
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
                tuning_results[name] = {"best_params": {"alpha": 10.0}}
            elif name == "RandomForest":
                model = RandomForestRegressor(
                    n_estimators=250, max_depth=15, min_samples_leaf=2,
                    random_state=RANDOM_STATE, n_jobs=-1,
                )
                tuning_results[name] = {"best_params": {"n_estimators": 250, "max_depth": 15}}
            elif name == "LightGBM":
                model = LGBMRegressor(
                    n_estimators=800, learning_rate=0.04, max_depth=5,
                    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=4, verbose=-1,
                )
                tuning_results[name] = {"best_params": {"n_estimators": 800, "learning_rate": 0.04}}
            elif name == "XGBoost":
                model = XGBRegressor(
                    n_estimators=800, learning_rate=0.04, max_depth=5,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0,
                    random_state=RANDOM_STATE, n_jobs=4, tree_method="hist", verbosity=0,
                )
                tuning_results[name] = {"best_params": {"n_estimators": 800, "learning_rate": 0.04}}
            elif name == "CatBoost":
                model = CatBoostRegressor(
                    iterations=800, learning_rate=0.04, depth=5,
                    l2_leaf_reg=3.0, random_state=RANDOM_STATE, thread_count=4, verbose=0,
                )
                tuning_results[name] = {"best_params": {"iterations": 800, "learning_rate": 0.04}}

        tuned_models[name] = model
        tuning_time = time.time() - t0
        tuning_results[name]["tuning_time_seconds"] = round(tuning_time, 1)
        print(f"    Tuning time: {tuning_time:.1f}s")

    # --- Cross-validate each tuned model ---
    cv_results = {}

    print(f"\n{'=' * 60}")
    print("CROSS-VALIDATION (5-Fold)")
    print("=" * 60)

    for name in model_names:
        print(f"\n{name}:")
        t0 = time.time()
        _, fold_scores, mean_score = cross_validate_model(
            tuned_models[name], X_dev, y_dev, cv,
        )
        cv_time = time.time() - t0
        cv_results[name] = {
            "mean": mean_score,
            "std": float(np.std(fold_scores)),
            "folds": [round(s, 5) for s in fold_scores],
            "cv_time_seconds": round(cv_time, 1),
        }

    # --- Select best model ---
    best_name = min(cv_results, key=lambda k: cv_results[k]["mean"])
    best_model = tuned_models[best_name]
    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_name} (CV log-RMSE: {cv_results[best_name]['mean']:.5f})")
    print("=" * 60)

    # --- Evaluate ALL models on holdout ---
    print(f"\n{'=' * 60}")
    print("HOLDOUT EVALUATION (All Models)")
    print("=" * 60)

    holdout_results = {}
    feature_importances = {}

    for name in model_names:
        print(f"\n{name}:")
        pipeline = build_pipeline(clone(tuned_models[name]), X_dev)
        pipeline.fit(X_dev, y_dev)

        results = evaluate_on_holdout(pipeline, X_holdout, y_holdout, y_holdout_raw)
        holdout_results[name] = results

        importances = extract_feature_importances(pipeline, X_dev)
        feature_importances[name] = importances

        print(f"    log-RMSE:   {results['log_rmse']:.5f}")
        print(f"    R²:         {results['r2']:.4f}")
        print(f"    MAE (USD):  ${results['mae_usd']:,.0f}")
        print(f"    RMSE (USD): ${results['rmse_usd']:,.0f}")
        print(f"    Median AE:  ${results['median_ae_usd']:,.0f}")

    # --- Retrain best model on ALL labeled data ---
    print(f"\n{'=' * 60}")
    print(f"PRODUCTION MODEL: Retraining {best_name} on full dataset")
    print("=" * 60)

    production_pipeline = build_pipeline(clone(best_model), X)
    production_pipeline.fit(X, y)

    # --- Serialize artifacts ---
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Pipeline
    pipeline_path = ARTIFACTS_DIR / "pipeline.joblib"
    joblib.dump(production_pipeline, pipeline_path)
    print(f"\nSaved pipeline: {pipeline_path}")

    # Feature config
    feature_config = extract_feature_config(X)
    feature_config["model_info"] = {
        "name": best_name,
        "cv_log_rmse": round(cv_results[best_name]["mean"], 5),
        "holdout_log_rmse": round(holdout_results[best_name]["log_rmse"], 5),
        "holdout_r2": round(holdout_results[best_name]["r2"], 4),
        "holdout_mae_usd": round(holdout_results[best_name]["mae_usd"], 0),
        "holdout_rmse_usd": round(holdout_results[best_name]["rmse_usd"], 0),
        "holdout_median_ae_usd": round(holdout_results[best_name]["median_ae_usd"], 0),
        "target_transform": "log1p",
        "training_samples": len(X),
        "n_features": len(X.columns),
        "feature_columns": X.columns.tolist(),
    }

    config_path = ARTIFACTS_DIR / "feature_config.json"
    with open(config_path, "w") as f:
        json.dump(feature_config, f, indent=2)
    print(f"Saved feature config: {config_path}")

    # Neighborhood stats
    neighborhood_stats = extract_neighborhood_stats(train_df)
    stats_path = ARTIFACTS_DIR / "neighborhood_stats.json"
    with open(stats_path, "w") as f:
        json.dump(neighborhood_stats, f, indent=2)
    print(f"Saved neighborhood stats: {stats_path}")

    # Neighborhood defaults
    neighborhood_defaults = extract_neighborhood_defaults(train_df)
    defaults_path = ARTIFACTS_DIR / "neighborhood_defaults.json"
    with open(defaults_path, "w") as f:
        json.dump(neighborhood_defaults, f, indent=2, default=str)
    print(f"Saved neighborhood defaults: {defaults_path}")

    # Model comparison
    comparison = {
        "dataset": "ames",
        "training_samples": len(X),
        "holdout_samples": len(X_holdout),
        "n_features_raw": len(X.columns),
        "target_transform": "log1p",
        "cv_folds": N_SPLITS,
        "optuna_trials": OPTUNA_TRIALS if HAS_OPTUNA else 0,
        "best_model": best_name,
        "ranking": sorted(model_names, key=lambda k: cv_results[k]["mean"]),
        "models": {},
    }
    for name in model_names:
        comparison["models"][name] = {
            "cv_log_rmse": round(cv_results[name]["mean"], 5),
            "cv_std": round(cv_results[name]["std"], 5),
            "cv_folds": cv_results[name]["folds"],
            "cv_time_seconds": cv_results[name]["cv_time_seconds"],
            "holdout_log_rmse": round(holdout_results[name]["log_rmse"], 5),
            "holdout_r2": round(holdout_results[name]["r2"], 4),
            "holdout_mae_usd": round(holdout_results[name]["mae_usd"], 0),
            "holdout_rmse_usd": round(holdout_results[name]["rmse_usd"], 0),
            "holdout_median_ae_usd": round(holdout_results[name]["median_ae_usd"], 0),
            "best_params": tuning_results[name].get("best_params", {}),
            "tuning_time_seconds": tuning_results[name].get("tuning_time_seconds", 0),
            "top_features": feature_importances.get(name, {}),
        }

    comparison_path = ARTIFACTS_DIR / "model_comparison_ames.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved model comparison: {comparison_path}")

    # --- Summary table ---
    print(f"\n{'=' * 60}")
    print("ALL MODELS — FINAL COMPARISON")
    print("=" * 60)
    print(f"\n  {'Model':<15s} {'CV log-RMSE':>12s} {'Holdout R²':>12s} {'MAE (USD)':>12s} {'Median AE':>12s}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for name in comparison["ranking"]:
        m = comparison["models"][name]
        marker = " ★" if name == best_name else ""
        print(
            f"  {name:<15s} {m['cv_log_rmse']:>12.5f} {m['holdout_r2']:>12.4f}"
            f" ${m['holdout_mae_usd']:>10,.0f} ${m['holdout_median_ae_usd']:>10,.0f}{marker}"
        )

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best:      {best_name}")
    print(f"  Pipeline:  {pipeline_path}")
    print(f"  Config:    {config_path}")
    print(f"  Stats:     {stats_path}")
    print(f"  Defaults:  {defaults_path}")
    print(f"  Compare:   {comparison_path}")
    print()


if __name__ == "__main__":
    main()
