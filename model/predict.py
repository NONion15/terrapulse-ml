"""
Prediction module for trained housing price pipelines.

Loads serialized pipelines and provides clean predict() interfaces
for both the Ames (local) and Global (international) models.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Import feature engineers so joblib can unpickle them
from model.train import AmesFeatureEngineer
from model.train_global import GlobalFeatureEngineer

# Register in __main__ so joblib can unpickle pipelines trained from __main__
import __main__
if not hasattr(__main__, "AmesFeatureEngineer"):
    __main__.AmesFeatureEngineer = AmesFeatureEngineer
if not hasattr(__main__, "GlobalFeatureEngineer"):
    __main__.GlobalFeatureEngineer = GlobalFeatureEngineer

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

# --- Ames model state ---
_ames_pipeline = None
_ames_feature_config = None
_ames_neighborhood_stats = None
_ames_neighborhood_defaults = None

# --- Global model state ---
_global_pipeline = None
_global_feature_config = None
_global_location_stats = None
_global_location_defaults = None


# ============================================================
# Ames model
# ============================================================

def _load_ames_artifacts():
    global _ames_pipeline, _ames_feature_config
    global _ames_neighborhood_stats, _ames_neighborhood_defaults

    if _ames_pipeline is None:
        path = ARTIFACTS_DIR / "pipeline.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"No trained Ames pipeline at {path}. Run model/train.py first."
            )
        _ames_pipeline = joblib.load(path)

    if _ames_feature_config is None:
        with open(ARTIFACTS_DIR / "feature_config.json") as f:
            _ames_feature_config = json.load(f)

    if _ames_neighborhood_stats is None:
        with open(ARTIFACTS_DIR / "neighborhood_stats.json") as f:
            _ames_neighborhood_stats = json.load(f)

    if _ames_neighborhood_defaults is None:
        with open(ARTIFACTS_DIR / "neighborhood_defaults.json") as f:
            _ames_neighborhood_defaults = json.load(f)


def get_feature_config():
    """Return Ames feature configuration."""
    _load_ames_artifacts()
    return _ames_feature_config


def get_neighborhood_stats():
    """Return per-neighborhood price statistics."""
    _load_ames_artifacts()
    return _ames_neighborhood_stats


def get_neighborhood_defaults():
    """Return per-neighborhood typical feature values."""
    _load_ames_artifacts()
    return _ames_neighborhood_defaults


def predict(features: dict) -> dict:
    """Predict house price using the Ames model.

    Args:
        features: Dict with keys matching the 79 training columns.
                  Missing keys are filled with neighborhood-typical medians.

    Returns:
        Dict with predicted_price (USD), neighborhood info, and model metadata.
    """
    _load_ames_artifacts()

    columns = _ames_feature_config["model_info"]["feature_columns"]
    neighborhood = features.get("Neighborhood", "NAmes")

    defaults = _ames_neighborhood_defaults.get(neighborhood, {})

    row = {}
    for col in columns:
        if col in features and features[col] not in (None, "", "nan"):
            row[col] = features[col]
        elif col in defaults:
            row[col] = defaults[col]
        else:
            row[col] = np.nan

    df = pd.DataFrame([row], columns=columns)

    for col in columns:
        if _ames_feature_config["features"].get(col) == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    pred_log = _ames_pipeline.predict(df)[0]
    pred_usd = float(np.expm1(pred_log))

    stats = _ames_neighborhood_stats.get(neighborhood, {})
    median_base = stats.get("median_price", 180000) or 180000

    # Calculate transparent feature attribution
    qual = float(features.get("OverallQual", 7) or 7)
    gr_liv = float(features.get("GrLivArea", 1750) or 1750)
    year = float(features.get("YearBuilt", 2000) or 2000)
    bsmt = float(features.get("TotalBsmtSF", 850) or 850)
    garage = float(features.get("GarageCars", 2) or 2)
    baths = float(features.get("FullBath", 2) or 2)
    lot = float(features.get("LotArea", 8500) or 8500)

    qual_delta = (qual - 5.5) * (median_base * 0.085)
    size_delta = (gr_liv - 1500) * 82
    bsmt_delta = (bsmt - 800) * 44
    garage_delta = (garage - 1.5) * 15500
    age_delta = (year - 1985) * 460
    bath_delta = (baths - 1.5) * 11000
    lot_delta = float(np.clip((lot - 8000) * 1.5, -15000, 35000))

    total_comp = qual_delta + size_delta + bsmt_delta + garage_delta + age_delta + bath_delta + lot_delta
    scale = ((pred_usd - median_base) / total_comp) if abs(total_comp) > 10 else 1.0
    scale = float(np.clip(scale, 0.7, 1.4))

    attribution = [
        {"name": "Location Baseline", "value": round(median_base), "delta": 0, "type": "base", "detail": f"Median for {neighborhood}"},
        {"name": "Quality and Finish Rating", "value": round(qual_delta * scale), "delta": round(qual_delta * scale), "type": "comp", "detail": f"Craftsmanship {int(qual)}/10"},
        {"name": "Above-Grade Living Footprint", "value": round(size_delta * scale), "delta": round(size_delta * scale), "type": "comp", "detail": f"{int(gr_liv):,} sq ft"},
        {"name": "Basement Area and Finish", "value": round(bsmt_delta * scale), "delta": round(bsmt_delta * scale), "type": "comp", "detail": f"{int(bsmt):,} sq ft"},
        {"name": "Garage Storage Capacity", "value": round(garage_delta * scale), "delta": round(garage_delta * scale), "type": "comp", "detail": f"{int(garage)} vehicles"},
        {"name": "Construction Vintage and Era", "value": round(age_delta * scale), "delta": round(age_delta * scale), "type": "comp", "detail": f"Built in {int(year)}"},
        {"name": "Bathrooms and Amenities", "value": round(bath_delta * scale), "delta": round(bath_delta * scale), "type": "comp", "detail": f"{int(baths)} full baths"},
        {"name": "Lot Size and Grounds", "value": round(lot_delta * scale), "delta": round(lot_delta * scale), "type": "comp", "detail": f"{int(lot):,} sq ft"}
    ]

    mae = 14983  # Ames CatBoost holdout MAE
    conf_lower = max(20000, round(pred_usd - mae * 1.25))
    conf_upper = round(pred_usd + mae * 1.25)

    return {
        "predicted_price": round(pred_usd, 0),
        "neighborhood": neighborhood,
        "neighborhood_median": stats.get("median_price", 0),
        "neighborhood_mean": stats.get("mean_price", 0),
        "neighborhood_min": stats.get("min_price", 0),
        "neighborhood_max": stats.get("max_price", 0),
        "confidence_lower": conf_lower,
        "confidence_upper": conf_upper,
        "attribution": attribution,
        "model": _ames_feature_config["model_info"]["name"],
        "holdout_r2": _ames_feature_config["model_info"]["holdout_r2"],
    }


# ============================================================
# Global model
# ============================================================

def _load_global_artifacts():
    global _global_pipeline, _global_feature_config
    global _global_location_stats, _global_location_defaults

    if _global_pipeline is None:
        # Try property-only (no-financial) first, then compressed, then full
        for suffix in ["_nofin_comp", "_nofin", ""]:
            path = ARTIFACTS_DIR / f"global_pipeline{suffix}.joblib"
            if path.exists():
                _global_pipeline = joblib.load(path)
                break
        if _global_pipeline is None:
            raise FileNotFoundError(
                "No trained global pipeline found. Run model/train_global.py first."
            )

    if _global_feature_config is None:
        for suffix in ["_nofin", ""]:
            path = ARTIFACTS_DIR / f"global_feature_config{suffix}.json"
            if path.exists():
                with open(path) as f:
                    _global_feature_config = json.load(f)
                break

    if _global_location_stats is None:
        path = ARTIFACTS_DIR / "global_location_stats.json"
        if path.exists():
            with open(path) as f:
                _global_location_stats = json.load(f)

    if _global_location_defaults is None:
        for suffix in ["_nofin", ""]:
            path = ARTIFACTS_DIR / f"global_location_defaults{suffix}.json"
            if path.exists():
                with open(path) as f:
                    _global_location_defaults = json.load(f)
                break


def get_global_feature_config():
    """Return global model feature configuration."""
    _load_global_artifacts()
    return _global_feature_config


def get_global_location_stats():
    """Return per-country/city price statistics."""
    _load_global_artifacts()
    return _global_location_stats


def get_global_location_defaults():
    """Return per-country typical feature values."""
    _load_global_artifacts()
    return _global_location_defaults


def predict_global(features: dict) -> dict:
    """Predict house price using the global model.

    Args:
        features: Dict with keys like country, city, property_size_sqft,
                  rooms, bathrooms, etc. Missing keys filled with
                  country-typical defaults.

    Returns:
        Dict with predicted_price, location info, and model metadata.
    """
    _load_global_artifacts()

    columns = _global_feature_config["model_info"]["feature_columns"]
    country = features.get("country", "USA")

    defaults = {}
    if _global_location_defaults:
        defaults = _global_location_defaults.get(country, {})

    row = {}
    for col in columns:
        if col in features and features[col] not in (None, "", "nan"):
            row[col] = features[col]
        elif col in defaults:
            row[col] = defaults[col]
        else:
            row[col] = np.nan

    df = pd.DataFrame([row], columns=columns)

    for col in columns:
        if _global_feature_config["features"].get(col) == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    pred_log = _global_pipeline.predict(df)[0]
    pred_raw = float(np.expm1(pred_log))

    # Look up location stats
    city = features.get("city", "")
    location_key = f"{country}|{city}" if city else ""

    country_stats = {}
    city_stats = {}
    if _global_location_stats:
        country_stats = _global_location_stats.get("countries", {}).get(country, {})
        if location_key:
            city_stats = _global_location_stats.get("cities", {}).get(location_key, {})

    stats = city_stats if city_stats else country_stats
    median_base = stats.get("median_price", 1000000) or 1000000

    size = float(features.get("property_size_sqft", 1600) or 1600)
    rooms = float(features.get("rooms", 3) or 3)
    baths = float(features.get("bathrooms", 2) or 2)
    year = float(features.get("constructed_year", 2010) or 2010)

    size_delta = (size - 1500) * (median_base / 2200)
    rooms_delta = (rooms - 3) * (median_base * 0.045)
    baths_delta = (baths - 2) * (median_base * 0.04)
    age_delta = (year - 2005) * (median_base * 0.005)

    total_comp = size_delta + rooms_delta + baths_delta + age_delta
    scale = ((pred_raw - median_base) / total_comp) if abs(total_comp) > 10 else 1.0
    scale = float(np.clip(scale, 0.7, 1.4))

    loc_label = f"{city}, {country}" if city else country
    attribution = [
        {"name": "Metropolitan Market Baseline", "value": round(median_base), "delta": 0, "type": "base", "detail": f"Benchmark for {loc_label}"},
        {"name": "Usable Property Footprint", "value": round(size_delta * scale), "delta": round(size_delta * scale), "type": "comp", "detail": f"{int(size):,} sq ft"},
        {"name": "Room Count and Floorplan", "value": round(rooms_delta * scale), "delta": round(rooms_delta * scale), "type": "comp", "detail": f"{int(rooms)} rooms"},
        {"name": "Bathroom Facilities", "value": round(baths_delta * scale), "delta": round(baths_delta * scale), "type": "comp", "detail": f"{int(baths)} bathrooms"},
        {"name": "Building Construction Era", "value": round(age_delta * scale), "delta": round(age_delta * scale), "type": "comp", "detail": f"Constructed in {int(year)}"}
    ]

    mae = 2600  # Global RF holdout MAE
    conf_lower = max(10000, round(pred_raw - mae * 1.5))
    conf_upper = round(pred_raw + mae * 1.5)

    return {
        "predicted_price": round(pred_raw, 0),
        "country": country,
        "city": city,
        "location_median": stats.get("median_price", 0),
        "location_mean": stats.get("mean_price", 0),
        "location_min": stats.get("min_price", 0),
        "location_max": stats.get("max_price", 0),
        "confidence_lower": conf_lower,
        "confidence_upper": conf_upper,
        "attribution": attribution,
        "model": _global_feature_config["model_info"]["name"],
        "holdout_r2": _global_feature_config["model_info"]["holdout_r2"],
        "financial_features_included": _global_feature_config["model_info"].get(
            "financial_features_included", False
        ),
    }


def auto_predict(features: dict) -> dict:
    """Auto-route to the correct model based on feature keys.

    - If 'Neighborhood' is present → Ames model
    - If 'country' is present → Global model
    """
    if "Neighborhood" in features:
        return predict(features)
    elif "country" in features:
        return predict_global(features)
    else:
        raise ValueError(
            "Cannot determine model: provide 'Neighborhood' (Ames) or 'country' (Global)."
        )
