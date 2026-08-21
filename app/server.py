"""
Flask server for the Real Estate Geospatial Intelligence & Prediction Platform.

Serves the interactive multi-scope map UI and handles prediction requests for both
Ames (parcel-level) and Global (40 international metropolitan markets).

Usage:
    python app/server.py [--port 8000]
"""

import json
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

try:
    from flask_cors import CORS
    cors_available = True
except ImportError:
    cors_available = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.predict import (
    auto_predict,
    get_feature_config,
    get_global_feature_config,
    get_global_location_defaults,
    get_global_location_stats,
    get_neighborhood_defaults,
    get_neighborhood_stats,
    predict,
    predict_global,
)

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)

if cors_available:
    CORS(app)

GEOJSON_PATH = PROJECT_ROOT / "data" / "neighborhoods.geojson"

CITY_COORDINATES = {
    "Australia|Brisbane": [-27.4698, 153.0251],
    "Australia|Melbourne": [-37.8136, 144.9631],
    "Australia|Sydney": [-33.8688, 151.2093],
    "Brazil|Rio de Janeiro": [-22.9068, -43.1729],
    "Brazil|São Paulo": [-23.5505, -46.6333],
    "Canada|Montreal": [45.5017, -73.5673],
    "Canada|Toronto": [43.6532, -79.3832],
    "Canada|Vancouver": [49.2827, -123.1207],
    "China|Beijing": [39.9042, 116.4074],
    "China|Shanghai": [31.2304, 121.4737],
    "China|Shenzhen": [22.5431, 114.0579],
    "France|Lyon": [45.7640, 4.8357],
    "France|Marseille": [43.2965, 5.3698],
    "France|Paris": [48.8566, 2.3522],
    "Germany|Berlin": [52.5200, 13.4050],
    "Germany|Frankfurt": [50.1109, 8.6821],
    "Germany|Munich": [48.1351, 11.5820],
    "India|Bangalore": [12.9716, 77.5946],
    "India|Chennai": [13.0827, 80.2707],
    "India|Delhi": [28.6139, 77.2090],
    "India|Hyderabad": [17.3850, 78.4867],
    "India|Mumbai": [19.0760, 72.8777],
    "India|Pune": [18.5204, 73.8567],
    "Japan|Kyoto": [35.0116, 135.7681],
    "Japan|Osaka": [34.6937, 135.5023],
    "Japan|Tokyo": [35.6762, 139.6503],
    "Singapore|Singapore": [1.3521, 103.8198],
    "South Africa|Cape Town": [-33.9249, 18.4241],
    "South Africa|Johannesburg": [-26.2041, 28.0473],
    "UAE|Abu Dhabi": [24.4539, 54.3773],
    "UAE|Dubai": [25.2048, 55.2708],
    "UK|Birmingham": [52.4862, -1.8904],
    "UK|Liverpool": [53.4084, -2.9916],
    "UK|London": [51.5074, -0.1278],
    "UK|Manchester": [53.4808, -2.2426],
    "USA|Chicago": [41.8781, -87.6298],
    "USA|Houston": [29.7604, -95.3698],
    "USA|Los Angeles": [34.0522, -118.2437],
    "USA|New York": [40.7128, -74.0060],
    "USA|San Francisco": [37.7749, -122.4194],
}

ESSENTIAL_FEATURES = [
    ("OverallQual", "Overall Quality", "slider", "Material and finish craftsmanship rating (1-10)"),
    ("GrLivArea", "Living Area (sq ft)", "number", "Above-grade ground living area"),
    ("YearBuilt", "Year Built", "number", "Original construction year"),
    ("TotalBsmtSF", "Basement Area (sq ft)", "number", "Total basement footprint"),
    ("GarageCars", "Garage Capacity", "select", "Vehicle storage capacity in garage"),
    ("FullBath", "Full Bathrooms", "select", "Full bathrooms with tub/shower above grade"),
    ("BedroomAbvGr", "Bedrooms", "select", "Total bedrooms above ground"),
    ("Fireplaces", "Fireplaces", "select", "Operational indoor fireplaces"),
    ("YearRemodAdd", "Year Remodeled", "number", "Major remodel or renovation year"),
    ("LotArea", "Lot Size (sq ft)", "number", "Total parcel land area"),
    ("OverallCond", "Overall Condition", "slider", "Current structural and aesthetic condition"),
    ("CentralAir", "Central Air", "select", "Central cooling HVAC system"),
    ("KitchenQual", "Kitchen Quality", "select", "Cabinetry, appliances, and layout grade"),
    ("BldgType", "Building Type", "select", "Architectural dwelling classification"),
    ("HouseStyle", "House Style", "select", "Story elevation and living structure style"),
]


AMES_HOUSES_PATH = PROJECT_ROOT / "data" / "ames_house_points.json"

def load_geojson():
    with open(GEOJSON_PATH) as f:
        return json.load(f)

def load_ames_houses():
    if AMES_HOUSES_PATH.exists():
        with open(AMES_HOUSES_PATH) as f:
            return json.load(f)
    return []


@app.route("/")
def index():
    geojson = load_geojson()
    ames_houses = load_ames_houses()
    ames_config = get_feature_config()
    ames_stats = get_neighborhood_stats()
    ames_defaults = get_neighborhood_defaults()

    try:
        global_config = get_global_feature_config()
        global_stats = get_global_location_stats()
        global_defaults = get_global_location_defaults()
    except Exception:
        global_config = {}
        global_stats = {}
        global_defaults = {}

    return render_template(
        "index.html",
        geojson_data=json.dumps(geojson),
        ames_house_points=json.dumps(ames_houses),
        feature_config=json.dumps(ames_config),
        neighborhood_stats=json.dumps(ames_stats),
        neighborhood_defaults=json.dumps(ames_defaults),
        essential_features=ESSENTIAL_FEATURES,
        model_info=ames_config.get("model_info", {}),
        global_config=json.dumps(global_config),
        global_stats=json.dumps(global_stats),
        global_defaults=json.dumps(global_defaults),
        city_coordinates=json.dumps(CITY_COORDINATES),
    )


# ============================================================
# Prediction Endpoints
# ============================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict using the Ames CatBoost pipeline."""
    try:
        features = request.get_json()
        if not features:
            return jsonify({"error": "No features provided"}), 400

        result = predict(features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/global", methods=["POST"])
def api_predict_global():
    """Predict using the Global 200k Random Forest pipeline."""
    try:
        features = request.get_json()
        if not features:
            return jsonify({"error": "No features provided"}), 400

        result = predict_global(features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/auto", methods=["POST"])
def api_predict_auto():
    """Auto-detect location type and route to optimal model."""
    try:
        features = request.get_json()
        if not features:
            return jsonify({"error": "No features provided"}), 400

        result = auto_predict(features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/features")
def api_features():
    return jsonify(get_feature_config())


@app.route("/api/neighborhoods")
def api_neighborhoods():
    return jsonify(get_neighborhood_stats())


@app.route("/api/ames/houses")
def api_ames_houses():
    return jsonify(load_ames_houses())


@app.route("/api/global/locations")
def api_global_locations():
    return jsonify({
        "stats": get_global_location_stats(),
        "coordinates": CITY_COORDINATES,
    })


@app.route("/api/model-comparison")
def api_model_comparison():
    result = {}
    artifacts = PROJECT_ROOT / "model" / "artifacts"

    for name in ["model_comparison_ames", "model_comparison_global_nofin"]:
        path = artifacts / f"{name}.json"
        if path.exists():
            with open(path) as f:
                result[name] = json.load(f)

    return jsonify(result)


@app.route("/healthz")
@app.route("/api/health")
def healthz():
    return jsonify({"status": "healthy", "service": "terrapulse-ml"}), 200


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Housing Price Predictor Server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  Intelligent Spatial ML Platform")
    print(f"  Local Address: http://localhost:{args.port}")
    print("=" * 50 + "\n")

    app.run(host=args.host, port=args.port, debug=False)
