# Intelligent Housing Price Prediction & Geospatial System

Production-grade residential property price prediction platform supporting both high-granularity US properties (Ames, Iowa) and international real estate across 13 countries and 40 global metropolitan markets.

Built with an automated ML pipeline comparing **5 competitive algorithms** (CatBoost, XGBoost, LightGBM, Random Forest, Ridge Regression) with **Optuna Bayesian hyperparameter optimization**, served via Flask with an interactive Leaflet.js geospatial interface.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train both models (or run individually)
python model/train.py              # Ames 5-model training & CatBoost artifact generation
python model/train_global.py --no-financial   # Global 200k 5-model training & artifact generation

# 3. Start the application
python app/server.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Model Benchmark & Assessment Comparisons

Both datasets were trained with **5-fold cross-validation** and evaluated against independent held-out test sets.

### 1. Ames Housing Dataset (1,460 Samples, 81 Features)

| Rank | Model Candidate | CV log-RMSE (5-Fold) | Holdout R² | Holdout MAE (USD) | Holdout Median AE | Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **CatBoost** | **0.11561** | **0.9094** | **$14,983** | **$8,798** | **Production Champion** |
| 🥈 | **XGBoost** | 0.12133 | 0.9052 | $15,433 | $8,346 | Strong L1/L2 Regularization |
| 🥉 | **LightGBM** | 0.12240 | 0.9013 | $16,309 | $9,743 | Fast Histogram Boosting |
| 4 | **Ridge Regression** | 0.13433 | 0.9114 | $15,795 | $10,736 | Linear Baseline |
| 5 | **Random Forest** | 0.13740 | 0.8913 | $16,588 | $8,829 | Bagging Baseline |

*Key Insights:*
- **CatBoost** delivered the lowest cross-validation log-RMSE (0.11561) and lowest holdout MAE ($14,983), handling high-cardinality neighborhood categoricals with minimal overfitting.
- Feature engineering (quality interaction terms, livable area ratios, basement finish ratios) boosted linear baseline R² from 0.86 to 0.9114.

---

### 2. Global Housing Dataset (200,000 Samples, 13 Countries, 40 Cities)

Evaluated in **Property-Only Mode** (excluding target-leaking loan/down-payment variables for honest generalization):

| Rank | Model Candidate | CV log-RMSE (5-Fold) | Holdout R² | Holdout MAE (USD) | Holdout MAPE | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **Random Forest** | **0.00877** | **0.9999** | **$2,600** | **0.4%** | **Production Champion** |
| 🥈 | **CatBoost** | 0.00917 | 0.9998 | $4,725 | 0.6% | Top Boosting Model |
| 🥉 | **XGBoost** | 0.00963 | 0.9999 | $4,266 | 0.5% | Top Boosting Model |
| 4 | **LightGBM** | 0.00977 | 0.9999 | $3,794 | 0.5% | Fast Large-Scale Boosting |
| 5 | **Ridge Regression** | 0.13512 | 0.9721 | $108,069 | 10.2% | Linear Baseline |

---

## Target Leakage Analysis

On the 200,000-row global dataset, exploratory correlation analysis revealed that `loan_amount` ($r = 0.938$) and `down_payment` ($r = 0.851$) are direct mathematical derivations of property price. 

To prevent data leakage in production:
1. **Property-Only Model (`--no-financial`)**: Excludes loan/down-payment metrics, relying purely on structural features (square footage, rooms, bathrooms, country, city, construction year, amenities, local risk indices).
2. **Full Model**: Available for evaluation where financing terms are known inputs.

---

## Project Architecture

```
ML_project/
├── train.csv                      # Ames training dataset (1,460 rows)
├── test.csv                       # Ames test set
├── global_house_purchase_dataset.csv # Global dataset (200,000 rows)
├── requirements.txt               # Dependencies (LightGBM, XGBoost, CatBoost, Optuna, etc.)
│
├── model/
│   ├── train.py                   # Ames 5-model training with Optuna
│   ├── train_global.py            # Global 200k 5-model training pipeline
│   ├── predict.py                 # Multi-model inference module with auto-routing
│   └── artifacts/
│       ├── pipeline.joblib        # Serialized Ames CatBoost pipeline
│       ├── feature_config.json    # Ames feature metadata
│       ├── neighborhood_stats.json # Ames neighborhood price statistics
│       ├── neighborhood_defaults.json # Ames smart default feature values
│       ├── model_comparison_ames.json # Full Ames 5-model metrics
│       ├── global_pipeline_nofin.joblib # Serialized Global Random Forest pipeline
│       ├── global_feature_config_nofin.json # Global feature metadata
│       ├── global_location_stats.json # Global country/city price benchmarks
│       └── model_comparison_global_nofin.json # Full Global 5-model metrics
│
└── app/
    ├── server.py                  # Flask server with REST prediction APIs
    ├── templates/
    │   └── index.html             # Map interface
    └── static/
        ├── css/style.css          # Dark theme layout
        └── js/app.js              # Map interactions & prediction handler
```

---

## API Endpoints

### 1. `POST /api/predict`
Predict price using the detailed Ames model.
```json
{
  "Neighborhood": "CollgCr",
  "OverallQual": 7,
  "GrLivArea": 1710,
  "YearBuilt": 2003,
  "TotalBsmtSF": 856,
  "GarageCars": 2,
  "FullBath": 2
}
```

### 2. `POST /api/predict/global`
Predict price using the Global model across 13 countries.
```json
{
  "country": "Germany",
  "city": "Berlin",
  "property_size_sqft": 1800,
  "rooms": 4,
  "bathrooms": 2,
  "property_type": "Apartment"
}
```

### 3. `POST /api/predict/auto`
Auto-routes to Ames or Global pipeline depending on provided features (`Neighborhood` vs `country`).

### 4. `GET /api/model-comparison`
Returns complete JSON assessment benchmarks and metrics for all trained models on both datasets.

---

## Retraining & Hyperparameter Tuning

```bash
# Retrain Ames model with Optuna Bayesian search
python model/train.py

# Retrain Global model (Property-only mode)
python model/train_global.py --no-financial

# Retrain Global model (With financial features)
python model/train_global.py
```
