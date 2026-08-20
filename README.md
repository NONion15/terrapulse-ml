# Ames Home Prediction Project

A machine learning project for residential real estate price estimation, built on the Ames, Iowa housing dataset with secondary validation on a 200,000-record international dataset.

This repository contains the end-to-end code for data preprocessing, feature engineering, model training with Bayesian hyperparameter optimization, and a lightweight web interface for interactive property evaluation and geospatial exploration.

---

## Table of Contents
1. Project Overview
2. Core Machine Learning Concepts
3. Models Evaluated and Comparison
4. Benchmark Results and Findings
5. Feature Engineering Pipeline
6. Target Leakage Prevention
7. Repository Structure
8. Quickstart and Installation Guide
9. Web Application Features

---

## 1. Project Overview

Real estate valuation typically relies on manual comparative market analysis (CMA), where an appraiser identifies comparable properties and adjusts for differences in square footage, condition, and location.

This project models that process mathematically using supervised machine learning algorithms. Instead of relying on hardcoded heuristics, the models learn the non-linear relationships between physical property characteristics, building materials, neighborhood locations, and historical transaction prices.

The system evaluates two distinct datasets:
- **Ames Housing Dataset (Primary Focus):** 1,460 residential properties from Ames, Iowa, containing 81 detailed attributes such as basement finishes, roof materials, masonry veneer, and zoning codes.
- **Global Housing Dataset (Scalability Benchmark):** 200,000 transaction records across 13 countries and 40 metropolitan markets, used to evaluate model stability on large-scale data.

---

## 2. Core Machine Learning Concepts

For readers new to machine learning, here is how the core pipeline operates:

### Features vs. Target
- **Features (X):** The input variables describing each home (living area, year built, overall quality rating, bathroom count, neighborhood).
- **Target (y):** The continuous numerical value we want to predict (Sale Price in US Dollars).

### Training vs. Testing Splits
To verify that the model actually learns underlying pricing patterns rather than simply memorizing the dataset (known as overfitting), we split the data:
- **Training Set (80%):** The historical records used by the algorithm to adjust its internal parameters.
- **Test Set (20%):** Held-out data that the model has never seen before, used to calculate honest evaluation scores.

### 5-Fold Cross-Validation
During hyperparameter tuning, we use 5-fold cross-validation. The training data is split into 5 equal subsets (folds). The model is trained on 4 folds and validated on the 5th, repeating 5 times so every data point is tested out-of-fold. The average score across all 5 folds provides a reliable estimate of generalization performance.

### Evaluation Metrics
- **R-squared (Coefficient of Determination):** Represents the proportion of variance in home prices explained by the model. An R-squared of 0.9094 means the model accounts for roughly 91% of the price variation in the market.
- **MAE (Mean Absolute Error):** The average dollar amount the prediction differs from the actual sale price (lower is better).
- **log-RMSE (Root Mean Squared Logarithmic Error):** Measures relative percentage error rather than absolute dollar error, preventing expensive luxury homes from dominating the loss calculation.

---

## 3. Models Evaluated and Comparison

We tested five regression algorithms to understand how different model architectures handle tabular real estate data:

### 1. Ridge Regression (Linear Baseline)
- **Concept:** Fits a linear equation with L2 weight regularization to penalize overly large coefficients.
- **Role:** Serves as a standard baseline to check whether complex tree ensembles genuinely outperform linear math.
- **Result:** Solid linear baseline (R-squared: 0.9114), but limited in capturing non-linear feature interactions without manual polynomial expansion.

### 2. Random Forest (Bagging Ensemble)
- **Concept:** Constructs a large ensemble (100+ trees) of independent decision trees trained on random subsets of data and features. Final predictions are computed by averaging all tree outputs.
- **Result:** Won 1st place on the 200,000-row global dataset (R-squared: 0.9999, MAE: $2,600). Excellent stability and resistance to variance on large datasets.

### 3. LightGBM (Histogram Gradient Boosting)
- **Concept:** A gradient boosting framework that buckets continuous values into discrete histogram bins and grows trees leaf-wise.
- **Result:** Blazing fast training times (<3 seconds on 200k rows) with competitive accuracy (R-squared: 0.9013 on Ames).

### 4. XGBoost (Extreme Gradient Boosting)
- **Concept:** Sequential gradient boosting with second-order Taylor expansion and explicit tree-pruning penalties.
- **Result:** Runner-up on Ames (R-squared: 0.9052, MAE: $15,433). Highly reliable and well-regularized.

### 5. CatBoost (Ames Dataset Champion)
- **Concept:** Gradient boosting designed specifically for categorical data. It converts categorical labels (neighborhoods, roof types, exterior materials) into target statistics dynamically during training, preventing prediction shifts.
- **Result:** Champion on Ames Housing. Achieved the lowest cross-validation log-RMSE (0.11561) and lowest holdout MAE ($14,983).

---

## 4. Benchmark Results and Findings

### Ames Housing Dataset (1,460 Samples, 81 Features)

| Rank | Model Name | 5-Fold CV log-RMSE | Holdout R-squared | Holdout MAE (USD) | Status |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | **CatBoost (Tuned)** | **0.11561** | **0.9094** | **$14,983** | Production Model |
| 2 | **XGBoost (Tuned)** | 0.12133 | 0.9052 | $15,433 | Runner-Up |
| 3 | **LightGBM (Tuned)** | 0.12240 | 0.9013 | $16,309 | Strong Contender |
| 4 | **Ridge Regression** | 0.13433 | 0.9114 | $15,795 | Linear Baseline |
| 5 | **Random Forest** | 0.13740 | 0.8913 | $16,588 | Bagging Baseline |

### Global Housing Dataset (200,000 Samples, 40 Cities)

| Rank | Model Name | 5-Fold CV log-RMSE | Holdout R-squared | Holdout MAE (USD) | Holdout MAPE |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **Random Forest** | **0.00877** | **0.9999** | **$2,600** | **0.4%** |
| 2 | **CatBoost** | 0.00917 | 0.9998 | $4,725 | 0.6% |
| 3 | **XGBoost** | 0.00963 | 0.9999 | $4,266 | 0.5% |
| 4 | **LightGBM** | 0.00977 | 0.9999 | $3,794 | 0.5% |

---

## 5. Feature Engineering Pipeline

Raw tabular features were transformed using domain-specific real estate logic in `model/train.py`:

```python
# 1. Total Enclosed Footprint (Above-ground living space + finished basement)
X['TotalFootprint'] = X['GrLivArea'] + X['TotalBsmtSF']

# 2. Quality and Condition Interaction (Build grade multiplied by upkeep rating)
X['QualCondScore'] = X['OverallQual'] * X['OverallCond']

# 3. Fractional Bathroom Count (Full baths weighted 1.0, half baths weighted 0.5)
X['TotalBath'] = X['FullBath'] + 0.5 * X['HalfBath']

# 4. Remodel Indicator (Flags whether a vintage home underwent modern renovation)
X['IsRemodeled'] = (X['YearRemodAdd'] != X['YearBuilt']).astype(int)

# 5. Room Density (Living space divided by total room count)
X['LivAreaPerRoom'] = X['GrLivArea'] / X['TotRmsAbvGrd']
```

These interaction terms improved linear model performance from an initial R-squared of 0.86 to over 0.91, giving the algorithms stronger structural signals.

---

## 6. Target Leakage Prevention

In exploratory data analysis of the global dataset, columns such as `loan_amount` and `down_payment` showed correlation coefficients of 0.938 and 0.851 with sale price.

In real-world mortgage underwriting, loan amounts and down payments are calculated *after* a property is appraised, not before. Training a model on loan amounts would create artificial accuracy (target leakage) because the model would simply invert the loan-to-value ratio rather than appraising property features.

To ensure production integrity:
- All financing columns were permanently excluded from the production pipeline (`--no-financial` mode).
- The models learn exclusively from intrinsic physical, structural, and geographic attributes.

---

## 7. Repository Structure

```
ames-home-prediction/
├── train.csv                          # Ames historical sales data (1,460 rows)
├── test.csv                           # Ames test dataset
├── global_house_purchase_dataset.csv  # 200k global real estate records
├── houseML.ipynb                      # Exploratory data analysis notebook
├── requirements.txt                   # Project dependencies
├── README.md                          # Documentation and project overview
│
├── model/                             # Machine learning pipelines
│   ├── train.py                       # Ames 5-model Optuna training script
│   ├── train_global.py                # Global 200k training script
│   ├── predict.py                     # Inference module with component attributions
│   └── artifacts/                     # Serialized pipelines and statistical benchmarks
│       ├── pipeline.joblib            # Trained CatBoost pipeline (1MB)
│       ├── feature_config.json        # Feature metadata and column configurations
│       ├── neighborhood_stats.json    # Neighborhood median and range benchmarks
│       ├── neighborhood_defaults.json # Statistical median defaults per neighborhood
│       └── model_comparison_ames.json # Full benchmark tournament metrics
│
├── data/                              # Geospatial data
│   ├── ames_house_points.json         # 1,460 house point coordinates and specs
│   └── neighborhoods.geojson          # Geographic polygon boundaries for Ames
│
└── app/                               # Web interface
    ├── server.py                      # Flask backend application
    ├── templates/
    │   └── index.html                 # 3-section user interface
    └── static/
        ├── css/style.css              # Application stylesheet
        └── js/app.js                  # Map rendering, spatial resolver, client logic
```

---

## 8. Quickstart and Installation Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12
- pip package manager

### 1. Clone Repository and Install Dependencies
```bash
git clone https://github.com/NONion15/terrapulse-ml.git
cd terrapulse-ml
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app/server.py
```
Open **http://localhost:8000** in your web browser.

### 3. (Optional) Retrain Models with Bayesian Optimization
```bash
# Train Ames models with 5-fold cross-validation and Optuna tuning
python model/train.py

# Train Global models in leakage-free property mode
python model/train_global.py --no-financial
```

---

## 9. Web Application Features

The interface is structured into three clean sections:

1. **Property Valuation Calculator:** A clean numerical interface with input sliders for quality ratings, living space, vintage, room counts, garage size, and expandable advanced architectural specifications. Displays real-time valuations, 95% confidence bounds, value attribution waterfall breakdowns, multi-model consensus comparisons, and a 10-year equity forecast.
2. **Geospatial Map Explorer:** Interactive Leaflet map displaying 1,460 individual Ames properties color-coded by valuation tier. Hovering over any dot displays detailed property specifications. Clicking any location on the map runs point-in-polygon resolution to dynamically load neighborhood statistical defaults and calculate live valuations.
3. **Model Documentation:** An educational overview explaining the regression algorithms, evaluation scorecards, feature engineering techniques, and target leakage prevention.
4. **Appraisal Certificate Generator:** Creates a printable valuation appraisal summary document with property specifications, component waterfall breakdown, and financing scenarios.

---

## License
Open-source under standard MIT licensing terms.
