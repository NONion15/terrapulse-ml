# 🏡 TerraPulse ML — Intelligent Real Estate Valuation & Learning Guide

> **A beginner-friendly, comprehensive guide to Machine Learning, Regression Algorithms, and Real Estate Valuation.**

Welcome to **TerraPulse ML**! This project is an end-to-end Machine Learning (ML) system that predicts residential property values. Whether you are a beginner taking your first steps into AI, a student preparing an assessment, or a developer exploring machine learning, this repository is designed to be **clear, educational, and easy to understand from scratch**.

---

## 📑 Table of Contents
1. [🌟 What is This Project? (The Big Picture)](#-what-is-this-project-the-big-picture)
2. [🧠 Machine Learning 101: Key Concepts Explained Simply](#-machine-learning-101-key-concepts-explained-simply)
3. [🤖 The 5 Algorithms We Tested (With Simple Analogies)](#-the-5-algorithms-we-tested-with-simple-analogies)
4. [🏆 Tournament Leaderboard: Which Algorithm Won?](#-tournament-leaderboard-which-algorithm-won)
5. [🔍 Smart Feature Engineering: Giving AI Common Sense](#-smart-feature-engineering-giving-ai-common-sense)
6. [⚠️ Target Leakage: The "Cheating on the Test" Trap](#️-target-leakage-the-cheating-on-the-test-trap)
7. [📁 Codebase Walkthrough: What Every File Does](#-codebase-walkthrough-what-every-file-does)
8. [🚀 Step-by-Step Quickstart Guide](#-step-by-step-quickstart-guide)
9. [🎨 How the 3-Section Web App Works](#-how-the-3-section-web-app-works)

---

## 🌟 What is This Project? (The Big Picture)

When a human real estate appraiser values a house, they don't guess blindly. They inspect the house (its square footage, number of bedrooms, materials, construction year, neighborhood) and compare it against **hundreds of past home sales** they remember.

**Machine Learning does the exact same thing using mathematics.**

Instead of writing thousands of rigid if-else rules (like *"if 3 bedrooms add $20,000"*), we feed the computer historical records of home sales. The computer studies the numbers, detects patterns, and learns how much each characteristic contributes to a home's market value.

This project trains and compares **5 competitive machine learning algorithms** across two diverse datasets:
- 🇺🇸 **Ames Housing Dataset (1,460 Homes):** Deep, granular US properties with 81 detailed features (fireplaces, basement finishes, roof styles, kitchen grades).
- 🌐 **Global Housing Dataset (200,000 Homes):** Large-scale international transactions spanning 13 countries and 40 metropolitan cities (Tokyo, London, New York, Berlin, Sydney, Dubai).

---

## 🧠 Machine Learning 101: Key Concepts Explained Simply

If you're new to AI and Data Science, here are the core concepts used throughout this project:

```
+-------------------------------------------------------------+
|                     MACHINE LEARNING FLOW                   |
|                                                             |
|   [ Historical Data ]                                       |
|    (Features + Prices)                                      |
|             │                                               |
|             ▼                                               |
|   [ Algorithm Training ] ───► Learns Mathematical Patterns  |
|             │                                               |
|             ▼                                               |
|   [ Validation (5-Fold) ] ──► Tests on Unseen "Exam" Data   |
|             │                                               |
|             ▼                                               |
|   [ Final Prediction ] ───► Estimates Price for New House   |
+-------------------------------------------------------------+
```

### 1. Features vs. Target
- **Features ($X$):** The clues/inputs (e.g., Living Area, Quality Rating, Bedrooms, Neighborhood).
- **Target ($y$):** The answer we want to predict (the **Sale Price** in dollars).

### 2. Training vs. Testing (Practice vs. The Final Exam)
If a student memorizes the exact questions and answers to a practice test, they might get 100%. But if they fail the actual final exam with new questions, they didn't really learn — they just memorized!
- In ML, memorizing is called **Overfitting**.
- To prevent this, we split the data: we let the algorithm study **80% of the houses** (Training Set), and grade its performance on the remaining **20% of houses it has never seen before** (Test Set).

### 3. 5-Fold Cross-Validation (Fair Grading)
Instead of testing on just one lucky slice, we divide the data into **5 equal buckets**:
- Round 1: Train on buckets 1-4, test on bucket 5.
- Round 2: Train on buckets 1, 2, 3, 5, test on bucket 4.
- ... Repeat 5 times!
- The average score tells us the true, honest performance of the algorithm.

### 4. How We Grade the Models (Scorecard Metrics)
- **$R^2$ Score (Accuracy %):** Measures how much of the variation in home prices our model explains. A score of `0.9094` means the model explains **90.94%** of all pricing differences in the market!
- **MAE (Mean Absolute Error):** The average dollar amount the prediction is off by (e.g., $\pm \$14,983$). Lower is better.
- **log-RMSE (Log Root Mean Squared Error):** Penalizes large percentage mistakes more heavily than small ones.

---

## 🤖 The 5 Algorithms We Tested (With Simple Analogies)

We evaluated 5 different regression algorithms to find out which one performs best:

### 📈 1. Ridge Regression (The Straight-Line Baseline)
- **The Analogy:** An appraiser who assigns a fixed price per unit (e.g., +$85/sq ft, +$15,000/bathroom).
- **How it works:** Draws a best-fit straight line through the data. It uses a mathematical penalty called *L2 Regularization* so no single feature gets an exaggerated, unrealistic weight.
- **Verdict:** Solid baseline, but struggles with complex curves and non-linear interactions.

### 🌲 2. Random Forest (The Committee of 100+ Trees)
- **The Analogy:** Instead of trusting 1 appraiser, you ask a committee of **100 independent decision trees**.
- **How it works:** Each tree asks a series of yes/no questions (*"Is living area > 1,800 sq ft? Is quality ≥ 7? Was it built after 2000?"*). Every tree casts a vote, and the average vote wins.
- **Verdict:** Won **1st place on the 200,000-row Global dataset** ($R^2 = 0.9999$). Highly resilient and virtually immune to erratic mistakes.

### ⚡ 3. LightGBM (The Lightning-Fast Speedster)
- **The Analogy:** A relay team where Student 2 focuses only on the questions Student 1 got wrong, and Student 3 focuses on what Student 2 missed.
- **How it works:** Sequentially corrects errors (Gradient Boosting). LightGBM sorts continuous numbers into discrete "bins" (histograms), making it blazingly fast.
- **Verdict:** Trained 200,000 rows in less than 3 seconds with excellent accuracy.

### 🎯 4. XGBoost (The Extreme Precision Master)
- **The Analogy:** A meticulous, ultra-cautious gradient booster.
- **How it works:** Similar to LightGBM, but uses stricter mathematical penalties on tree depth. If a branch doesn't provide significant proof of improving accuracy, XGBoost cuts (prunes) it off immediately.
- **Verdict:** Runner-up on Ames ($R^2 = 0.9052$), world-renowned across competitive Kaggle competitions.

### 👑 5. CatBoost (The Categorical Master — Ames Champion)
- **The Analogy:** An appraiser who is a genius at understanding words and labels (neighborhood names, roof styles, kitchen grades, zoning codes).
- **How it works:** Traditional algorithms struggle with text and require complex encoding. CatBoost specializes in converting text categories into intelligent mathematical relationships on the fly without memorizing the training data.
- **Verdict:** Won **1st place on Ames Housing** with the lowest error rate ($\text{MAE} = \$14,983$, $R^2 = 0.9094$).

---

## 🏆 Tournament Leaderboard: Which Algorithm Won?

### 🇺🇸 Ames Housing Tournament (1,460 Properties, 81 Features)

| Rank | Algorithm | 5-Fold CV log-RMSE | Holdout Accuracy ($R^2$) | Average Dollar Error (MAE) | Verdict |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 🥇 | **CatBoost (Tuned)** | **0.11561** | **90.94%** | **$14,983** | 🏆 **Production Champion** |
| 🥈 | **XGBoost (Tuned)** | 0.12133 | 90.52% | $15,433 | Strong Regularization |
| 🥉 | **LightGBM (Tuned)** | 0.12240 | 90.13% | $16,309 | Fast Histogram Boosting |
| 4 | **Ridge Regression** | 0.13433 | 91.14% | $15,795 | Linear Baseline |
| 5 | **Random Forest** | 0.13740 | 89.13% | $16,588 | Bagging Baseline |

---

### 🌐 Global Housing Tournament (200,000 Properties, 40 International Cities)

| Rank | Algorithm | 5-Fold CV log-RMSE | Holdout Accuracy ($R^2$) | Average Dollar Error (MAE) | Percentage Error |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **Random Forest (Tuned)** | **0.00877** | **99.99%** | **$2,600** | **0.4%** |
| 🥈 | **CatBoost** | 0.00917 | 99.98% | $4,725 | 0.6% |
| 🥉 | **XGBoost** | 0.00963 | 99.99% | $4,266 | 0.5% |
| 4 | **LightGBM** | 0.00977 | 99.99% | $3,794 | 0.5% |

---

## 🔍 Smart Feature Engineering: Giving AI Common Sense

Raw data alone doesn't always reflect how humans evaluate homes. In `model/train.py`, we created **12 domain-specific engineered features**:

```python
# 1. Total Enclosed Footprint (Combines main floor + finished basement)
X['TotalFootprint'] = X['GrLivArea'] + X['TotalBsmtSF']

# 2. Quality × Condition Multiplier (Captures non-linear maintenance impact)
X['QualCondScore'] = X['OverallQual'] * X['OverallCond']

# 3. Fractional Bathroom Count (Weights full baths vs half baths)
X['TotalBath'] = X['FullBath'] + 0.5 * X['HalfBath']

# 4. Renovation Indicator (Flags whether a vintage home was modernized)
X['IsRemodeled'] = (X['YearRemodAdd'] != X['YearBuilt']).astype(int)

# 5. Room Density (Living space divided by total room count)
X['LivAreaPerRoom'] = X['GrLivArea'] / X['TotRmsAbvGrd']
```

These smart clues boosted our baseline linear $R^2$ accuracy from **86% to 91%**!

---

## ⚠️ Target Leakage: The "Cheating on the Test" Trap

When training machine learning models, **Target Leakage** happens when a feature accidentally contains the answer to what you are trying to predict.

In the 200,000-row Global dataset, the raw CSV contained columns named `loan_amount` and `down_payment`. Our correlation analysis revealed:
- `loan_amount` had a **0.938 correlation** ($94\%$) with `price`.
- `down_payment` had a **0.851 correlation** ($85\%$) with `price`.

### Why is this cheating?
In the real world, a mortgage lender calculates your loan amount **after** the house price is already set! If an AI uses the loan amount to predict the price, it isn't actually appraising the house — it's just reading the bank's receipt.

### The Fix (`--no-financial` Mode):
In our production pipeline, we strictly purged `loan_amount` and `down_payment`. The AI is forced to predict prices honestly using only physical, architectural, and geographic facts (square footage, rooms, year, city, country, risk indices).

---

## 📁 Codebase Walkthrough: What Every File Does

Here is an overview of the directory structure:

```
terrapulse-ml/
├── train.csv                          # Ames, Iowa historical sales (1,460 rows)
├── test.csv                           # Ames test set for Kaggle benchmarks
├── global_house_purchase_dataset.csv  # 200k global records across 13 countries
├── houseML.ipynb                      # Jupyter notebook for exploratory data analysis
├── requirements.txt                   # Python packages (scikit-learn, CatBoost, etc.)
├── README.md                          # This comprehensive guide
│
├── model/                             # 🧠 MACHINE LEARNING BACKEND
│   ├── train.py                       # Trains Ames 5-model tournament with Optuna
│   ├── train_global.py                # Trains Global 200k tournament (leakage-free)
│   ├── predict.py                     # Inference engine with waterfall attribution
│   └── artifacts/                     # Trained weights & metadata
│       ├── pipeline.joblib            # Champion CatBoost serialized pipeline (1MB)
│       ├── global_pipeline_nofin_comp.joblib # Champion Random Forest pipeline (94MB)
│       ├── feature_config.json        # Ames feature configuration & medians
│       ├── neighborhood_stats.json    # Ames spatial price baselines
│       ├── model_comparison_ames.json # Full benchmark tournament metrics
│       └── model_comparison_global_nofin.json # Global benchmark metrics
│
├── data/                              # 🗺️ GEOSPATIAL DATA
│   ├── ames_house_points.json         # 1,460 individual house coordinates & tiers
│   └── neighborhoods.geojson          # Geographic boundaries
│
└── app/                               # 💻 USER INTERFACE & WEB SERVER
    ├── server.py                      # Flask backend server
    ├── templates/
    │   └── index.html                 # 3-Section interactive web interface
    └── static/
        ├── css/style.css              # Clean dark-mode glassmorphic styling
        └── js/app.js                  # Client-side map & reactive calculation engine
```

---

## 🚀 Step-by-Step Quickstart Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/NONion15/terrapulse-ml.git
cd terrapulse-ml
pip install -r requirements.txt
```

### 2. Start the Interactive Web Application
```bash
python app/server.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser!

### 3. (Optional) Retrain the Machine Learning Models
You can run the full 5-model tournament and Bayesian hyperparameter tuning anytime:
```bash
# Train Ames Housing Models (Takes ~2 minutes with Optuna)
python model/train.py

# Train Global 200k Models (Leakage-Free Property Mode)
python model/train_global.py --no-financial
```

---

## 🎨 How the 3-Section Web App Works

The application is structured into 3 distinct, uncluttered sections:

### 1. ⚡ Quick Number-Only Predictor (Section 1)
- **Spacious Sliders & Inputs:** Adjust Quality (1-10), Living Area (sqft), Year Built, Bedrooms, Bathrooms, Garage, and Basement.
- **Live Valuation Card:** Instantly updates price with dynamic ticker animation, 95% Confidence Interval, Price/SqFt rate, and monthly mortgage calculations.
- **Archetype Presets:** 1-click presets for *🏰 Luxury Estate*, *🏡 Suburban Family*, *🏢 Starter Home*, and *🛠️ Value-Add*.

### 2. 🗺️ Geospatial Dot Explorer (Section 2)
- **House Dot Pointers:** Renders **1,460 individual house dots** across Ames, color-coded into 5 valuation tiers:
  - 🟢 **Value Tier** (< $140K)
  - 🔵 **Starter Tier** ($140K – $190K)
  - 🟣 **Mid Tier** ($190K – $250K)
  - 🟡 **Upper Tier** ($250K – $330K)
  - 🔴 **Luxury Tier** (> $330K)
- **Click & Hover:** Hover over any dot to view its price snapshot. Click any dot to open the side inspector and customize its specifications.
- **Global Hubs:** Switch to Global mode to explore 40 international metropolitan cities.

### 3. 📖 Machine Learning Guide (Section 3)
- An on-page educational guide explaining ML concepts, algorithm comparisons, feature engineering, and valuation breakdowns without technical jargon.

### 📄 Official Appraisal Report:
- Generate a printable **Valuation Appraisal Certificate** complete with confidence intervals, structural specs, and a component value breakdown.

---

## 📜 License & Credits
- Built with **Python 3.12**, **Flask**, **CatBoost**, **Scikit-Learn**, **Leaflet.js**, and **Optuna**.
- Created as an open-source educational project for Machine Learning in Real Estate.
