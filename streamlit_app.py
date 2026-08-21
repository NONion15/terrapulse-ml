"""
Streamlit Web Application for TerraPulse Real Estate Valuation & Geospatial ML.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
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

# Page configuration
st.set_page_config(
    page_title="TerraPulse ML - Real Estate Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
    }
</style>
""", unsafe_allow_html=True)

# App Title & Header
st.markdown('<div class="main-header">🏠 TerraPulse ML Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Real Estate Valuation & Geospatial Predictive Intelligence</div>', unsafe_allow_html=True)

# Load metadata
@st.cache_data
def load_app_data():
    try:
        ames_config = get_feature_config()
        ames_stats = get_neighborhood_stats()
        ames_defaults = get_neighborhood_defaults()
    except Exception as e:
        ames_config, ames_stats, ames_defaults = {}, {}, {}

    try:
        global_config = get_global_feature_config()
        global_stats = get_global_location_stats()
        global_defaults = get_global_location_defaults()
    except Exception as e:
        global_config, global_stats, global_defaults = {}, {}, {}

    return ames_config, ames_stats, ames_defaults, global_config, global_stats, global_defaults

ames_config, ames_stats, ames_defaults, global_config, global_stats, global_defaults = load_app_data()

# Navigation tabs
tab_ames, tab_global, tab_benchmarks, tab_about = st.tabs([
    "🏡 Ames Residential Predictor",
    "🌍 Global 40-Metropolitan Predictor",
    "📊 ML Model Benchmarks",
    "ℹ️ Architecture & Methodology"
])

# ==============================================================================
# TAB 1: Ames Predictor
# ==============================================================================
with tab_ames:
    st.subheader("Residential Property Valuation (Ames, Iowa)")
    st.caption("Powered by Tuned CatBoost Regressor with 81 Tabular Property Features")
    
    col_input, col_result = st.columns([1.2, 1], gap="large")
    
    with col_input:
        st.markdown("#### 🛠️ Property Specifications")
        
        # Neighborhood selector
        neighborhood_list = sorted(list(ames_stats.keys())) if ames_stats else ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst"]
        selected_neighborhood = st.selectbox(
            "Neighborhood Area",
            neighborhood_list,
            index=neighborhood_list.index("CollgCr") if "CollgCr" in neighborhood_list else 0,
            help="Select the specific Ames zoning neighborhood."
        )
        
        n_stat = ames_stats.get(selected_neighborhood, {})
        if n_stat:
            st.info(f"📍 **{selected_neighborhood} Market Context**: Median Sale Price **${n_stat.get('median_price', 0):,}** (Average ${n_stat.get('mean_price', 0):,}) across {n_stat.get('count', 0)} sales.")
        
        col1, col2 = st.columns(2)
        with col1:
            overall_qual = st.slider("Craftsmanship & Quality (1-10)", min_value=1, max_value=10, value=7, help="Overall material quality and finishes rating.")
            gr_liv_area = st.number_input("Living Area (Above Ground Sq Ft)", min_value=400, max_value=6000, value=1850, step=50)
            year_built = st.number_input("Construction Year", min_value=1870, max_value=2026, value=2004, step=1)
            garage_cars = st.selectbox("Garage Vehicle Capacity", [0, 1, 2, 3, 4], index=2)
            
        with col2:
            overall_cond = st.slider("Structural Condition (1-10)", min_value=1, max_value=10, value=6, help="Current physical and aesthetic condition.")
            total_bsmt = st.number_input("Total Basement Area (Sq Ft)", min_value=0, max_value=4000, value=950, step=50)
            full_baths = st.selectbox("Full Bathrooms Above Grade", [1, 2, 3, 4], index=1)
            lot_area = st.number_input("Lot Size (Sq Ft)", min_value=1000, max_value=100000, value=9200, step=100)
            
        with st.expander("Advanced Property Features"):
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                year_remod = st.number_input("Remodel / Addition Year", min_value=1950, max_value=2026, value=max(2004, year_built))
                bedrooms = st.selectbox("Bedrooms Above Ground", [1, 2, 3, 4, 5, 6], index=2)
                fireplaces = st.selectbox("Fireplaces", [0, 1, 2, 3], index=1)
            with col_adv2:
                central_air = st.selectbox("Central Air Conditioning", ["Y", "N"], index=0)
                kitchen_qual = st.selectbox("Kitchen Quality Rating", ["Ex", "Gd", "TA", "Fa"], index=1)
                bldg_type = st.selectbox("Dwelling Classification", ["1Fam", "TwnhsE", "Twnhs", "Duplex", "2fmCon"], index=0)

    with col_result:
        st.markdown("#### 🎯 ML Valuation Output")
        
        input_payload = {
            "Neighborhood": selected_neighborhood,
            "OverallQual": overall_qual,
            "OverallCond": overall_cond,
            "GrLivArea": gr_liv_area,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod,
            "TotalBsmtSF": total_bsmt,
            "GarageCars": garage_cars,
            "FullBath": full_baths,
            "BedroomAbvGr": bedrooms,
            "LotArea": lot_area,
            "Fireplaces": fireplaces,
            "CentralAir": central_air,
            "KitchenQual": kitchen_qual,
            "BldgType": bldg_type,
        }
        
        try:
            pred_result = predict(input_payload)
            predicted_price = pred_result["predicted_price"]
            conf_low = pred_result.get("confidence_interval", {}).get("lower", predicted_price * 0.92)
            conf_high = pred_result.get("confidence_interval", {}).get("upper", predicted_price * 1.08)
            price_per_sqft = predicted_price / max(1, gr_liv_area)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Estimated Market Value</div>
                <div class="metric-value">${predicted_price:,.0f}</div>
                <div style="color: #94A3B8; font-size: 0.95rem; margin-top: 0.5rem;">
                    Confidence Interval (±1 MAE): <strong>${conf_low:,.0f} – ${conf_high:,.0f}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Price per Living Sq Ft", f"${price_per_sqft:,.2f}/sq ft")
            with res_col2:
                model_name = pred_result.get("model_info", {}).get("model_name", "CatBoost Champion")
                st.metric("Inference Engine", model_name)
                
            # Attribution waterfall / table
            st.markdown("##### 🔍 Value Decomposition")
            attributions = pred_result.get("attribution", [])
            if attributions:
                attr_df = pd.DataFrame(attributions)
                attr_df["Value Impact"] = attr_df["delta"].apply(lambda d: f"+${d:,}" if d > 0 else (f"-${abs(d):,}" if d < 0 else "Base"))
                st.dataframe(
                    attr_df[["name", "detail", "Value Impact"]].rename(columns={
                        "name": "Feature Component",
                        "detail": "Input Spec",
                    }),
                    use_container_width=True,
                    hide_index=True
                )
        except Exception as e:
            st.error(f"Inference error: {e}")

# ==============================================================================
# TAB 2: Global Predictor
# ==============================================================================
with tab_global:
    st.subheader("International Real Estate Valuation (40 Global Metropolitans)")
    st.caption("Trained on 200,000 international transaction records across 13 countries")
    
    g_col_in, g_col_out = st.columns([1.2, 1], gap="large")
    
    with g_col_in:
        locations_list = sorted(list(global_stats.keys())) if global_stats else [
            "USA|New York", "USA|San Francisco", "UK|London", "France|Paris", 
            "Japan|Tokyo", "Australia|Sydney", "Canada|Toronto", "Germany|Berlin"
        ]
        
        selected_location = st.selectbox(
            "Metropolitan City",
            locations_list,
            index=0,
            help="Select city and country market."
        )
        
        g_stat = global_stats.get(selected_location, {})
        if g_stat:
            st.info(f"🌐 **{selected_location} Market Overview**: Median Sale Price **${g_stat.get('median_price', 0):,}** | Mean **${g_stat.get('mean_price', 0):,}**")
            
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            g_size = st.number_input("Total Living Footprint (Sq Ft)", min_value=200, max_value=15000, value=1200, step=50)
            g_beds = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=1)
            g_baths = st.selectbox("Bathrooms", [1, 2, 3, 4, 5], index=1)
        with g_col2:
            g_year = st.number_input("Building Vintage (Year Built)", min_value=1850, max_value=2026, value=2015, step=1)
            g_cond = st.slider("Property Condition Rating (1-5)", min_value=1, max_value=5, value=4)
            g_type = st.selectbox("Dwelling Type", ["Apartment", "Single-Family Home", "Townhouse", "Condo"], index=0)

    with g_col_out:
        st.markdown("#### 🎯 Global Valuation Output")
        
        if selected_location:
            country_val, city_val = selected_location.split("|")
            g_payload = {
                "Country": country_val,
                "City": city_val,
                "Size_sqft": g_size,
                "Bedrooms": g_beds,
                "Bathrooms": g_baths,
                "Year_Built": g_year,
                "Condition_Score": g_cond,
                "Property_Type": g_type,
            }
            
            try:
                g_result = predict_global(g_payload)
                g_price = g_result["predicted_price"]
                g_per_sqft = g_price / max(1, g_size)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Estimated Market Valuation</div>
                    <div class="metric-value">${g_price:,.0f} USD</div>
                    <div style="color: #94A3B8; font-size: 0.95rem; margin-top: 0.5rem;">
                        Market Benchmark Engine: <strong>Random Forest Ensemble (200k Records)</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Valuation / Sq Ft", f"${g_per_sqft:,.2f}/sq ft")
                
                # Currency conversions
                st.markdown("##### 💱 Foreign Exchange Context")
                fx_rates = {"EUR": 0.92, "GBP": 0.79, "JPY": 154.5, "CAD": 1.36, "AUD": 1.52}
                fx_cols = st.columns(3)
                fx_cols[0].metric("EUR (€)", f"€{g_price * fx_rates['EUR']:,.0f}")
                fx_cols[1].metric("GBP (£)", f"£{g_price * fx_rates['GBP']:,.0f}")
                fx_cols[2].metric("JPY (¥)", f"¥{g_price * fx_rates['JPY']:,.0f}")
            except Exception as e:
                st.error(f"Global inference error: {e}")

# ==============================================================================
# TAB 3: Model Benchmarks
# ==============================================================================
with tab_benchmarks:
    st.subheader("5-Model Bayesian Optimization Benchmark Results")
    st.markdown("All models evaluated with **5-Fold Cross Validation** on holdout test partitions.")
    
    ames_benchmarks = pd.DataFrame([
        {"Model": "CatBoost (Champion)", "5-Fold CV log-RMSE": 0.11561, "Holdout R²": 0.9094, "Holdout MAE": "$14,983", "Status": "Production Champion"},
        {"Model": "XGBoost", "5-Fold CV log-RMSE": 0.12133, "Holdout R²": 0.9052, "Holdout MAE": "$15,433", "Status": "Runner-Up"},
        {"Model": "LightGBM", "5-Fold CV log-RMSE": 0.12240, "Holdout R²": 0.9013, "Holdout MAE": "$16,309", "Status": "Fastest Booster"},
        {"Model": "Ridge Regression", "5-Fold CV log-RMSE": 0.13433, "Holdout R²": 0.9114, "Holdout MAE": "$15,795", "Status": "Linear Baseline"},
        {"Model": "Random Forest", "5-Fold CV log-RMSE": 0.13740, "Holdout R²": 0.8913, "Holdout MAE": "$16,588", "Status": "Bagging Baseline"},
    ])
    
    st.dataframe(ames_benchmarks, use_container_width=True, hide_index=True)
    
    st.markdown("#### 📈 Accuracy Comparison (Holdout R²)")
    chart_data = pd.DataFrame({
        "Model": ames_benchmarks["Model"],
        "R-Squared": [0.9094, 0.9052, 0.9013, 0.9114, 0.8913]
    }).set_index("Model")
    st.bar_chart(chart_data)

# ==============================================================================
# TAB 4: About
# ==============================================================================
with tab_about:
    st.subheader("System Architecture & Implementation")
    st.markdown("""
    - **Modeling Core:** Supervised non-linear regression using CatBoost with symmetric decision trees, Target Encoding, and Bayesian hyperparameter tuning.
    - **Target Leakage Prevention:** Strict cross-validation partitioning ensures zero data leakage between train and holdout splits.
    - **Confidence Estimation:** Calibrated holdout MAE bands calculate transparent valuation ranges.
    """)

# Footer
st.markdown("---")
st.caption("TerraPulse Real Estate Machine Learning Platform • Powered by Streamlit & Scikit-Learn")
