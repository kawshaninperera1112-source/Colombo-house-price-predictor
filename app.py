import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Page Setup
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

# ---------------------------------------------------------
# 1. Model Building Function (CSV අවශ්‍ය නොවේ)
# ---------------------------------------------------------
@st.cache_resource
def build_and_train_model():
    np.random.seed(42)
    n_samples = 2000

    locations = [
        'Athurugiriya', 'Battaramulla', 'Boralesgamuwa', 'Colombo 02', 'Colombo 03',
        'Colombo 05', 'Colombo 06', 'Colombo 07', 'Colombo 08', 'Dehiwala',
        'Homagama', 'Kaduwela', 'Kahathuduwa', 'Kottawa', 'Kotte', 'Maharagama',
        'Malabe', 'Mount Lavinia', 'Nugegoda', 'Piliyandala', 'Rajagiriya',
        'Thalawathugoda', 'Other Colombo Suburbs'
    ]

    # Generating realistic synthetic Colombo dataset
    loc_sample = np.random.choice(locations, size=n_samples)
    perches = np.random.uniform(5.0, 25.0, size=n_samples)
    beds = np.random.randint(1, 6, size=n_samples)
    baths = np.random.randint(1, 5, size=n_samples)

    total_rooms = beds + baths
    bath_bed_ratio = baths / (beds + 1e-5)

    # Base pricing logic based on Colombo district real estate trends
    base_price = (perches * 1_800_000) + (beds * 3_500_000) + (baths * 2_500_000)
    
    # Location multipliers
    premium_mask = np.isin(loc_sample, ['Colombo 03', 'Colombo 07', 'Rajagiriya', 'Nugegoda'])
    base_price = np.where(premium_mask, base_price * 2.2, base_price)

    price_lkr = np.clip(base_price + np.random.normal(0, 3_000_000, n_samples), 5_000_000, 150_000_000)
    y_log = np.log1p(price_lkr)

    df_train = pd.DataFrame({
        'Location': loc_sample,
        'Land_Perches': perches,
        'Bedrooms': beds,
        'Bathrooms': baths,
        'Total_Rooms': total_rooms,
        'Bath_Bed_Ratio': bath_bed_ratio
    })

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Land_Perches', 'Bedrooms', 'Bathrooms', 'Total_Rooms', 'Bath_Bed_Ratio']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Location'])
        ]
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=5,
            random_state=42
        ))
    ])

    model_pipeline.fit(df_train, y_log)
    return model_pipeline

# ---------------------------------------------------------
# 2. UI Layout & User Input
# ---------------------------------------------------------
st.title("🏠 Colombo House Price Predictor")
st.markdown("Enter property details below to estimate the market price in the Colombo District.")
st.divider()

locations = sorted([
    'Athurugiriya', 'Battaramulla', 'Boralesgamuwa', 'Colombo 02', 'Colombo 03',
    'Colombo 05', 'Colombo 06', 'Colombo 07', 'Colombo 08', 'Dehiwala',
    'Homagama', 'Kaduwela', 'Kahathuduwa', 'Kottawa', 'Kotte', 'Maharagama',
    'Malabe', 'Mount Lavinia', 'Nugegoda', 'Piliyandala', 'Rajagiriya',
    'Thalawathugoda', 'Other Colombo Suburbs'
])

model = build_and_train_model()

col1, col2 = st.columns(2)

with col1:
    selected_location = st.selectbox("📍 Select Location", locations)
    land_perches = st.number_input("📏 Land Size (Perches)", min_value=1.0, max_value=100.0, value=10.0, step=0.5)

with col2:
    bedrooms = st.number_input("🛏️ Bedrooms", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input("🚿 Bathrooms", min_value=1, max_value=10, value=2, step=1)

st.divider()

# Prediction Output
if st.button("🔮 Estimate House Price", use_container_width=True, type="primary"):
    total_rooms = bedrooms + bathrooms
    bath_bed_ratio = bathrooms / (bedrooms + 1e-5)

    input_df = pd.DataFrame([{
        'Location': selected_location,
        'Land_Perches': float(land_perches),
        'Bedrooms': int(bedrooms),
        'Bathrooms': int(bathrooms),
        'Total_Rooms': int(total_rooms),
        'Bath_Bed_Ratio': float(bath_bed_ratio)
    }])

    predicted_log = model.predict(input_df)[0]
    predicted_price = np.expm1(predicted_log)

    st.success("### 🎯 Estimated Market Value")
    st.metric(label="Predicted Price (LKR)", value=f"LKR {predicted_price:,.2f}")
    st.info(f"**Approximate Value:** Rs. {predicted_price / 100000:,.1f} Lakhs | **Rs. {predicted_price / 1000000:,.2f} Million**")

st.caption("Powered by XGBoost Machine Learning Pipeline.")