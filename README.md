# 🏠 Colombo Residential House Price Predictor

An end-to-end Machine Learning web application designed to estimate residential property prices in the Colombo District, Sri Lanka. Built using an optimized **XGBoost Regressor** pipeline and deployed interactively with **Streamlit**.

🌐 **Live Demo:** [Colombo House Price Predictor App](https://colombo-house-price-predictor-dduvq7fareregvlahx6wwn.streamlit.app/)

---

## 📌 Project Overview

Predicting real estate prices in Sri Lanka can be challenging due to wide variations in location value, property specs, and market dynamics. This project leverages historical scraped property listings (from Ikman.lk) to train a machine learning regression model that accurately predicts home market values based on key attributes.

### Key Features
- **Location-Aware Pricing:** Covers major Colombo suburbs and prime urban zones (e.g., Colombo 03, Colombo 07, Rajagiriya, Maharagama, etc.).
- **Feature-Engineered Pipeline:** Incorporates combined domain metrics such as `Total_Rooms` and `Bath_Bed_Ratio`.
- **Log-Transformed Target:** Handles price skewness using logarithmic scale transformation during training for stable predictions.
- **Interactive UI:** Simple, real-time predictions powered by Streamlit.

---

## 📊 Dataset & Model Architecture

- **Domain:** Colombo Real Estate Market
- **Features Used:**
  - `Location` (Categorical - One-Hot Encoded)
  - `Land_Perches` (Numerical - Standard Scaled)
  - `Bedrooms` & `Bathrooms` (Numerical)
  - `Total_Rooms` & `Bath_Bed_Ratio` (Engineered Features)
- **Target Variable:** `Price_LKR` (Filtered for mid-range market: 5M – 150M LKR)
- **Model:** **XGBoost Regressor**
  - Evaluation Metrics: Log $R^2 \approx 0.49$, MAE $\approx 14.1\text{M LKR}$

---

## 🛠️ Tech Stack

- **Language:** Python 3.13
- **Machine Learning:** XGBoost, Scikit-Learn, NumPy, Pandas
- **Web Framework:** Streamlit
- **Version Control & Hosting:** Git, GitHub, Streamlit Community Cloud

---

## 🚀 Local Installation & Setup

To run this project locally on your machine, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/kawshaninperera1112-source/Colombo-house-price-predictor.git](https://github.com/kawshaninperera1112-source/Colombo-house-price-predictor.git)
   cd Colombo-house-price-predictor
