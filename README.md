# 🏠 Colombo Residential House Price Predictor

An end-to-end machine learning project that estimates residential house prices in the Colombo District, Sri Lanka. Built with an **XGBoost** regression pipeline and deployed as an interactive **Streamlit** web app.

🌐 **Live Demo:** [Colombo House Price Predictor App](https://colombo-house-price-predictor-dduvq7fareregvlahx6wwn.streamlit.app/)
*(Free-tier Streamlit apps go to sleep when idle, so the first load may take a few seconds.)*

<!-- Add a screenshot of the app here, for example: ![App Screenshot](app_screenshot.png) -->

---

## 📌 Project Overview

House prices in Sri Lanka vary widely with location, land size and property specifications, and there are few public tools for estimating them. This project scrapes property listings from **Ikman.lk**, cleans and models the data, and serves the resulting model through a simple web interface.

**What it demonstrates:** data cleaning of messy scraped text, outlier handling, exploratory analysis, feature engineering, model comparison against baselines, cross-validation, and deployment.

### Key Features

- **Location-aware pricing:** 15 Colombo postal zones (Colombo 01-15) and 17 major suburbs (e.g., Nugegoda, Malabe, Piliyandala, Thalawathugoda), plus an "Other Colombo" group.
- **Feature engineering:** combined metrics `Total_Rooms` and `Bath_Bed_Ratio`.
- **Log-transformed target:** reduces the influence of very expensive listings and stabilises training.
- **Fair evaluation:** scores are reported on unseen test data and compared with simple baselines, not just raw model output.
- **Interactive UI:** real-time predictions through Streamlit.

---

## 📊 Dataset

| Item | Detail |
|---|---|
| Source | Property listings scraped from Ikman.lk |
| Raw listings | 2,520 |
| After cleaning and outlier removal | 2,392 |
| Modelling range (LKR 5M - 150M) | 2,248 |
| Train / test split | 1,798 / 450 (80/20, `random_state=42`) |

**Cleaning steps**
1. Extract **location** by keyword matching on the listing URL and heading.
2. Parse **land size (perches)**, **bedrooms**, **bathrooms** and **price** from text with regular expressions.
3. Drop rows without price, bedrooms or bathrooms; fill missing land size with the median of the same location.
4. Remove unrealistic records: price outside LKR 5M-500M, bedrooms/bathrooms outside 1-10.
5. Remove per-location price outliers (more than 2 standard deviations from the location mean).

> The raw scraped file is not included in this repository.

---

## 🤖 Model

| Item | Detail |
|---|---|
| Algorithm | XGBoost Regressor (in a scikit-learn `Pipeline`) |
| Target | `Price_LKR` (log-transformed with `log1p`, converted back with `expm1`) |
| Market focus | LKR 5M - 150M (mid-range market) |
| Hyperparameters | `n_estimators=400`, `learning_rate=0.03`, `max_depth=5`, `subsample=0.8`, `colsample_bytree=0.8` |

**Features**

| Feature | Type | Processing |
|---|---|---|
| `Location` | Categorical | One-hot encoded |
| `Land_Perches` | Numerical | Standard scaled |
| `Bedrooms`, `Bathrooms` | Numerical | Standard scaled |
| `Total_Rooms` | Engineered | Bedrooms + Bathrooms |
| `Bath_Bed_Ratio` | Engineered | Bathrooms / Bedrooms |

---

## 📈 Model Performance

Evaluated on a held-out test set (20%, n = 450), all values in LKR on the original price scale:

| Model | R² | MAE (LKR M) | Median APE |
|---|---|---|---|
| Baseline: overall median | -0.06 | 21.49 | 37.1% |
| Baseline: location median | 0.16 | 18.74 | 32.2% |
| Random Forest | 0.46 | 14.44 | 23.0% |
| **XGBoost (final)** | **0.49** | **14.12** | 23.5% |

**5-fold cross-validation (XGBoost):** MAE LKR 14.38M ± 0.27M, R² 0.48 ± 0.04, consistent with the test-set result.

**How to read this**
- The final model reduces error by about **34%** compared with always guessing the median price, and by about **25%** compared with guessing the median price of the house's location.
- A typical prediction is off by roughly **23% of the true price** (median absolute percentage error), so results are **rough estimates, not valuations**.
- XGBoost and Random Forest perform similarly; the gap is small relative to the variation across cross-validation folds.

<details>
<summary><b>Model development steps</b> (click to expand)</summary>

| Step | R² | MAE (LKR M) |
|---|---|---|
| Random Forest, basic features | 0.39 | 23.10 |
| Random Forest, engineered features + log target | 0.43 | 21.75 |
| XGBoost, same features | 0.46 | 21.26 |
| XGBoost, mid-range market only (final) | 0.49 | 14.12 |

The first three rows use the full price range (up to LKR 500M); the last row uses the mid-range test set, so MAE is not directly comparable across those rows. See the notebook for details.

</details>

---

## ⚠️ Limitations

- Prices are online **asking prices**, not final sale prices.
- **Floor area** was available for only 3 of 2,520 listings, so it could not be used, even though it strongly affects price.
- Many listings had no land size; these were **imputed with the location median**.
- The model is trained on **LKR 5M - 150M** only. Predictions for luxury or very cheap properties are unreliable.
- Locations are matched by keyword from listing text, so some listings fall under "Other Colombo" and occasional mislabelling is possible.
- Property age, condition, road access and nearby amenities are not captured.

**Future work:** add a "land size missing" flag, check for duplicate listings, tune hyperparameters, and use finer location features such as distance to the city centre.

---

## 📂 Project Structure

```
Colombo-house-price-predictor/
├── app.py                        # Streamlit web app
├── colombo_house_price.ipynb     # Data cleaning, EDA, modelling and evaluation
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🚀 Local Installation & Setup

```bash
git clone https://github.com/kawshaninperera1112-source/Colombo-house-price-predictor.git
cd Colombo-house-price-predictor
pip install -r requirements.txt
streamlit run app.py
```

To reproduce the analysis, open `colombo_house_price.ipynb` in Jupyter or Google Colab. It expects the raw scraped file `colombo.csv` in the working directory.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Machine Learning:** XGBoost, scikit-learn, NumPy, pandas
- **Visualisation:** Matplotlib, Seaborn
- **Web App:** Streamlit
- **Hosting:** GitHub, Streamlit Community Cloud

---

## 👩‍💻 Author

**Kawshani Perera**
BICT (Hons) undergraduate, Rajarata University of Sri Lanka

[GitHub](https://github.com/kawshaninperera1112-source) | [LinkedIn](https://www.linkedin.com/in/kawshaniperera-916b5a279)

---

*This project is for educational purposes. Predictions are estimates and should not be used as financial or property valuation advice.*
