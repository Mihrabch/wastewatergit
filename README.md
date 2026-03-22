# Wastewater-Based COVID-19 Hospitalization Forecasting

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Accepted-green?style=flat)

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| Waste Water Treatment Plants | 6 |
| ZIP Codes Covered | 43 |
| Predictive Lag Discovered | 7 – 21 days |
| Best Model | Random Forest |
| Forecast Accuracy | 90.80% (WWTP) and 77.27% (Zipcode) at 14-day ahead |
| Study Region & Period | South Carolina · 2020 – 2021 |

---

## What This Does

Predicts WWTP & ZIP-code-level COVID-19 hospitalizations from SARS-CoV-2 RNA concentrations measured in wastewater — days before hospitalizations occur.

**Pipeline:**
1. Load raw RNA wastewater data from 6 treatment plants
2. Using R - language, apply spline smoothing to denoise the signal
3. Using R - language, Interpolation for missing values
4. Using Python map each treatment plant to its corresponding ZIP codes
5. Merge with EHR hospitalization records
6. Discover optimal lag (7–21 days) between wastewater signal and hospitalizations
7. Train and compare ML models across lags
8. Evaluate with weekly percentage agreement metric

---

## Models

- Random Forest Regressor
- Poisson Regression (statsmodels)
- Hyperparameter tuning via `HalvingGridSearchCV`

---

## Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `Statsmodels` `Matplotlib` `Seaborn` 

---

## Repo Structure

```
├── notebook/
│   └── wastewater_forecasting.ipynb   # Main pipeline
├── data/                               # Input data (not included — proprietary EHR)
├── notebook.html                       # Rendered notebook (no setup needed)
└── README.md
```

---

## Dataset is restricted and can not be shared.

Open `notebook.html` directly in any browser — to understand.

---

## Author

**Mohammad Mihrab Uddin Chowdhury**  
[mihrabch.github.io](https://mihrabch.github.io) · [LinkedIn](https://linkedin.com/in/mihrab-chowdhury) · [GitHub](https://github.com/Mihrabch)
