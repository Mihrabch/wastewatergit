# Wastewater-Based COVID-19 Hospitalization Forecasting

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![R](https://img.shields.io/badge/R-Spline%20Smoothing-276DC3?style=flat&logo=r)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Accepted-green?style=flat)

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| Wastewater Treatment Plants | 6 |
| ZIP Codes Covered | 43 |
| Predictive Lag Discovered | 7 – 21 days |
| Best Model | Random Forest |
| Forecast Accuracy | 90.80% (WWTP) · 77.27% (ZIP code) at 14-day ahead |
| Study Region & Period | South Carolina · 2020 – 2021 |

---

## What This Does

Predicts WWTP & ZIP-code-level COVID-19 hospitalizations from SARS-CoV-2 RNA concentrations measured in wastewater — days before hospitalizations occur.

**Pipeline:**
1. Load raw RNA wastewater data from 6 treatment plants
2. Apply spline smoothing and interpolation for missing values *(R)*
3. Map each treatment plant to its corresponding ZIP codes *(Python)*
4. Merge with EHR hospitalization records
5. Discover optimal lag (7–21 days) between wastewater signal and hospitalizations
6. Train and compare ML models across lags
7. Evaluate with weekly percentage agreement metric

---

## Models

- Random Forest Regressor
- Poisson Regression (statsmodels)
- Hyperparameter tuning via `HalvingGridSearchCV`

---

## Stack

`R` `Python` `Pandas` `NumPy` `Scikit-learn` `Statsmodels` `Matplotlib` `Seaborn`

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

> Dataset and final version of the code are restricted and cannot be shared (EHR Data)
> Open `notebook.ipynb` to view the full pipeline.

---

## Author

**Mohammad Mihrab Uddin Chowdhury**  
[mihrabch.github.io](https://mihrabch.github.io) · [LinkedIn](https://linkedin.com/in/mihrab-chowdhury) · [GitHub](https://github.com/Mihrabch)
