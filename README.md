# Wastewater-Based COVID-19 Hospitalization Forecasting

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Under%20Accepted-green?style=flat)

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| Treatment Plants | 6 |
| ZIP Codes Covered | 43 |
| Predictive Lag Discovered | 7 – 21 days |
| Forecast Error Reduction | ↓ 18% vs baseline |
| Study Region & Period | South Carolina · 2020 – 2021 |

---

## What This Does

Predicts ZIP-code-level COVID-19 hospitalizations from SARS-CoV-2 RNA concentrations measured in wastewater — days before hospitalizations occur.

**Pipeline:**
1. Load raw RNA wastewater data from 6 treatment plants
2. Apply spline smoothing to denoise the signal
3. Map each treatment plant to its corresponding ZIP codes
4. Merge with EHR hospitalization records
5. Discover optimal lag (7–21 days) between wastewater signal and hospitalizations
6. Train and compare ML models across lags
7. Evaluate with weekly percentage agreement metric

---

## Models

- Random Forest Regressor
- Gradient Boosting Regressor
- Poisson Regression (statsmodels)
- Voting Ensemble
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

## View Without Running

Open `notebook.html` directly in any browser — no Python setup needed.

---

## Author

**Mohammad Mihrab Uddin Chowdhury**  
[mihrabch.github.io](https://mihrabch.github.io) · [LinkedIn](https://linkedin.com/in/mihrab-chowdhury) · [GitHub](https://github.com/Mihrabch)
