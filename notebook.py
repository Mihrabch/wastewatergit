"""
Wastewater-Based COVID-19 Hospitalization Forecasting
======================================================
6 treatment plants · 43 ZIP codes · 7–21 day predictive lag
90.80% accuracy (WWTP) · 77.27% (ZIP code) · South Carolina 2020–21

- R : spline smoothing and interpolation of raw RNA data
- Python : feature engineering, modeling, evaluation

Stack: Python · R · Scikit-learn · Statsmodels · Pandas
       Random Forest · Poisson Regression · HalvingGridSearchCV
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import HalvingGridSearchCV

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────

lag_range = [5, 7, 14, 21]

wwtp_zip_map = {
    'WWTP1': [...],
    'WWTP2': [...],
    'WWTP3': [...],
    'WWTP4': [...],
    'WWTP5': [...],
    'WWTP6': [...],
}


# ── Load Data ─────────────────────────────────────────────────────────────────

hosp_data = pd.read_csv('')

wwtp1_plm = pd.read_excel('', usecols=['Date', 'Yhatu', 'Observed_RNA'])
wwtp2_plm = pd.read_excel('', usecols=['Date', 'Yhatu', 'Observed_RNA'])
wwtp3_plm = pd.read_excel('', usecols=['Date', 'Yhatu', 'Observed_RNA'])
wwtp4_plm = pd.read_excel('', usecols=['Date', 'Yhatu', 'Observed_RNA'])
wwtp5_plm = pd.read_excel('', usecols=['Date', 'Yhatu', 'Observed_RNA'])
wwtp6_plm = pd.read_excel('', usecols=['Date', 'Yhatu', 'Observed_RNA'])

wwtp1_plm.rename(columns={'Yhatu': 'WWTP1_rna', 'Observed_RNA': 'Observed_rna_WWTP1'}, inplace=True)
wwtp2_plm.rename(columns={'Yhatu': 'WWTP2_rna', 'Observed_RNA': 'Observed_rna_WWTP2'}, inplace=True)
wwtp3_plm.rename(columns={'Yhatu': 'WWTP3_rna', 'Observed_RNA': 'Observed_rna_WWTP3'}, inplace=True)
wwtp4_plm.rename(columns={'Yhatu': 'WWTP4_rna', 'Observed_RNA': 'Observed_rna_WWTP4'}, inplace=True)
wwtp5_plm.rename(columns={'Yhatu': 'WWTP5_rna', 'Observed_RNA': 'Observed_rna_WWTP5'}, inplace=True)
wwtp6_plm.rename(columns={'Yhatu': 'WWTP6_rna', 'Observed_RNA': 'Observed_rna_WWTP6'}, inplace=True)

merged_rna = (wwtp1_plm
              .merge(wwtp2_plm, on='Date', how='outer')
              .merge(wwtp3_plm, on='Date', how='outer')
              .merge(wwtp4_plm, on='Date', how='outer')
              .merge(wwtp5_plm, on='Date', how='outer')
              .merge(wwtp6_plm, on='Date', how='outer'))

merged_rna['Date'] = pd.to_datetime(merged_rna['Date'])


# ── Spline-Smoothed RNA Trend Plot ────────────────────────────────────────────

wwtp_rna_cols      = [col for col in merged_rna.columns if '_rna' in col and 'Observed' not in col]
wwtp_display_names = {f'{w}_rna': w for w in wwtp_zip_map}

merged_rna_filled             = merged_rna.copy()
observed_cols                 = [col for col in merged_rna.columns if 'Observed' in col]
merged_rna_filled[observed_cols] = merged_rna_filled[observed_cols].fillna(0)

plt.figure(figsize=(14, 6))
wwtp_colors = {}
for col in wwtp_rna_cols:
    line, = plt.plot(merged_rna_filled['Date'], merged_rna_filled[col],
                     label=wwtp_display_names.get(col, col))
    wwtp_colors[col] = line.get_color()
for col in observed_cols:
    base = col.replace('Observed_rna_', '') + '_rna'
    plt.plot(merged_rna_filled['Date'], merged_rna_filled[col],
             linestyle='--', color=wwtp_colors.get(base, 'black'),
             label=f'{wwtp_display_names.get(base, base)} (observed)')
plt.xlabel('Date'); plt.ylabel('RNA Copies')
plt.title('WWTP RNA Trends Over Time')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


# ── prepare_wwtp_data ─────────────────────────────────────────────────────────

def prepare_wwtp_data(merged_rna, wwtp_zip_map, start_date, end_date):
    merged_rna = merged_rna[(merged_rna['Date'] >= start_date) & (merged_rna['Date'] <= end_date)]
    wwdataframe = {}
    for wwtp in wwtp_zip_map:
        wwdataframe[wwtp] = merged_rna[['Date', f'{wwtp}_rna']].copy()
    return wwdataframe


# ── process_hospitalization_data ──────────────────────────────────────────────

def process_hospitalization_data(hosp_data, start_date, end_date):
    def create_label(row):
        if row['SEX'] == 'M':   return 'M'
        elif row['SEX'] == 'F': return 'F'
        else:                   return 'U'

    hosp_data['Gender_Label'] = hosp_data.apply(create_label, axis=1)
    hospitalization_data      = hosp_data[['ZIP', 'ADMD_new', 'Gender_Label']]

    hosp_dataframes = {}
    for wwtp, zip_codes in wwtp_zip_map.items():
        filtered_hosp             = hospitalization_data[hospitalization_data['ZIP'].isin(zip_codes)].copy()
        filtered_hosp['ADMD_new'] = pd.to_datetime(filtered_hosp['ADMD_new'])
        filtered_hosp             = filtered_hosp[(filtered_hosp['ADMD_new'] >= start_date) &
                                                   (filtered_hosp['ADMD_new'] <= end_date)]
        daily_hosp                = pd.crosstab(filtered_hosp['ADMD_new'], filtered_hosp['Gender_Label']).reset_index()

        gender_cols                 = [c for c in ['F', 'M', 'U'] if c in daily_hosp.columns]
        daily_hosp['Daily_ADMD']    = daily_hosp[gender_cols].sum(axis=1)
        daily_hosp['Smoothed_ADMD'] = daily_hosp['Daily_ADMD'].copy()

        wwdataframe[wwtp]['Date']  = pd.to_datetime(wwdataframe[wwtp]['Date'])
        daily_hosp['ADMD_new']     = pd.to_datetime(daily_hosp['ADMD_new'])
        non_matching_dates         = wwdataframe[wwtp][~wwdataframe[wwtp]['Date'].isin(daily_hosp['ADMD_new'])]['Date']
        new_rows = pd.DataFrame({'ADMD_new': non_matching_dates, 'F': 0, 'M': 0, 'U': 0,
                                  'Daily_ADMD': 0, 'Smoothed_ADMD': 0})
        daily_hosp = pd.concat([daily_hosp, new_rows], ignore_index=True)
        daily_hosp = daily_hosp.sort_values('ADMD_new').reset_index(drop=True)

        daily_hosp['Smoothed_ADMD'] = daily_hosp['Smoothed_ADMD'].rolling(window=7).mean()
        for i in range(6):
            daily_hosp.loc[i, 'Smoothed_ADMD'] = daily_hosp['Daily_ADMD'].iloc[:i+1].mean()

        hosp_dataframes[wwtp] = daily_hosp
    return hosp_dataframes


# ── build_combined_wwtp_dataframe ─────────────────────────────────────────────

def build_combined_wwtp_dataframe(wwtp_zip_map, wwdataframe, hosp_dataframes):
    combined_df = pd.DataFrame()
    for wwtp in wwtp_zip_map:
        merged_df = pd.merge(
            wwdataframe[wwtp][['Date', f'{wwtp}_rna']],
            hosp_dataframes[wwtp][['ADMD_new', 'Smoothed_ADMD']],
            left_on='Date', right_on='ADMD_new', how='outer'
        ).rename(columns={f'{wwtp}_rna': f'SARS_CoV_2_{wwtp}',
                           'Smoothed_ADMD': f'Smoothed_ADMD_{wwtp}'})

        if combined_df.empty:
            combined_df = merged_df
        else:
            combined_df = pd.merge(combined_df,
                                   merged_df[['Date', f'SARS_CoV_2_{wwtp}', f'Smoothed_ADMD_{wwtp}']],
                                   on='Date', how='outer')
    return combined_df


# ── log_tranform ──────────────────────────────────────────────────────────────

def log_tranform(combined_df):
    sars_cov2_cols_train = [col for col in combined_df.columns if re.search(r'SARS_CoV_2_WWTP\d+', col)]
    combined_df[sars_cov2_cols_train] = np.log1p(combined_df[sars_cov2_cols_train].to_numpy())
    return combined_df


# ── build_train_test_dataframe ────────────────────────────────────────────────

def build_train_test_dataframe(combined_df):
    split_ratio = 0.9
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    split_idx   = int(len(combined_df) * split_ratio)
    split_date  = combined_df.loc[split_idx, 'Date'] if split_idx < len(combined_df) else combined_df['Date'].max()
    train_combined_df = combined_df[combined_df['Date'] <= split_date].copy()
    test_combined_df  = combined_df[combined_df['Date'] >  split_date].copy()
    print(f"Split Date = {split_date.date()}, Train shape = {train_combined_df.shape}, Test shape = {test_combined_df.shape}")
    return train_combined_df, test_combined_df, split_date


# ── generate_lagged_training_data ─────────────────────────────────────────────

def generate_lagged_training_data(train_combined_df, wwtp_zip_map, lag_range):
    lagged_train_data_dict = {}
    lagged_train_sars_dict = {}
    last_train_rna  = {}
    last_train_days = {}

    for lag in lag_range:
        lagged_train_df   = pd.DataFrame()
        lagged_train_dfss = pd.DataFrame()

        for wwtp in wwtp_zip_map:
            lagged_train_df[f'Lagged_Smoothed_ADMD_{wwtp}_{lag}'] = train_combined_df[f'Smoothed_ADMD_{wwtp}'].iloc[lag:].reset_index(drop=True)
            lagged_train_df['Date Hosp']                           = train_combined_df['Date'].iloc[lag:].reset_index(drop=True)
            lagged_train_dfss[f'SARS_CoV_2_{wwtp}']               = train_combined_df[f'SARS_CoV_2_{wwtp}'].iloc[:-lag].reset_index(drop=True)
            lagged_train_dfss['Date']                              = train_combined_df['Date'].iloc[:-lag].reset_index(drop=True)
            last_train_rna[(wwtp, lag)]                            = train_combined_df[f'SARS_CoV_2_{wwtp}'].iloc[-lag:].reset_index(drop=True)
            last_train_days[(wwtp, lag)]                           = train_combined_df['Date'].iloc[-lag:].reset_index(drop=True)

        lagged_train_data_dict[lag] = lagged_train_df.dropna().reset_index(drop=True)
        lagged_train_sars_dict[lag] = lagged_train_dfss.dropna().reset_index(drop=True)

    return lagged_train_data_dict, lagged_train_sars_dict, last_train_rna, last_train_days


# ── generate_lagged_test_data ─────────────────────────────────────────────────

def generate_lagged_test_data(test_combined_df, wwtp_zip_map, lag_range, last_train_rna, last_train_days):
    lagged_test_data_dict = {}
    lagged_test_sars_dict = {}

    for lag in lag_range:
        lagged_test_df   = pd.DataFrame()
        lagged_test_dfss = pd.DataFrame()

        for wwtp in wwtp_zip_map:
            lagged_test_df[f'Lagged_Smoothed_ADMD_{wwtp}_{lag}'] = test_combined_df[f'Smoothed_ADMD_{wwtp}'].reset_index(drop=True)
            lagged_test_df['Date Hosp']                           = test_combined_df['Date'].reset_index(drop=True)
            lagged_test_dfss[f'SARS_CoV_2_{wwtp}']               = pd.concat([
                last_train_rna[(wwtp, lag)],
                test_combined_df[f'SARS_CoV_2_{wwtp}'].iloc[:-lag].reset_index(drop=True)
            ]).reset_index(drop=True)
            lagged_test_dfss['Date'] = pd.concat([
                last_train_days[(wwtp, lag)],
                test_combined_df['Date'].iloc[:-lag].reset_index(drop=True)
            ]).reset_index(drop=True)

        lagged_test_data_dict[lag] = lagged_test_df.reset_index(drop=True)
        lagged_test_sars_dict[lag] = lagged_test_dfss.dropna().reset_index(drop=True)

    return lagged_test_data_dict, lagged_test_sars_dict


# ── combine_lagged_dataframes ─────────────────────────────────────────────────

def combine_lagged_dataframes(lag_range, lagged_train_data_dict, lagged_train_sars_dict,
                               lagged_test_data_dict, lagged_test_sars_dict):
    train_combined_df_for_lag = {}
    test_combined_df_for_lag  = {}
    for lag in lag_range:
        train_combined_df_for_lag[lag] = pd.concat([lagged_train_data_dict[lag], lagged_train_sars_dict[lag]], axis=1)
        test_combined_df_for_lag[lag]  = pd.concat([lagged_test_data_dict[lag],  lagged_test_sars_dict[lag]],  axis=1)
    return train_combined_df_for_lag, test_combined_df_for_lag


# ── run_models_for_all_wwtps ──────────────────────────────────────────────────

def run_models_for_all_wwtps(train_combined_df_for_lag, test_combined_df_for_lag,
                              lag_range, wwtp_zip_map, models):
    def percentage_agreement(y_test, y_pred):
        min_values = np.minimum(y_test, y_pred)
        max_values = np.maximum(y_test, y_pred)
        max_values[max_values == 0] = 1e-9
        return (min_values / max_values) * 100

    y_predslag        = {}
    y_testlag         = {}
    results_lag       = []
    poisson_params_list = []

    drop_cols = lambda lag: ['Date', 'Date Hosp'] + [f'Lagged_Smoothed_ADMD_{w}_{lag}' for w in wwtp_zip_map]

    for lag in lag_range:
        for wwtp in wwtp_zip_map:
            y_predslag.setdefault(wwtp, {}).setdefault(lag, {})
            y_testlag.setdefault(wwtp,  {}).setdefault(lag, {})

    for lag in lag_range:
        print(f'\nProcessing data for lag {lag}...\n')
        for wwtp, zip_codes in wwtp_zip_map.items():
            X_train = train_combined_df_for_lag[lag].drop(columns=drop_cols(lag))
            y_train = train_combined_df_for_lag[lag][f'Lagged_Smoothed_ADMD_{wwtp}_{lag}'].round().astype(int)
            X_test  = test_combined_df_for_lag[lag].drop(columns=drop_cols(lag))
            y_test  = test_combined_df_for_lag[lag][f'Lagged_Smoothed_ADMD_{wwtp}_{lag}'].round().astype(int)
            y_testlag[wwtp][lag] = y_test

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_predslag[wwtp][lag][name] = y_pred

                if name == 'Poisson Regression':
                    poisson_params_list.append({'WWTP': wwtp, 'Lag': lag, 'Model': name,
                                                'Intercept': model.intercept_,
                                                'Coefficients': model.coef_.tolist()})
                pa = percentage_agreement(y_test, y_pred)
                results_lag.append({'WWTP': wwtp, 'Model': name, 'Lag': lag, 'Percentage Agreement': pa})

    results_df        = pd.DataFrame(results_lag)
    poisson_params_df = pd.DataFrame(poisson_params_list)
    return results_df, y_predslag, y_testlag, poisson_params_df


# ── plot_predictions_and_hospitalizations ─────────────────────────────────────

def plot_predictions_and_hospitalizations(train_combined_df_for_lag, test_combined_df_for_lag,
                                           lagged_test_sars_dict, y_predslag, wwtp_zip_map,
                                           lag_range, models, show_rna=True):
    wwtp_display_names = {f'{w}_rna': w for w in wwtp_zip_map}

    for lag in lag_range:
        for wwtp in wwtp_zip_map.keys():
            train_dates     = train_combined_df_for_lag[lag]['Date Hosp']
            y_train         = train_combined_df_for_lag[lag][f'Lagged_Smoothed_ADMD_{wwtp}_{lag}']
            test_dates      = test_combined_df_for_lag[lag]['Date Hosp']
            y_test          = test_combined_df_for_lag[lag][f'Lagged_Smoothed_ADMD_{wwtp}_{lag}']
            test_start_date = pd.to_datetime(test_dates.iloc[0])

            fig, ax1 = plt.subplots(figsize=(14, 6))
            ax1.plot(train_dates, y_train,       label='Train Hosp', color='green',      linewidth=2)
            ax1.plot(test_dates,  y_test.values, label='Test Hosp',  color='red',        linewidth=2)
            ax1.axvline(x=test_start_date, linestyle='--', linewidth=3, color='lightgreen', label='Hosp Train/Test')

            for model_name in models.keys():
                if model_name in y_predslag[wwtp][lag]:
                    ax1.plot(test_dates, y_predslag[wwtp][lag][model_name],
                             linestyle='--', linewidth=2, label=f'{model_name}')

            rna_df     = lagged_test_sars_dict[lag]
            rna_col    = f'SARS_CoV_2_{wwtp}'
            rna_dates  = pd.to_datetime(rna_df['Date'].reset_index(drop=True))
            rna_values = rna_df[rna_col].reset_index(drop=True)

            ax2 = ax1.twinx()
            ax2.plot(rna_dates, rna_values, color='blue', alpha=0.6, linewidth=2, label='RNA Copies')
            ax2.axvline(x=rna_dates.iloc[0], linestyle=':', linewidth=3, color='grey', label='RNA Train/Test')
            ax2.set_ylabel('SARS-CoV-2 RNA Copies', color='blue', fontsize=25)
            ax2.tick_params(axis='y', labelcolor='blue', labelsize=20)

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=15)

            all_values = pd.concat([y_train, y_test] +
                                   [pd.Series(y_predslag[wwtp][lag][m]) for m in models if m in y_predslag[wwtp][lag]])
            max_val  = np.nanmax(all_values)
            interval = max_val / 4
            ax1.set_yticks(np.round(np.arange(0, max_val + interval, interval)).astype(int))
            ax1.tick_params(axis='y', labelsize=20)
            ax1.tick_params(axis='x', labelsize=15)
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

            plt.title(f'{wwtp_display_names.get(f"{wwtp}_rna", wwtp)} | Lag: {lag} | Hospitalizations and RNA Split', fontsize=30)
            ax1.set_xlabel('Date', fontsize=20)
            ax1.set_ylabel('Hospitalizations', fontsize=20)
            ax1.grid(True)
            fig.tight_layout()
            plt.show()


# ── evaluate_weekly_percentage_agreement ──────────────────────────────────────

def evaluate_weekly_percentage_agreement(y_predslag, y_testlag, lag_range, wwtp_zip_map, models):
    def percentage_agreement(y_test, y_pred):
        min_value = min(y_test, y_pred)
        max_value = max(y_test, y_pred)
        if max_value == 0: max_value = 1e-9
        return (min_value / max_value) * 100

    export_data        = []
    y_pred_weekly_sums = {}

    for lag in lag_range:
        for wwtp, zip_codes in wwtp_zip_map.items():
            for model_name in models:
                y_pred_model      = pd.Series(np.ravel(y_predslag[wwtp][lag][model_name]))
                y_test_wwtp       = pd.Series(np.ravel(y_testlag[wwtp][lag]))
                y_pred_weekly_sum = y_pred_model.groupby(y_pred_model.index // 7).sum()
                y_test_weekly_sum = y_test_wwtp.groupby(y_test_wwtp.index   // 7).sum()

                y_pred_weekly_sums.setdefault(wwtp, {}).setdefault(lag, {})[model_name] = y_pred_weekly_sum

                for week in range(len(y_test_weekly_sum)):
                    pa_week = percentage_agreement(y_test_weekly_sum.iloc[week], y_pred_weekly_sum.iloc[week])
                    export_data.append({'WWTP': wwtp, 'Model': model_name, 'Lag': lag,
                                        'Week': week + 1, 'Percentage Agreement': pa_week})

    export_df = pd.DataFrame(export_data)
    return export_df, y_pred_weekly_sums


# ── summarize_model_performance ───────────────────────────────────────────────

def summarize_model_performance(export_df,
                                 models_to_plot=['Poisson Regression', 'Random Forest'],
                                 output_prefix='WWTP_Model',
                                 save_excel=True,
                                 save_plot=True,
                                 plot_filename='median_percentage_agreement_by_lag.png'):
    wwtp_model_avg_pa = export_df.groupby(['WWTP', 'Model', 'Lag'])['Percentage Agreement'].mean().reset_index()
    wwtp_model_avg_pa['Percentage Agreement'] = wwtp_model_avg_pa['Percentage Agreement'].round(2)

    model_summary_from_wwtp_avg = wwtp_model_avg_pa.groupby(['Model', 'Lag'])['Percentage Agreement'].agg(
        Min='min', Q1=lambda x: x.quantile(0.25), Mean='mean',
        Median='median', Q3=lambda x: x.quantile(0.75), Max='max'
    ).reset_index()
    model_summary_from_wwtp_avg['IQR'] = model_summary_from_wwtp_avg['Q3'] - model_summary_from_wwtp_avg['Q1']
    model_summary_from_wwtp_avg        = model_summary_from_wwtp_avg.round(2)

    filtered_stats = model_summary_from_wwtp_avg[model_summary_from_wwtp_avg['Model'].isin(models_to_plot)]
    pivot_df       = filtered_stats.pivot(index='Lag', columns='Model', values='Median')

    plt.figure(figsize=(10, 6))
    for model in models_to_plot:
        plt.plot(pivot_df.index, pivot_df[model], marker='o', label=model)
    plt.title('Median Percentage Agreement by Lag')
    plt.xlabel('Lag (days)')
    plt.ylabel('Median Percentage Agreement (%)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

    return wwtp_model_avg_pa, model_summary_from_wwtp_avg


# ── tune_random_forest_with_percentage_agreement ──────────────────────────────

def tune_random_forest_with_percentage_agreement(train_combined_df_for_lag, test_combined_df_for_lag,
                                                  wwtp_zip_map, lag_range):
    def percentage_agreement(y_true, y_pred):
        min_vals = np.minimum(y_true, y_pred)
        max_vals = np.maximum(y_true, y_pred)
        return np.mean(min_vals / max_vals) * 100

    pa_scorer     = make_scorer(percentage_agreement, greater_is_better=True)
    rf_param_grid = {'n_estimators': [1500, 2000, 3000, 4000, 5000], 'max_depth': [10, None]}
    results_lag   = []
    best_rf_params = {}

    for lag in lag_range:
        print(f'\nProcessing data for lag {lag}...\n')
        for wwtp, zip_codes in wwtp_zip_map.items():
            feature_cols = [col for col in train_combined_df_for_lag[lag].columns
                            if 'Lagged_Smoothed_ADMD_' not in col and col not in ['Date', 'Date Hosp']]
            X_train = train_combined_df_for_lag[lag][feature_cols]
            y_train = train_combined_df_for_lag[lag][f'Lagged_Smoothed_ADMD_{wwtp}_{lag}'].round().astype(int)

            rf_search = HalvingGridSearchCV(RandomForestRegressor(random_state=42),
                                            rf_param_grid, scoring=pa_scorer, cv=5, factor=2, n_jobs=-1)
            rf_search.fit(X_train, y_train)
            results_lag.append({'wwtp': wwtp, 'lag': lag})
            best_rf_params[(wwtp, lag)] = rf_search.best_params_

    return results_lag, best_rf_params


# ── Running Functions ─────────────────────────────────────────────────────────

start_date = ''
end_date   = ''

wwdataframe     = prepare_wwtp_data(merged_rna, wwtp_zip_map, start_date, end_date)
hosp_dataframes = process_hospitalization_data(hosp_data, start_date, end_date)
combined_df     = build_combined_wwtp_dataframe(wwtp_zip_map, wwdataframe, hosp_dataframes)
combined_dff    = log_tranform(combined_df)

train_combined_df, test_combined_df, split_date = build_train_test_dataframe(combined_dff)

lagged_train_data_dict, lagged_train_sars_dict, last_train_rna, last_train_days = generate_lagged_training_data(
    train_combined_df=train_combined_df, wwtp_zip_map=wwtp_zip_map, lag_range=lag_range)

lagged_test_data_dict, lagged_test_sars_dict = generate_lagged_test_data(
    test_combined_df=test_combined_df, wwtp_zip_map=wwtp_zip_map, lag_range=lag_range,
    last_train_rna=last_train_rna, last_train_days=last_train_days)

train_combined_df_for_lag, test_combined_df_for_lag = combine_lagged_dataframes(
    lag_range, lagged_train_data_dict, lagged_train_sars_dict,
    lagged_test_data_dict, lagged_test_sars_dict)

models = {
    'Random Forest'     : RandomForestRegressor(random_state=42),
    'Poisson Regression': PoissonRegressor()
}

results_df, y_predslag, y_testlag, poisson_params_df = run_models_for_all_wwtps(
    train_combined_df_for_lag, test_combined_df_for_lag, lag_range, wwtp_zip_map, models)

plot_predictions_and_hospitalizations(
    train_combined_df_for_lag=train_combined_df_for_lag,
    test_combined_df_for_lag=test_combined_df_for_lag,
    lagged_test_sars_dict=lagged_test_sars_dict,
    y_predslag=y_predslag,
    wwtp_zip_map=wwtp_zip_map,
    lag_range=lag_range,
    models=models,
    show_rna=True)

export_df, y_pred_weekly_sums = evaluate_weekly_percentage_agreement(
    y_predslag=y_predslag, y_testlag=y_testlag,
    lag_range=lag_range, wwtp_zip_map=wwtp_zip_map, models=models)

wwtp_model_avg_pa, model_summary_from_wwtp_avg = summarize_model_performance(
    export_df,
    models_to_plot=['Poisson Regression', 'Random Forest'],
    output_prefix='WWTP_Model',
    plot_filename='median_percentage_agreement_by_lag.png')
