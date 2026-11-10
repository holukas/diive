"""
=====================================================
XGBoost Gap-Filling with Wet Nights from Lloyd-Taylor
=====================================================

Gap-fill NEE with long-term XGBoost, then fill the wet night gaps from a
Lloyd-Taylor respiration model, and compare both fills.

An open-path gas analyser cannot measure in rain and fog. XGBoost learns these
nights only from the few humid nights the analyser did measure, and tends to
fill them too low. ``LongTermGapFillingXGBoostWetNightTS`` keeps the XGBoost fill
for all other gaps and fills the wet night gaps from a temperature response
fitted to the dry measured nights: E0 per year, the reference respiration per
month (Lloyd and Taylor 1994).

This example runs the plain ``LongTermGapFillingXGBoostTS`` and the wet night
variant with the same features and settings, and compares them in the wet night
gaps.
"""

# %%
# Setup
# =====
#
# The example data are from CH-DAV, a forest site with a closed-path analyser.
# Its wet nights are not missing more often than its dry nights, so this example
# shows how the method works and how the two fills differ, not the low bias of an
# open-path fill. Run it on your own open-path data to see that.

import matplotlib.pyplot as plt
import pandas as pd

import diive as dv
from diive.gapfilling import LongTermGapFillingXGBoostTS, LongTermGapFillingXGBoostWetNightTS

# %%
# Load and prepare data
# ^^^^^^^^^^^^^^^^^^^^^
#
# One year of half-hourly NEE, with air temperature, VPD, global radiation and
# precipitation.

TARGET_COL = 'NEE_CUT_REF_orig'
TA_COL = 'Tair_f'
PREC_COL = 'PREC_TOT_T1_25+20_1'

df = dv.load_exampledata_parquet()
df = df.loc[df.index.year == 2020, [TARGET_COL, TA_COL, 'VPD_f', 'Rg_f', PREC_COL]].copy()

print(f"Data: {len(df)} records from {df.index.min().date()} to {df.index.max().date()}")
print(f"Missing values in target: {df[TARGET_COL].isnull().sum()}")

# %%
# Mark night and wet records
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Which records count as wet is the caller's choice. Here a record is wet when it
# rained in the previous 6 hours (12 half-hours, the record itself included).
# Other choices are a wet-window signal of the analyser or an overcast sky. The
# Lloyd-Taylor model is fitted to the measured nights that are not wet (pass
# ``dry=`` to choose them yourself).

night = df['Rg_f'] < 10
wet = df[PREC_COL].rolling(12, min_periods=1).sum() > 0
gap = df[TARGET_COL].isna()

print(f"Night records:              {int(night.sum())}")
print(f"Wet night records:          {int((night & wet).sum())}")
print(f"Wet night gaps:             {int((night & wet & gap).sum())}")
print(f"Dry measured night records: {int((night & ~wet & ~gap).sum())}")

# %%
# Engineer features
# ^^^^^^^^^^^^^^^^^
#
# Both fills use the same features. Precipitation is not a feature here, it only
# marks the wet records.

engineer = dv.gapfilling.FeatureEngineer(
    target_col=TARGET_COL,
    features_lag=[-1, 1],
    vectorize_timestamps=True,
    sanitize_timestamp=False,  # Example data are already clean
)
df_engineered = engineer.fit_transform(df.drop(columns=[PREC_COL]))

print(f"Engineered features: {df_engineered.shape[1] - 1}")

# %%
# Plain XGBoost
# ^^^^^^^^^^^^^

XGB_KWARGS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    early_stopping_rounds=10,
    random_state=42,
    n_jobs=-1,
)

xgb = LongTermGapFillingXGBoostTS(input_df=df_engineered, target_col=TARGET_COL, **XGB_KWARGS)
xgb.run()

print(f"Plain XGBoost: {xgb.gapfilled_.name}")

# %%
# XGBoost with wet nights from Lloyd-Taylor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The same XGBoost fill runs first. Then the night gaps marked as wet get the
# Lloyd-Taylor estimate. A month with fewer than ``min_records`` dry measured
# nights uses the reference respiration of the year. A year with fewer than that
# keeps the XGBoost fill.

xgbwn = LongTermGapFillingXGBoostWetNightTS(
    input_df=df_engineered,
    target_col=TARGET_COL,
    wet=wet,
    ta=df[TA_COL],
    night=night,
    min_records=30,
    **XGB_KWARGS,
)
xgbwn.run()

print(f"Wet night variant: {xgbwn.gapfilled_.name}")
print(f"Wet night gaps filled from Lloyd-Taylor: {xgbwn.n_wetnight_filled_}")
print("\nLloyd-Taylor parameters per year (E0 in degC, RREF_YEAR at 15 degC):")
print(xgbwn.wetnight_params_)

# %%
# Flags
# ^^^^^
#
# The flag is 0 for measured, 1 for XGBoost, 2 for the XGBoost fallback and 3 for
# a wet night gap filled from the Lloyd-Taylor model.

flag = xgbwn.get_flag()
print(flag.value_counts().sort_index())

# %%
# Compare the two fills in the wet night gaps
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Outside the wet night gaps both fills are the same. In the wet night gaps,
# compare the mean of each fill with the measured wet and dry nights.

lt = flag == 3
gf_xgb = xgb.gapfilled_
gf_xgbwn = xgbwn.gapfilled_
measured = df[TARGET_COL]

same_elsewhere = (gf_xgb[~lt] - gf_xgbwn[~lt]).abs().max()
print(f"Largest difference outside the wet night gaps: {same_elsewhere:.2e}")

summary = pd.Series({
    'Measured, dry nights': measured[night & ~wet].mean(),
    'Measured, wet nights': measured[night & wet].mean(),
    'Plain XGBoost, wet night gaps': gf_xgb[lt].mean(),
    'Lloyd-Taylor, wet night gaps': gf_xgbwn[lt].mean(),
}, name='Mean NEE (umol m-2 s-1)')
print(summary.round(2))

# Annual sums in g C m-2 (umol CO2 m-2 s-1 to g C m-2 per half-hour)
to_gc = 0.02161926
print(f"\nAnnual NEE, plain XGBoost:     {gf_xgb.sum() * to_gc:8.1f} g C m-2")
print(f"Annual NEE, wet night variant: {gf_xgbwn.sum() * to_gc:8.1f} g C m-2")

# %%
# Monthly means in the wet night gaps
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The Lloyd-Taylor fill follows the temperature of each wet night and the
# respiration level of the dry nights of the same month.

monthly = pd.DataFrame({
    'Plain XGBoost': gf_xgb[lt],
    'Lloyd-Taylor': gf_xgbwn[lt],
}).groupby(lambda ts: ts.month).mean()

fig, ax = plt.subplots(figsize=(9, 4.5))
monthly.plot(ax=ax, marker='o', color=['#2196F3', '#F44336'])
ax.set_xlabel('Month')
ax.set_ylabel(r'NEE ($\mathrm{\mu mol\ m^{-2}\ s^{-1}}$)')
ax.set_title('Mean fill of the wet night gaps')
ax.set_xticks(range(1, 13))
fig.tight_layout()
plt.show()

# %%
# Cumulative NEE
# ^^^^^^^^^^^^^^

df_cumulative = pd.DataFrame({
    'Plain XGBoost': gf_xgb,
    'Wet nights from Lloyd-Taylor': gf_xgbwn,
}).multiply(to_gc)

dv.plotting.Cumulative(
    df=df_cumulative,
    units=r'($\mathrm{gC\ m^{-2}}$)',
    start_year=2020,
    end_year=2020,
).plot()

print("[OK] XGBoost gap-filling with wet nights from Lloyd-Taylor complete.")
