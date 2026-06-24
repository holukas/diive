"""
======================
Select Records By Condition
======================

Select records of one variable based on the value of another variable.

Given a target variable and a condition variable, keep only the target records
where the condition variable falls within a chosen [lower, upper] range. This is
a non-destructive conditional subselection (the input is never modified).

Best for: Isolating a variable under specific conditions, e.g. CO2 fluxes only
during warm periods, or radiation only on high-VPD days.
"""

# %%
# Load data
# ^^^^^^^^^

import diive as dv

df = dv.load_exampledata_parquet()

# %%
# Keep NEE only where air temperature is between 10 and 20 degrees C
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Default (set_to_nan=True) keeps the full time index; out-of-range
# records become NaN, so the regular 30-minute spacing is preserved.
nee_warm = dv.keep_records_where(
    df, target='NEE_CUT_REF_f', condition_var='Tair_f',
    lower=10, upper=20, verbose=True,
)

print(f"\nKept {nee_warm.notna().sum()} of {len(nee_warm)} records.")

# %%
# Drop the non-matching records instead of masking them
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# With set_to_nan=False only the matching records are returned.
nee_warm_dropped = dv.keep_records_where(
    df, target='NEE_CUT_REF_f', condition_var='Tair_f',
    lower=10, upper=20, set_to_nan=False,
)

print(f"\nResult length after dropping: {len(nee_warm_dropped)}")

# %%
# Use an open (one-sided) range
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Omit a bound to leave that side open: here, NEE wherever Tair_f >= 20.
nee_hot = dv.keep_records_where(
    df, target='NEE_CUT_REF_f', condition_var='Tair_f', lower=20,
)

print(f"\nMean NEE where Tair_f >= 20: {nee_hot.mean():.3f}")

# %%
# Visualize which records were selected
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Plot a short window so individual selected points are visible. The top panel
# shows the condition variable with the [lower, upper] band shaded; the bottom
# shows the target. Markers highlight the records where the condition variable
# falls inside the band, making the selection rule obvious.

import matplotlib.pyplot as plt

# A narrow band slices through the diurnal temperature wave, so the selected
# records form a clear, readable subset rather than nearly everything.
lower, upper = 15, 20
window = df.loc['2018-06-15':'2018-06-22']

# Selected records: condition variable within [lower, upper] (set_to_nan keeps
# the index aligned, so the non-NaN positions mark the selection).
selected = dv.keep_records_where(
    window, target='NEE_CUT_REF_f', condition_var='Tair_f',
    lower=lower, upper=upper,
)
mask = selected.notna()

fig, (ax_cond, ax_target) = plt.subplots(
    2, 1, figsize=(16, 8), sharex=True, constrained_layout=True)

# Condition variable with the selection band
ax_cond.plot(window.index, window['Tair_f'], color='#90A4AE', lw=1, label='Tair_f')
ax_cond.axhspan(lower, upper, color='#FFC107', alpha=0.2,
                label=f'band [{lower}, {upper}] C')
ax_cond.scatter(window.index[mask], window['Tair_f'][mask],
                color='#F44336', s=18, zorder=5, label='selected')
ax_cond.set_ylabel('Tair_f (C)')
ax_cond.set_title('Condition variable: records inside the band are selected')
ax_cond.legend(loc='upper right')

# Target variable, same records marked
ax_target.plot(window.index, window['NEE_CUT_REF_f'], color='#90A4AE', lw=1,
               label='NEE_CUT_REF_f')
ax_target.scatter(window.index[mask], window['NEE_CUT_REF_f'][mask],
                  color='#F44336', s=18, zorder=5, label='selected')
ax_target.set_ylabel('NEE_CUT_REF_f')
ax_target.set_title('Target variable: only the selected records are kept')
ax_target.legend(loc='upper right')

fig.show()
