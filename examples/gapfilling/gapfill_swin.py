"""
=====================================
SW_IN Gap-Filling (Physics + XGBoost)
=====================================

Gap-fill shortwave incoming radiation with physics-aware partitioning
combined with XGBoost.

Nighttime gaps are set to zero (no solar radiation after sunset). Daytime
gaps are filled with XGBoost. The two parts are assembled into one complete,
physically consistent time series.

Three configurations are run in turn, each differing from the first by exactly
one setting, and each reporting its own features as it goes:

1. the defaults, with no context driver -- the climatology ceiling;
2. the same, plus ``reduce_features=True``;
3. the same, plus a second radiation sensor via ``context_df``.

They are compared at the end on withheld daytime-gap RMSE computed at runtime.

Part of the diive library: https://github.com/holukas/diive
"""

# %%
# The climatology ceiling
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# SW_IN is exactly zero at night. ``SWINGapFillerXGBoost`` uses potential
# radiation (``SW_IN_POT``, from lat/lon) to split the record into daytime and
# nighttime, sets nighttime gaps to zero, and fills daytime gaps with XGBoost.
#
# With no context drivers, every feature the model sees is a deterministic
# function of the timestamp: SW_IN_POT, the timestamp features and the record
# number are all fixed once the time is known. The model can therefore only
# reproduce a climatology -- the expected SW_IN for that time of day and year.
# It cannot know whether a gap was overcast or clear, because nothing in its
# inputs carries the sky state. More timestamp-derived features (lags, rolling
# means, EMAs of SW_IN_POT) do not help: they are the same function again.
#
# What breaks the ceiling is a second radiation measurement passed through
# ``context_df``. In preference order: a co-located sensor (pyranometer or
# PPFD) seeing the same sky, a nearby station's radiation, or reanalysis SW_IN
# such as ERA5-Land. All carry synoptic sky state that the timestamp cannot.

import textwrap

import matplotlib.pyplot as plt
import numpy as np

import diive as dv

# %%
# Setup
# ^^^^^
#
# The dataset carries ``Rg_f`` (the target) and ``PPFD``, a second independent
# radiation sensor at the same site, nearly gap-free in 2020.
#
# Beyond ``lat``/``lon``/``utc_offset`` the gap-filler needs nothing: it ships
# SW_IN defaults for the XGBoost hyperparameters and the feature windows.
#
# One default is worth watching. ``n_estimators`` is an upper budget, not a tree
# count: ``early_stopping_rounds`` halts boosting once the held-out error stops
# improving. Whether it gets the chance depends on the data, so each run reports
# the trees it actually built. The SHAP pass costs time linear in the tree count,
# so a config that runs to the budget is markedly slower.

SITE_LAT = 46.815  # CH-DAV Davos, Switzerland
SITE_LON = 9.855
SITE_UTC_OFFSET = 1
TARGET_COL = 'Rg_f'      # Shortwave incoming radiation (W/m2)
CONTEXT_COL = 'PPFD'     # Second radiation sensor, carries the sky state

# The class carries the SW_IN defaults for the tree budget, depth, early stopping
# and the seed, so only n_jobs is left to set here: a machine choice, not a
# modelling one.
XGB_KWARGS = dict(n_jobs=-1)

df = dv.load_exampledata_parquet()
df = df[df.index.year == 2020].copy()

# Keep an untouched copy of the target to score the fill against.
truth = df[TARGET_COL].copy()

# Introduce artificial gaps: randomly remove 15% of observed values.
rng = np.random.default_rng(seed=42)
observed_idx = df[TARGET_COL].dropna().index
gap_idx = rng.choice(observed_idx, size=int(0.15 * len(observed_idx)), replace=False)
df.loc[gap_idx, TARGET_COL] = np.nan

print(f"Records: {len(df)}")
print(f"Gaps in {TARGET_COL}: {df[TARGET_COL].isnull().sum()} "
      f"({100 * df[TARGET_COL].isnull().mean():.1f}%)")
print(f"Gaps in {CONTEXT_COL}: {df[CONTEXT_COL].isnull().sum()}")

print("\nXGBoost parameters from the class (no need to pass these):")
for key, value in dv.gapfilling.SWINGapFillerXGBoost._XGB_DEFAULTS.items():
    print(f"  {key:22s} {value}")
print("set by this example:")
for key, value in XGB_KWARGS.items():
    print(f"  {key:22s} {value}")


# %%
# Scoring, and a helper to run one configuration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The true SW_IN at every artificial gap is known, so the fill can be measured
# directly. Nighttime gaps are trivially zero, so only *daytime* gaps are
# scored -- that is where the model has to work.
#
# All three configurations go through ``run_config``, so they differ only in the
# keywords each one passes.

WIDTH = 80


def section(title):
    """Print a banner so each configuration's output is easy to find."""
    print("\n" + "=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def print_features(title, names):
    """Print a wrapped, comma-separated feature list under a counted heading."""
    print(f"\n{title} ({len(names)}):")
    print(textwrap.fill(", ".join(names), width=78,
                        initial_indent="    ", subsequent_indent="    "))


def daytime_gap_rmse(result):
    """RMSE of the fill against truth, at artificial daytime gap records."""
    swinpot = result.gapfilling_df['SW_IN_POT']
    daytime = swinpot >= 0.001
    idx = df.index.isin(gap_idx) & daytime.values
    err = result.gapfilled[idx] - truth[idx]
    return float(np.sqrt((err ** 2).mean())), int(idx.sum())


def run_config(label, **kwargs):
    """Build, run and score one SWINGapFillerXGBoost configuration."""
    gf = dv.gapfilling.SWINGapFillerXGBoost(
        series=df[TARGET_COL],
        lat=SITE_LAT,
        lon=SITE_LON,
        utc_offset=SITE_UTC_OFFSET,
        verbose=1,
        **kwargs,
        **XGB_KWARGS,
    )
    gf.run()
    r = gf.results
    rmse, n_gaps = daytime_gap_rmse(r)
    r2 = r.scores_traintest.get('r2', float('nan')) if r.scores_traintest else float('nan')
    n_features = int(len(r.feature_importances)) if r.feature_importances is not None else 0
    # Trees actually built, versus the n_estimators budget: early stopping at work.
    # gf.kwargs is the effective config -- class defaults merged with any override.
    n_trees = r.model.get_booster().num_boosted_rounds()
    budget = gf.kwargs['n_estimators']
    print(f"\n{label}: RMSE {rmse:.1f} W/m2 | held-out R2 {r2:.3f} | "
          f"{n_features} features | {n_trees} of {budget} trees "
          f"| scored over {n_gaps} records")
    return dict(label=label, rmse=rmse, r2=r2, n_features=n_features,
                n_trees=n_trees, budget=budget, interp=gf.interpolate_short_gaps,
                results=r)


# %%
# Configuration 1: the defaults
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Nothing is passed but the series and the site coordinates. This is the
# baseline the other two are measured against, and the climatology ceiling in
# practice: the model sees ``SW_IN_POT`` plus timestamp features and nothing
# else, so every input is a deterministic function of the timestamp.
#
# One default is worth naming here, because it differs in configuration 3:
# ``interpolate_short_gaps='auto'`` resolves to *on* (a 2-record limit), since
# there is no ``context_df`` to resolve short gaps better.

section("Configuration 1: defaults, no context driver")
ceiling = run_config("1 no context (ceiling)")

print(f"\ninterpolate_short_gaps='auto' resolved to: {ceiling['interp']} records")
print_features("features in the final model",
               list(ceiling['results'].feature_importances.index))


# %%
# Configuration 2: the defaults plus SHAP feature reduction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Identical to configuration 1 except for ``reduce_features=True``. This trims
# the feature list rather than adding information, so it cannot lift the
# climatology ceiling -- what it buys is a smaller, cheaper model.
#
# Reduction adds a ``.RANDOM`` column of pure noise, scores every feature by SHAP
# importance, and keeps only those clearing the noise by a margin. The cut-off is
# *not* the random column's importance itself but
#
#     threshold = random_importance + shap_threshold_factor * random_SD
#
# with ``shap_threshold_factor`` defaulting to 0.5. That is a stricter bar than
# it sounds: several features below score above ``.RANDOM`` and are still
# dropped. Lower the factor to keep more of them.

section("Configuration 2: defaults + reduce_features=True")
reduced = run_config("2 no context + reduce_features", reduce_features=True)

r_reduced = reduced['results']
fi_reduction = r_reduced.feature_importances_reduction

SHAP_THRESHOLD_FACTOR = 0.5  # the SWINGapFillerXGBoost default
random_importance = fi_reduction.loc['.RANDOM', 'SHAP_IMPORTANCE']
random_sd = fi_reduction.loc['.RANDOM', 'SHAP_SD']
threshold = random_importance + SHAP_THRESHOLD_FACTOR * random_sd

print_features("scored before reduction (incl. the .RANDOM benchmark)",
               list(fi_reduction.index))
print_features("kept", r_reduced.accepted_features)
print_features("dropped", r_reduced.rejected_features)

print(f"\n.RANDOM importance {random_importance:.2f}, SD {random_sd:.2f}"
      f"  ->  keep threshold = {random_importance:.2f} + "
      f"{SHAP_THRESHOLD_FACTOR} * {random_sd:.2f} = {threshold:.2f}")
print("\nSHAP importances at reduction time, strongest first:")
verdict = {name: ('.RANDOM (benchmark)' if name == '.RANDOM'
                  else 'kept' if name in r_reduced.accepted_features else 'dropped')
           for name in fi_reduction.index}
print(fi_reduction.assign(VERDICT=fi_reduction.index.map(verdict)).to_string())


# %%
# Configuration 3: the defaults plus a second radiation sensor
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Identical to configuration 1 except for ``context_df=df[['PPFD']]``. PPFD is a
# second, independent radiation sensor at the same site: unlike any timestamp
# feature it measures the sky state directly, which is what breaks the ceiling.
# The feature engineer derives rolling, EMA and SD variants of it, so the feature
# count roughly doubles.
#
# Note the knock-on effect: ``interpolate_short_gaps='auto'`` now resolves to
# *off*, because a strong context sensor resolves short gaps better than
# clearness-index interpolation, which would overwrite those fills.

section("Configuration 3: defaults + PPFD context sensor")
context = run_config("3 PPFD context", context_df=df[[CONTEXT_COL]])

print(f"\ninterpolate_short_gaps='auto' resolved to: {context['interp']}")
print_features("features in the final model",
               list(context['results'].feature_importances.index))

print(f"\nResult columns: {list(context['results'].gapfilling_df.columns)}")
print("\nTop 10 features by SHAP importance:")
print(context['results'].feature_importances.head(10).to_string())


# %%
# Comparison
# ^^^^^^^^^^
#
# All numbers come from this run: withheld daytime-gap RMSE (lower is better),
# the daytime model's held-out R2, the feature count, and the trees early
# stopping settled on.

runs = [ceiling, reduced, context]

section("SWINGapFillerXGBoost: feature reduction, and a second radiation sensor")
print(f"{'config':<30}{'RMSE W/m2':>11}{'held-R2':>9}{'n_feat':>8}{'trees':>8}")
print("-" * WIDTH)
for run in runs:
    print(f"{run['label']:<30}{run['rmse']:>11.1f}{run['r2']:>9.3f}"
          f"{run['n_features']:>8d}{run['n_trees']:>8d}")
print("-" * WIDTH)


def vs_ceiling(run):
    """Change in RMSE relative to the ceiling run, stated with its direction."""
    change = 100 * (run['rmse'] / ceiling['rmse'] - 1)
    return (f"RMSE {ceiling['rmse']:.1f} -> {run['rmse']:.1f} W/m2 "
            f"({abs(change):.0f}% {'higher' if change > 0 else 'lower'})")


print(f"Feature reduction: {vs_ceiling(reduced)}, "
      f"{ceiling['n_features']} -> {reduced['n_features']} features")
print(f"Second sensor:     {vs_ceiling(context)}")

maxed = [r['label'] for r in runs if r['n_trees'] >= r['budget']]
print(f"Early stopping: settled below the {ceiling['budget']}-tree budget"
      if not maxed else
      f"Early stopping never triggered for {maxed} -- n_estimators was the "
      f"binding limit (raise it, or lower early_stopping_rounds)")


# %%
# Observed versus gap-filled
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Three panels: the observed record with gaps, the no-context fill (still on
# the ceiling), and the PPFD-context fill (ceiling broken).

fig, axes = plt.subplots(1, 3, figsize=(20, 5),
                         gridspec_kw={'wspace': 0.2},
                         constrained_layout=True)

dv.plotting.HeatmapDateTime(series=df[TARGET_COL]).plot(
    ax=axes[0], zlabel=r'$\mathrm{W\ m^{-2}}$')
axes[0].set_title('Observed SW_IN\n(with gaps)', fontsize=11, fontweight='bold')

dv.plotting.HeatmapDateTime(series=ceiling['results'].gapfilled).plot(
    ax=axes[1], zlabel=r'$\mathrm{W\ m^{-2}}$')
axes[1].set_title(f"Gap-filled, no context\nRMSE {ceiling['rmse']:.0f} W/m2",
                  fontsize=11, fontweight='bold')

dv.plotting.HeatmapDateTime(series=context['results'].gapfilled).plot(
    ax=axes[2], zlabel=r'$\mathrm{W\ m^{-2}}$')
axes[2].set_title(f"Gap-filled, PPFD context\nRMSE {context['rmse']:.0f} W/m2",
                  fontsize=11, fontweight='bold')

fig.suptitle('SW_IN Gap-Filling: the second radiation sensor breaks the '
             'climatology ceiling', fontsize=13, fontweight='bold')
plt.show()

print("\nSW_IN gap-filling example complete.")
