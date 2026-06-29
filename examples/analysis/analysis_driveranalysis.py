"""
================================
Driver Attribution by Epistemic Level (EXPERIMENTAL)
================================

.. note::

    EXPERIMENTAL feature. ``DriverAnalysis`` lives in the
    ``dv.analysis.experimental`` subnamespace, not the stable ``dv.analysis`` API,
    and its interface may change without a deprecation cycle. Instantiating it
    emits a one-time ``ExperimentalWarning``.

Attribute a flux time series to its candidate drivers and read the evidence by
*epistemic level* — association, temporal prediction, causation — instead of as a
single importance ranking.

``DriverAnalysis`` triangulates several methods on top of one shared, time-aware,
eddy-covariance-correct preprocessing pass:

- **Association (Layer 1):** SHAP importance (vs a ``.RANDOM`` benchmark) and
  correlation-robust ALE response curves.
- **Temporal prediction (Layer 2):** lagged importance (response timescale),
  scale-resolved importance (STL components and temporal aggregations), and
  regime-stratified importance.
- **Causation (Layer 3, opt-in):** a deseasonalized Granger sanity check.
  (Granger is predictive, not causal evidence on its own.)

The headline output is the convergence/divergence summary: where methods agree
you can trust the attribution, and where they disagree is the scientific signal.
SHAP and ALE are association-level diagnostics and are never presented as causal.

Best for: deciding which environmental variables actually drive a flux — and how
confident the evidence lets you be.
"""

# %%
# Load and prepare data
# ^^^^^^^^^^^^^^^^^^^^^^

import diive as dv

df = dv.load_exampledata_parquet()

# Keep one growing season to keep the example fast; in practice use more data.
df = df.loc[(df.index.year == 2020) & (df.index.month >= 5) & (df.index.month <= 5)].copy()

# Target: original (unfilled) net ecosystem exchange. Drop low-quality records so
# we attribute observed fluxes, not gap-filled ones.
target = df['NEE_CUT_REF_orig'].copy()
target[df['QCF_NEE'] > 0] = float('nan')
target.name = 'NEE'

# Candidate drivers: air temperature, VPD, global radiation, soil water content,
# plus a deliberately irrelevant column to show what "not a driver" looks like.
drivers = df[['Tair_f', 'VPD_f', 'Rg_f', 'SWC_FF0_0.15_1']].copy()
drivers.columns = ['TA', 'VPD', 'SW_IN', 'SWC']

print(f"Target: {target.notna().sum()} valid records; {len(drivers.columns)} drivers")

# %%
# Run the association + temporal layers
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``lags`` asks the temporal layer for the per-driver response timescale across
# the previous 6 hours (12 half-hourly records). Bootstrapping reports ranking
# stability (SHAP fluctuates a few percent between fits).

da = dv.analysis.experimental.DriverAnalysis(
    target=target,
    drivers=drivers,
    model='rf',
    lags=list(range(-12, 1)),
    n_bootstrap=5,
    verbose=2,
).run(levels=('static', 'temporal'))

# %%
# Convergence/divergence summary
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Per-driver verdict, sorted by verdict then SHAP rank. Diverging rows (where
# methods contradict each other) are where the science is.

da.summary()

# %%
# The synthesis table in full
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

conv = da.results.convergence
print(conv[['shap_relevant', 'ale_direction', 'ale_relevant', 'dominant_lag',
            'timescale', 'scale_dependence', 'regime_dependence',
            'agreement', 'verdict']])

# %%
# Headline plot: convergence grid
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Driver x method relevance grid (green = relevant, grey = weak, red = not),
# with ALE direction glyphs. Disagreement across a row flags a non-robust driver.

da.plot_convergence(showplot=True)

# %%
# Layer 1: SHAP importance and an ALE response curve
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

da.plot_importance(showplot=True)

# ALE describes how the model responds to VPD (association, not causation).
ale_vpd = da.results.ale['VPD']
print(f"VPD ALE range = {ale_vpd.ale_range:.3g}, "
      f"shape = {ale_vpd.direction(da.ale_range_threshold)}")
ale_vpd.plot(showplot=True)

# %%
# Layer 1: 2D ALE for an interaction (VPD x TA)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Second-order ALE isolates the interaction: a flat surface means the two drivers
# act additively.

ale2d = da.ale_2d('VPD', 'TA')
print(f"VPD x TA interaction strength = {ale2d.interaction_strength:.3g}")
ale2d.plot(showplot=True)

# %%
# Layer 2: response-timescale fingerprint and scale-resolved importance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

da.plot_lagged(showplot=True)
da.plot_scale_resolved(showplot=True)

# %%
# Layer 3 (opt-in): deseasonalized Granger sanity check
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Granger is a cheap, caveated check — NOT causal evidence on its own. Inputs are
# STL-deseasonalized first (shared seasonality is the classic spurious-Granger
# trap).

granger = da.granger()
print(granger)
