# Flux post-processing chain: methods

This page explains the methods of the flux post-processing chain (`diive.flux.postprocessingchain`)
in more detail than the docstrings: what each step does, why, and what to check. The functions and
their arguments are in the {doc}`API reference <api_reference>`.

The chain follows the levels of the Swiss FluxNet flux processing chain:

| Level | What it does | Function |
|---|---|---|
| L2 | Quality flags from the EddyPro output | `run_level2` |
| L3.1 | Adds the storage term | `run_level31` |
| L3.2 | Removes outliers | `make_level32_detector`, `run_level32` |
| L3.3 | Removes records with weak turbulence (USTAR filter) | `run_level33_constant_ustar` |
| L4.1 | Fills the gaps | `run_level41_xgb`, `run_level41_rf`, `run_level41_mds` |
| L4.2 | Partitions NEE into GPP and RECO | `run_level42_partitioning` |
| | Random, USTAR and joint uncertainty | `RandomUncertaintyPAS20`, `JointUncertaintyPAS20` |

Each step takes the chain container and returns a new one, so a step can be repeated without
running the earlier ones again.

(storage-term)=
## Level 3.1: storage term

The storage term is added to the measured flux (`FC + SC_SINGLE` gives `NEE`). A single-point
storage term is the change in the concentration at the measurement height between two consecutive
records, times the height of the air column below. It is only as good as the concentration, so
`run_level31` can screen it in two steps before it is added. The values both steps remove are
filled with a rolling median, as missing values are, so every measured flux keeps a storage term.
The flag column `FLAG_..._ISFILLED` records what happened to each value: 0 measured, 1 filled
because missing, 2 replaced by a spike block mean, 3 discarded next to a gap and filled.

(storage-restarts)=
### Next to gaps in the concentration

Option `concentration_col`, e.g. `'CO2_MOLAR_DENSITY'`.

After a signal loss an analyser can restart with readings far from the real concentration and
return to it over several hours. Before a signal loss it can drift away in the same way. The
storage term needs the concentration of two consecutive records, so the jump across the gap is
never counted, while the return or the drift is: it enters the storage term as a large flux of one
sign. Over years these one-sided changes add up to a storage term that no longer sums to about
zero, and they shift the annual sums of the flux.

For each record next to a gap, the screening asks whether the concentration is normal: within
`n_sigma` (3) robust standard deviations (1.4826 times the median absolute deviation) of the median
concentration at the same time of day, over `window_days` (7) days on either side. After a gap, the
storage values up to and including the first normal record are discarded: the first normal record
still uses the concentration before it. Before a gap (`before_gaps`, on by default), the records
outside the band are discarded. At most `max_records` (48) records are discarded per gap. Settings
go into `restart_screening`.

Check the result with `plot_screening`: the cumulative storage term at measured fluxes should
stay near zero over a year. What remains of a drift usually lies in periods where the analyser was
failing for weeks; the fluxes of such periods are removed by the quality flags anyway, so their
storage term does not enter the flux.

(storage-spikes)=
### Spikes

Option `screen_storage_term=True`.

A spike in the storage term would turn a good flux record into an outlier that Level 3.2 removes,
so the flux record would be lost. The spikes are found with a Hampel filter on the double
differences of the storage term (13 days, 5.5 sigma, one pass). Only isolated spikes are replaced:
a run of more than `max_run` (3) flagged values is a noisy period, not a spike, and stays as
measured. Each run, together with one record on either side, is replaced by the mean of its
measured values. The storage term is the change in concentration, so the sum over such a block is
the real concentration change across it; keeping the sum keeps the storage term balanced, and a
single spike is spread over three records.

In `plot_screening`, the mean daily cycle should stay positive at night and dip in the morning,
when the CO2 that built up overnight is released. The spikes should spread over the day with both
signs; a cluster of negative spikes in the morning would mean that real release was replaced.

(outlier-tests)=
## Level 3.2: outlier tests

`make_level32_detector` returns a `StepwiseOutlierDetection` for the flux. Each test draws its
plot first and removes nothing until it is accepted with `addflag()`. A record counts for the
first accepted test that removed it, and each test runs on what the earlier ones kept.
`showplot_overview` shows all removals by test, month and hour.

(abslim)=
### Absolute limits

`flag_outliers_abslim_test` removes values outside fixed limits. It catches extreme values in
periods that the Hampel filters cannot judge, because the bad values fill their window.

(hampel-double-differences)=
### Hampel filter on double differences

`flag_outliers_hampel_test(use_differencing=True)` compares each record with its neighbours: the
double difference of a record is large when the record differs from both. A record is removed
when its double difference lies more than `n_sigma` robust standard deviations from the median of
the window, similar to the despiking of Papale et al. (2006). It finds single spikes. A record whose neighbours are missing has no double difference
and is not tested, so runs of two values between gaps pass.

(hampel-time-of-day)=
### Hampel filter per time of day

`flag_outliers_hampel_test(by_time_of_day=True)` compares each value with the values of the same
half-hour, and of `time_of_day_slots` half-hours before and after it, over the window. It finds
values that are wrong for their time of day, such as noon-like uptake at sunrise, and runs of high
values that the double differences do not catch. A record with fewer than `min_records` values in
its pool is not tested.

(negative-nighttime)=
### Negative values at night

At night there is no photosynthesis, so the flux is respiration and its true value is zero or
above. Random noise around a small respiration still pushes some records below zero, so a negative
value is plausible if it lies within a few times the noise of zero. Removing every negative value
would cut off one side of the noise and raise the nighttime flux. `flag_outliers_negative_night_test`
therefore removes only nighttime values below `-z` times an uncertainty. Daytime values are not
tested, because by day both signs are real, and high nighttime values are real respiration.

`nighttime_uncertainty` gives four uncertainties to choose from:

- `FS`: the uncertainty EddyPro reports for each half-hour. It covers only the sampling error of
  that half-hour's covariance (Finkelstein and Sims 2001), so it is small.
- `PAS20`: the random uncertainty of FLUXNET2015 and ONEFlux (Pastorello et al. 2020), the spread
  of measured fluxes under similar conditions within days of each record. It also holds real
  differences between the fluxes, so it is a generous estimate of the random error.
- `MAX`: the larger of the two for each record.
- `POOLED`: the robust spread (1.4826 times the median absolute deviation) of all nighttime values
  within `pooled_days` (15) days on either side.

The per-record uncertainties fail where they are needed most. PAS20 is the spread of similar
fluxes, so in a short noisy stretch, for example while an analyser fails, the bad values widen
their own uncertainty and pass the test. The pooled spread does not depend on the records next to
a value: a few bad values hardly move the median of a month of nights. It holds the real
night-to-night variation of respiration besides the random error, so it is larger than the noise,
and a value below about four times it cannot come from noise. The plot reports how many removals
noise alone would explain.

Compute the uncertainties after the other tests are accepted, so that outliers do not widen them.

(ustar-filter)=
## Level 3.3: USTAR filter

`run_level33_constant_ustar` removes records with weak turbulence, where the friction velocity
(USTAR) is below a threshold, by day and by night, as ONEFlux does. Records without USTAR are
removed too. Unlike ONEFlux, the first record above the threshold after a period below it is kept.
Each threshold is one scenario, usually the 16th, 50th and 84th percentile of a threshold detection.
From here on each scenario has its own series, so `data.filteredseries` is empty.

(gapfilling-input)=
### Records for gap-filling and highest-quality records

After Level 3.3 every measured record that failed a test or the USTAR filter is removed. The
records left depend on the USTAR scenario and on the quality flags accepted by day and by night
(`daytime_accept_qcf_below`, `nighttime_accept_qcf_below`; with 2, QCF 0 and 1 are kept). They are
what the gap-filling starts from: columns ending in `_L3.3_<scenario>_QCF`. The highest-quality
records, QCF 0, passed every test without a warning; use them for analyses that need measured
values only: columns ending in `_L3.3_<scenario>_QCF0`. `report_gapfilling_input` and
`plot_gapfilling_input` show both, `report_gaps` the gaps left.

(xgboost-gapfilling)=
## Level 4.1: gap-filling with XGBoost

`run_level41_xgb` (and `run_level41_rf` for a random forest) gives each year its own model for each
USTAR scenario, trained on that year and its two neighbouring years. The features are the meteo
drivers and features engineered from them (lags, rolling statistics, timestamp terms).

- **Feature selection.** SHAP values select the features: a feature is kept if it is clearly more
  important than a random variable. With `shared_features=True` every scenario then uses the
  features that any scenario selected, so the scenarios differ in their records, not in their
  features.
- **Training and test.** A random 75 % of the complete records trains the model, the other 25 % tests
  it. The number of trees is either chosen once for all models by blocked cross-validation
  (`n_estimators_cv`, see below) or by early stopping for each model on 10 % of its training
  records, up to `n_estimators`.
- **Final fit.** The gaps are filled by a model refitted on all complete records with the number of
  trees early stopping chose. Its scores on all measured records are therefore in-sample; the
  held-out test scores describe the fit.
- **Fallback.** Where a driver is missing, a second model on timestamp features fills the record.

`report_model_overview` lists for each year the years it was trained on, the features, the
training and test records, the trees, the test scores and the fallback records.

(tree-selection)=
### Number of trees from blocked cross-validation

Option `n_estimators_cv=True` of `run_level41_xgb`.

After the feature selection, XGBoost is cross-validated on the training records of each yearly
pool of one USTAR scenario (the middle one by default). The folds hold out whole blocks of days
(`block_days`, 14) instead of single records, and the error on the held-out blocks is recorded
after every tree. The error first falls as trees are added, reaches a minimum, and then rises
again slowly: the extra trees fit patterns that help with single missing records but not with
gaps of days. The number of trees is the minimum of the median error curve of all pools,
rounded to ten (with `tolerance` above zero, the smallest number within that fraction of the
minimum). Every year and scenario is then trained with that fixed number and no early stopping. `n_estimators` sets the largest
number tested. The random test records are not used, so the test scores stay out-of-sample.

Compared with early stopping for each model, this has five advantages:

1. **No random undersized models.** With early stopping, each model picks its own number of
   trees on a small random validation set, and a few unusual records in that set can stop the
   training far too early. Such a model fits worse than its neighbours, and its annual sum can
   fall out of line with the other years and scenarios. With one fixed number this cannot
   happen.
2. **Cleaner scenario differences.** All years and USTAR scenarios get the same model size. The
   scenarios then differ only in the records the USTAR filter keeps, not in the size of their
   models, so the USTAR uncertainty reflects the threshold and the scenarios keep their order.
3. **Tuned for real gaps.** Half-hourly fluxes are autocorrelated. In a random split of single
   records, the neighbours of almost every held-out record are in the training data, so the
   number of trees is tuned for short, easy gaps. Held-out blocks of days look like real gaps.
4. **A score for long gaps.** The error on the held-out blocks is reported next to the random
   test score (`report_tree_selection`). It shows how much worse the model fills gaps of days to
   weeks, which decide the annual sums.
5. **One documented choice.** One number with its error curve (`plot_tree_selection`) replaces a
   separate, unreported choice for every model, which makes the setting easier to report and to
   reproduce.

It costs the run time of the cross-validation, a few minutes for a record of ten years. Each
year still has a single model, so some model variance remains in the sums.
`report_gapfilling_feature_importances` and `plot_feature_ranks_per_year` show which features
carry the models.

(mds-gapfilling)=
## Level 4.1: gap-filling with MDS

`run_level41_mds` uses marginal distribution sampling (Reichstein et al. 2005): each gap is filled
with the mean of measured fluxes under similar conditions (incoming radiation, air temperature and
VPD within set tolerances), in a time window that widens until enough similar records are found.
`mode='diive'` uses the windows of ONEFlux and takes the symmetric mean of Vekuri et al. (2023) by
day; `mode='oneflux'` reproduces ONEFlux. `fill_long_gaps=True` fills every gap, also long ones,
for products that need complete years. With the QC flags of the drivers (`swin_qc`, `ta_qc`,
`vpd_qc`) only measured driver values count as similar. MDS predicts each measured record without
using it, so its scores are out-of-sample.

MDS accepts a mean of as few as two similar records (`avg_min_n_vals`, as in ONEFlux). In a long
gap the window can hold only a few records from the gap's edges, and a few bad values among them
set the fill for days. The fill flag encodes the method and the window, `method * 1000 + days`, so
a long run of the same value with a high window shows where this happened.

(method-comparison)=
### Comparing gap-filling methods

`report_gapfilling_comparison` and `plot_gapfilling_comparison` compare two methods that filled
the same gaps. The difference between their sums comes from the filled values alone; the table
splits it into daytime and nighttime gaps and by gap length, and scores both methods on the same
held-out records, the test records of the ML model. These are random single half-hours, so the
scores describe short gaps, while the sums depend more on the long ones.

(partitioning)=
## Level 4.2: partitioning

`run_level42_partitioning` splits NEE into GPP and RECO for each USTAR scenario, with the nighttime
method (Reichstein et al. 2005), which fits the temperature response of the nighttime flux, and
the daytime method (Lasslop et al. 2010), which fits a light response curve to the daytime flux.
Both fit the measured flux and use the gap-filled flux to compute GPP. `mode='oneflux'` follows
the ONEFlux code (output columns `_NT_OF`, `_DT_OF`), `mode='reddyproc'` REddyProc. ONEFlux itself
partitions a flux filled with MDS, so a partitioning of an ML-filled flux is not identical to
ONEFlux output.

(uncertainty)=
## Uncertainty

The uncertainty of the gap-filled flux follows Pastorello et al. (2020) and has two parts.

- **Random uncertainty** (`RandomUncertaintyPAS20`, after Hollinger and Richardson 2005). For a
  measured record it is the standard deviation of the measured fluxes within ±7 days and ±1 hour
  under similar conditions (TA, VPD and SW_IN, with the MDS tolerances), if there are more than five
  such records (method 1). Otherwise, and for every gap-filled record, it is the median of these
  standard deviations for similar flux values within ±14 days (method 2, as in the ONEFlux code).
  diive adds two steps so that every record gets a value: the same median over the whole record
  (method 3), then the median for the closest flux values (method 4). The gap-filled records depend
  on the filled values, so the random uncertainty is computed for each gap-filling method.
- **USTAR uncertainty** (`JointUncertaintyPAS20`): half the difference between the series filtered
  at the 84th and at the 16th percentile USTAR threshold. FLUXNET2015 filters the flux at 40
  thresholds and takes the 16th and 84th percentile of the 40 results; with three fixed thresholds
  the outer two scenarios stand in for these percentiles.

The joint uncertainty combines both for each record: √(random² + USTAR²). All three are 1-sigma
values in the flux units.

For sums over a day or a year the two parts add up differently. Random errors are independent
between records, so they add in quadrature. The USTAR threshold applies to every record at once,
so its part is half the difference between the sums of the two outer scenarios. The difference
between gap-filling methods is not part of the joint uncertainty, as in FLUXNET2015; report it
beside it.

## References

- Finkelstein, P. L. and Sims, P. F. (2001): Sampling error in eddy correlation flux measurements.
  J. Geophys. Res. 106, 3503-3509.
- Hollinger, D. Y. and Richardson, A. D. (2005): Uncertainty in eddy covariance measurements and
  its application to physiological models. Tree Physiology 25, 873-885.
- Lasslop, G. et al. (2010): Separation of net ecosystem exchange into assimilation and respiration
  using a light response curve approach. Global Change Biology 16, 187-208.
- Papale, D. et al. (2006): Towards a standardized processing of Net Ecosystem Exchange measured
  with eddy covariance technique. Biogeosciences 3, 571-583.
- Pastorello, G. et al. (2020): The FLUXNET2015 dataset and the ONEFlux processing pipeline for
  eddy covariance data. Scientific Data 7, 225.
- Reichstein, M. et al. (2005): On the separation of net ecosystem exchange into assimilation and
  ecosystem respiration. Global Change Biology 11, 1424-1439.
- Vekuri, H. et al. (2023): A widely-used eddy covariance gap-filling method creates systematic
  bias in carbon balance estimates. Scientific Reports 13, 1720.
