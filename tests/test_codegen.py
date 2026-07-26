"""
TEST_CODEGEN: reproducible-script generation across the library
===============================================================

Pure (no GUI, no data, no plotting) tests for the ``*_to_code`` functions that
back every "Copy Python" button in the GUI. Each one turns a settings dict into
a runnable snippet, so a point-and-click run stays reproducible.

Three checks per generator:

1. the emitted snippet is syntactically valid (``compile``),
2. it contains the call it is supposed to reproduce,
3. every keyword it passes to a ``dv.*`` callable is actually accepted by that
   callable's signature (:func:`_assert_kwargs_accepted`).

Check 3 is the one that catches real drift: a renamed or removed library
parameter leaves the generator emitting a snippet that still compiles but raises
``TypeError`` the moment a user runs it. Compiling alone would not notice.

The flux-chain generators (``chain_to_code``, ``level2_to_code`` ...
``level41_to_code``) live in ``tests/test_flux_codegen.py``; this module covers
everything else. :class:`TestCodegenCompleteness` enforces that split, so a new
``*_to_code`` function cannot land untested.

Run: pytest tests/test_codegen.py -v
"""
from __future__ import annotations

import ast
import inspect
import pathlib
import unittest

import diive as dv

# --- Shared helpers ---------------------------------------------------------


def _dotted(node: ast.AST) -> str | None:
    """Dotted source name of an attribute chain, or None if not a plain chain."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _resolve(dotted: str):
    """Resolve a ``dv.<...>`` dotted name against the real package, else None."""
    parts = dotted.split(".")
    if parts[0] != "dv":
        return None
    obj = dv
    for part in parts[1:]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


#: A ``**`` catch-all with this name does NOT mean "accepts anything". Every
#: outlier detector takes ``**legacy`` and immediately calls
#: ``reject_legacy_params``, which maps a removed parameter name to its
#: replacement and raises the normal unexpected-keyword error for everything
#: else -- so the named parameters really are the complete accepted set. Without
#: this carve-out the signature check silently skips every detector constructor.
_REJECTING_VAR_KEYWORD = "legacy"


def _accepted_kwargs(func) -> set[str] | None:
    """Parameter names ``func`` accepts, or None when it takes a real ``**kwargs``."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    names = set()
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            if param.name != _REJECTING_VAR_KEYWORD:
                return None  # genuine passthrough (e.g. XGBoostTS -> XGBRegressor)
            continue
        names.add(param.name)
    return names


class _CallCollector(ast.NodeVisitor):
    """Collect ``(callable_description, resolved_object, keyword_names)`` for
    every call in the snippet that targets a resolvable ``dv.*`` object.

    Handles both the constructor (``dv.plotting.X(...)``) and the phase-2 method
    chained onto it (``dv.plotting.X(...).plot(...)``), which is where the
    plotting generators put most of their keywords.
    """

    def __init__(self):
        self.found = []

    def visit_Call(self, node: ast.Call):
        kwnames = [kw.arg for kw in node.keywords if kw.arg is not None]
        dotted = _dotted(node.func)
        if dotted:
            obj = _resolve(dotted)
            if obj is not None:
                target = obj.__init__ if inspect.isclass(obj) else obj
                self.found.append((dotted, target, kwnames))
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call):
            # X(...).method(...) -- resolve X, then look up the method on it.
            owner = _dotted(node.func.value.func)
            if owner:
                cls = _resolve(owner)
                method = getattr(cls, node.func.attr, None) if cls is not None else None
                if method is not None:
                    self.found.append((f"{owner}.{node.func.attr}", method, kwnames))
        self.generic_visit(node)


class CodegenTestCase(unittest.TestCase):
    """Base class carrying the three shared assertions."""

    def assertRunnable(self, code: str) -> ast.Module:
        """The snippet compiles, is non-empty, and ends in a newline."""
        self.assertTrue(code.endswith("\n"), "snippet should end in a newline")
        self.assertTrue(code.strip(), "snippet should not be empty")
        compile(code, "<gen>", "exec")
        return ast.parse(code)

    def assertKwargsAccepted(self, code: str, *, expect_dv_calls: bool = True):
        """Every keyword passed to a resolvable ``dv.*`` callable really exists.

        Silently skips anything that cannot be resolved (locally-named variables,
        pandas calls, optional-extra classes), so the check never produces a
        false failure -- it only fires on a genuine generator/library mismatch.

        ``expect_dv_calls=False`` for snippets that deliberately import their
        functions by name instead of going through the ``dv`` namespace (the
        flux-chain generators do this), where finding no ``dv.*`` call is correct
        rather than a sign the snippet shape drifted.
        """
        collector = _CallCollector()
        collector.visit(ast.parse(code))
        if expect_dv_calls:
            self.assertTrue(collector.found,
                            "no resolvable dv.* call found -- did the snippet shape change?")
        for dotted, target, kwnames in collector.found:
            accepted = _accepted_kwargs(target)
            if accepted is None:
                continue
            for kwname in kwnames:
                self.assertIn(
                    kwname, accepted,
                    f"{dotted}() is passed '{kwname}', which its signature does not accept")

    def check(self, code: str, *must_contain: str,
              expect_dv_calls: bool = True) -> str:
        """Run all three checks and return the code for further assertions."""
        self.assertRunnable(code)
        self.assertKwargsAccepted(code, expect_dv_calls=expect_dv_calls)
        for needle in must_contain:
            self.assertIn(needle, code)
        return code


# --- Plotting ---------------------------------------------------------------

# Union of the opts keys the plotting generators read. They index `opts[...]`
# directly, so a missing key is a KeyError -- one shared dict keeps each test to
# the arguments it actually cares about. Values are deliberately all non-None so
# every optional `_kw_lines` branch is exercised.
_PLOT_OPTS = {
    "_format": {"title": "My title", "xlabel": "x", "ylabel": "y"},
    # heatmaps / colorbars
    "ax_orientation": "vertical", "cmap": "RdYlBu_r", "vmin": -10.0, "vmax": 10.0,
    "color_bad": "#EEEEEE", "zlabel": "flux", "cb_digits_after_comma": 1,
    "cb_extend": "both", "show_colormap": True, "show_less_xticklabels": False,
    "show_values": True, "show_values_n_dec_places": 2, "show_values_fontsize": 6,
    "cb_labelsize": 10, "minticks": 3, "maxticks": 12,
    "agg": "mean", "ranks": False,
    # timeseries / diel cycle
    "drop_gaps": True, "color_by": None, "color": "#2196F3", "linewidth": 1.5,
    "alpha": 0.8, "marker": "o", "markersize": 4, "color_by_cmap": "viridis",
    "band": "sd", "each_month": True,
    # cumulative / waterfall
    "series_units": "gC m-2", "yearly_end_date": "08-31", "show_reference": True,
    "highlight_year": 2021, "digits_after_comma": 1, "units": "gC m-2",
    "show_title": True, "fill": True, "resample": "D",
    "uptake_is_negative": True, "color_uptake": "#4CAF50",
    "color_release": "#F44336", "bar_width": 0.8, "show_connectors": True,
    # histogram
    "n_bins": 30, "highlight_peak": True, "show_zscores": True,
    "show_zscore_values": True, "show_info": True, "show_counts": True,
    # ridgeline
    "how": "monthly", "hspace": -0.5, "shade_percentile": 90,
    "show_mean_line": True, "ascending": True, "kd_kwargs": {"bw_adjust": 0.5},
    # shifted distribution
    "ref_period": (2016, 2018), "comp_period": (2019, 2021),
    "ref_label": "Reference", "comp_label": "Comparison",
    "zone_labels": True, "show_legend": True, "show_xaxis": True,
    "show_yaxis": True,
    # hexbin / heatmap xyz
    "gridsize": 25, "normalize_axes": True, "mincnt": 5,
    "binning_type": "equal_width", "aggfunc": "mean", "min_n_vals_per_bin": 3,
    # wind rose
    "n_sectors": 16, "z_agg": "mean", "show_colorbar": True, "cb_label": "z",
    "max_sector_labels": 8,
    # tree ring
    "resample_freq": "D", "show_month_labels": True, "show_month_lines": True,
    "show_year_labels": True, "show_year_separators": True,
    "year_label_frequency": 2, "style": "filled", "amplitude_scale": 1.2,
    "ring_width": 0.9,
}


def _opts(**overrides) -> dict:
    opts = dict(_PLOT_OPTS)
    opts.update(overrides)
    return opts


class TestPlottingCodegen(CodegenTestCase):
    """The 16 generators in ``diive/core/plotting/codegen.py``."""

    def test_heatmap_datetime(self):
        from diive.core.plotting.codegen import heatmap_datetime_to_code
        code = self.check(heatmap_datetime_to_code("TA", _opts()),
                          "dv.plotting.HeatmapDateTime(", "df['TA'],",
                          "ax_orientation='vertical'", "plt.show()")
        self.assertIn("format_style=dv.plotting.FormatStyle(", code)
        self.assertIn("cmap='RdYlBu_r'", code)

    def test_heatmap_datetime_omits_none_kwargs(self):
        from diive.core.plotting.codegen import heatmap_datetime_to_code
        # None means "library default" -- the line is skipped entirely.
        code = self.check(heatmap_datetime_to_code("TA", _opts(cmap=None, vmin=None)))
        self.assertNotIn("cmap=", code)
        self.assertNotIn("vmin=", code)

    def test_heatmap_datetime_default_format_is_clean(self):
        from diive.core.plotting.codegen import heatmap_datetime_to_code
        # A plot left at the house style emits no FormatStyle at all.
        code = self.check(heatmap_datetime_to_code("TA", _opts(_format=None)))
        self.assertNotIn("FormatStyle", code)

    def test_heatmap_yearmonth(self):
        from diive.core.plotting.codegen import heatmap_yearmonth_to_code
        self.check(heatmap_yearmonth_to_code("TA", _opts()),
                   "dv.plotting.HeatmapYearMonth(", "agg='mean'", "ranks=False")

    def test_timeseries(self):
        from diive.core.plotting.codegen import timeseries_to_code
        self.check(timeseries_to_code("TA", _opts()),
                   "dv.plotting.TimeSeries(", "drop_gaps=True", "linewidth=1.5")

    def test_timeseries_titles_default_to_varname(self):
        from diive.core.plotting.codegen import timeseries_to_code
        code = self.check(timeseries_to_code("TA", _opts(_format={"title": None})))
        self.assertIn("title='TA'", code)

    def test_timeseries_with_color_by(self):
        from diive.core.plotting.codegen import timeseries_to_code
        code = self.check(timeseries_to_code("TA", _opts(color_by="SW_IN")),
                          "color_series=df['SW_IN']", "color_label='SW_IN'")
        self.assertIn("cmap='viridis'", code)

    def test_dielcycle(self):
        from diive.core.plotting.codegen import dielcycle_to_code
        self.check(dielcycle_to_code("TA", _opts()),
                   "dv.plotting.DielCycle(", "agg='mean'", "band='sd'")

    def test_cumulative_year(self):
        from diive.core.plotting.codegen import cumulative_year_to_code
        self.check(cumulative_year_to_code("NEE", _opts()),
                   "dv.plotting.CumulativeYear(", "highlight_year=2021")

    def test_cumulative_takes_a_frame_not_a_series(self):
        from diive.core.plotting.codegen import cumulative_to_code
        # Cumulative plots one curve per column, so the snippet must double-bracket.
        self.check(cumulative_to_code("NEE", _opts()),
                   "dv.plotting.Cumulative(", "df[['NEE']],")

    def test_waterfall(self):
        from diive.core.plotting.codegen import waterfall_to_code
        self.check(waterfall_to_code("NEE", _opts()),
                   "dv.plotting.WaterfallPlot(", "uptake_is_negative=True")

    def test_histogram(self):
        from diive.core.plotting.codegen import histogram_to_code
        self.check(histogram_to_code("TA", _opts()),
                   "dv.plotting.HistogramPlot(", "method='n_bins'", "n_bins=30")

    def test_ridgeline_uses_its_own_figure(self):
        from diive.core.plotting.codegen import ridgeline_to_code
        # RidgeLinePlot builds its own gridspec, so the snippet passes fig, not ax.
        code = self.check(ridgeline_to_code("TA", _opts()),
                          "dv.plotting.RidgeLinePlot(", "fig = plt.figure(", "fig=fig,")
        self.assertNotIn("ax=ax,", code)

    def test_shifted_distribution(self):
        from diive.core.plotting.codegen import shifted_distribution_to_code
        self.check(shifted_distribution_to_code("TA", _opts()),
                   "dv.plotting.ShiftedDistributionPlot(",
                   "ref_period=(2016, 2018)", "comp_period=(2019, 2021)")

    def test_hexbin(self):
        from diive.core.plotting.codegen import hexbin_to_code
        self.check(hexbin_to_code("TA", "VPD", "NEE", _opts()),
                   "dv.plotting.HexbinPlot(", "gridsize=25", "mincnt=5",
                   "dropna(subset=['TA', 'VPD'])")

    def test_heatmap_xyz_grids_first(self):
        from diive.core.plotting.codegen import heatmap_xyz_to_code
        # HeatmapXYZ needs pre-aggregated input, so a GridAggregator comes first.
        self.check(heatmap_xyz_to_code("TA", "VPD", "NEE", _opts()),
                   "dv.analysis.GridAggregator(",
                   "dv.plotting.HeatmapXYZ.from_gridaggregator(",
                   "binning_type='equal_width'")

    def test_windrose_without_z(self):
        from diive.core.plotting.codegen import windrose_to_code
        code = self.check(windrose_to_code("WS", "WD", None, _opts()),
                          "dv.plotting.WindRosePlot(", "n_sectors=16",
                          '"projection": "polar"')
        self.assertNotIn("z_agg=", code)

    def test_windrose_with_z(self):
        from diive.core.plotting.codegen import windrose_to_code
        self.check(windrose_to_code("WS", "WD", "NEE", _opts()),
                   "z=df['NEE']", "z_agg='mean'")

    def test_treering_filled(self):
        from diive.core.plotting.codegen import treering_to_code
        code = self.check(treering_to_code("TA", _opts(style="filled")),
                          "dv.plotting.TreeRingPlot(", ").plot(")
        self.assertNotIn("amplitude_scale", code)

    def test_treering_line_uses_plot_line(self):
        from diive.core.plotting.codegen import treering_to_code
        # The line style is a different renderer with extra parameters.
        self.check(treering_to_code("TA", _opts(style="line")),
                   ").plot_line(", "amplitude_scale=1.2", "ring_width=0.9")

    def test_datetime_surface(self):
        from diive.core.plotting.codegen import datetime_surface_to_code
        self.check(datetime_surface_to_code("TA", cmap="magma"),
                   "dv.plotting.datetime_surface_grid(df['TA'])",
                   "projection='3d'", "cmap='magma'")

    def test_surface_xyz(self):
        from diive.core.plotting.codegen import surface_xyz_to_code
        self.check(surface_xyz_to_code("TA", "VPD", "NEE", n_bins=20, aggfunc="median"),
                   "dv.analysis.GridAggregator(", "n_bins=20", "aggfunc='median'",
                   "agg.df_agg_wide")

    def test_scatter(self):
        from diive.core.plotting.scatter import scatter_to_code
        self.check(scatter_to_code("TA", "NEE", nbins=10, binagg="mean"),
                   "dv.plotting.ScatterXY(", "nbins=10", "binagg='mean'")

    def test_scatter_with_z_and_format(self):
        from diive.core.plotting.scatter import scatter_to_code
        code = self.check(
            scatter_to_code("TA", "NEE", "SW_IN",
                            format_kwargs={"title": "T", "xlabel": None}),
            "z=df['SW_IN']", "format_style=dv.plotting.FormatStyle(title='T')")
        self.assertNotIn("xlabel", code)  # None fields dropped


# --- Outlier detection ------------------------------------------------------


class TestOutlierCodegen(CodegenTestCase):
    """The 10 generators in ``diive/preprocessing/outlier_detection/codegen.py``."""

    def test_hampel(self):
        from diive.preprocessing.outlier_detection.codegen import hampel_to_code
        self.check(hampel_to_code({"n_sigma": 4.0, "window_length": 48}, var_name="TA"),
                   "dv.outliers.Hampel(", "series = df['TA']",
                   "n_sigma=4.0", "window_length=48",
                   "cleaned = h.filteredseries")

    def test_hampel_omits_default_valued_kwargs(self):
        from diive.preprocessing.outlier_detection.codegen import hampel_to_code
        from diive.preprocessing.outlier_detection.hampel import Hampel
        default_sigma = inspect.signature(Hampel.__init__).parameters["n_sigma"].default
        code = self.check(hampel_to_code({"n_sigma": default_sigma, "window_length": 48}))
        self.assertNotIn("n_sigma=", code)
        self.assertIn("window_length=48", code)

    def test_repeat_false_is_rendered(self):
        from diive.preprocessing.outlier_detection.codegen import hampel_to_code
        self.assertIn(").run(repeat=False)", hampel_to_code({}, repeat=False))
        self.assertIn(").run()", hampel_to_code({}, repeat=True))

    def test_localsd(self):
        from diive.preprocessing.outlier_detection.codegen import localsd_to_code
        self.check(localsd_to_code({"n_sd": 3.0, "winsize": 96}, var_name="TA"),
                   "dv.outliers.LocalSD(", "n_sd=3.0", "winsize=96")

    def test_lof(self):
        from diive.preprocessing.outlier_detection.codegen import lof_to_code
        self.check(lof_to_code({"n_neighbors": 25, "contamination": 0.02}),
                   "dv.outliers.LocalOutlierFactor(", "n_neighbors=25")

    def test_absolutelimits(self):
        from diive.preprocessing.outlier_detection.codegen import absolutelimits_to_code
        self.check(absolutelimits_to_code({"minval": -50.0, "maxval": 50.0}),
                   "dv.outliers.AbsoluteLimits(", "minval=-50.0", "maxval=50.0")

    def test_zscore(self):
        from diive.preprocessing.outlier_detection.codegen import zscore_to_code
        # 4 is the library default and would be omitted -- use a differing value.
        self.check(zscore_to_code({"thres_zscore": 5}), "dv.outliers.zScore(",
                   "thres_zscore=5")

    def test_zscorerolling(self):
        from diive.preprocessing.outlier_detection.codegen import zscorerolling_to_code
        self.check(zscorerolling_to_code({"thres_zscore": 4, "winsize": 48}),
                   "dv.outliers.zScoreRolling(", "winsize=48")

    def test_zscoreincrements(self):
        from diive.preprocessing.outlier_detection.codegen import zscoreincrements_to_code
        self.check(zscoreincrements_to_code({"thres_zscore": 5}),
                   "dv.outliers.zScoreIncrements(")

    def test_trimlow(self):
        from diive.preprocessing.outlier_detection.codegen import trimlow_to_code
        self.check(trimlow_to_code({"lower_limit": 0.0}), "dv.outliers.TrimLow(",
                   "lower_limit=0.0")

    def test_manualremoval_ignores_repeat(self):
        from diive.preprocessing.outlier_detection.codegen import manualremoval_to_code
        # ManualRemoval flags fixed timestamps, so repeat is meaningless for it.
        code = self.check(
            manualremoval_to_code({"remove_dates": ["2021-06-01 12:00:00"]},
                                  repeat=False),
            "dv.outliers.ManualRemoval(", "remove_dates=")
        self.assertIn(").run()", code)
        self.assertNotIn("repeat=False", code)

    def test_stepwise_chain(self):
        from diive.preprocessing.outlier_detection.codegen import stepwise_to_code
        code = self.check(
            stepwise_to_code(
                [{"method": "flag_outliers_hampel_test",
                  "kwargs": {"n_sigma": 4.0}},
                 {"method": "flag_outliers_zscore_test",
                  "kwargs": {"thres_zscore": 4}},
                 {"method": "flag_missingvals_test", "kwargs": {}}],
                var_name="TA", site_lat=46.6, site_lon=9.8, utc_offset=1,
                load_hint="dv.load_parquet('data.parquet')"),
            "StepwiseOutlierDetection(", "FlagQCF(",
            "df = dv.load_parquet('data.parquet')")
        # One addflag() per committed test; a no-kwarg test renders bare.
        self.assertEqual(code.count("sod.addflag()"), 3)
        self.assertIn("sod.flag_missingvals_test()", code)

    def test_stepwise_appends_corrections(self):
        from diive.preprocessing.outlier_detection.codegen import stepwise_to_code
        code = self.check(
            stepwise_to_code(
                [{"method": "flag_missingvals_test", "kwargs": {}}],
                var_name="SW_IN", site_lat=46.6, site_lon=9.8, utc_offset=1,
                corrections=[{"key": "radiation_zero_offset", "kwargs": {}}]),
            "corrected = ")


# --- Gap-filling ------------------------------------------------------------


class TestGapfillingCodegen(CodegenTestCase):
    """The 6 untested generators in ``diive/gapfilling/codegen.py``."""

    def test_xgboost(self):
        from diive.gapfilling.codegen import xgboost_gapfill_to_code
        code = self.check(
            xgboost_gapfill_to_code("NEE", ["TA", "SW_IN", "VPD"],
                                    {"n_estimators": 200, "random_state": 42},
                                    load_hint="dv.load_parquet('data.parquet')"),
            "dv.gapfilling.XGBoostTS(", "target = 'NEE'",
            "features = ['TA', 'SW_IN', 'VPD']", "n_estimators=200",
            "df = dv.load_parquet('data.parquet')", "_gfXG")
        self.assertNotIn("reduce_features", code)

    def test_xgboost_with_reduction(self):
        from diive.gapfilling.codegen import xgboost_gapfill_to_code
        self.check(xgboost_gapfill_to_code("NEE", ["TA"], {}, reduce=True,
                                           shap_threshold_factor=0.7),
                   "model.reduce_features(shap_threshold_factor=0.7)")

    def test_randomforest(self):
        from diive.gapfilling.codegen import randomforest_gapfill_to_code
        self.check(randomforest_gapfill_to_code("NEE", ["TA"], {"n_estimators": 99}),
                   "dv.gapfilling.RandomForestTS(", "_gfRF")

    def test_ml_gapfill_shared_renderer(self):
        from diive.gapfilling.codegen import ml_gapfill_to_code
        self.check(ml_gapfill_to_code("XGBoostTS", "_gfXG", "NEE", ["TA"], {}),
                   "dv.gapfilling.XGBoostTS(", "input_df=df[[target] + features]")

    def test_mds(self):
        from diive.gapfilling.codegen import mds_gapfill_to_code
        code = self.check(
            mds_gapfill_to_code("NEE", "SW_IN", "TA", "VPD_kPa",
                                {"swin_tol": 50, "ta_tol": 2.5}),
            "dv.gapfilling.FluxMDS(", "swin='SW_IN'", "ta='TA'",
            "vpd='VPD_kPa'", "swin_tol=50")
        # MDS has no feature list and no SHAP reduction.
        self.assertNotIn("features", code)
        self.assertNotIn("reduce_features", code)

    def test_longterm_xgboost_passes_reduction_to_run(self):
        from diive.gapfilling.codegen import longterm_xgboost_gapfill_to_code
        # The long-term classes take reduction in run(), not as a separate call.
        code = self.check(
            longterm_xgboost_gapfill_to_code("NEE", ["TA"], {}, reduce=True),
            "dv.gapfilling.LongTermGapFillingXGBoostTS(",
            "model.run(reduce_features=True", "scores_per_year")
        self.assertNotIn("model.reduce_features(", code)

    def test_longterm_randomforest(self):
        from diive.gapfilling.codegen import longterm_randomforest_gapfill_to_code
        self.check(longterm_randomforest_gapfill_to_code("NEE", ["TA"], {}),
                   "dv.gapfilling.LongTermGapFillingRandomForestTS(", "model.run()")

    def test_longterm_ml_shared_renderer(self):
        from diive.gapfilling.codegen import longterm_ml_gapfill_to_code
        self.check(longterm_ml_gapfill_to_code("LongTermGapFillingXGBoostTS", "_gfXG",
                                               "NEE", ["TA"], {}),
                   "dv.gapfilling.LongTermGapFillingXGBoostTS(")

    def test_feature_engineer(self):
        from diive.core.ml.feature_engineer import feature_engineer_to_code
        self.check(feature_engineer_to_code(["TA", "SW_IN"],
                                            {"target_col": "_target_",
                                             "features_lag": [-1, -2]}),
                   "FeatureEngineer(", "'TA'", "'SW_IN'")


# --- Variables --------------------------------------------------------------


class TestVariablesCodegen(CodegenTestCase):
    """The 3 generators in ``diive/variables/utilities.py``."""

    def test_combine_variables(self):
        from diive.variables.utilities import combine_variables_to_code
        self.check(combine_variables_to_code("TA_1", "TA_2", method="add",
                                             keep_overlap_only=False, name="TA_SUM"),
                   "dv.variables.combine_variables(", "method='add'",
                   "keep_overlap_only=False")

    def test_combine_variables_default_method(self):
        from diive.variables.utilities import combine_variables_to_code
        self.check(combine_variables_to_code("TA_1", "TA_2"),
                   "dv.variables.combine_variables(")

    def test_potrad(self):
        from diive.variables.utilities import potrad_to_code
        self.check(potrad_to_code(46.6, 9.8, 1, name="SW_IN_POT"),
                   "dv.variables.potrad(", "46.6", "9.8")

    def test_calc_vpd(self):
        from diive.variables.utilities import calc_vpd_from_ta_rh_to_code
        self.check(calc_vpd_from_ta_rh_to_code("TA", "RH", name="VPD_kPa"),
                   "dv.variables.calc_vpd_from_ta_rh(", "'TA'", "'RH'")


# --- Analysis ---------------------------------------------------------------


class TestAnalysisCodegen(CodegenTestCase):
    """The 5 generators scattered across ``diive/analysis/``."""

    def test_compound_extremes(self):
        from diive.analysis.compoundextremes import compound_extremes_to_code
        self.check(compound_extremes_to_code("VPD", "SWC", agg="daily",
                                             threshold=1.5),
                   "dv.analysis.CompoundExtremes(")

    def test_rank_drivers(self):
        from diive.analysis.correlation import rank_drivers_to_code
        self.check(rank_drivers_to_code("NEE", method="spearman", max_lag=4),
                   "dv.analysis.rank_drivers(", "method='spearman'", "max_lag=4")

    def test_rank_drivers_without_plot(self):
        from diive.analysis.correlation import rank_drivers_to_code
        code = self.check(rank_drivers_to_code("NEE", with_plot=False))
        self.assertNotIn("ScatterXY", code)

    def test_gapstats(self):
        from diive.analysis.gapfinder import gapstats_to_code
        self.check(gapstats_to_code("NEE", long_gap_records=96),
                   "dv.analysis.GapStats(", "96")

    def test_spectrogram(self):
        from diive.analysis.harmonic import spectrogram_to_code
        self.check(spectrogram_to_code("TA", nperseg=512, window="hamming"),
                   "dv.analysis.spectrogram(", "nperseg=512", "window='hamming'")

    def test_seasonal_trend(self):
        from diive.analysis.seasonaltrend import seasonal_trend_to_code
        self.check(seasonal_trend_to_code("TA", seasonal_period=180),
                   "dv.analysis.SeasonalTrendDecomposition(")

    def test_seasonal_trend_anomalies_view(self):
        from diive.analysis.seasonaltrend import seasonal_trend_to_code
        self.check(seasonal_trend_to_code("TA", view="anomalies",
                                          reference_start_year=2016,
                                          reference_end_year=2020),
                   "2016")


# --- Flux -------------------------------------------------------------------


class TestFluxCodegen(CodegenTestCase):
    """Flux generators outside the chain levels (those live in test_flux_codegen)."""

    def test_randunc(self):
        from diive.flux.lowres.codegen import randunc_to_code
        self.check(randunc_to_code("NEE", "NEE_f", "TA", "VPD", "SW_IN",
                                   load_hint="dv.load_parquet('data.parquet')"),
                   "RandomUncertaintyPAS20(", "df = dv.load_parquet('data.parquet')")

    def test_randunc_vpd_unit_flag(self):
        from diive.flux.lowres.codegen import randunc_to_code
        code = self.check(randunc_to_code("NEE", "NEE_f", "TA", "VPD", "SW_IN",
                                          vpd_in_kpa=False))
        self.assertIn("vpd_in_kpa=False", code)

    def test_jointunc(self):
        from diive.flux.lowres.codegen import jointunc_to_code
        self.check(jointunc_to_code("NEE_RANDUNC", "NEE_CUT_16", "NEE_CUT_84",
                                    divisor=1.349),
                   "JointUncertaintyPAS20(", "1.349")

    def test_partitioning_all_four_ports(self):
        from diive.flux.partitioning.codegen import partitioning_to_code
        expected = {
            "NT_OF": "partition_nee_nighttime_oneflux",
            "NT_RP": "partition_nee_nighttime_reddyproc",
            "DT_RP": "partition_nee_daytime_reddyproc",
            "DT_OF": "partition_nee_daytime_oneflux",
        }
        picks = {"nee": "NEE", "ta": "TA", "sw_in": "SW_IN", "nee_f": "NEE_f",
                 "ta_f": "TA_f", "sw_in_f": "SW_IN_f", "vpd": "VPD",
                 "nee_sd": "NEE_SD"}
        for suffix, func in expected.items():
            with self.subTest(method=suffix):
                self.check(partitioning_to_code(suffix, picks, lat=46.6, lon=9.8,
                                                utc_offset=1),
                           f"dv.flux.{func}(")

    def test_level42(self):
        from diive.flux.fluxprocessingchain import level42_to_code
        code = self.check(
            level42_to_code(
                init_kwargs=dict(fluxcol="FC", site_lat=46.6, site_lon=9.8,
                                 utc_offset=1),
                level2_settings={"ssitc": {"apply": True,
                                           "setflag_timeperiod": None}},
                level31_kwargs={},
                level32_steps=[{"method": "flag_outliers_hampel_test",
                                "kwargs": {}}],
                level33_kwargs={"thresholds": [0.18],
                                "threshold_labels": ["CUT_50"]},
                level41_cfg={"methods": ["mds"],
                             "mds": {"swin": "SW_IN", "ta": "TA", "vpd": "VPD"}},
                level42_cfg={"variants": ["nt_of"], "gapfill_method": "mds"}),
            "run_level42_nighttime_oneflux(", "final_df = data.fpc_df",
            expect_dv_calls=False)
        # L4.2 renders the whole chain before it.
        self.assertIn("run_level2(", code)
        self.assertIn("run_level41_mds(", code)


# --- Misc -------------------------------------------------------------------


class TestMiscCodegen(CodegenTestCase):

    def test_select_records(self):
        from diive.core.dfun.frames import select_records_to_code
        self.check(
            select_records_to_code("NEE", [
                {"cond": "TA", "lower": 5, "upper": 25, "inclusive": "both",
                 "mode": "keep"},
                {"cond": "SW_IN", "lower": 0, "upper": 100, "inclusive": "both",
                 "mode": "remove"},
            ]),
            "dv.keep_records_where(", "invert=True")

    def test_select_records_with_no_steps_is_a_copy(self):
        from diive.core.dfun.frames import select_records_to_code
        code = select_records_to_code("NEE", [])
        self.assertRunnable(code)
        self.assertIn("df['NEE'].copy()", code)

    def test_corrections(self):
        from diive.preprocessing.corrections.codegen import corrections_to_code
        code = corrections_to_code(
            [{"key": "radiation_zero_offset", "kwargs": {"clamp_negatives": False}}],
            site_lat=46.6, site_lon=9.8, utc_offset=1)
        self.assertRunnable("import diive as dv\n" + code)
        self.assertIn("corrected = cleaned.copy()", code)
        self.assertIn("dv.corrections.", code)

    def test_corrections_empty_chain(self):
        from diive.preprocessing.corrections.codegen import corrections_to_code
        code = corrections_to_code([], site_lat=46.6, site_lon=9.8, utc_offset=1)
        # Nothing to apply -> nothing (or a bare copy) to render; must not crash.
        self.assertIsInstance(code, str)


# --- Completeness guard -----------------------------------------------------


class TestCodegenCompleteness(unittest.TestCase):
    """Every ``*_to_code`` in the library must be named by a codegen test file.

    Without this, a new "Copy Python" button ships a generator nobody tests --
    which is exactly how the 47-function gap this module closes came about.
    """

    # Test files that jointly own the codegen surface.
    _TEST_FILES = ("test_codegen.py", "test_flux_codegen.py")

    def test_every_codegen_function_is_covered(self):
        repo = pathlib.Path(__file__).resolve().parent.parent
        sources = "\n".join(
            (repo / "tests" / name).read_text(encoding="utf-8")
            for name in self._TEST_FILES)

        missing = []
        for path in sorted((repo / "diive").rglob("*.py")):
            if "gui" in path.parts:
                continue  # the GUI calls generators, it does not define them
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in tree.body:
                if (isinstance(node, ast.FunctionDef)
                        and node.name.endswith("_to_code")
                        and node.name not in sources):
                    missing.append(f"{node.name} ({path.relative_to(repo).as_posix()})")

        self.assertEqual(
            [], missing,
            "codegen functions with no test -- add one to tests/test_codegen.py:\n  "
            + "\n  ".join(missing))


if __name__ == "__main__":
    unittest.main()
