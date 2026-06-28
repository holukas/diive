"""
CORE.PLOTTING.CODEGEN: RUNNABLE SNIPPETS FOR THE PLOT TYPES
===========================================================

One ``*_to_code`` function per diive plot class. Each returns a self-contained,
runnable Python snippet that reproduces what the GUI's plotting tabs render: the
two-phase ``dv.plotting.X(data...).plot(...)`` call with the same data-render and
``FormatStyle`` arguments.

These live in the library (not the GUI) so the GUI only *asks* for the code and
copies it — it never builds the script itself (the GUI <-> library separation
rule). Each function takes the variable name(s) plus the GUI settings panel's
``values()`` dict, so the GUI wiring stays a thin one-liner.

The scatter codegen is the older sibling in ``scatter.py`` (:func:`scatter_to_code`),
kept there for backwards compatibility; the rest live here.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations


def _fmt_line(format_kwargs: dict | None) -> str | None:
    """A ``format_style=dv.plotting.FormatStyle(...)`` line for the non-None
    fields, or None when nothing differs from the house style (so a plot left at
    the default chrome produces a clean call)."""
    fmt = {k: v for k, v in (format_kwargs or {}).items() if v is not None}
    if not fmt:
        return None
    args = ", ".join(f"{k}={v!r}" for k, v in fmt.items())
    return f"    format_style=dv.plotting.FormatStyle({args}),"


def _kw_lines(kwargs: dict) -> list[str]:
    """One ``key=repr(value),`` line per kwarg whose value is not None.

    None means "library default" for every data-render argument here, so a
    skipped line plots identically to passing None — it just keeps the snippet
    clean.
    """
    return [f"    {k}={v!r}," for k, v in kwargs.items() if v is not None]


def _script(class_name: str, ctor_lines: list[str], plot_lines: list[str], *,
            fig_line: str = "fig, ax = plt.subplots(figsize=(8, 8))",
            pre_lines: list[str] | None = None,
            plot_method: str = "plot") -> str:
    """Assemble the imports + figure setup + ``Class(...).plot(...)`` call.

    ``plot_method`` names the phase-2 method to call (default ``"plot"``); a few
    classes expose alternative renderers (e.g. ``TreeRingPlot.plot_line``).
    """
    pre = ("\n".join(pre_lines) + "\n") if pre_lines else ""
    return (
        "import matplotlib.pyplot as plt\n"
        "import diive as dv\n"
        "\n"
        + pre
        + f"{fig_line}\n"
        f"dv.plotting.{class_name}(\n"
        + "\n".join(ctor_lines) + "\n"
        f").{plot_method}(\n"
        + "\n".join(plot_lines) + "\n"
        ")\n"
        "plt.show()\n"
    )


def heatmap_datetime_to_code(varname: str, opts: dict, *, df_name: str = "df") -> str:
    """Reproduce a :class:`HeatmapDateTime` (date x time-of-day) plot."""
    ctor = [
        f"    {df_name}[{varname!r}],",
        f"    ax_orientation={opts['ax_orientation']!r},",
    ]
    plot = ["    ax=ax,", "    fig=fig,"]
    fmt = _fmt_line(opts.get("_format"))
    if fmt:
        plot.append(fmt)
    plot += _kw_lines({
        "cmap": opts["cmap"], "vmin": opts["vmin"], "vmax": opts["vmax"],
        "color_bad": opts["color_bad"], "zlabel": opts["zlabel"],
        "cb_digits_after_comma": opts["cb_digits_after_comma"],
        "cb_extend": opts["cb_extend"], "show_colormap": opts["show_colormap"],
        "show_less_xticklabels": opts["show_less_xticklabels"],
        "show_values": opts["show_values"],
        "show_values_n_dec_places": opts["show_values_n_dec_places"],
        "show_values_fontsize": opts["show_values_fontsize"],
        "cb_labelsize": opts["cb_labelsize"],
        "minticks": opts["minticks"], "maxticks": opts["maxticks"],
    })
    return _script("HeatmapDateTime", ctor, plot)


def heatmap_yearmonth_to_code(varname: str, opts: dict, *, df_name: str = "df") -> str:
    """Reproduce a :class:`HeatmapYearMonth` (year x month aggregate) plot."""
    ctor = [
        f"    {df_name}[{varname!r}],",
        f"    agg={opts['agg']!r},",
        f"    ranks={opts['ranks']!r},",
        f"    ax_orientation={opts['ax_orientation']!r},",
    ]
    plot = ["    ax=ax,", "    fig=fig,"]
    fmt = _fmt_line(opts.get("_format"))
    if fmt:
        plot.append(fmt)
    plot += _kw_lines({
        "cmap": opts["cmap"], "vmin": opts["vmin"], "vmax": opts["vmax"],
        "color_bad": opts["color_bad"], "zlabel": opts["zlabel"],
        "cb_digits_after_comma": opts["cb_digits_after_comma"],
        "cb_extend": opts["cb_extend"], "show_colormap": opts["show_colormap"],
        "show_less_xticklabels": opts["show_less_xticklabels"],
        "show_values": opts["show_values"],
        "show_values_n_dec_places": opts["show_values_n_dec_places"],
        "show_values_fontsize": opts["show_values_fontsize"],
        "cb_labelsize": opts["cb_labelsize"],
    })
    return _script("HeatmapYearMonth", ctor, plot)


def timeseries_to_code(varname: str, opts: dict, *, df_name: str = "df") -> str:
    """Reproduce a :class:`TimeSeries` line plot (with optional colour-by)."""
    ctor = [f"    {df_name}[{varname!r}],",
            f"    drop_gaps={opts['drop_gaps']!r},"]
    color_by = opts.get("color_by")
    if color_by:
        ctor.append(f"    color_series={df_name}[{color_by!r}],")

    fmt = dict(opts.get("_format") or {})
    fmt.setdefault("title", None)
    if fmt["title"] is None:
        fmt["title"] = varname  # the GUI defaults the title to the variable name
    plot = ["    ax=ax,"]
    fmt_line = _fmt_line(fmt)
    if fmt_line:
        plot.append(fmt_line)
    plot += _kw_lines({
        "color": opts.get("color"), "linewidth": opts["linewidth"],
        "alpha": opts["alpha"], "marker": opts["marker"],
        "markersize": opts["markersize"],
    })
    if color_by:
        plot += _kw_lines({"cmap": opts["color_by_cmap"], "color_label": color_by})
    return _script("TimeSeries", ctor, plot)


def dielcycle_to_code(varname: str, opts: dict, *, df_name: str = "df") -> str:
    """Reproduce a :class:`DielCycle` plot."""
    ctor = [f"    {df_name}[{varname!r}],"]
    fmt = dict(opts.get("_format") or {})
    if fmt.get("title") is None:
        fmt["title"] = varname
    plot = ["    ax=ax,"]
    fmt_line = _fmt_line(fmt)
    if fmt_line:
        plot.append(fmt_line)
    plot += _kw_lines({
        "agg": opts["agg"], "band": opts["band"], "each_month": opts["each_month"],
        "cmap": opts["cmap"], "marker": opts["marker"],
        "markersize": opts["markersize"],
    })
    return _script("DielCycle", ctor, plot)


def cumulative_year_to_code(varname: str, opts: dict, *, df_name: str = "df") -> str:
    """Reproduce a :class:`CumulativeYear` plot."""
    ctor = [f"    {df_name}[{varname!r}],"]
    ctor += _kw_lines({
        "series_units": opts["series_units"],
        "yearly_end_date": opts["yearly_end_date"],
        "show_reference": opts["show_reference"],
        "highlight_year": opts["highlight_year"],
    })
    plot = ["    ax=ax,", "    showplot=False,"]
    fmt_line = _fmt_line(opts.get("_format"))
    if fmt_line:
        plot.append(fmt_line)
    plot += _kw_lines({"digits_after_comma": opts["digits_after_comma"]})
    return _script("CumulativeYear", ctor, plot)


def cumulative_to_code(varname: str, opts: dict, *, df_name: str = "df") -> str:
    """Reproduce a :class:`Cumulative` (running total across the whole record)."""
    # Cumulative takes a DataFrame (one curve per column); a single-column frame
    # plots one cumulative curve.
    ctor = [f"    {df_name}[[{varname!r}]],"]
    ctor += _kw_lines({"units": opts["units"]})
    plot = ["    ax=ax,", "    showplot=False,"]
    fmt_line = _fmt_line(opts.get("_format"))
    if fmt_line:
        plot.append(fmt_line)
    plot += _kw_lines({
        "digits_after_comma": opts["digits_after_comma"],
        "show_title": opts["show_title"],
        "fill": opts["fill"],
    })
    return _script("Cumulative", ctor, plot)


def waterfall_to_code(varname: str, opts: dict, *, df_name: str = "df") -> str:
    """Reproduce a :class:`WaterfallPlot` (cumulative budget of period contributions)."""
    ctor = [f"    {df_name}[{varname!r}],"]
    ctor += _kw_lines({
        "series_units": opts["series_units"],
        "resample": opts["resample"],
        "agg": opts["agg"],
        "uptake_is_negative": opts["uptake_is_negative"],
    })
    plot = ["    ax=ax,", "    showplot=False,"]
    fmt_line = _fmt_line(opts.get("_format"))
    if fmt_line:
        plot.append(fmt_line)
    plot += _kw_lines({
        "digits_after_comma": opts["digits_after_comma"],
        "color_uptake": opts["color_uptake"],
        "color_release": opts["color_release"],
        "bar_width": opts["bar_width"],
        "show_connectors": opts["show_connectors"],
    })
    return _script("WaterfallPlot", ctor, plot)


def histogram_to_code(varname: str, opts: dict, *, df_name: str = "df") -> str:
    """Reproduce a :class:`HistogramPlot`."""
    ctor = [
        f"    {df_name}[{varname!r}].dropna(),",
        "    method='n_bins',",
        f"    n_bins={opts['n_bins']!r},",
    ]
    fmt = dict(opts.get("_format") or {})
    if fmt.get("title") is None:
        fmt["title"] = varname
    plot = ["    ax=ax,"]
    fmt_line = _fmt_line(fmt)
    if fmt_line:
        plot.append(fmt_line)
    plot += _kw_lines({
        "highlight_peak": opts["highlight_peak"],
        "show_zscores": opts["show_zscores"],
        "show_zscore_values": opts["show_zscore_values"],
        "show_info": opts["show_info"], "show_counts": opts["show_counts"],
        "show_title": opts["show_title"],
    })
    return _script("HistogramPlot", ctor, plot)


def ridgeline_to_code(varname: str, opts: dict, *, df_name: str = "df") -> str:
    """Reproduce a :class:`RidgeLinePlot` (single-figure, stacked densities)."""
    ctor = [f"    {df_name}[{varname!r}].dropna(),"]
    plot = ["    fig=fig,", "    showplot=False,"]
    fmt_line = _fmt_line(opts.get("_format"))
    if fmt_line:
        plot.append(fmt_line)
    plot += _kw_lines({
        "how": opts["how"], "hspace": opts["hspace"],
        "shade_percentile": opts["shade_percentile"],
        "show_mean_line": opts["show_mean_line"], "ascending": opts["ascending"],
        "kd_kwargs": opts["kd_kwargs"],
    })
    return _script("RidgeLinePlot", ctor, plot,
                   fig_line="fig = plt.figure(figsize=(9, 11))")


def shifted_distribution_to_code(varname: str, opts: dict, *,
                                 df_name: str = "df") -> str:
    """Reproduce a :class:`ShiftedDistributionPlot` (reference vs comparison period)."""
    ctor = [
        f"    series={df_name}[{varname!r}],",
        f"    ref_period={tuple(opts['ref_period'])!r},",
        f"    comp_period={tuple(opts['comp_period'])!r},",
    ]
    plot = ["    ax=ax,"]
    fmt_line = _fmt_line(opts.get("_format"))
    if fmt_line:
        plot.append(fmt_line)
    plot += _kw_lines({
        "ref_label": opts["ref_label"], "comp_label": opts["comp_label"],
        "zone_labels": opts["zone_labels"],
        "show_legend": opts["show_legend"], "show_title": opts["show_title"],
        "show_xaxis": opts["show_xaxis"], "show_yaxis": opts["show_yaxis"],
    })
    return _script("ShiftedDistributionPlot", ctor, plot,
                   fig_line="fig, ax = plt.subplots(figsize=(16, 7))")


def hexbin_to_code(xcol: str, ycol: str, zcol: str, opts: dict, *,
                   df_name: str = "df") -> str:
    """Reproduce a :class:`HexbinPlot` (z aggregated into 2D x/y bins)."""
    pre = [
        f"sub = {df_name}[[{xcol!r}, {ycol!r}, {zcol!r}]]"
        f".dropna(subset=[{xcol!r}, {ycol!r}])",
    ]
    ctor = [
        f"    x=sub[{xcol!r}],",
        f"    y=sub[{ycol!r}],",
        f"    z=sub[{zcol!r}],",
        f"    gridsize={opts['gridsize']!r},",
        f"    normalize_axes={opts['normalize_axes']!r},",
        f"    mincnt={opts['mincnt']!r},",
    ]
    plot = ["    ax=ax,", "    fig=fig,"]
    fmt_line = _fmt_line(opts.get("_format"))
    if fmt_line:
        plot.append(fmt_line)
    plot += _kw_lines({
        "cmap": opts["cmap"], "vmin": opts["vmin"], "vmax": opts["vmax"],
        "color_bad": opts["color_bad"], "zlabel": opts["zlabel"],
        "cb_digits_after_comma": opts["cb_digits_after_comma"],
        "cb_extend": opts["cb_extend"], "show_colormap": opts["show_colormap"],
        "show_values": opts["show_values"],
        "show_values_n_dec_places": opts["show_values_n_dec_places"],
        "show_values_fontsize": opts["show_values_fontsize"],
        "cb_labelsize": opts["cb_labelsize"],
    })
    return _script("HexbinPlot", ctor, plot, pre_lines=pre)


def heatmap_xyz_to_code(xcol: str, ycol: str, zcol: str, opts: dict, *,
                        df_name: str = "df") -> str:
    """Reproduce a :class:`HeatmapXYZ` (z aggregated into a 2D grid of x/y bins).

    The raw x/y/z variables are first binned and aggregated with
    :class:`~diive.analysis.gridaggregator.GridAggregator`, because ``HeatmapXYZ``
    needs pre-aggregated input (one z value per unique (x, y) bin).
    """
    pre = [
        f"sub = {df_name}[[{xcol!r}, {ycol!r}, {zcol!r}]]"
        f".dropna(subset=[{xcol!r}, {ycol!r}])",
        "agg = dv.analysis.GridAggregator(",
        f"    x=sub[{xcol!r}], y=sub[{ycol!r}], z=sub[{zcol!r}],",
        f"    binning_type={opts['binning_type']!r}, n_bins={opts['n_bins']!r},",
        f"    aggfunc={opts['aggfunc']!r}, "
        f"min_n_vals_per_bin={opts['min_n_vals_per_bin']!r},",
        ")",
    ]
    ctor = [
        "    agg,",
        f"    {xcol!r},",
        f"    {ycol!r},",
        f"    {zcol!r},",
    ]
    plot = ["    ax=ax,", "    fig=fig,"]
    fmt_line = _fmt_line(opts.get("_format"))
    if fmt_line:
        plot.append(fmt_line)
    plot += _kw_lines({
        "cmap": opts["cmap"], "vmin": opts["vmin"], "vmax": opts["vmax"],
        "color_bad": opts["color_bad"], "zlabel": opts["zlabel"],
        "cb_digits_after_comma": opts["cb_digits_after_comma"],
        "cb_extend": opts["cb_extend"], "show_colormap": opts["show_colormap"],
        "show_values": opts["show_values"],
        "show_values_n_dec_places": opts["show_values_n_dec_places"],
        "show_values_fontsize": opts["show_values_fontsize"],
        "cb_labelsize": opts["cb_labelsize"],
    })
    return _script("HeatmapXYZ.from_gridaggregator", ctor, plot, pre_lines=pre)


def windrose_to_code(valuecol: str, winddircol: str, zcol: str | None, opts: dict, *,
                     df_name: str = "df") -> str:
    """Reproduce a :class:`WindRosePlot` (polar, per-sector aggregate)."""
    ctor = [
        f"    series={df_name}[{valuecol!r}],",
        f"    wind_dir={df_name}[{winddircol!r}],",
        f"    agg={opts['agg']!r},",
        f"    n_sectors={opts['n_sectors']!r},",
    ]
    if zcol:
        ctor.append(f"    z={df_name}[{zcol!r}],")
        ctor.append(f"    z_agg={opts['z_agg']!r},")
    plot = ["    ax=ax,"]
    fmt_line = _fmt_line(opts.get("_format"))
    if fmt_line:
        plot.append(fmt_line)
    plot += _kw_lines({
        "cmap": opts["cmap"], "color": opts["color"],
        "vmin": opts["vmin"], "vmax": opts["vmax"],
        "show_colorbar": opts["show_colorbar"], "cb_label": opts["cb_label"],
        "cb_digits_after_comma": opts["cb_digits_after_comma"],
        "max_sector_labels": opts["max_sector_labels"],
    })
    return _script(
        "WindRosePlot", ctor, plot,
        fig_line='fig, ax = plt.subplots(figsize=(9, 9), '
                 'subplot_kw={"projection": "polar"})')


def treering_to_code(varname: str, opts: dict, *, df_name: str = "df") -> str:
    """Reproduce a :class:`TreeRingPlot` (concentric annual rings on a polar axis).

    ``opts['style']`` selects the renderer: ``'filled'`` -> ``plot`` (a colour
    mesh per ring), ``'line'`` -> ``plot_line`` (one radial line trace per year,
    wiggling by the data value). The line style adds a few extra parameters.
    """
    ctor = [
        f"    df={df_name}[[{varname!r}]],",
        f"    value_col={varname!r},",
        f"    resample_freq={opts['resample_freq']!r},",
    ]
    plot = ["    ax=ax,"]
    fmt_line = _fmt_line(opts.get("_format"))
    if fmt_line:
        plot.append(fmt_line)
    kwargs = {
        "cmap": opts["cmap"], "vmin": opts["vmin"], "vmax": opts["vmax"],
        "show_month_labels": opts["show_month_labels"],
        "show_month_lines": opts["show_month_lines"],
        "show_year_labels": opts["show_year_labels"],
        "show_year_separators": opts["show_year_separators"],
        "year_label_frequency": opts["year_label_frequency"],
        "cb_label": opts["cb_label"],
        "cb_digits_after_comma": opts["cb_digits_after_comma"],
        "cb_labelsize": opts["cb_labelsize"],
    }
    if opts["style"] == "line":
        kwargs.update({
            "linewidth": opts["linewidth"], "alpha": opts["alpha"],
            "amplitude_scale": opts["amplitude_scale"],
            "ring_width": opts["ring_width"],
        })
    plot += _kw_lines(kwargs)
    return _script(
        "TreeRingPlot", ctor, plot,
        fig_line='fig, ax = plt.subplots(figsize=(10, 10), '
                 'subplot_kw={"projection": "polar"})',
        plot_method=("plot_line" if opts["style"] == "line" else "plot"))


def datetime_surface_to_code(varname: str, *, cmap: str = "viridis",
                             df_name: str = "df") -> str:
    """Reproduce the 3-D date x time-of-day relief surface.

    The GUI renders it with PyVista (GPU); this snippet rebuilds the same numeric
    grid via :func:`~diive.plotting.datetime_surface_grid` and draws a matplotlib
    3-D surface so it runs anywhere without the optional 3-D extra.
    """
    return (
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import diive as dv\n"
        "\n"
        f"grid = dv.plotting.datetime_surface_grid({df_name}[{varname!r}])\n"
        "xx, yy = np.meshgrid(grid.x_hours, grid.y_days)\n"
        "z = np.ma.masked_invalid(grid.z)\n"
        "\n"
        "fig = plt.figure(figsize=(10, 7))\n"
        "ax = fig.add_subplot(111, projection='3d')\n"
        f"ax.plot_surface(xx, yy, z, cmap={cmap!r})\n"
        "ax.set_xlabel('Time of day (hours)')\n"
        "ax.set_ylabel('Days since start')\n"
        f"ax.set_zlabel({varname!r})\n"
        f"ax.set_title('3D surface - ' + {varname!r})\n"
        "plt.show()\n"
    )
