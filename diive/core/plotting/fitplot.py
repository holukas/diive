def fitplot(
        flux_bts_results: dict,
        ax,
        highlight_year: int = None,
        color: str = 'black',
        color_fitline: str = 'black',
        label: str = 'label',
        edgecolor: str = 'none',
        marker: str = 'o',
        alpha: float = 1
):
    # x/y
    _numvals_y = len(flux_bts_results['y'])
    _startyear_y = flux_bts_results['y'].index.year[0]
    _endyear_y = flux_bts_results['y'].index.year[-1]
    line_xy = ax.scatter(flux_bts_results['x'],
                         flux_bts_results['y'],
                         edgecolor=edgecolor, color=color, alpha=alpha, s=60,
                         label=f"{label} {_startyear_y}-{_endyear_y} ({_numvals_y} days)",
                         zorder=1, marker=marker)

    # Highlight year
    if highlight_year:
        import pandas as pd
        _subset_year = pd.DataFrame()
        _subset_year['x'] = flux_bts_results['x']
        _subset_year['y'] = flux_bts_results['y']
        _subset_year.index = pd.to_datetime(_subset_year.index)
        _subset_year = _subset_year.loc[_subset_year.index.year == highlight_year, :]
        _numvals_y = len(_subset_year['y'])
        line_highlight = ax.scatter(_subset_year['x'],
                                    _subset_year['y'],
                                    edgecolor='#455A64', color='#FFD54F',  # amber 300
                                    alpha=1, s=100,
                                    label=f"{label} {highlight_year} ({_numvals_y} days)",
                                    zorder=100, marker=marker)

    # Fit
    a = flux_bts_results['fit_params_opt'][0]
    b = flux_bts_results['fit_params_opt'][1]
    c = flux_bts_results['fit_params_opt'][2]
    label_fit = rf"$y = {a:.2f}x^2 + {b:.2f}x + {c:.2f}$"
    line_fit, = ax.plot(flux_bts_results['fit_df']['fit_x'],
                        flux_bts_results['fit_df']['nom'],
                        c=color_fitline, lw=3, zorder=99, alpha=1, label=label_fit)

    # Fit confidence region
    # Uncertainty lines (95% confidence)
    line_fit_ci = ax.fill_between(flux_bts_results['fit_df']['fit_x'],
                                  flux_bts_results['fit_df']['nom_lower_ci95'],
                                  flux_bts_results['fit_df']['nom_upper_ci95'],
                                  alpha=.2, color=color_fitline, zorder=99,
                                  label="95% confidence region")

    # Fit prediction interval
    # Lower prediction band (95% confidence)
    ax.plot(flux_bts_results['fit_df']['fit_x'],
            flux_bts_results['fit_df']['lower_predband'],
            color=color_fitline, ls='--', zorder=99, lw=2,
            label="95% prediction interval")
    # Upper prediction band (95% confidence)
    line_fit_pb, = ax.plot(flux_bts_results['fit_df']['fit_x'],
                           flux_bts_results['fit_df']['upper_predband'],
                           color=color_fitline, ls='--', zorder=99, lw=2,
                           label="95% prediction interval")

    if highlight_year:
        return line_xy, line_fit, line_fit_ci, line_fit_pb, line_highlight

    return line_xy, line_fit, line_fit_ci, line_fit_pb

