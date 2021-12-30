from pathlib import Path

import pandas as pd

import pkgs.dfun
from pkgs.dfun.stats import CalcTimeSeriesStats


def statsbox_txt(focus_stats_df, stat_name, stat_label, prev_stats_available, show_prev_stats):
    """Add stats into GUI statsbox"""

    current_value = focus_stats_df[stat_name].iloc[-1]
    stat_label.setToolTip(str(current_value))
    if prev_stats_available:
        if show_prev_stats:
            prev_value = focus_stats_df[stat_name].iloc[-2]
            delta = current_value - prev_value
            sign = '+' if delta > 0 else ''

            red = '#e57373'
            green = '#81c784'
            grey = '#455a64'  # blue grey 700

            if delta == 0:
                delta_color = grey
            elif delta < 0:
                delta_color = red
            else:
                delta_color = green

            stat_label.setText(f'{current_value:.2f} <font color=\"{delta_color}\">{sign}{delta:.2f}</font>')
        else:
            stat_label.setText(f'{current_value}')
    else:
        try:
            stat_label.setText('{:.2f}'.format(current_value))
        except (ValueError, TypeError) as e:
            stat_label.setText('{}'.format(current_value))


def statsboxes_contents(self):
    """Insert stats into GUI statsboxes"""

    stats_len = len(self.focus_stats_df)
    prev_stats_available = True if stats_len > 1 else False

    # Dict of statsboxes
    # Contains: column name in stats_df: [stats label, show delta in label]
    statsboxesdict = {
        'run_id': [self.lbl_run_id, False],
        'startdate': [self.lbl_startdate_val, False],
        'enddate': [self.lbl_enddate_val, False],
        'period': [self.lbl_period_val, False],
        'nov': [self.lbl_nov_val, True],
        'dtype': [self.lbl_type_val, False],
        'missing': [self.lbl_missingvalues_val, True],
        'missing_perc': [self.lbl_missingvaluesperc_val, True],
        'mean': [self.lbl_mean_val, True],
        'sd': [self.lbl_sd_val, True],
        'sd/mean': [self.lbl_sdmean_val, True],
        'median': [self.lbl_median_val, True],
        'max': [self.lbl_max_val, True],
        'p95': [self.lbl_p95_val, True],
        'p75': [self.lbl_p75_val, True],
        'p25': [self.lbl_p25_val, True],
        'p05': [self.lbl_p05_val, True],
        'min': [self.lbl_min_val, True],
        # 'mad': [self.lbl_mad_val, True],
        'cumsum': [self.lbl_cumsum_val, True],
        'dataset_freq': [self.lbl_dataset_freq_val, False],
        'dataset_num_cols': [self.lbl_dataset_num_cols_val, False],
        'dataset_seasons_available': [self.lbl_dataset_seasons_available_val, False],
        'dataset_events_available': [self.lbl_dataset_events_available_val, False],
        'dataset_current_outdir': [self.lbl_dataset_current_outdir, False],
        'dataset_current_filetype': [self.lbl_dataset_current_filetpye, False],
    }

    # Update all statsboxes in GUI
    for key, value in statsboxesdict.items():
        statsbox_txt(focus_stats_df=self.focus_stats_df,
                     stat_name=key,
                     stat_label=value[0],
                     prev_stats_available=prev_stats_available,
                     show_prev_stats=value[1])


class AssembleStats:
    """Collect stats for statsboxes"""

    def __init__(
            self,
            series: pd.Series,
            run_id: str,
            data_df: pd.DataFrame,
            project_outdir: Path,
            filetype_config: dict,
            prev_stats_df: pd.DataFrame
    ):
        self.series = series
        self.run_id = run_id
        self.data_df = data_df
        self.project_outdir = project_outdir
        self.filetype_config = filetype_config
        self.prev_stats_df = prev_stats_df

        self.stats_df = pd.DataFrame()

        self._assemble()

    def get(self):
        return self.stats_df

    def _assemble(self):
        timeseries_stats = CalcTimeSeriesStats(series=self.series).get()

        dataset_run_stats = CalcDatasetRunStats(run_id=self.run_id,
                                                data_df=self.data_df,
                                                project_outdir=self.project_outdir,
                                                filetype_config=self.filetype_config).get()
        stats_df = pd.concat([timeseries_stats, dataset_run_stats], axis=1)

        self.stats_df = self.prev_stats_df.append(stats_df, ignore_index=True)


class CalcDatasetRunStats():
    """Calc stats relevant to the current run and dataset"""

    def __init__(self, run_id, data_df, project_outdir, filetype_config):
        self.data_df = data_df.copy()
        self.run_id = run_id
        self.project_outdir = project_outdir
        self.filetype_config = filetype_config

        self.stats_df = pd.DataFrame()

        self._calc()

    def get(self):
        return self.stats_df

    def _calc(self):
        self.stats_df.loc[0, 'run_id'] = self.run_id

        # todo check if same as in config file
        self.stats_df.loc[0, 'dataset_freq'] = pkgs.dfun.frames.infer_freq(self.data_df.index)

        self.stats_df.loc[0, 'dataset_num_cols'] = len(self.data_df.columns)
        self.stats_df.loc[0, 'dataset_seasons_available'] = self._detect_if_seasons_available(df=self.data_df)
        self.stats_df.loc[0, 'dataset_events_available'] = self._detect_if_events_available(df=self.data_df)
        self.stats_df.loc[0, 'dataset_current_outdir'] = self.project_outdir
        self.stats_df.loc[0, 'dataset_current_filetype'] = self.filetype_config['GENERAL']['NAME']

    def _detect_if_seasons_available(self, df):
        collist = df.columns.to_list()
        occurrences = sum('GRP_SEASON_' in s[0] for s in collist)
        available = 'Yes' if occurrences > 0 else 'No'
        return available

    def _detect_if_events_available(self, df):
        collist = df.columns.to_list()
        occurrences = sum('[event]' in s[1] for s in collist)
        available = 'Yes' if occurrences > 0 else 'No'
        return available
