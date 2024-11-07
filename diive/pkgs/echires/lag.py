import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from pandas import DataFrame
from scipy.signal import find_peaks

from diive.core.plotting.plotfuncs import default_format


# todo logger: setup_dyco

class MaxCovariance:

    def __init__(
            self,
            df: DataFrame,
            var_reference: str,
            var_lagged: str,
            lgs_winsize_from: int = -1000,
            lgs_winsize_to: int = 1000,
            shift_stepsize: int = 1,
            segment_name: str = "segment_name_here",
    ):
        """ Determine the time lag between two variables by finding the maximum
         covariance.

        Args:
            df: Input data.
            var_reference: Name of reference variable in *df*. Lag of *var_lagged* will be determined in
                relation to this reference.
            var_lagged: Name of lagged variable in *df*. Lag will be determined in relation *var_reference*.
            lgs_winsize_from: Start of lag search time window (in number of records).
            lgs_winsize_to: End of lag search time window (in number of records).
            shift_stepsize: Step-size for lag search (in number of records).
            segment_name:
        """

        self.segment_df = df
        self.var_reference = var_reference
        self.var_lagged = var_lagged
        self.lgs_winsize_from = lgs_winsize_from
        self.lgs_winsize_to = lgs_winsize_to
        self.shift_stepsize = shift_stepsize  # Negative moves lagged values "upwards" in column
        self.segment_name = segment_name

        # Init new variables
        self.props_peak_auto = None
        self._cov_df = DataFrame()
        self.idx_peak_cov_abs_max = None
        self.idx_peak_auto = None
        self.idx_instantaneous_default_lag = None

    @property
    def cov_df(self) -> DataFrame:
        """Overall flag, calculated from individual flags from multiple iterations."""
        if not isinstance(self._cov_df, DataFrame):
            raise Exception('No covariance results available.')
        return self._cov_df

    def get(self):
        """Get covariance results as DataFrame and info about the
        covariance peak"""
        return self.cov_df, self.props_peak_auto

    def run(self):
        """Execute processing stack"""

        # Reset covariance results
        self._cov_df = DataFrame()

        # Setup
        self._cov_df = self._setup_lagsearch_df()

        # Detect and flag max covariance peak
        self._cov_df = self._find_max_cov_peak()

        # Detect and flag automatic peak
        self._cov_df, self.props_peak_auto = self.find_auto_peak()

        # Get indices of peaks and instantaneous default lag
        self.idx_peak_cov_abs_max = \
            self.get_peak_idx(cov_df=self.cov_df, flag_col='flag_peak_max_cov_abs')
        self.idx_peak_auto = \
            self.get_peak_idx(cov_df=self.cov_df, flag_col='flag_peak_auto')
        self.idx_instantaneous_default_lag = \
            self.get_peak_idx(cov_df=self.cov_df, flag_col='flag_instantaneous_default_lag')

    def _setup_lagsearch_df(self) -> DataFrame:
        """
        Setup DataFrame that collects lagsearch results

        Returns
        -------
        pandas DataFrame prepared for storing segment lag search results
        """
        df = DataFrame(columns=['index', 'segment_name', 'shift', 'cov', 'cov_abs',
                                'flag_peak_max_cov_abs', 'flag_peak_auto'])
        df['shift'] = range(int(self.lgs_winsize_from),
                            int(self.lgs_winsize_to) + self.shift_stepsize,
                            self.shift_stepsize)
        df['index'] = pd.NaT
        # df['index'] = np.nan
        df['segment_name'] = self.segment_name
        df['cov'] = np.nan
        df['cov_abs'] = np.nan
        df['flag_peak_max_cov_abs'] = False  # Flag True = found peak
        df['flag_peak_auto'] = False
        df['flag_instantaneous_default_lag'] = False
        return df

    def find_auto_peak(self):
        """Automatically find peaks in covariance time series.

        The found peak is flagged TRUE in *cov_df*.

        Peaks are searched automatically using scipy's .find_peaks method.
        The peak_score is calculated for each peak, the top-scoring peaks
        are retained. If the previously calculated max covariance peak is
        part of the top-scoring peaks, the record at which these two peaks
        were found is flagged in cov_df. Basically, this method works as a
        validation step to detect high-quality covariance peaks that are
        later used to calculate default lag times.

        Returns
        -------
        cov_df
        props_peak_df

        """

        cov_df = self.cov_df.copy()

        found_peaks_idx, found_peaks_dict = find_peaks(cov_df['cov_abs'],
                                                       height=0, width=0, prominence=0)
        found_peaks_props_df = DataFrame.from_dict(found_peaks_dict)
        found_peaks_props_df['idx_in_cov_df'] = found_peaks_idx

        # Calculate peak score, a combination of peak_height, prominences and width_heights
        found_peaks_props_df['peak_score'] = found_peaks_props_df['prominences'] \
                                             * found_peaks_props_df['width_heights'] \
                                             * found_peaks_props_df['peak_heights']
        found_peaks_props_df['peak_score'] = found_peaks_props_df['peak_score'] ** .5  # Make numbers smaller
        found_peaks_props_df['peak_rank'] = found_peaks_props_df['peak_score'].rank(ascending=False)

        score_threshold = found_peaks_props_df['peak_score'].quantile(0.9)
        top_scoring_peaks_df = found_peaks_props_df.loc[found_peaks_props_df['peak_score'] >= score_threshold]
        top_scoring_peaks_df = top_scoring_peaks_df.sort_values(by=['peak_score', 'prominences', 'width_heights'],
                                                                ascending=False)

        idx_peak_cov_abs_max = \
            self.get_peak_idx(cov_df=cov_df, flag_col='flag_peak_max_cov_abs')

        # Check if peak of max absolute covariance is also in auto-detected peaks
        if idx_peak_cov_abs_max in top_scoring_peaks_df['idx_in_cov_df'].values:
            props_peak_df = top_scoring_peaks_df.iloc[
                top_scoring_peaks_df['idx_in_cov_df'].values == idx_peak_cov_abs_max]
            props_peak_df = props_peak_df.iloc[0]
            peak_idx = int(props_peak_df['idx_in_cov_df'])
            cov_df.loc[peak_idx, 'flag_peak_auto'] = True
        else:
            props_peak_df = DataFrame()

        return cov_df, props_peak_df

    def _find_max_cov_peak(self) -> DataFrame:
        """Find maximum absolute covariance.

        Returns
        -------
        pandas DataFrame with segment lag search results
        """

        cov_df = self.cov_df.copy()

        _index = self.segment_df.index.copy()
        _var_lagged = self.segment_df[self.var_lagged].copy()
        _var_reference = self.segment_df[self.var_reference].copy()

        # Check if data column is empty
        if _var_lagged.dropna().empty:
            pass

        else:

            # start_time = time.time()

            for ix, row in cov_df.iterrows():
                shift = int(row['shift'])
                try:
                    if shift < 0:
                        index_shifted = _index[-shift]  # Note the negative sign
                        # index_shifted = _index.iloc[-shift]  # Note the negative sign
                        # index_shifted = str(_segment_df['index'].iloc[-shift])  # Note the negative sign
                        # index_shifted = str(_segment_df['index'][-shift])  # Note the negative sign
                    else:
                        # todo why time?
                        index_shifted = pd.NaT

                    # print(index_shifted)

                    # # Shift and calculate covariance (pandas)
                    _scalar_data_shifted = _var_lagged.shift(shift)
                    cov = _var_reference.cov(_scalar_data_shifted)

                    # # Shift and calculate covariance (numpy)
                    # # As an alternative, covariance can be calculated using numpy. While
                    # # this approach might be faster in many cases, it is slower here, I
                    # # assume because of the presence of NaNs in the data that have to be
                    # # masked first.
                    # # Requires masking the NaN values in _scalar_data_shifted_array.
                    # _scalar_data_shifted = _var_lagged.shift(shift)
                    # _var_reference_array = _var_reference.values
                    # _scalar_data_shifted_array = _scalar_data_shifted.values
                    # _masked_scalar_data_shifted_array = (
                    #     np.ma.array(_scalar_data_shifted_array, mask=np.isnan(_scalar_data_shifted_array)))
                    # # Calculating the covariance, extract from covariance matrix
                    # cov = np.ma.cov(_var_reference_array, _masked_scalar_data_shifted_array)[0,1]

                    cov_df.loc[cov_df['shift'] == row['shift'], 'cov'] = cov
                    cov_df.loc[cov_df['shift'] == row['shift'], 'index'] = index_shifted

                except IndexError:
                    # If not enough data in the file to perform the shift, continue
                    # to the next shift and try again. This can happen for the last
                    # segments in each file, when there is no more data available
                    # at the end.
                    continue

            # # Timing (for testing)
            # end_time = time.time()
            # execution_time = end_time - start_time
            # print(f"Function execution time: {execution_time:.6f} seconds")

            # Results
            cov_df['cov_abs'] = cov_df['cov'].abs()
            cov_max_ix = cov_df['cov_abs'].idxmax()
            cov_df.loc[cov_max_ix, 'flag_peak_max_cov_abs'] = True

        return cov_df

    @staticmethod
    def get_peak_idx(cov_df, flag_col):
        """
        Search a boolean column for *True* and return index if successful

        Parameters
        ----------
        cov_df: pandas DataFrame
        flag_col: str
            Column name in *df*.

        Returns
        -------
        The index where flag_col is *True*.

        """
        # Peak of maximum covariance
        if True in cov_df[flag_col].values:
            idx = cov_df.loc[cov_df[flag_col] == True, :].index.values[0]
        else:
            idx = False
        return idx

    def plot_scatter_cov(self, title: str = None, txt_info: str = "", outpath: str = None, outname: str = None):
        """Make scatter plot with z-values as colors and display found max covariance."""

        # Setup figure
        fig = plt.figure(facecolor='white', figsize=(16, 9))
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        # gs.update(wspace=0.1, hspace=0.1, left=0.1, right=0, top=0.9, bottom=0.1)
        ax = fig.add_subplot(gs[0, 0])

        if title:
            font = {'family': 'sans-serif', 'color': 'black', 'weight': 'bold', 'size': 20, 'alpha': 1, }
            ax.set_title(title, fontdict=font)

        # Covariance and shift data
        x_shift = self.cov_df.loc[:, 'shift']
        y_cov = self.cov_df.loc[:, 'cov']
        z_cov_abs = self.cov_df.loc[:, 'cov_abs']

        # Main plot: covariances per shift, vals from abs cov as scatter point colors
        ax.scatter(x_shift, y_cov, c=z_cov_abs,
                   alpha=0.9, edgecolors='none',
                   marker='o', s=24, cmap='coolwarm', zorder=98)

        # Use abs cov also as line colors
        self.z_as_colored_lines(fig=fig, ax=ax,
                                x=x_shift,
                                y=y_cov,
                                z=z_cov_abs)

        # Markers for points of interest, e.g. peaks
        txt_info = self.mark_max_cov_abs_peak(ax=ax, txt_info=txt_info)
        txt_info = self.mark_auto_detected_peak(ax=ax, txt_info=txt_info)
        txt_info = self.mark_instantaneous_default_lag(ax=ax, txt_info=txt_info)

        # Add info text
        ax.text(0.02, 0.98, txt_info,
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, size=10, color='black', backgroundcolor='none', zorder=100)

        # Format & legend
        default_format(ax=ax, ax_labels_fontcolor='black', ax_labels_fontsize=12,
                       ax_xlabel_txt='lag [records]', ax_ylabel_txt='covariance', txt_ylabel_units='')
        ax.legend(frameon=False, loc='upper right').set_zorder(100)

        fig.tight_layout()

        if outpath:
            self._save_cov_plot(fig=fig, outpath=outpath, outname=outname)
        else:
            fig.show()

    def _save_cov_plot(self, fig, outpath, outname):
        """
        Save covariance plot for segment to png

        Parameters
        ----------
        fig: Figure
            The plot that is saved

        Returns
        -------
        None
        """
        outpath = outpath / outname
        fig.savefig(f"{outpath}", format='png', bbox_inches='tight', facecolor='w',
                    transparent=True, dpi=100)
        plt.close(fig)
        return

    def z_as_colored_lines(self, fig, ax, x, y, z):
        """
        Add z values as colors to line plot

        From: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html
        Create a set of line segments so that we can color them individually
        This creates the points as a N x 1 x 2 array so that we can stack points
        together easily to get the segments. The segments array for line collection
        needs to be (numlines) x (points per line) x 2 (for x and y)

        Parameters
        ----------
        fig: Figure
        ax: axis
        x: x values, shift
        y: y values, covariance
        z: z values, absolute covariance

        Returns
        -------
        None
        """

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(z.min(), z.max())
        lc = LineCollection(segments, cmap='coolwarm', norm=norm)
        # Set the values used for colormapping
        lc.set_array(z)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        cbar = fig.colorbar(line, ax=ax, location='right', pad=0.03)
        cbar.set_label('absolute covariance', rotation=90)

    def mark_max_cov_abs_peak(self, ax, txt_info):
        """
        Mark peak of max absolute covariance

        Parameters
        ----------
        ax: axis
        txt_info: str
            Info text about segment lag search that is shown in the plot

        Returns
        -------
        Extended info text
        """
        if self.idx_peak_cov_abs_max:
            ax.scatter(self.cov_df.iloc[self.idx_peak_cov_abs_max]['shift'],
                       self.cov_df.iloc[self.idx_peak_cov_abs_max]['cov'],
                       alpha=1, edgecolors='red', marker='o', s=72, c='red',
                       label='maximum absolute covariance', zorder=99)
            txt_info += \
                f"\nFOUND PEAK MAX ABS COV\n" \
                f"    cov {self.cov_df.iloc[self.idx_peak_cov_abs_max]['cov']:.3f}\n" \
                f"    record {self.cov_df.iloc[self.idx_peak_cov_abs_max]['shift']}\n"
        else:
            txt_info += "\n(!)NO PEAK MAX ABS COV FOUND\n"
        return txt_info

    def mark_instantaneous_default_lag(self, ax, txt_info):
        """
        Mark instantaneous default time lag

        Used in Phase 3

        Parameters
        ----------
        ax: axis
        txt_info: str
            Info text about segment lag times, shown in plot

        Returns
        -------
        Extended info text
        """
        if self.idx_instantaneous_default_lag:
            ax.scatter(self.cov_df.iloc[self.idx_instantaneous_default_lag]['shift'],
                       self.cov_df.iloc[self.idx_instantaneous_default_lag]['cov'],
                       alpha=1, edgecolors='green', marker='o', s=200, c='None',
                       label='instantaneous default lag', zorder=90)
            txt_info += \
                f"\nTIME LAG SET TO DEFAULT (PHASE 3)\n" \
                f"    lag time was set to default\n" \
                f"    cov {self.cov_df.iloc[self.idx_instantaneous_default_lag]['cov']:.3f}\n" \
                f"    record {self.cov_df.iloc[self.idx_instantaneous_default_lag]['shift']}\n"
        return txt_info

    def mark_auto_detected_peak(self, ax, txt_info):
        """
        Mark auto-detected peak

        Parameters
        ----------
        ax: axis
        txt_info: str
            Info text about segment lag search that is shown in the plot

        Returns
        -------
        Extended info text
        """
        if self.idx_peak_auto:
            ax.scatter(self.cov_df.iloc[self.idx_peak_auto]['shift'],
                       self.cov_df.iloc[self.idx_peak_auto]['cov'],
                       alpha=1, edgecolors='black', marker='o', s=200, c='None',
                       label='auto-detected peak', zorder=90)
            txt_info += \
                f"\nFOUND AUTO-PEAK\n" \
                f"    cov {self.cov_df.iloc[self.idx_peak_auto]['cov']:.3f}\n" \
                f"    record {self.cov_df.iloc[self.idx_peak_auto]['shift']}\n" \
                f"    peak_score {self.props_peak_auto['peak_score']:.0f}\n" \
                f"    peak_rank {self.props_peak_auto['peak_rank']:.0f}\n" \
                f"    peak_height {self.props_peak_auto['peak_heights']:.0f}\n" \
                f"    prominence {self.props_peak_auto['prominences']:.0f}\n" \
                f"    width {self.props_peak_auto['widths']:.0f}\n" \
                f"    width_height {self.props_peak_auto['width_heights']:.0f}\n"
        else:
            txt_info += "\n(!)NO AUTO-PEAK FOUND\n"
        return txt_info


def example():
    from pathlib import Path
    from diive.core.io.filereader import ReadFileType
    from diive.core.io.filereader import search_files

    OUTDIR = r'P:\Flux\RDS_calculations\DEG_EddyMercury\Magic file for Diive\OUT'
    SEARCHDIRS = [r'P:\Flux\RDS_calculations\DEG_EddyMercury\Magic file for Diive\IN']
    PATTERN = 'DEG_*.csv'
    FILEDATEFORMAT = 'DEG_%Y%m%d%H%M.csv'
    FILE_GENERATION_RES = '6h'
    DATA_NOMINAL_RES = 0.05
    FILES_HOW_MANY = 1
    FILETYPE = 'ETH-MERCURY-CSV-20HZ'
    DATA_SPLIT_DURATION = '30min'
    DATA_SPLIT_OUTFILE_PREFIX = 'CH-DAS_'
    DATA_SPLIT_OUTFILE_SUFFIX = '_30MIN-SPLIT'

    # from diive.core.io.filesplitter import FileSplitterMulti
    # fsm = FileSplitterMulti(
    #     outdir=OUTDIR,
    #     searchdirs=SEARCHDIRS,
    #     filename_pattern=PATTERN,
    #     filename_date_format=FILEDATEFORMAT,
    #     file_generation_freq=FILE_GENERATION_RES,
    #     data_nominal_res=DATA_NOMINAL_RES,
    #     files_split_how_many=FILES_HOW_MANY,
    #     filetype=FILETYPE,
    #     data_split_duration=DATA_SPLIT_DURATION,
    #     data_split_outfile_prefix=DATA_SPLIT_OUTFILE_PREFIX,
    #     data_split_outfile_suffix=DATA_SPLIT_OUTFILE_SUFFIX
    # )
    # fsm.run()

    filelist = search_files(searchdirs=r'P:\Flux\RDS_calculations\DEG_EddyMercury\Magic file for Diive\OUT\splits',
                            pattern='DEG_*.csv.gz')

    # Settings
    # U = 'U_[HS50-A]'  # Name of the horizontal wind component measured in x-direction, measured in units of m s-1
    # V = 'V_[HS50-A]'  # Name of the horizontal wind component measured in y-direction, measured in units of m s-1
    # W = 'W_[HS50-A]'  # Name of the vertical wind component measured in z-direction, measured in units of m s-1
    # C = 'CO2_DRY_[IRGA72-A]'  # Name of the measured dry mole fraction, here in umol CO2 mol-1

    for filepath in filelist:
        # Read file
        df = pd.read_csv(filepath)
        df = df.replace(-9999, np.nan)
        # df, meta = ReadFileType(filepath=filepath,
        #                         filetype='GENERIC-CSV-HEADER-1ROW-TS-END-FULL-NS-20HZ',
        #                         data_nrows=None,
        #                         output_middle_timestamp=True).get_filedata()

        # # Already done in input files:
        # from diive.pkgs.echires.windrotation import WindRotation2D
        # # Wind rotation for turbulent fluctuations
        # u = df[U].copy()
        # v = df[V].copy()
        # w = df[W].copy()
        # c = df[C].copy()
        # wr = WindRotation2D(u=u, v=v, w=w, c=c)
        # primes_df = wr.get_primes()
        #
        # # Add turbulent fluctuations to file data
        # merged_df = pd.concat([primes_df, df], axis=1)

        # Find maximum covariance
        mc = MaxCovariance(
            df=df,
            var_reference='z_TURB',
            var_lagged='Lumex_Hg0_microgram_m3_TURB',
            lgs_winsize_from=-300,
            lgs_winsize_to=300,
            shift_stepsize=1,
            segment_name="test"
        )

        mc.run()

        mc.plot_scatter_cov(title=str(Path(filepath.name)))
        cov_df, props_peak_auto = mc.get()
        cov_df.to_csv(Path(OUTDIR, 'cov_df.csv'), index=False)
        foundlag = cov_df.loc[cov_df['flag_peak_max_cov_abs'] == True]
        lag = cov_df.iloc[foundlag.index]['shift']
        lag = lag.tolist()[0]
        print(lag)

if __name__ == '__main__':
    example()
