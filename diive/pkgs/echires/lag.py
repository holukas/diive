import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from pandas import DataFrame
from scipy.signal import find_peaks


# todo logger: setup_dyco

class MaxCovariance:
    """
    Determine the time lag for each file by calculating covariances
    and finding covariance peaks
    """

    def __init__(
            self,
            segment_df: DataFrame,
            var_reference: str,
            var_lagged: str,
            lgs_winsize_from: int = -1000,
            lgs_winsize_to: int = 1000,
            shift_stepsize: int = 1,
            segment_name: str = "segment_name_here",
    ):

        self.segment_df = segment_df
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
            raise Exception(f'No covariance results available.')
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

    def plot(self):
        # todo Plot covariance
        self.make_scatter_cov()

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

        _segment_df = self.segment_df.copy()
        _segment_df['index'] = _segment_df.index

        # Check if data column is empty
        if _segment_df[self.var_lagged].dropna().empty:
            pass

        else:
            for ix, row in cov_df.iterrows():
                shift = int(row['shift'])
                try:
                    if shift < 0:
                        index_shifted = _segment_df['index'].iloc[-shift]  # Note the negative sign
                        # index_shifted = str(_segment_df['index'].iloc[-shift])  # Note the negative sign
                        # index_shifted = str(_segment_df['index'][-shift])  # Note the negative sign
                    else:
                        # todo why time?
                        index_shifted = pd.NaT
                    scalar_data_shifted = _segment_df[self.var_lagged].shift(shift)
                    # cov = _segment_df[ref_sig].corr(scalar_data_shifted)
                    cov = _segment_df[self.var_reference].cov(scalar_data_shifted)
                    cov_df.loc[cov_df['shift'] == row['shift'], 'cov'] = cov
                    cov_df.loc[cov_df['shift'] == row['shift'], 'index'] = index_shifted

                except IndexError:
                    # If not enough data in the file to perform the shift, continue
                    # to the next shift and try again. This can happen for the last
                    # segments in each file, when there is no more data available
                    # at the end.
                    continue

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

    def make_scatter_cov(self):
        """Make scatter plot with z-values as colors and display found max covariance."""

        # Setup figure
        fig = plt.figure(facecolor='white', figsize=(16, 9))
        gs = gridspec.GridSpec(1, 1)  # rows, cols
        gs.update(wspace=0.3, hspace=0.3, left=0.03, right=0.97, top=0.97, bottom=0.03)
        ax = fig.add_subplot(gs[0, 0])

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

        # txt_info = \
        #     # f"PHASE: {self.phase}\n" \
        #     # f"Iteration: {self.iteration}\n" \
        #     f"Time lag search window: from {self.lgs_winsize_from} to {self.lgs_winsize_to} records\n" \
        #     f"Segment name: {self.segment_name}\n" \
        #     # f"Segment start: {self.segment_start}\n" \
        #     # f"Segment end: {self.segment_end}\n" \
        #     # f"File: {self.filename} - File date: {self.file_idx}\n" \
        #     f"Lag search step size: {self.shift_stepsize} records\n"

        txt_info = \
            f"Time lag search window: from {self.lgs_winsize_from} to {self.lgs_winsize_to} records\n" \
            f"Segment name: {self.segment_name}\n" \
            f"Lag search step size: {self.shift_stepsize} records\n"

        # Markers for points of interest, e.g. peaks
        txt_info = self.mark_max_cov_abs_peak(ax=ax, txt_info=txt_info)
        txt_info = self.mark_auto_detected_peak(ax=ax, txt_info=txt_info)
        txt_info = self.mark_instantaneous_default_lag(ax=ax, txt_info=txt_info)

        # Add info text
        ax.text(0.02, 0.98, txt_info,
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, size=10, color='black', backgroundcolor='none', zorder=100)

        # Format & legend
        default_format(ax=ax, label_color='black', fontsize=12,
                       txt_xlabel='lag [records]', txt_ylabel='covariance', txt_ylabel_units='-')
        ax.legend(frameon=False, loc='upper right').set_zorder(100)

        fig.show()


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
        cbar = fig.colorbar(line, ax=ax)
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
            txt_info += \
                f"\n(!)NO PEAK MAX ABS COV FOUND\n"
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
            txt_info += \
                f"\n(!)NO AUTO-PEAK FOUND\n"
        return txt_info




def default_format(ax, fontsize=12, label_color='black',
                   txt_xlabel='', txt_ylabel='', txt_ylabel_units='',
                   width=1, length=5, direction='in', colors='black', facecolor='white'):
    """Apply default format to plot."""
    ax.set_facecolor(facecolor)
    ax.tick_params(axis='x', width=width, length=length, direction=direction, colors=colors, labelsize=fontsize,
                   top=True)
    ax.tick_params(axis='y', width=width, length=length, direction=direction, colors=colors, labelsize=fontsize,
                   right=True)
    format_spines(ax=ax, color=colors, lw=1)
    if txt_xlabel:
        ax.set_xlabel(txt_xlabel, color=label_color, fontsize=fontsize, fontweight='bold')
    if txt_ylabel and txt_ylabel_units:
        ax.set_ylabel(f'{txt_ylabel}  {txt_ylabel_units}', color=label_color, fontsize=fontsize, fontweight='bold')
    if txt_ylabel and not txt_ylabel_units:
        ax.set_ylabel(f'{txt_ylabel}', color=label_color, fontsize=fontsize, fontweight='bold')


def format_spines(ax, color, lw):
    """Set color and linewidth of axis spines"""
    spines = ['top', 'bottom', 'left', 'right']
    for spine in spines:
        ax.spines[spine].set_color(color)
        ax.spines[spine].set_linewidth(lw)


def example():
    from diive.pkgs.echires.windrotation import WindRotation2D
    from diive.core.io.filereader import ReadFileType

    # Settings
    SOURCEFILE = r"F:\Sync\luhk_work\20 - CODING\27 - VARIOUS\dyco\_testdata\CH-DAS_202308281300.csv.gz"
    U = 'U_[R350-B]'
    V = 'V_[R350-B]'
    W = 'W_[R350-B]'
    C = 'CH4_DRY_[QCL-C2]'

    # Read file
    df, meta = ReadFileType(filepath=SOURCEFILE,
                            filetype='ETH-SONICREAD-BICO-CSVGZ-20HZ',
                            data_nrows=None,
                            output_middle_timestamp=True).get_filedata()

    # Wind rotation for turbulent fluctuations
    u = df[U].copy()
    v = df[V].copy()
    w = df[W].copy()
    c = df[C].copy()
    wr = WindRotation2D(u=u, v=v, w=w, c=c)
    w_prime, c_prime = wr.get_wc_primes()

    subset = pd.concat([w_prime, c_prime], ignore_index=False, axis=1)

    mc = MaxCovariance(
        segment_df=subset,
        var_reference=str(w_prime.name),
        var_lagged=str(c_prime.name),
        lgs_winsize_from=-1000,
        lgs_winsize_to=1000,
        shift_stepsize=1,
        segment_name="test"
    )

    mc.run()

    mc.plot()

    cov_df, props_peak_auto = mc.get()

    print(w_prime, c_prime)


if __name__ == '__main__':
    example()
