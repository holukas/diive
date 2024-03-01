import pandas as pd
from bokeh.models import BoxZoomTool, PanTool, ResetTool, WheelZoomTool, WheelPanTool, UndoTool, \
    RedoTool, SaveTool, HoverTool, BoxSelectTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show
from bokeh.plotting import output_notebook
from pandas import Series

import diive.core.plotting.plotfuncs as pf
import diive.core.plotting.styles.LightTheme as theme

output_notebook()


class TimeSeries:
    def __init__(self,
                 series: Series,
                 ax=None,
                 series_units: str = None):
        """
        Plot timeseries

        Args:
            series: Data for plotting
            ax: Axis to show the matplotlib plot
            series_units: Units of *series*
        """
        self.series = series.copy()
        self.ax = ax  # todo should be required when calling .plot() instead of here
        self.series_units = series_units
        self.varname = series.name

        self.series = self.series.dropna()

    def plot_interactive(self, height: int = 600, width: int = 1200):
        # output_file(filename=r"M:\Downloads\_temp\bokeh.html", title="Static HTML file")

        # Bokeh needs dataframe
        df = pd.DataFrame()
        df['date'] = self.series.index
        df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d %H:%M")
        df['value'] = self.series.values

        # Convert dataframe for bokeh
        source = ColumnDataSource(df)

        # todo selection: https://docs.bokeh.org/en/latest/docs/user_guide/interaction/tools.html#ug-interaction-tools-pandrag
        p = figure(height=height,
                   width=width,
                   title=f"{self.series.name} ({self.series.index.name})",
                   tools=[
                       HoverTool(
                           tooltips=[('Date', '@date{%F %T}'),
                                     ('Value', '@value')],
                           formatters={'@date': 'datetime'},
                           # mode='vline'
                           mode='mouse'
                       ),
                       BoxZoomTool(),
                       ResetTool(),
                       PanTool(),
                       BoxSelectTool(),
                       WheelZoomTool(),
                       WheelPanTool(),
                       UndoTool(),
                       RedoTool(),
                       # ZoomInTool(),  # not working, plot will not show
                       # ZoomOutTool(),  # not working, plot will not show
                       SaveTool()],
                   x_axis_type='datetime')

        p.line(x='date', y='value', line_width=2, source=source, color='#455A64')
        p.circle(x='date', y='value', size=5, source=source, color='#455A64')

        p.yaxis.axis_label = self.series.name

        # # https://stackoverflow.com/questions/61340741/get-bokehs-selection-in-notebook
        # from bokeh.models import CustomJS
        # # make a custom javascript callback that exports the indices of the selected points to the Jupyter notebook
        # callback = CustomJS(args=dict(s=source),
        #                     code="""
        #                          console.log('Running CustomJS callback now.');
        #                          var indices = s.selected.indices;
        #                          var kernel = IPython.notebook.kernel;
        #                          kernel.execute("selected_indices = " + indices)
        #                          """)
        #
        # # set the callback to run when a selection geometry event occurs in the figure
        # p.js_on_event('selectiongeometry', callback)
        # selected_indices

        # # Add hover tooltip
        # hover = HoverTool(
        #     tooltips=[
        #         ('Date', '@date{%F %T}'),
        #         ('Value', '@value')
        #     ],
        #     formatters={
        #         '@date':'datetime'
        #     },
        #     # Display a tooltip whenever the cursor is vertically in line with a glyph
        #     mode='vline'
        # )

        # hover.formatters = {'@date': 'datetime'}
        # p.add_tools(hover)

        # HoverTool(tooltips=[('date', '@DateTime{%F}')],
        #           formatters={'@DateTime': 'datetime'})

        # Show plot
        show(p)

    def plot(self):
        """Plot data using matplotlib"""
        # Create axis
        if self.ax:
            # If ax is given, plot directly to ax, no fig needed
            self.fig = None
            # self.ax = self.ax
            self.showplot = False
        else:
            # If no ax is given, create fig and ax and then show the plot
            self.fig, self.ax = pf.create_ax()
            self.showplot = True
        color_list = theme.colorwheel_36()  # get some colors
        label = self.series.name
        self.ax.plot_date(x=self.series.index,
                          y=self.series,
                          color=color_list[0], alpha=1,
                          lw=theme.WIDTH_LINE_DEFAULT,
                          fmt='-', markeredgecolor='none', ms=0,
                          zorder=99, label=label)
        self._apply_format()
        if self.showplot:
            self.fig.show()

    def _apply_format(self):
        """Format matplotlib plot"""

        # ymin = self.series.min()
        # ymax = self.series.max()
        # self.ax.set_ylim(ymin, ymax)

        pf.add_zeroline_y(ax=self.ax, data=self.series)

        pf.default_format(ax=self.ax,
                          ax_xlabel_txt='Date',
                          ax_ylabel_txt=self.varname,
                          txt_ylabel_units=self.series_units)

        pf.default_legend(ax=self.ax,
                          labelspacing=0.2,
                          ncol=1)

        pf.nice_date_ticks(ax=self.ax, minticks=3, maxticks=20, which='x', locator='auto')

        if self.showplot:
            title = f"{self.series.name}"
            self.fig.suptitle(title, fontsize=theme.FIGHEADER_FONTSIZE)
            self.fig.tight_layout()


def example():
    # Test data
    from diive.configs.exampledata import load_exampledata_DIIVE_CSV_30MIN
    data_df, metadata_df = load_exampledata_DIIVE_CSV_30MIN()
    # series_units = metadata_df.loc[series_col]['UNITS']

    # # Plot to existing ax
    # fig, ax = pf.create_ax()
    # # series_units = r'($\mathrm{gC\ m^{-2}}$)'
    # TimeSeries(ax=ax,
    #            series=data_df['NEE_CUT_REF_f'],
    #            series_units=None).plot()
    # fig.tight_layout()
    # fig.show()

    # Plot interactive
    from bokeh.plotting import output_file
    output_file(filename=r"F:\Downloads\_temp\bokeh.html", title="Static HTML file")
    TimeSeries(series=data_df['NEE_CUT_REF_f']).plot_interactive()


if __name__ == '__main__':
    example()
