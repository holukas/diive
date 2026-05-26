"""
QCF - OVERALL QUALITY CONTROL FLAG

Calculate overall quality flag (QCF) by combining multiple individual test flags
into a single quality indicator. Supports daytime/nighttime separation and USTAR
filtering scenarios.

Key concepts:
    - QCF flag values: 0 (good), 1 (marginal), 2 (poor/rejected)
    - Hard flags (value 2) indicate critical quality issues
    - Soft flags (value 1) indicate minor quality concerns
    - Flag sums drive the overall QCF decision logic

Typical workflow:
    1. Create FlagQCF instance with DataFrame containing individual test flags
    2. Call calculate() to compute QCF from flag combinations
    3. Access results via properties: flagqcf, filteredseries, filteredseries_hq
    4. Use reporting methods for diagnostics: report_qcf_flags(), report_qcf_evolution()

Example:
    >>> qcf = FlagQCF(
    ...     df=data_with_flags,
    ...     series=flux_series,
    ...     outname='NEE',
    ...     swinpot=sw_in_pot  # Optional: enables daytime/nighttime separation
    ... )
    >>> qcf.calculate(daytime_accept_qcf_below=2, nighttime_accept_qcf_below=2)
    >>> quality_controlled_flux = qcf.filteredseries  # NaN for rejected values
    >>> highest_quality_flux = qcf.filteredseries_hq  # NaN for any QCF > 0
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from diive.core.base.identify import identify_flagcols
from diive.core.funcs.funcs import validate_id_string
from diive.core.plotting.heatmap_datetime import HeatmapDateTime
from diive.core.utils.console import console as _console, detail
from diive.variables import daytime_nighttime_flag_from_swinpot


class FlagQCF:
    """Calculate overall quality flag (QCF) from individual test flags.

    **REQUIRED:** Input DataFrame must contain individual test flag columns with
    naming pattern FLAG_*_TEST (e.g., FLAG_L41_NEE_RANGE_TEST, FLAG_L41_NEE_SONIC_TEST).
    Each test flag column must have values: 0=pass, 1=soft warning, 2=hard fail.

    This class automatically identifies all FLAG_*_TEST columns, counts hard/soft
    flags for each record, and calculates QCF based on flag sums. Supports optional
    daytime/nighttime separation and USTAR filtering scenarios.

    Flag Values & Sums:
        Individual flags: 0=pass, 1=soft warning (minor issue), 2=hard fail (critical)
        Flag sums count how many tests raised flags:
            - Sum of all flags: Total number of raised flags
            - Sum of hard flags: Count of critical issues (value 2)
            - Sum of soft flags: Count of minor issues (value 1)

    QCF Decision Logic (based on sums):
        QCF = 0: All flags pass (total flag sum = 0)
        QCF = 1: 1-3 soft flags raised AND no hard flags
        QCF = 2: >3 soft flags OR >=2 hard flags OR rejected by daytime/nighttime logic

    Data Access:
        - flagqcf: Overall QCF flag for each record
        - filteredseries: Data with QCF=2 records set to NaN (general use)
        - filteredseries_hq: Only QCF=0 records (strict quality)
        - flags: Full DataFrame with all individual flags, flag sums, and QCF results
    """

    def __init__(self,
                 df: DataFrame,
                 target_col: str,
                 outname: str = None,
                 swinpot_col: str = None,
                 idstr: str = None,
                 nighttime_threshold: float = 50,
                 ustar_scenarios: list = None
                 ):
        """Initialize QCF calculator.

        Args:
            df: DataFrame containing individual test flag columns following pattern
                FLAG_*_{idstr}_{target_col}_TEST (e.g., FLAG_TEST1_L41_NEE_TEST). Values must be 0/1/2.
            target_col: Column name of the target variable to calculate QCF for (e.g., 'NEE' or 'FC').
                This name must appear in flag column names for proper identification.
            outname: Output name for QCF flag columns. If None, uses target_col name.
            idstr: ID string identifier that appears in flag column names. This string
                is used to identify which flags belong together and to name output
                columns. Examples: '_L41', '_L41_FC', '_L4.1'. If None, auto-detected
                from flag columns. All FLAG_*_{idstr}_{target_col}_TEST columns are identified
                and used for QCF calculation.
            swinpot_col: Column name of solar potential radiation for daytime/nighttime
                separation. If provided, enables separate acceptance thresholds for day/night.
            nighttime_threshold: Solar radiation threshold (W/m²) below which records
                are considered nighttime. Default 50 W/m².
            ustar_scenarios: List of USTAR filtering scenario names (e.g., ['CUT_50', 'CUT_75']).
                If provided, automatically excludes irrelevant USTAR flags for current scenario.

        Raises:
            ValueError: If multiple USTAR scenarios are detected in idstr.
            KeyError: If target_col or swinpot_col column not found in DataFrame.

        Example:
            >>> qcf = FlagQCF(
            ...     df=data,  # Must contain FLAG_TEST1_L41_NEE_TEST, FLAG_TEST2_L41_NEE_TEST, etc.
            ...     target_col='NEE',  # Column name of variable to check
            ...     swinpot_col='SW_IN_POT',  # Optional: for daytime/nighttime separation
            ...     idstr='_L41'  # Identifies flags as FLAG_*_L41_NEE_TEST pattern
            ... )
        """
        if target_col not in df.columns:
            raise KeyError(f"Column '{target_col}' not found in DataFrame")
        if swinpot_col and swinpot_col not in df.columns:
            raise KeyError(f"Column '{swinpot_col}' not found in DataFrame")

        self.df = df.copy()  # Original data
        self.series = df[target_col].copy()
        self.series_name = target_col
        self.swinpot_data = df[swinpot_col].copy() if swinpot_col else None
        self.ustar_scenarios = ustar_scenarios  # Required to get the correct USTAR FLAG_ columns for each scenario

        self.outname = outname if outname else target_col

        self.idstr = validate_id_string(idstr=idstr)

        # Identify FLAG columns
        # If there are different USTAR scenarios, all flag columns that are not relevant for
        # the current scenario must be removed.

        # First check if there are USTAR scenarios
        if ustar_scenarios:
            # Get ID of USTAR scenario, e.g. 'CUT_50'. Info about the scenario is in the ID string.
            # Therefore, here we can check if any USTAR scenario string appears in the ID string.
            current_ustar_scenario = [u for u in self.ustar_scenarios if u in self.idstr]
            # Make sure there is only one detected scenario
            if len(current_ustar_scenario) == 1:
                current_ustar_scenario = str(current_ustar_scenario[0])
            else:
                raise ValueError(f"(!)More than one USTAR scenario detected: "
                                 f"current_ustar_scenario={current_ustar_scenario}.")
            exclude_ustar_ids = self.ustar_scenarios.copy()
            exclude_ustar_ids.remove(current_ustar_scenario)
        else:
            exclude_ustar_ids = None
            # current_ustar_scenario = None

        flagcols = identify_flagcols(df=df, seriescol=target_col, exclude_ustar_ids=exclude_ustar_ids)
        self._flags_df = df[flagcols].copy()

        # Detect daytime and nighttime
        if self.swinpot_data is not None:
            self.daytime, self.nighttime = \
                daytime_nighttime_flag_from_swinpot(swinpot=self.swinpot_data, nighttime_threshold=nighttime_threshold)
        else:
            self.daytime = None
            self.nighttime = None

        # Generate QCF column names
        self.filteredseriescol = f"{self.outname}{self.idstr}_QCF"  # Quality-controlled flux
        self.filteredseriescol_hq = f"{self.outname}{self.idstr}_QCF0"  # Quality-controlled flux, highest quality
        self.flagqcfcol = f"FLAG{self.idstr}_{self.outname}_QCF"  # Overall flag
        self.sumflagscol = f'SUM{self.idstr}_{self.outname}_FLAGS'
        self.sumhardflagscol = f'SUM{self.idstr}_{self.outname}_HARDFLAGS'
        self.sumsoftflagscol = f'SUM{self.idstr}_{self.outname}_SOFTFLAGS'

        self.daytime_accept_qcf_below = None
        self.nighttime_accept_qcf_below = None

    @property
    def flags(self) -> DataFrame:
        """Return dataframe containing all test flags and calculated QCF results.

        Returns:
            DataFrame with original test flags plus calculated QCF columns:
                - FLAG*_QCF: Overall quality flag (0/1/2)
                - SUM*_FLAGS: Total flag count
                - SUM*_HARDFLAGS: Count of hard flags (value 2)
                - SUM*_SOFTFLAGS: Count of soft flags (value 1)
                - *_QCF: Quality-controlled series (NaN for QCF=2)
                - *_QCF0: Highest-quality series (NaN for QCF>0)
        """
        if not isinstance(self._flags_df, DataFrame):
            raise Exception('Results for flags are empty')
        return self._flags_df

    @property
    def filteredseries(self) -> Series:
        """Return quality-controlled series with rejected records as NaN.

        Records with QCF=2 (poor quality) are set to NaN. Use this for general
        analysis where marginal data (QCF=1) is acceptable.

        Returns:
            Series with original values where QCF<2, NaN elsewhere.
        """
        return self.flags[self.filteredseriescol]

    @property
    def filteredseries_hq(self) -> Series:
        """Return highest-quality series with only QCF=0 records.

        Records with QCF>0 (any flags) are set to NaN. Use this for stringent
        analysis requiring only the best-quality data.

        Returns:
            Series with original values where QCF=0, NaN elsewhere.
        """
        return self.flags[self.filteredseriescol_hq]

    @property
    def flagqcf(self) -> Series:
        """Return overall QCF flag for each record.

        QCF values:
            0 = Good quality (all flags pass)
            1 = Marginal quality (minor issues, 1-3 soft flags)
            2 = Poor quality (critical issues or too many soft flags)

        Returns:
            Series with QCF values (0, 1, 2, or NaN if no flags available).
        """
        return self.flags[self.flagqcfcol]

    def get(self) -> DataFrame:
        """Return original DataFrame with calculated QCF results appended.

        Combines the original input data with all new QCF-related columns:
        QCF flag, flag sums, and quality-controlled series variants.

        Returns:
            DataFrame with original columns plus new QCF columns.
        """
        returndf = self.df.copy()  # Main data
        newcols = [col for col in self.flags.columns if col not in returndf]
        newcolsdf = self.flags[newcols].copy()
        returndf = pd.concat([returndf, newcolsdf], axis=1)  # Add new columns to main data
        [detail(f"Added column {c}.") for c in newcols]
        return returndf

    def calculate(self,
                  daytime_accept_qcf_below: int = 2,
                  nighttime_accept_qcf_below: int = 2):
        """Calculate QCF from test flags and generate quality-controlled series.

        Orchestrates the complete QCF workflow:
        1. Sum hard (value 2) and soft (value 1) flags
        2. Apply QCF decision logic
        3. Apply daytime/nighttime-specific thresholds if swinpot was provided
        4. Generate filtered series with rejected values as NaN

        Args:
            daytime_accept_qcf_below: Accept daytime records where QCF < this value.
                Default 2 (rejects only QCF=2). Set to 1 to also reject QCF=1.
            nighttime_accept_qcf_below: Accept nighttime records where QCF < this value.
                Default 2. Separate threshold useful for day/night quality differences.

        Note:
            Only applies daytime/nighttime logic if swinpot was provided at init.
            Otherwise, all records with QCF>=2 are rejected uniformly.
        """
        self.daytime_accept_qcf_below = daytime_accept_qcf_below
        self.nighttime_accept_qcf_below = nighttime_accept_qcf_below
        self._flags_df = self._calculate_flagsums(df=self._flags_df)
        self._flags_df = self._calculate_flag_qcf(df=self._flags_df)
        self._add_series()
        self._calculate_series_qcf()

    def _add_series(self):
        """Add original series to flags dataframe for QC processing."""
        self._flags_df[self.series_name] = self.series.copy()

    def _calculate_series_qcf(self):
        """Generate quality-controlled series variants (QCF and QCF0)."""
        # Accepted-quality fluxes
        self._flags_df[self.filteredseriescol] = self._flags_df[self.series_name].copy()
        ix = self._flags_df[self.flagqcfcol] == 2
        self._flags_df.loc[ix, self.filteredseriescol] = np.nan

        # Highest-quality fluxes
        self._flags_df[self.filteredseriescol_hq] = self._flags_df[self.series_name].copy()
        ix = self._flags_df[self.flagqcfcol] > 0
        self._flags_df.loc[ix, self.filteredseriescol_hq] = np.nan

    def report_qcf_flags(self):
        """Print detailed statistics for each test flag.

        Generates comprehensive breakdown of each test flag:
        - Individual test statistics (pass/warn/fail counts)
        - Two reports: WITH and FOR available records
        - Optional daytime/nighttime separation

        Shows which tests are raising flags and their impact.
        """
        _console.print(f"\n\n{'═' * 100}")
        _console.print(f"  INDIVIDUAL TEST FLAG STATISTICS: {self.series_name}")
        _console.print(f"{'═' * 100}")

        flagcols = identify_flagcols(df=self.flags, seriescol=self.series_name)

        # Filter to only test flags (exclude QCF)
        test_flagcols = [c for c in flagcols if str(c).endswith('_TEST')]

        if not test_flagcols:
            _console.print("No test flags found.")
            return

        _console.print(f"\n  Total test flags: {len(test_flagcols)}")

        # === REPORT 1: FLAGS WITH MISSING VALUES ===
        _console.print(f"\n\n  ┌─ REPORT 1A: ALL RECORDS (INCLUDING MISSING VALUES)")
        _console.print(f"  │")
        for col in test_flagcols:
            self._flagstats_dt_nt(col=col, df=self.flags)

        # === REPORT 2: FLAGS FOR AVAILABLE RECORDS ===
        _console.print(f"\n  ┌─ REPORT 1B: AVAILABLE RECORDS ONLY (EXCLUDING MISSING VALUES)")
        _console.print(f"  │")
        _df = self.flags.copy()
        ix_missing_vals = _df[self.series_name].isnull()
        _df = _df[~ix_missing_vals].copy()
        for col in test_flagcols:
            self._flagstats_dt_nt(col=col, df=_df)

        # === SUMMARY ===
        _console.print(f"\n{'═' * 100}\n")

    def _flagstats_dt_nt(self, col: str, df: DataFrame):
        """Print flag statistics overall, daytime, and nighttime (if available)."""
        # Extract test name from column
        test_name = col.replace('FLAG_', '').replace('_TEST', '')
        _console.print(f"\n  ├─ {test_name}")
        _console.print(f"  │  {'Period':<15} │ {'Pass (0)':<15} │ {'Warn (1)':<15} │ {'Fail (2)':<15} │ {'Missing':<12}")
        _console.print(f"  │  {'-' * 85}")

        flag = df[col]
        self._flagstats(flag=flag, prefix="OVERALL", indent="  │  ")

        if isinstance(self.daytime, Series):
            flag = df[col].loc[self.daytime == 1]
            if len(flag) > 0:
                self._flagstats(flag=flag, prefix="DAYTIME", indent="  │  ")

        if isinstance(self.nighttime, Series):
            flag = df[col].loc[self.nighttime == 1]
            if len(flag) > 0:
                self._flagstats(flag=flag, prefix="NIGHTTIME", indent="  │  ")

    def report_qcf_evolution(self):
        """Print how QCF evolves as tests are applied sequentially.

        Shows cumulative impact of each test flag: how many additional records get
        flagged as you add each test. Helps identify which tests are most impactful
        on data rejection and understand QC filtering progression.

        Output:
            - Sequential table: for each test, available vs rejected records with percentages
            - Impact summary: tests ranked by number of records they newly reject
            - Impact levels: HIGH (>=5 records), MODERATE (>=2), LOW (<2)
        """
        _console.print(f"\n\n{'=' * 120}")
        _console.print(f"QCF EVOLUTION: SEQUENTIAL TEST APPLICATION")
        _console.print(f"{'=' * 120}")
        _console.print(f"Shows how QCF flag distribution changes as tests are applied sequentially")
        _console.print(f"Target variable: {self.series_name}\n")

        flagcols = identify_flagcols(df=self.flags, seriescol=self.series_name)
        flagcols = [c for c in flagcols if str(c).startswith('FLAG_') and (str(c).endswith('_TEST'))]
        allflags_df = self.flags[flagcols].copy()

        ix_missing_vals = self.df[self.series_name].isnull()
        allflags_df = allflags_df[~ix_missing_vals].copy()  # Ignore missing values

        n_vals = len(allflags_df)
        _console.print(f"Measured records (excluding missing): {n_vals}\n")

        # Track cumulative rejections
        n_flag2_prev = 0

        # Header for combined results table
        _console.print(
            f"{'Step':<5} {'Test Name':<38} {'New Rej':<8} {'Available':<12} {'Rejected':<12} {'Avail %':<9} {'Rej %':<9} {'QCF Distribution':<15}")
        _console.print(f"{'':5} {'':38} {'':8} {'(count)':<12} {'(count)':<12} {'':9} {'':9} {'0/1/2':<15}")
        _console.print(f"{'-' * 120}")

        for ix_test, test_col in enumerate(flagcols, 1):
            prog_testcols = flagcols[:ix_test]
            prog_df = allflags_df[prog_testcols].copy()

            # Calculate QCF with tests so far
            prog_df = self._calculate_flagsums(df=prog_df)
            prog_df = self._calculate_flag_qcf(df=prog_df)

            # Count QCF distribution
            n_flag0 = (prog_df[self.flagqcfcol] == 0).sum()
            n_flag1 = (prog_df[self.flagqcfcol] == 1).sum()
            n_flag2 = (prog_df[self.flagqcfcol] == 2).sum()

            # Available (pass) vs Rejected records
            n_available = n_flag0 + n_flag1
            n_rejected = n_flag2
            perc_available = (n_available / n_vals) * 100 if n_vals > 0 else 0
            perc_rejected = (n_rejected / n_vals) * 100 if n_vals > 0 else 0

            # New rejections from this test
            n_new_rejected = n_flag2 - n_flag2_prev

            # Extract test name (remove FLAG_, _TEST)
            test_name = test_col.replace('FLAG_', '').replace('_TEST', '')
            if len(test_name) > 35:
                test_name = test_name[:32] + "..."

            qcf_dist = f"{n_flag0}/{n_flag1}/{n_flag2}"
            _console.print(
                f"{ix_test:<5} {test_name:<38} {n_new_rejected:<8} {n_available:<12} {n_rejected:<12} {perc_available:<9.2f} {perc_rejected:<9.2f} {qcf_dist:<15}")

            n_flag2_prev = n_flag2

        _console.print(f"\n{'=' * 120}\n")

    def _flagstats(self, flag: Series, prefix: str, indent: str = "  "):
        """Print flag value counts in table format (0=pass, 1=warn, 2=fail)."""
        n_values = len(flag)
        flagcounts = flag.value_counts().to_dict()
        flagmissing = flag.isnull().sum()

        # Count each flag value
        n_pass = flagcounts.get(0, 0)
        n_warn = flagcounts.get(1, 0)
        n_fail = flagcounts.get(2, 0)

        # Calculate percentages
        perc_pass = (n_pass / n_values) * 100 if n_values > 0 else 0
        perc_warn = (n_warn / n_values) * 100 if n_values > 0 else 0
        perc_fail = (n_fail / n_values) * 100 if n_values > 0 else 0
        perc_miss = (flagmissing / n_values) * 100 if n_values > 0 else 0

        # Format as table row with better spacing
        pass_str = f"{n_pass:>5} ({perc_pass:>5.1f}%)"
        warn_str = f"{n_warn:>5} ({perc_warn:>5.1f}%)"
        fail_str = f"{n_fail:>5} ({perc_fail:>5.1f}%)"
        miss_str = f"{flagmissing:>4} ({perc_miss:>5.1f}%)"

        _console.print(f"{indent}{prefix:<15} │ {pass_str:<15} │ {warn_str:<15} │ {fail_str:<15} │ {miss_str:<12}")

    def report_qcf_series(self):
        """Print comprehensive summary statistics for quality-controlled series.

        Shows data availability, QCF distribution, and quality assessment:
        - Time period coverage
        - Data availability at each filtering stage
        - QCF flag distribution (good/marginal/poor)
        - Data loss breakdown and quality assessment
        - Useful for overall data quality assessment

        Useful for quickly assessing overall data quality and coverage impact.
        """
        _console.print(f"\n\n{'=' * 70}")
        _console.print(f"QCF QUALITY CONTROL REPORT: {self.series_name}")
        _console.print(f"{'=' * 70}")

        series = self.flags[self.series_name]
        seriesqcf = self.flags[self.filteredseriescol]
        qcf_flags = self.flags[self.flagqcfcol]

        # === TIME PERIOD ===
        start = series.index[0].strftime('%Y-%m-%d %H:%M')
        end = series.index[-1].strftime('%Y-%m-%d %H:%M')
        duration_days = (series.index[-1] - series.index[0]).days

        _console.print(f"\n[1] TIME PERIOD")
        _console.print(f"    Start:  {start}")
        _console.print(f"    End:    {end}")
        _console.print(f"    Duration: {duration_days} days ({len(series)} records)")

        # === DATA AVAILABILITY STAGES ===
        n_potential = len(series)
        n_measured = len(series.dropna())
        n_missed = n_potential - n_measured
        n_available = len(seriesqcf.dropna())
        n_rejected = n_measured - n_available

        perc_measured = (n_measured / n_potential) * 100
        perc_missed = (n_missed / n_potential) * 100
        perc_available = (n_available / n_measured) * 100 if n_measured > 0 else 0
        perc_rejected = (n_rejected / n_measured) * 100 if n_measured > 0 else 0

        _console.print(f"\n[2] DATA AVAILABILITY STAGES")
        _console.print(f"    Potential records (all time slots): {n_potential}")
        _console.print(f"    Measured records (data exists):     {n_measured:>4} ({perc_measured:>6.2f}% of potential)")
        _console.print(f"    Missing records (gaps/gaps):        {n_missed:>4} ({perc_missed:>6.2f}% of potential)")
        _console.print(f"    |__ After QC (QCF < 2):             {n_available:>4} ({perc_available:>6.2f}% of measured)")
        _console.print(f"    |__ Rejected by QC (QCF >= 2):      {n_rejected:>4} ({perc_rejected:>6.2f}% of measured)")

        # === QCF FLAG DISTRIBUTION ===
        n_qcf0 = (qcf_flags == 0).sum()
        n_qcf1 = (qcf_flags == 1).sum()
        n_qcf2 = (qcf_flags == 2).sum()

        perc_qcf0 = (n_qcf0 / n_measured * 100) if n_measured > 0 else 0
        perc_qcf1 = (n_qcf1 / n_measured * 100) if n_measured > 0 else 0
        perc_qcf2 = (n_qcf2 / n_measured * 100) if n_measured > 0 else 0

        _console.print(f"\n[3] QCF FLAG DISTRIBUTION (for measured records)")
        _console.print(f"    QCF=0 (Good quality):     {n_qcf0:>4} ({perc_qcf0:>6.2f}%) - All tests pass")
        _console.print(f"    QCF=1 (Marginal quality): {n_qcf1:>4} ({perc_qcf1:>6.2f}%) - Minor quality issues")
        _console.print(f"    QCF=2 (Poor quality):     {n_qcf2:>4} ({perc_qcf2:>6.2f}%) - Critical issues or too many warnings")

        # === DATA LOSS ANALYSIS ===
        data_loss_perc = (n_rejected / n_measured * 100) if n_measured > 0 else 0

        _console.print(f"\n[4] DATA LOSS ANALYSIS")
        _console.print(f"    Records retained after QC: {n_available}/{n_measured} ({perc_available:.2f}%)")
        _console.print(f"    Data loss from QC:         {n_rejected}/{n_measured} ({data_loss_perc:.2f}%)")
        _console.print(
            f"    Final data coverage:       {n_available}/{n_potential} ({(n_available / n_potential) * 100:.2f}% of potential)")

        _console.print(f"\n{'=' * 70}\n")

    def _calculate_flag_qcf(self, df: DataFrame) -> DataFrame:
        """Calculate QCF flag (0/1/2) using hierarchical decision rules and apply day/night thresholds."""

        # QCF is NaN if no flag is available
        df[self.flagqcfcol] = np.nan

        # QCF is 0 if all flags show zero
        ix = df[self.sumflagscol] == 0
        df.loc[ix, self.flagqcfcol] = 0

        # QCF is 2 if more than three soft flags were raised
        ix = df[self.sumsoftflagscol] > 3
        df.loc[ix, self.flagqcfcol] = 2

        # QCF is 2 if at least one hard flag was raised
        ix = df[self.sumhardflagscol] >= 2
        df.loc[ix, self.flagqcfcol] = 2

        # QCF is 1 if no hard flag and max. three soft flags and
        # min. one soft flag were raised
        ix = (df[self.sumsoftflagscol] <= 3) & (df[self.sumsoftflagscol] >= 1) & (df[self.sumhardflagscol] == 0)
        df.loc[ix, self.flagqcfcol] = 1

        # Flag daytime values based on param
        if isinstance(self.daytime, Series):
            ix = (df[self.flagqcfcol] >= self.daytime_accept_qcf_below) & (self.daytime == 1)
            df.loc[ix, self.flagqcfcol] = 2

        # Flag nighttime values based on param
        if isinstance(self.nighttime, Series):
            ix = (df[self.flagqcfcol] >= self.nighttime_accept_qcf_below) & (self.nighttime == 1)
            df.loc[ix, self.flagqcfcol] = 2

        # Daytime and nighttime flags are only calculated when swinpot is provided.
        # This means that if both do not exist, no separation into daytime and nighttime
        # was done. In that case, all records where QCF = 2 are rejected.
        if not isinstance(self.daytime, Series) and not isinstance(self.nighttime, Series):
            default_accept_qcf_below = 2
            ix = (df[self.flagqcfcol] >= default_accept_qcf_below)
            df.loc[ix, self.flagqcfcol] = 2

        return df

    def _calculate_flagsums(self, df: DataFrame) -> DataFrame:
        """Sum hard flags (value 2) and soft flags (value 1) across all FLAG_*_TEST columns."""

        # Get variables that contain flag data (individual quality test results)
        onlyflagtests = [c for c in df.columns if str(c).startswith('FLAG_') and str(c).endswith('_TEST')]
        onlyflagtests_df = df[onlyflagtests].copy()

        # Build flag sums from flag data
        sumhardflags = onlyflagtests_df[onlyflagtests_df == 2].sum(axis=1)  # The sum of all flags that show 2
        sumsoftflags = onlyflagtests_df[onlyflagtests_df == 1].sum(axis=1)  # The sum of all flags that show 1
        sumflags = sumhardflags.add(sumsoftflags)  # Sum of all flags

        # Add flag sums to overall flag dataframe
        df[self.sumhardflagscol] = sumhardflags
        df[self.sumsoftflagscol] = sumsoftflags
        df[self.sumflagscol] = sumflags
        return df

    def showplot_qcf_heatmaps(self, maxabsval: float = None, figsize: tuple = (18, 8)):
        """Display 4-panel heatmap showing data before/after QC and flag distribution.

        Panels (left to right):
        1. Original series (before QC)
        2. Quality-controlled series (after QC, NaN where QCF=2)
        3. Flag sums (total count of raised flags)
        4. QCF overall flag (0=good, 1=marginal, 2=poor)

        Uses datetime heatmap layout (e.g., day-of-year vs hour-of-day) for visual patterns.

        Args:
            maxabsval: Max absolute value for colorbar symmetry. If None, auto-scales.
                Useful for comparing multiple panels with same scale.
            figsize: Figure dimensions (width, height) in inches. Default (18, 8).
        """
        fig = plt.figure(facecolor='white', figsize=figsize)
        gs = gridspec.GridSpec(1, 4)  # rows, cols
        gs.update(wspace=0.4, hspace=0, left=0.03, right=0.97, top=0.9, bottom=0.1)
        ax_before = fig.add_subplot(gs[0, 0])
        ax_after = fig.add_subplot(gs[0, 1], sharey=ax_before)
        ax_flagsum = fig.add_subplot(gs[0, 2], sharey=ax_before)
        ax_flag = fig.add_subplot(gs[0, 3], sharey=ax_before)

        # Heatmaps
        vmin = -maxabsval if maxabsval else None
        vmax = maxabsval if maxabsval else None
        HeatmapDateTime(series=self.flags[self.series_name]).plot(ax=ax_before, vmin=vmin, vmax=vmax,
                                                                  cb_digits_after_comma=0)
        HeatmapDateTime(series=self.flags[self.filteredseriescol]).plot(ax=ax_after, vmin=vmin, vmax=vmax,
                                                                        cb_digits_after_comma=0)
        HeatmapDateTime(series=self.flags[self.sumflagscol]).plot(ax=ax_flagsum,
                                                                  cb_digits_after_comma=0)
        HeatmapDateTime(series=self.flags[self.flagqcfcol]).plot(ax=ax_flag,
                                                                 cb_digits_after_comma=0)
        plt.setp(ax_after.get_yticklabels(), visible=False)
        plt.setp(ax_flagsum.get_yticklabels(), visible=False)
        plt.setp(ax_flag.get_yticklabels(), visible=False)
        ax_after.axes.get_yaxis().get_label().set_visible(False)
        ax_flagsum.axes.get_yaxis().get_label().set_visible(False)
        ax_flag.axes.get_yaxis().get_label().set_visible(False)

        fig.show()

    def showplot_qcf_timeseries(self, figsize=(16, 20)):
        """Display all QCF data columns as individual time series subplots.

        Creates subplots for: original series, test flags, flag sums, and QCF flag.
        Useful for detailed inspection of temporal patterns and data quality.

        Args:
            figsize: Figure dimensions (width, height) in inches. Default (16, 20).
        """
        self.flags.plot(subplots=True, figsize=figsize)
        plt.show()
