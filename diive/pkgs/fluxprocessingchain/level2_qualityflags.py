"""
Performs Level 2 quality control (QC) flag generation for flux data derived from EddyPro outputs.

This module provides the `FluxQualityFlagsEddyPro` class, which encapsulates
various QC tests described in flux processing literature (e.g., VM97) and
specific checks related to EddyPro outputs. These tests assess data quality
based on criteria like signal strength, angle of attack, wind steadiness,
and statistical properties of the raw data.

Reference:
(VM97) Vickers, Dean, and L. Mahrt. 1997. “Quality Control and Flux Sampling
    Problems for Tower and Aircraft Data.” JOURNAL OF ATMOSPHERIC AND
    OCEANIC TECHNOLOGY 14:15.
"""

import pandas as pd
from pandas import DataFrame

from diive.core.funcs.funcs import validate_id_string
from diive.pkgs.qaqc.eddyproflags import (flags_vm97_eddypro_fluxnetfile_tests,
                                          flag_fluxbasevar_completeness_eddypro_test, \
                                          flag_spectral_correction_factor_eddypro_test, flag_ssitc_eddypro_test,
                                          flag_angle_of_attack_eddypro_test,
                                          flag_steadiness_horizontal_wind_eddypro_test,
                                          flag_signal_strength_eddypro_test)
from diive.pkgs.qaqc.flags import MissingValues


class FluxQualityFlagsEddyPro:
    """
    Performs quality control flag creation for flux data calculated from EddyPro outputs.

    This class provides methods for applying various quality control tests to flux data
    calculated by EddyPro. The tests result in the computation of quality flags that
    can help identify unreliable or low-quality data in flux measurements. Results are
    stored in a separate dataframe, which can be accessed using the `results` property.

    Attributes:
        fluxcol (str): Name of the flux variable in the input dataframe.
        dfin (DataFrame): Input dataframe containing EddyPro flux calculation results.
            A copy of this dataframe is maintained internally to perform operations.
        idstr (str): Suffix appended to generated flag column names for identification.
            Validated for usage within flags.
        fluxbasevar (str): Name of the base variable used to calculate the flux, such
            as 'CO2_CONC' for CO2 flux calculations.
        _results (DataFrame): Internal dataframe used to store the original flux
            column and all generated flag columns.
        """

    def __init__(self,
                 dfin: DataFrame,
                 fluxcol: str,
                 fluxbasevar: str,
                 idstr: str = None):
        """
        Initializes the class with a dataframe, flux column name, base variable for flux calculation,
        and an optional identifier string.

        Args:
            dfin (DataFrame): Input dataframe to be processed.
            fluxcol (str): The name of the column in the dataframe that contains flux data.
            fluxbasevar (str): The base variable used for the flux calculations.
            idstr (str, optional): An optional identifier string that will be validated.
        """
        self.fluxcol = fluxcol
        self.dfin = dfin.copy()
        self.idstr = validate_id_string(idstr=idstr)
        self.fluxbasevar = fluxbasevar

        # Collect flags together with flux and potential radiation in separate dataframe
        self._results = self.dfin[[fluxcol]].copy()

    @property
    def results(self) -> DataFrame:
        """
        Returns the results as a DataFrame.

        The method fetches the results for flux flags and ensures they are
        stored as a DataFrame. If the `results` attribute is not of type
        DataFrame, an exception will be raised.

        Raises:
            Exception: If the results for flux flags are empty or not
            stored as a DataFrame.

        Returns:
            DataFrame: The results for flux flags in a pandas DataFrame format.
        """
        if not isinstance(self._results, DataFrame):
            raise Exception('Results for flux flags are empty')
        return self._results

    def angle_of_attack_test(
            self,
            application_dates: list or None = None
    ):
        """
        Calculates and applies the flag for angle of attack tests.

        This test flags data based on the angle of attack values provided by
        EddyPro, which can indicate poor sensor orientation or flow distortion.

        Args:
            application_dates (list | None): A list of specific dates for which the angle
                of attack test should be applied. If None, the test is applied to
                all available dates.

        Returns:
            None: This function does not return a value. It updates the instance's
            internal `_results` dataframe with the calculated flag.
        """
        flag = flag_angle_of_attack_eddypro_test(df=self.dfin, flux=self.fluxcol,
                                                 idstr=self.idstr, application_dates=application_dates)
        self._results[flag.name] = flag

    def steadiness_of_horizontal_wind(self):
        """
        Evaluates the steadiness of horizontal wind using an external test function and stores
        the resulting flag within the class instance.

        This test identifies periods where the horizontal wind is not stationary,
        which violates assumptions of eddy covariance.

        Returns:
            None: This method does not return a value. The evaluation result is stored internally
            in the `_results` attribute with the flag name as the key.
        """
        flag = flag_steadiness_horizontal_wind_eddypro_test(df=self.dfin, flux=self.fluxcol,
                                                            idstr=self.idstr)
        self._results[flag.name] = flag

    def raw_data_screening_vm97_tests(
            self,
            spikes: bool = True,
            amplitude: bool = False,
            dropout: bool = True,
            abslim: bool = False,
            skewkurt_hf: bool = False,
            skewkurt_sf: bool = False,
            discont_hf: bool = False,
            discont_sf: bool = False,
    ):
        """
        Performs raw data screening using the VM97 tests.

        This function applies a series of data screening tests to raw flux data
        based on the VM97 methodologies. These tests check for statistical
        anomalies in the high-frequency raw data that was used to compute the
        fluxes. The applied tests can be individually enabled or disabled
        through function parameters. The results of the tests are combined and
        stored internally.

        Reference:
        Vickers, Dean, and L. Mahrt. 1997. “Quality Control and Flux Sampling
            Problems for Tower and Aircraft Data.” JOURNAL OF ATMOSPHERIC AND
            OCEANIC TECHNOLOGY 14:15.

        Args:
            spikes (bool): If True, applies the spikes detection test.
            amplitude (bool): If True, applies the amplitude threshold test.
            dropout (bool): If True, applies the dropout detection test.
            abslim (bool): If True, applies the absolute limits check.
            skewkurt_hf (bool): If True, applies skewness and kurtosis check for
                high-frequency data.
            skewkurt_sf (bool): If True, applies skewness and kurtosis check for
                low-frequency data.
            discont_hf (bool): If True, applies discontinuity test for
                high-frequency data.
            discont_sf (bool): If True, applies discontinuity test for
                low-frequency data.

        Returns:
            None: Modifies the `_results` attribute in-place.
        """
        kwargs = dict(
            df=self.dfin,
            flux=self.fluxcol,
            fluxbasevar=self.fluxbasevar,
            idstr=self.idstr,
            spikes=spikes,
            amplitude=amplitude,
            dropout=dropout,
            abslim=abslim,
            skewkurt_hf=skewkurt_hf,
            skewkurt_sf=skewkurt_sf,
            discont_hf=discont_hf,
            discont_sf=discont_sf,
        )
        flags = flags_vm97_eddypro_fluxnetfile_tests(**kwargs)
        self._results = pd.concat([self.results, flags], axis=1)

    def signal_strength_test(self,
                             signal_strength_col: str,
                             method: str,
                             threshold: int):
        """
        Tests the signal strength of flux data using the specified method and threshold.

        Flag flux values where signal strength / AGC (Automatic Gain Control) is
        not sufficient (e.g., too high or too low), which often indicates
        measurement issues with the gas analyzer (e.g., dirty lenses).

        This method applies the `flag_signal_strength_eddypro_test` function on the
        given flux data column, signal strength column, and specific parameters.
        The result of the test is stored in the `self._results` dataframe.

        Args:
            signal_strength_col (str): Name of the column containing signal strength
                data to be tested (e.g., 'AGC_LI7500').
            method (str): Name of the method to be used for testing signal strength
                Can be 'discard above' or 'discard below' the <threshold>.
            threshold (int): Threshold for discarding data.

        Returns:
            None: Modifies the `_results` attribute in-place.
        """
        flag = flag_signal_strength_eddypro_test(
            df=self.dfin, var_col=self.fluxcol, idstr=self.idstr,
            signal_strength_col=signal_strength_col,
            method=method, threshold=threshold
        )
        self._results[flag.name] = flag

    def spectral_correction_factor_test(self,
                                        thres_good: int = 2,
                                        thres_ok: int = 4):
        """
        Evaluates the spectral correction factor test for data quality assessment.

        Spectral correction factor test flag is created based on results from EddyPro,
        categorizing data quality into good, ok, and bad.

        Large correction factors can indicate significant flux loss that was not
        properly compensated for. It updates the results within the class instance
        by performing the test on the provided input data.

        Reference:
        (SAB18) Sabbatini, Simone, Ivan Mammarella, Nicola Arriga, Gerardo Fratini,
            Alexander Graf, Lukas Hörtnagl, Andreas Ibrom, Bernard Longdoz, Matthias Mauder,
            Lutz Merbold, Stefan Metzger, Leonardo Montagnani, Andrea Pitacco,
            Corinna Rebmann, Pavel Sedlák, Ladislav Šigut, Domenico Vitale, and
            Dario Papale. 2018. “Eddy Covariance Raw Data Processing for CO2 and Energy
            Fluxes Calculation at ICOS Ecosystem Stations.” International Agrophysics 32(4):495–515.
            doi:10.1515/intag-2017-0043.


        Args:
            thres_good (int, optional): Threshold value below which the data quality is considered good.
                Default is 2.
            thres_ok (int, optional): Threshold value below which the data quality is considered ok,
                but above the `thres_good`. Default is 4.

        Returns:
            None: Modifies the `_results` attribute in-place.
        """
        flag = flag_spectral_correction_factor_eddypro_test(
            df=self.dfin, flux=self.fluxcol, idstr=self.idstr,
            thres_good=thres_good, thres_ok=thres_ok)
        self._results[flag.name] = flag

    def missing_vals_test(self):
        """
        Applies a missing values test to the primary flux column.

        Assign flag = 2 for missing records, flag = 0 for non-missing records.

        Returns:
            None: Modifies the `_results` attribute in-place.
        """
        flagtest = MissingValues(series=self.dfin[self.fluxcol].copy(), idstr=self.idstr)
        flagtest.calc(repeat=False)
        flag = flagtest.get_flag()
        self._results[flag.name] = flag

    def ssitc_test(self, setflag_timeperiod: dict = None):
        """
        Applies the Steady State and Integral Turbulence Characteristics (SSITC) test.

        This method calls the external `flag_ssitc_eddypro_test` function, which
        evaluates the flux data based on steady state and integral turbulence
        criteria (Mauder & Foken, 2004). The resulting flag is added to the
        internal `_results` dataframe.

        Args:
            setflag_timeperiod (dict, optional): A dictionary defining specific
                time periods where the flag should be set manually. Defaults to None.

        Returns:
            None: Modifies the `_results` attribute in-place.
        """
        flag = flag_ssitc_eddypro_test(df=self.dfin, flux=self.fluxcol, idstr=self.idstr,
                                       setflag_timeperiod=setflag_timeperiod)
        self._results[flag.name] = flag

    def gas_completeness_test(self, thres_good: float = 0.99, thres_ok: float = 0.97):
        flag = flag_fluxbasevar_completeness_eddypro_test(df=self.dfin, flux=self.fluxcol,
                                                          fluxbasevar=self.fluxbasevar,
                                                          idstr=self.idstr,
                                                          thres_good=thres_good, thres_ok=thres_ok)
        self._results[flag.name] = flag


def example():
    pass


if __name__ == '__main__':
    example()
