from typing import Literal

import pandas as pd
from pandas import DataFrame

from diive.core.funcs.funcs import validate_id_string
from diive.pkgs.qaqc.eddyproflags import (flags_vm97_eddypro_fulloutputfile_tests, \
                                          flags_vm97_eddypro_fluxnetfile_tests, flag_fluxbasevar_completeness_eddypro_test, \
                                          flag_spectral_correction_factor_eddypro_test, flag_ssitc_eddypro_test,
                                          flag_angle_of_attack_eddypro_test,
                                          flag_steadiness_horizontal_wind_eddypro_test,
                                          flag_signal_strength_eddypro_test)
from diive.pkgs.qaqc.flags import MissingValues


class FluxQualityFlagsEddyPro:

    def __init__(self,
                 dfin: DataFrame,
                 fluxcol: str,
                 fluxbasevar: str,
                 idstr: str = None):
        """
        Create QCF (quality-control flag) for selected flags, calculated
        from EddyPro's _fluxnet_ or _full_output_ results files.

        Args:
            dfin: Dataframe containing EddyPro flux calculation results.
            fluxcol: Name of the flux variable in *dfin*.
            idstr: Suffix added to output variable names.
            fluxbasevar: Name of the variable that was used to calculate the flux, e.g. 'CO2_CONC' for CO2 flux.
        """
        self.fluxcol = fluxcol
        self.dfin = dfin.copy()
        self.idstr = validate_id_string(idstr=idstr)
        self.fluxbasevar = fluxbasevar

        # Collect flags together with flux and potential radiation in separate dataframe
        self._results = self.dfin[[fluxcol]].copy()

    @property
    def results(self) -> DataFrame:
        """Return dataframe containing flags"""
        if not isinstance(self._results, DataFrame):
            raise Exception('Results for flux flags are empty')
        return self._results

    def angle_of_attack_test(
            self,
            application_dates: list or None = None
    ):
        flag = flag_angle_of_attack_eddypro_test(df=self.dfin, flux=self.fluxcol,
                                                 idstr=self.idstr, application_dates=application_dates)
        self._results[flag.name] = flag

    def steadiness_of_horizontal_wind(self):
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
        flag = flag_signal_strength_eddypro_test(
            df=self.dfin, var_col=self.fluxcol, idstr=self.idstr,
            signal_strength_col=signal_strength_col,
            method=method, threshold=threshold
        )
        self._results[flag.name] = flag

    def spectral_correction_factor_test(self,
                                        thres_good: int = 2,
                                        thres_ok: int = 4):
        flag = flag_spectral_correction_factor_eddypro_test(
            df=self.dfin, flux=self.fluxcol, idstr=self.idstr,
            thres_good=thres_good, thres_ok=thres_ok)
        self._results[flag.name] = flag

    def missing_vals_test(self):
        flagtest = MissingValues(series=self.dfin[self.fluxcol].copy(), idstr=self.idstr)
        flagtest.calc(repeat=False)
        flag = flagtest.get_flag()
        self._results[flag.name] = flag

    def ssitc_test(self, setflag_timeperiod: dict = None):
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
