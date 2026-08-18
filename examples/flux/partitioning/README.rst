NEE Partitioning Examples
=========================

Splitting net ecosystem exchange (NEE) into gross primary production (GPP) and ecosystem
respiration (RECO). Four faithful ports of the standard reference routines: two nighttime
(Reichstein et al. 2005) and two daytime (Lasslop et al. 2010), one of each from ONEFlux (``_OF``)
and from REddyProc (``_RP``).

**5 examples, one per method plus a comparison.**

Contents
--------

- **partitioning_nighttime_oneflux.py** — ONEFlux nighttime, per calendar year, output tagged
  ``*_NT_OF``.
- **partitioning_nighttime_reddyproc.py** — REddyProc nighttime, one temperature sensitivity for
  the whole record, output tagged ``*_NT_RP``.
- **partitioning_daytime_reddyproc.py** — REddyProc daytime light-response curves, ``*_DT_RP``.
- **partitioning_daytime_oneflux.py** — ONEFlux daytime, ``*_DT_OF``.
- **partitioning_comparison.py** — all four on the same data, side by side.

Because each method tags its output columns differently, all four can be kept in one dataframe and
compared directly.

Units
-----

Inputs are in physical units: air temperature in °C, VPD in kPa, radiation in W m-2. The ports
state the units they expect in their docstrings and do not validate them, so passing hPa where kPa
is expected produces a plausible-looking wrong answer.

Running Examples
----------------

.. code-block:: bash

   uv run python examples/flux/partitioning/partitioning_comparison.py
   uv run python examples/run_all_examples.py
