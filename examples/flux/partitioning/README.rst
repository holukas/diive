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

One entry point
---------------

``dv.flux.PartitioningNEE`` (or the function ``dv.flux.partition_nee``) runs any of the four ports.
``method`` picks ``'nighttime'`` or ``'daytime'``, ``mode`` picks ``'oneflux'`` (default, the ICOS
and FLUXNET2015 approach) or ``'reddyproc'``. The results are those of the matching class:

.. code-block:: python

   import diive as dv

   part = dv.flux.PartitioningNEE(
       nee=df['NEE_CUT_REF_orig'], method='nighttime', mode='oneflux',
       ta=df['Tair_orig'], sw_in=df['Rg_orig'], nee_f=df['NEE_CUT_REF_f'],
       ta_f=df['Tair_f'], lat=46.815,
   ).run()
   part.reco, part.gpp   # RECO_NT_OF, GPP_NT_OF

In the post-processing chain the same switches are ``FluxConfig(partition_method=...,
partition_mode=...)`` and ``run_level42_partitioning(data, method=..., mode=...)``.

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
