.. _api_reference:

=============
API Reference
=============

``import diive as dv`` exposes ten domain namespaces. Each page below documents
one of them, generated from that namespace's ``__all__``.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :ref:`dv.outliers <api_outliers>`
     - Detection methods that flag implausible records, most of them with an optional daytime/nighttime split.
   * - :ref:`dv.gapfilling <api_gapfilling>`
     - Machine-learning and meteorological-similarity gap-filling, plus the feature engineering that feeds the models.
   * - :ref:`dv.flux <api_flux>`
     - Eddy covariance flux post-processing: the processing chain, u* threshold detection, NEE partitioning and uncertainty.
   * - :ref:`dv.analysis <api_analysis>`
     - Time series analysis: correlation, causality, gaps, binning, decomposition and compound extremes.
   * - :ref:`dv.plotting <api_plotting>`
     - Plot types. Every class follows the two-phase pattern: data and computation parameters at construction, styling and rendering in ``plot()``.
   * - :ref:`dv.times <api_times>`
     - Timestamp sanitizing, frequency detection, resampling and formatting.
   * - :ref:`dv.variables <api_variables>`
     - Derived variables and feature calculations, from potential radiation and VPD to day/night flags.
   * - :ref:`dv.corrections <api_corrections>`
     - Offset and gain corrections applied to measured series.
   * - :ref:`dv.qaqc <api_qaqc>`
     - Quality flags, flag aggregation and the stepwise meteo screening pipeline.
   * - :ref:`dv.events <api_events>`
     - Event markers for instants and periods, and their conversion to flags and plot overlays.

.. toctree::
   :maxdepth: 1

   api/toplevel
   api/outliers
   api/gapfilling
   api/flux
   api/analysis
   api/plotting
   api/times
   api/variables
   api/corrections
   api/qaqc
   api/events
