Preprocessing Examples
======================

Cleaning measured data before analysis: correcting known sensor issues, flagging implausible
records, and aggregating those flags into an overall quality judgement.

The examples are grouped into three subsections, in the order they are normally applied.

Contents
--------

**Corrections** (``corrections/``, 7 examples) — fix a known, systematic error in the measurement:
sensor offsets against a replicate, wind direction offsets, the nighttime zero offset of a
radiation sensor, and setting values to a threshold or to missing.

**Outlier detection** (``outlier_detection/``, 9 examples) — flag records that cannot be right:
absolute limits, Hampel, local standard deviation, z-score variants, local outlier factor,
trimming and manual removal, plus chaining several methods with ``StepwiseOutlierDetection``.

**Quality control** (``qaqc/``, 3 examples) — aggregate the individual test flags into one overall
quality flag with ``FlagQCF``, and screen meteorological data.

Running Examples
----------------

.. code-block:: bash

   uv run python examples/preprocessing/outlier_detection/outlier_hampel.py
   uv run python examples/run_all_examples.py
