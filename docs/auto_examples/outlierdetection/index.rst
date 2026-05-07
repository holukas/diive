:orphan:

# Outlier Detection Examples

Quality control and outlier detection methods for time series data.

## Overview

10+ robust outlier detection algorithms:

- **Absolute Limits** — Min/max threshold filtering with separate day/night thresholds
- **Hampel Filter** — Spike detection using Median Absolute Deviation (MAD)
- **Local Standard Deviation** — Adaptive rolling window thresholds
- **Z-Score** — Global, rolling, and day/night-stratified z-score detection
- **Z-Score Increments** — Detect abrupt changes in values
- **Local Outlier Factor (LOF)** — Density-based anomaly detection
- **Manual Removal** — Explicit outlier flagging and removal
- **Trim Filter** — Symmetric removal of extreme values
- **Step-Wise Orchestration** — Chain multiple methods for progressive filtering

## Use Cases

- Quality assurance for eddy covariance flux measurements
- Identifying instrument malfunctions and data artifacts
- Progressive multi-stage outlier filtering
- Robust data cleaning before gap-filling


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This module demonstrates the ManualRemoval class for explicitly removing data points or date ranges from a time series.">

.. only:: html

  .. image:: /auto_examples/outlierdetection/images/thumb/sphx_glr_manualremoval_thumb.png
    :alt:

  :doc:`/auto_examples/outlierdetection/manualremoval`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Manual Outlier Removal examples.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This module demonstrates the LocalOutlierFactor class for identifying outliers based on local density deviations. Two modes are available: - Global mode: Single LOF threshold for entire series - Day/night mode: Separate thresholds for daytime/nighttime periods">

.. only:: html

  .. image:: /auto_examples/outlierdetection/images/thumb/sphx_glr_lof_thumb.png
    :alt:

  :doc:`/auto_examples/outlierdetection/lof`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Local Outlier Factor (LOF) outlier detection examples.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Examples demonstrating the z-score increments method for outlier detection. Identifies outliers based on abrupt changes between consecutive values.">

.. only:: html

  .. image:: /auto_examples/outlierdetection/images/thumb/sphx_glr_incremental_thumb.png
    :alt:

  :doc:`/auto_examples/outlierdetection/incremental`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Outlier Detection: Z-Score Increments Examples</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Z-score detects outliers as values that deviate significantly from the mean (measured in standard deviations). Supports global, daytime/nighttime, and rolling window approaches.">

.. only:: html

  .. image:: /auto_examples/outlierdetection/images/thumb/sphx_glr_zscore_thumb.png
    :alt:

  :doc:`/auto_examples/outlierdetection/zscore`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Examples: Z-Score Outlier Detection</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Trim filter removes values below a threshold, then removes an equal number of values from the high end (trimmed mean approach).">

.. only:: html

  .. image:: /auto_examples/outlierdetection/images/thumb/sphx_glr_trim_thumb.png
    :alt:

  :doc:`/auto_examples/outlierdetection/trim`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Examples: Trimming Outliers (Trim Filter)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Examples demonstrating absolute value limits for outlier detection, with separate daytime/nighttime thresholds.">

.. only:: html

  .. image:: /auto_examples/outlierdetection/images/thumb/sphx_glr_absolutelimits_thumb.png
    :alt:

  :doc:`/auto_examples/outlierdetection/absolutelimits`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Outlier Detection: Absolute Limits Examples</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Examples demonstrating the local standard deviation method for outlier detection. Identifies values that deviate significantly from rolling window statistics.">

.. only:: html

  .. image:: /auto_examples/outlierdetection/images/thumb/sphx_glr_localsd_thumb.png
    :alt:

  :doc:`/auto_examples/outlierdetection/localsd`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Outlier Detection: Local Standard Deviation Examples</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Examples demonstrating the Hampel filter (Median Absolute Deviation) for robust outlier detection with optional daytime/nighttime separation.">

.. only:: html

  .. image:: /auto_examples/outlierdetection/images/thumb/sphx_glr_hampel_thumb.png
    :alt:

  :doc:`/auto_examples/outlierdetection/hampel`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Outlier Detection: Hampel Filter Examples</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The StepwiseOutlierDetection class chains multiple outlier detection methods sequentially. Each method operates on results from the previous one, progressively filtering outliers. This example shows 6 methods with full parameter signatures:">

.. only:: html

  .. image:: /auto_examples/outlierdetection/images/thumb/sphx_glr_stepwise_thumb.png
    :alt:

  :doc:`/auto_examples/outlierdetection/stepwise`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Examples demonstrating step-wise outlier detection orchestration.</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/outlierdetection/manualremoval
   /auto_examples/outlierdetection/lof
   /auto_examples/outlierdetection/incremental
   /auto_examples/outlierdetection/zscore
   /auto_examples/outlierdetection/trim
   /auto_examples/outlierdetection/absolutelimits
   /auto_examples/outlierdetection/localsd
   /auto_examples/outlierdetection/hampel
   /auto_examples/outlierdetection/stepwise


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: outlierdetection_python.zip </auto_examples/outlierdetection/outlierdetection_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: outlierdetection_jupyter.zip </auto_examples/outlierdetection/outlierdetection_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
