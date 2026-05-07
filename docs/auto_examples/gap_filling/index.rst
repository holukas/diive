:orphan:

# Gap-Filling Examples

Machine learning and statistical methods for gap-filling in time series data.

## Overview

Comprehensive gap-filling workflows with multiple algorithms:

- **Random Forest Gap-Filling** — ML-based gap-filling with 8-stage feature engineering (R² 0.60-0.80)
- **XGBoost Gap-Filling** — Gradient boosting approach with hyperparameter optimization (R² 0.65-0.85)
- **Marginal Distribution Sampling (MDS)** — Meteorological similarity matching, no training required
- **Linear Interpolation** — Simple conservative gap-filling for small gaps
- **Method Comparison** — Side-by-side evaluation of ML vs. MDS approaches
- **Feature Engineering** — Harmonized 8-stage pipeline: lags, rolling stats, differencing, EMA, polynomial terms, STL, timestamps, record number

## Use Cases

- Filling flux measurement gaps in eddy covariance data
- Preparing gap-free datasets for ecosystem analyses
- Comparing gap-filling method performance
- Long-term flux dataset completion


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Gap-filling after Reichstein et al (2005): https://doi.org/10.1111/j.1365-2486.2005.001002.x">

.. only:: html

  .. image:: /auto_examples/gap_filling/images/thumb/sphx_glr_mds_thumb.png
    :alt:

  :doc:`/auto_examples/gap_filling/mds`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Marginal Distribution Sampling (MDS) gap-filling examples.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates simple linear interpolation for filling small gaps in time series data, with configurable limits on gap size. Includes two examples showing conservative (limit=1) vs. generous (limit=5) gap-filling strategies.">

.. only:: html

  .. image:: /auto_examples/gap_filling/images/thumb/sphx_glr_interpolate_thumb.png
    :alt:

  :doc:`/auto_examples/gap_filling/interpolate`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Linear interpolation gap-filling examples.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="XGBoost is a gradient boosting approach for gap-filling time series data. Effective for non-linear relationships, complex temporal interactions, and data with outliers.">

.. only:: html

  .. image:: /auto_examples/gap_filling/images/thumb/sphx_glr_xgboost_ts_thumb.png
    :alt:

  :doc:`/auto_examples/gap_filling/xgboost_ts`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">XGBoost gap-filling examples.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Random Forest is a robust, interpretable machine learning approach for gap-filling time series data. Effective for non-linear relationships and feature interactions.">

.. only:: html

  .. image:: /auto_examples/gap_filling/images/thumb/sphx_glr_randomforest_ts_thumb.png
    :alt:

  :doc:`/auto_examples/gap_filling/randomforest_ts`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Random Forest gap-filling examples.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates three gap-filling methods with cumulative flux curves. Uses one month of data for fast execution. Methods are evaluated on the same data with performance metrics and cumulative carbon flux visualization.">

.. only:: html

  .. image:: /auto_examples/gap_filling/images/thumb/sphx_glr_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/gap_filling/comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comparison of gap-filling methods: MDS, Random Forest, and XGBoost.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates the performance improvement of FluxMDS (optimized vectorization) compared to the original _FluxMDS implementation.">

.. only:: html

  .. image:: /auto_examples/gap_filling/images/thumb/sphx_glr_mds_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/gap_filling/mds_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">FluxMDS Performance Comparison: Original vs Optimized.</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/gap_filling/mds
   /auto_examples/gap_filling/interpolate
   /auto_examples/gap_filling/xgboost_ts
   /auto_examples/gap_filling/randomforest_ts
   /auto_examples/gap_filling/comparison
   /auto_examples/gap_filling/mds_comparison


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: gap_filling_python.zip </auto_examples/gap_filling/gap_filling_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: gap_filling_jupyter.zip </auto_examples/gap_filling/gap_filling_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
