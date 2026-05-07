:orphan:

# Flux Quality and Analysis Examples

Flux-specific quality control, analysis, and processing workflows.

## Overview

Specialized methods for eddy covariance flux data:

- **High-Quality Flux Analysis** — Robust outlier detection for CO₂ and energy fluxes
- **Self-Heating Correction** — Sonic anemometer temperature correction (SCOP methodology)
- **Uncertainty Quantification** — Random uncertainty estimation (PAS20 method)
- **USTAR Threshold Detection** — Friction velocity filtering for nighttime fluxes
- **Flux Variable Detection** — Identify and extract flux measurement variables

## Use Cases

- Multi-level flux processing workflows
- Identifying valid/invalid flux measurements
- Correcting instrument-specific biases
- Quality assessment of ecosystem flux data
- Uncertainty quantification for publications


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Detects which base variable (e.g., CO2, H2O) was used to calculate a given flux variable (e.g., FC, FH2O, LE). Useful for understanding measurement nomenclature in eddy covariance data files.">

.. only:: html

  .. image:: /auto_examples/flux/images/thumb/sphx_glr_common_thumb.png
    :alt:

  :doc:`/auto_examples/flux/common`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Examples for flux variable utilities and base variable detection.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates robust outlier detection for CO2 flux (NEE) using Hampel filter (Median Absolute Deviation) with automatic day/night separation based on solar geometry. The Hampel method is ideal for removing measurement spikes while preserving ecosystem signal.">

.. only:: html

  .. image:: /auto_examples/flux/images/thumb/sphx_glr_hqflux_thumb.png
    :alt:

  :doc:`/auto_examples/flux/hqflux`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Examples for high-quality flux analysis with outlier detection.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates the RandomUncertaintyPAS20 class for computing flux measurement uncertainty across multiple error sources (instrumental, statistical, gap-filling).">

.. only:: html

  .. image:: /auto_examples/flux/images/thumb/sphx_glr_uncertainty_thumb.png
    :alt:

  :doc:`/auto_examples/flux/uncertainty`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Random uncertainty quantification for eddy covariance flux measurements.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates SCOP (Self-heating Correction for Open-Path) methodology to remove spurious CO2 flux measurements caused by sun-induced heating of instrument surfaces.">

.. only:: html

  .. image:: /auto_examples/flux/images/thumb/sphx_glr_selfheating_thumb.png
    :alt:

  :doc:`/auto_examples/flux/selfheating`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Self-heating correction examples for open-path IRGA sensors.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates USTAR (friction velocity) threshold determination using multiple temperature classes (Papale et al., 2006) and applying constant USTAR thresholds to create multiple flux scenarios for uncertainty analysis.">

.. only:: html

  .. image:: /auto_examples/flux/images/thumb/sphx_glr_ustarthreshold_thumb.png
    :alt:

  :doc:`/auto_examples/flux/ustarthreshold`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">USTAR threshold detection and filtering for low-turbulence flux data.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Comprehensive example demonstrating: - USTAR threshold detection with forward/back modes - Bootstrap uncertainty estimation - Visualization of NEE response to USTAR stratification">

.. only:: html

  .. image:: /auto_examples/flux/images/thumb/sphx_glr_ustar_mp_detection_thumb.png
    :alt:

  :doc:`/auto_examples/flux/ustar_mp_detection`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">USTAR Moving Point Detection - Complete Workflow (Papale et al., 2006)</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/flux/common
   /auto_examples/flux/hqflux
   /auto_examples/flux/uncertainty
   /auto_examples/flux/selfheating
   /auto_examples/flux/ustarthreshold
   /auto_examples/flux/ustar_mp_detection


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: flux_python.zip </auto_examples/flux/flux_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: flux_jupyter.zip </auto_examples/flux/flux_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
