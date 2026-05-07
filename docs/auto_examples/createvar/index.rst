:orphan:

# Variable Creation Examples

Create derived variables and transformations for ecosystem flux analysis.

## Overview

Generate new variables from raw measurements:

- **Unit Conversions** — Temperature, latent heat, evapotranspiration conversions
- **Air Properties** — Aerodynamic resistance, dry air density
- **Vapor Pressure Deficit (VPD)** — Calculate and gap-fill VPD
- **Lagged Variants** — Create time-shifted variables for temporal analysis
- **Noise Generation** — Synthetic data creation and noise injection
- **Potential Radiation** — Solar radiation calculations
- **Day/Night Flags** — Solar geometry-based day/night classification
- **Time Since Event** — Count records since condition met

## Use Cases

- Feature engineering for machine learning models
- Unit standardization across datasets
- Creating environmental context variables
- Temporal feature creation for time series analysis


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Run this script to see daytime/nighttime flag results:     python examples/createvar/daynightflag.py">

.. only:: html

  .. image:: /auto_examples/createvar/images/thumb/sphx_glr_daynightflag_thumb.png
    :alt:

  :doc:`/auto_examples/createvar/daynightflag`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Examples for daytime/nighttime flag calculation using daynightflag module.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Run this script to see air calculation results:     python examples/createvar/air.py">

.. only:: html

  .. image:: /auto_examples/createvar/images/thumb/sphx_glr_air_thumb.png
    :alt:

  :doc:`/auto_examples/createvar/air`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Examples for air variable calculations using air module.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Run this script to see conversion results:     python examples/createvar/conversions.py">

.. only:: html

  .. image:: /auto_examples/createvar/images/thumb/sphx_glr_conversions_thumb.png
    :alt:

  :doc:`/auto_examples/createvar/conversions`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Examples for unit conversions and variable transformations using conversions module.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="VPD can be calculated from air temperature (TA) and relative humidity (RH), which are widely available measurements in ecosystem monitoring networks.">

.. only:: html

  .. image:: /auto_examples/createvar/images/thumb/sphx_glr_vpd_thumb.png
    :alt:

  :doc:`/auto_examples/createvar/vpd`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Examples for VPD (Vapor Pressure Deficit) calculations using calc_vpd_from_ta_rh.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Run this script to see lagged variants examples:     python examples/createvar/laggedvariants.py">

.. only:: html

  .. image:: /auto_examples/createvar/images/thumb/sphx_glr_laggedvariants_thumb.png
    :alt:

  :doc:`/auto_examples/createvar/laggedvariants`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Examples for lagged variants creation using laggedvariants module.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Run this script to see potential radiation examples:     python examples/createvar/potentialradiation.py">

.. only:: html

  .. image:: /auto_examples/createvar/images/thumb/sphx_glr_potentialradiation_thumb.png
    :alt:

  :doc:`/auto_examples/createvar/potentialradiation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Examples for potential radiation calculations using potentialradiation module.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="TimeSince counts consecutive records since the last occurrence of a condition by tracking when values fall outside a specified limit range. Useful for: - Dry period detection (time since last precipitation &gt; 0) - Frost period detection (time since freezing temperature &lt;= 0°C) - Warm spell analysis - Event-based time tracking">

.. only:: html

  .. image:: /auto_examples/createvar/images/thumb/sphx_glr_timesince_thumb.png
    :alt:

  :doc:`/auto_examples/createvar/timesince`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Examples for time-since calculations using TimeSince class.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Run this script to see noise examples:     python examples/createvar/noise.py">

.. only:: html

  .. image:: /auto_examples/createvar/images/thumb/sphx_glr_noise_thumb.png
    :alt:

  :doc:`/auto_examples/createvar/noise`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Examples for noise generation and addition using noise module.</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/createvar/daynightflag
   /auto_examples/createvar/air
   /auto_examples/createvar/conversions
   /auto_examples/createvar/vpd
   /auto_examples/createvar/laggedvariants
   /auto_examples/createvar/potentialradiation
   /auto_examples/createvar/timesince
   /auto_examples/createvar/noise


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: createvar_python.zip </auto_examples/createvar/createvar_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: createvar_jupyter.zip </auto_examples/createvar/createvar_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
