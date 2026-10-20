.. _installation:

============
Installation
============

diive needs **Python 3.12 or 3.13**. The steps below install Python for you, so you do not need
to install it separately.

Install diive into its own **environment**: a folder with one Python installation and the packages
you add to it. This keeps diive from clashing with other Python projects on your computer.

Library and desktop app with conda
==================================

This is the recommended way for new Python users. `conda <https://conda-forge.org/download/>`_
manages the environment, and `pip <https://pip.pypa.io/>`_ installs diive from
`PyPI <https://pypi.org/project/diive/>`_.

1. **Install conda.** Download and install `Miniforge <https://conda-forge.org/download/>`_, a free
   conda distribution (on Windows, the x86_64 installer). On Windows, keep the default options. On
   macOS and Linux, answer **yes** when the installer asks whether to initialize conda.
2. **Open a terminal with conda.** On Windows, open **Miniforge Prompt** from the Start menu. On
   macOS and Linux, open a new terminal window.
3. **Create an environment** named ``diive`` and switch to it. Confirm with ``y`` when conda asks:

   .. code-block:: bash

      conda create -n diive python=3.12
      conda activate diive

   The prompt now starts with ``(diive)``. Every following command runs inside this environment.
4. **Install diive** with the desktop app. The download takes a few minutes:

   .. code-block:: bash

      python -m pip install "diive[gui]"

   For the library only, use ``python -m pip install diive``. Writing ``python -m pip`` makes sure
   the package goes into the active environment.
5. **Start the desktop app:**

   .. code-block:: bash

      diive-gui

   The first time, the app opens with the bundled example dataset. Later it reopens the project
   you used last.

**Starting the app later.** Open the terminal (step 2) and run:

.. code-block:: bash

   conda activate diive
   diive-gui

**Updating to the latest release.**

.. code-block:: bash

   conda activate diive
   python -m pip install --upgrade "diive[gui]"

**Removing the environment.** This deletes the environment and all its packages:

.. code-block:: bash

   conda deactivate
   conda env remove -n diive

diive is installed with pip, also inside a conda environment. Install further packages in this
environment with pip as well. Installing the same packages with conda afterwards can replace the
versions diive needs. The diive package on conda-forge is outdated, so do not install diive with
``conda install``.

Optional components
===================

``pip install diive`` installs the library and everything it needs for processing and plotting:
pandas, NumPy, scikit-learn, XGBoost, SHAP, statsmodels, matplotlib, seaborn, Bokeh, PyArrow and a
few smaller packages. This is sufficient for using diive in scripts. The other parts are optional:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Install
     - Contains
   * - ``diive``
     - The library only
   * - ``"diive[gui]"``
     - The library and the desktop app (PySide6)
   * - ``"diive[gui,gui3d]"``
     - Adds 3-D surface plots to the app (PyVista and VTK, large download)
   * - ``"diive[db]"``
     - Adds access to InfluxDB v2 databases (``influxdb-client``)
   * - ``"diive[all]"``
     - All of the above

Keep the double quotes around names with brackets. Some shells need them.

The InfluxDB connection is a specialized feature for groups that keep their measurements in an
InfluxDB v2 database. Most users do not need it. It is used by the library (``InfluxIO``), the
app's **File > Database** tabs, the database source of the app's meteo screening tab and the
database notebooks in ``notebooks/``. Without it, these parts of the app show a short notice
instead.

Other Python environments
=========================

diive works in any environment with Python 3.12 or 3.13, for example a conda environment you
already use or one made with ``venv``. Activate it and run:

.. code-block:: bash

   python -m pip install "diive[gui]"

A new environment is safer than an existing one: diive needs recent versions of pandas, NumPy and
PyArrow, and pip replaces older versions in the environment.

Jupyter
=======

JupyterLab is not part of diive. Install it in the diive environment and start it there:

.. code-block:: bash

   python -m pip install jupyterlab
   jupyter lab

If you start Jupyter from another environment, make the diive environment available there as a
kernel instead. Run this in the diive environment, then choose **Python (diive)** in Jupyter:

.. code-block:: bash

   python -m pip install ipykernel
   python -m ipykernel install --user --name diive --display-name "Python (diive)"

Newest code from GitHub
=======================

PyPI has the latest release. New work reaches GitHub earlier: it lands in the ``indev`` branch
first and moves to ``main`` with each merge. To install it into your environment, you need
`git <https://git-scm.com/downloads>`_:

.. code-block:: bash

   python -m pip install "diive[gui] @ git+https://github.com/holukas/diive.git"         # main branch
   python -m pip install "diive[gui] @ git+https://github.com/holukas/diive.git@indev"   # indev branch

Run the same command again to update.

Development setup
=================

To work on the code, run the examples or run the tests, clone the repository and install it in
editable mode, together with the development tools (tests, linting, documentation, Jupyter).
Changes to the code then take effect without installing again. New work happens in the ``indev``
branch.

In a conda or other Python environment:

.. code-block:: bash

   git clone https://github.com/holukas/diive.git
   cd diive
   git checkout indev
   python -m pip install --upgrade pip              # --group needs pip 25.1 or newer
   python -m pip install -e ".[gui]" --group dev

With `uv <https://docs.astral.sh/uv/>`_, which also installs Python itself:

.. code-block:: bash

   git clone https://github.com/holukas/diive.git
   cd diive
   git checkout indev
   uv sync --extra gui

Get the newest code later with ``git pull``. The contributing guide on GitHub
(`CONTRIBUTING.md <https://github.com/holukas/diive/blob/main/CONTRIBUTING.md>`_) describes the
tests, the code style and further uv options.

Verifying the installation
==========================

.. code-block:: bash

   python -c "import diive; print(diive.__version__)"

This prints the installed version. Load the bundled example data to see that the library works:

.. code-block:: python

   import diive as dv

   df = dv.load_exampledata_parquet()
   print(f"Loaded {len(df)} records")
   print(df.head())

Common problems:

- **The environment is not active.** Run ``conda activate diive`` or activate your own
  environment. With uv, put ``uv run`` in front of the command.
- **pip installs an old diive version or finds none.** The environment has a Python version other
  than 3.12 or 3.13. Check with ``python --version`` and create a new environment with
  ``python=3.12``.
- **No module named 'PySide6' when starting the app.** The desktop app was not installed. Install
  ``"diive[gui]"``.
- **diive-gui is not found.** Start the app with ``python -m diive.gui._cli`` instead.

Next steps
==========

After installation, continue with the :ref:`Getting Started <getting_started>` guide or browse
the :ref:`Example Gallery <examples_gallery>`.
