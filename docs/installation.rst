.. _installation:

============
Installation
============

diive needs **Python 3.12 or 3.13**.

Required tools
==============

Each installation option uses only some of these tools, as listed in the last column.

.. list-table::
   :widths: 12 58 30
   :header-rows: 1

   * - Tool
     - What it does
     - Needed for
   * - `conda <https://conda-forge.org/download/>`_
     - Creates separate Python environments, each with its own Python version and packages.
     - Option A (recommended for new Python users)
   * - `pip <https://pip.pypa.io/>`_
     - Installs Python packages, such as diive, from `PyPI <https://pypi.org/project/diive/>`_.
       It comes with Python.
     - Options A, B and C
   * - `git <https://git-scm.com/downloads>`_
     - Downloads the source code from GitHub and keeps it up to date.
     - Options C and D
   * - `uv <https://docs.astral.sh/uv/>`_
     - Installs Python, creates the environment and installs the packages, all in one tool.
     - Option D (development)

An **environment** is a folder with one Python installation and the packages you add to it.
Installing diive into its own environment keeps it from clashing with other Python projects on
your computer.

Basic installation and optional components
==========================================

``pip install diive`` installs the library and everything it needs for processing and plotting:
pandas, NumPy, scikit-learn, XGBoost, SHAP, statsmodels, matplotlib, seaborn, Bokeh, PyArrow and a
few smaller packages. This is sufficient for using diive in scripts.

It does **not** install the desktop app, the 3-D views, the database connection or Jupyter. These
are optional components:

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Purpose
     - Install
     - Adds
   * - Use diive in Python scripts
     - ``diive``
     - The library (basic install)
   * - Use the desktop app
     - ``"diive[gui]"``
     - PySide6 (Qt)
   * - See 3-D surface plots in the app
     - ``"diive[gui,gui3d]"``
     - PyVista and VTK (large)
   * - Connect to an InfluxDB v2 database (see note below)
     - ``"diive[db]"``, with the app ``"diive[gui,db]"``
     - ``influxdb-client``
   * - Everything above
     - ``"diive[gui,gui3d,db]"``
     -
   * - Run Jupyter notebooks, e.g. those in ``notebooks/``
     - ``jupyterlab`` (a separate package)
     - JupyterLab

Put package names with brackets in double quotes, as shown, so that the command works in every
shell (Command Prompt, PowerShell, bash, zsh).

The InfluxDB connection is a specialized feature for groups that keep their measurements in an
InfluxDB v2 database. Most users do not need it. It is used by the library (``InfluxIO``), the
app's **File > Database** tabs, the database source of the app's meteo screening tab and the
database notebooks in ``notebooks/``. Without it, these parts of the app show a short notice
instead.

Option A: conda (recommended for new Python users)
==================================================

1. **Install conda.** Download and install `Miniforge <https://conda-forge.org/download/>`_, a free
   conda distribution. On Windows, keep the default options. On macOS and Linux, answer **yes**
   when the installer asks whether to initialize conda.
2. **Open a terminal with conda.** On Windows, open **Miniforge Prompt** from the Start menu. On
   macOS and Linux, open a new terminal window.
3. **Create an environment** named ``diive`` with Python 3.12, and switch to it:

   .. code-block:: bash

      conda create -n diive python=3.12
      conda activate diive

   The prompt now starts with ``(diive)``. Every following command runs inside this environment.
4. **Install diive with pip.** diive is published on PyPI, so it is installed with pip, not with
   conda. For the library and the desktop app:

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

**Using the library.** The library is available in the same environment, in scripts, in Python or in Jupyter:

.. code-block:: bash

   python -m pip install jupyterlab   # once, only if you want notebooks
   jupyter lab

**Updating to the latest release.**

.. code-block:: bash

   conda activate diive
   python -m pip install --upgrade "diive[gui]"

**Removing the environment.** This deletes the environment and all its packages:

.. code-block:: bash

   conda deactivate
   conda env remove -n diive

In this environment, install further packages with pip as well. Installing the same packages with
conda afterwards can replace the versions diive needs.

Option B: pip in an existing environment
========================================

If you already have Python 3.12 or 3.13 and an environment you want to use (for example one made
with ``venv``), activate it and run:

.. code-block:: bash

   python -m pip install "diive[gui]"   # or just: diive
   diive-gui                            # starts the desktop app

Option C: development version from GitHub
=========================================

PyPI has the latest release. New work reaches GitHub earlier: it lands in the ``indev`` branch
first and moves to ``main`` with each merge. To install from GitHub, you need
`git <https://git-scm.com/downloads>`_.

To install directly from GitHub, run this inside your environment (for example the conda environment from
Option A):

.. code-block:: bash

   python -m pip install "diive[gui] @ git+https://github.com/holukas/diive.git"         # main branch
   python -m pip install "diive[gui] @ git+https://github.com/holukas/diive.git@indev"   # indev branch

Alternatively, clone the repository with git and install it from the local folder. With ``-e`` (editable), changes to
the code, or new code fetched with ``git pull``, take effect without installing again:

.. code-block:: bash

   git clone https://github.com/holukas/diive.git
   cd diive
   python -m pip install -e ".[gui]"

To get the newest code later, run ``git pull`` in the ``diive`` folder. Switch to the development
branch with ``git checkout indev``.

Option D: development setup with uv
===================================

uv installs a matching Python version itself and keeps the environment in the folder ``.venv``.
Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ and
`git <https://git-scm.com/downloads>`_, then:

.. code-block:: bash

   git clone https://github.com/holukas/diive.git
   cd diive
   uv sync --extra gui    # library, desktop app and the development tools
   uv run diive-gui       # starts the desktop app

``uv run`` runs a command inside the project environment, for example ``uv run pytest tests/`` or
``uv run python examples/gapfilling/gapfill_randomforest.py``. Plain ``uv sync`` installs the
library and the development tools (tests, linting, docs, Jupyter). Optional
components:

.. list-table::
   :widths: 12 53 35
   :header-rows: 1

   * - Part
     - Adds
     - Command
   * - ``gui``
     - The desktop app (PySide6)
     - ``uv sync --extra gui``
   * - ``gui3d``
     - 3-D surface plots in the app (PyVista, VTK)
     - ``uv sync --extra gui3d``
   * - ``db``
     - InfluxDB v2 access (``influxdb-client``)
     - ``uv sync --group db``
   * - ``dev``
     - Tests, linting, docs and Jupyter
     - included in ``uv sync``
   * - ``build``
     - PyInstaller, to build the standalone Windows app
     - ``uv sync --group build``

Combine them as needed, e.g. ``uv sync --extra gui --extra gui3d --group db``.
``uv sync --all-extras --all-groups`` installs everything. ``--all-extras`` alone leaves out
``db``, ``dev`` and ``build``, because uv counts those as groups.

``db`` is both an extra and a group on purpose. When you work on diive itself, use the group
(``uv sync --group db``). A project that depends on diive asks for the extra (``diive[db]``),
because a group never reaches the published package.

Standalone Windows app
======================

The desktop app can also be built as a Windows program that runs without Python. In PowerShell,
after ``uv sync --extra gui --group build``, run ``.\packaging\build_gui.ps1 -Clean``. Details:
`packaging/README.md <https://github.com/holukas/diive/blob/main/packaging/README.md>`_.

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

If the version check fails, or ``diive-gui`` is not found:

- **The environment is not active.** Run ``conda activate diive`` (Option A) or activate your own
  environment. With uv, put ``uv run`` in front of the command.
- **No module named 'PySide6' when starting the app.** The desktop app was not installed. Install
  ``"diive[gui]"``.
- **diive-gui is still not found.** Start the app with ``python -m diive.gui._cli`` instead.

Next steps
==========

After installation, continue with the :ref:`Getting Started <getting_started>` guide or browse
the :ref:`Example Gallery <examples_gallery>`.
