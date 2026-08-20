Flux Processing Chain Examples
==============================

Post-processing of Level-1 eddy covariance fluxes through the Swiss FluxNet levels: quality flag
expansion (L2), storage correction (L3.1), outlier removal (L3.2), u\* filtering (L3.3),
gap-filling (L4.1) and NEE partitioning (L4.2).

**4 examples covering both entry points into the chain.**

Contents
--------

- **fluxprocessingchain_composable.py** — the composable per-level API: one function per level,
  each taking the ``FluxLevelData`` container and returning a new one. Full control over every
  detector and model setting, and the path to branch several gap-filling methods from the same
  filtered state.
- **fluxprocessingchain_runchain.py** — the single-call driver: one ``FluxConfig``, one ``run_chain``.
  Fixed sensible defaults, much less to write.
- **fluxprocessingchain_level2.py** — Level 2 on its own, showing the per-test flag settings.
- **fluxprocessingchain_partitioning.py** — Level 4.2, splitting gap-filled NEE into GPP and RECO.

Which entry point?
------------------

``run_chain`` when the standard pipeline is what you want. The composable functions when you need to
override anything ``FluxConfig`` does not expose — a specific outlier detector, model
hyperparameters, or a custom sequence of Level-3.2 tests.

Running Examples
----------------

.. code-block:: bash

   uv run python examples/flux/fluxprocessingchain/fluxprocessingchain_composable.py
   uv run python examples/run_all_examples.py
