"""
TEST_DOCSTRING_EXAMPLES: DOCSTRING SAMPLES ACTUALLY RUN
=======================================================

Executes the ``>>>`` samples in library docstrings with the :mod:`doctest`
module. The companion ``test_docstring_refs`` resolves what a sample *names*;
this one runs what a sample *does*, which is the only way a stale call shape
(a renamed keyword, a dropped parameter) gets caught. Two such samples were
broken when this test was written: ``FlagQCF`` was documented with a ``series=``
argument it has not taken for a long time, and ``classify_variable('TA_f')`` was
documented as returning ``None`` when it returns a meteo class.

Every sample-bearing module is discovered by parsing the tree, so a **new
sample is executed automatically**. A sample that cannot run here must be listed
in :data:`SKIP` with a reason — there is no silent filter, and a ``SKIP`` entry
that no longer matches a real sample fails the test too.

``diive.gui`` is out of scope: it needs the optional PySide6 dependency and its
docstrings carry no samples.

Part of the diive library: https://github.com/holukas/diive
"""

import ast
import doctest
import importlib
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / 'diive'

# Samples kept for readers but not executed here, each with the reason it cannot
# run cheaply and silently. Keys are doctest names (module path, then the
# qualified name of the object whose docstring holds the sample).
SKIP = {
    'diive.analysis.granger.GrangerCausality':
        "statsmodels' grangercausalitytests prints its own report to stdout",
    'diive.core.ml.results.GapFillingResult':
        "trains a random forest on the full bundled record",
    'diive.flux.fluxprocessingchain.container.add_driver':
        "needs a FluxLevelData built by init_flux_data",
    'diive.flux.fluxprocessingchain.run_chain.run_chain':
        "runs the whole six-level chain on EddyPro-FLUXNET input",
    'diive.core.plotting.timeseries.TimeSeries.plot_interactive':
        "bokeh show() opens a browser tab",
    'diive.core.plotting.timeseries.TimeSeries.plot_rangetool':
        "bokeh show() opens a browser tab",
    'diive.flux.lowres.ustar_bootstrap.UstarBootstrapThresholds':
        "100 bootstrap iterations over the full record",
    'diive.flux.lowres.ustar_mp_detection.UstarMovingPointDetection':
        "seasonal threshold detection over the full record",
    'diive.flux.lowres.ustar_vekuri_detection.UstarVekuriThresholdDetection':
        "seasonal threshold detection over the full record",
    'diive.flux.partitioning.daytime_oneflux.DaytimePartitioningOneFlux':
        "partitioning port, ~20 s per year, plus a parquet load",
    'diive.flux.partitioning.daytime_reddyproc.DaytimePartitioningReddyProc':
        "partitioning port, ~20 s per year, plus a parquet load",
    'diive.flux.partitioning.nighttime_oneflux.NighttimePartitioningOneFlux':
        "partitioning port, ~20 s per year, plus a parquet load",
    'diive.flux.partitioning.nighttime_reddyproc.NighttimePartitioningReddyProc':
        "partitioning port, ~20 s per year, plus a parquet load",
}


def _modules_with_samples():
    """Yield the dotted name of every diive module holding a ``>>>`` sample.

    Parsed with ``ast`` rather than imported, so the scan itself pulls in
    nothing.
    """
    for path in sorted(PACKAGE_ROOT.rglob('*.py')):
        parts = path.relative_to(PACKAGE_ROOT).parts
        if parts[0] == 'gui':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node)
            if doc and '>>>' in doc:
                stem = parts[:-1] + (path.stem,)
                yield 'diive.' + '.'.join(p for p in stem if p != '__init__')
                break


def _collect():
    """Return (runnable, skipped) lists of DocTest objects with examples."""
    finder = doctest.DocTestFinder(exclude_empty=True)
    runnable, skipped = [], []
    for modname in _modules_with_samples():
        module = importlib.import_module(modname)
        for test in finder.find(module, modname):
            if not test.examples:
                continue
            (skipped if test.name in SKIP else runnable).append(test)
    return runnable, skipped


class TestDocstringExamplesRun(unittest.TestCase):
    """Run every docstring sample that is not explicitly skipped."""

    @classmethod
    def setUpClass(cls):
        cls.runnable, cls.skipped = _collect()

    def tearDown(self):
        plt.close('all')

    def test_samples_are_found(self):
        self.assertGreater(len(self.runnable), 5,
                           "expected several runnable samples; the scan is wrong")

    def test_skip_list_has_no_stale_entries(self):
        found = {t.name for t in self.skipped}
        self.assertEqual(sorted(set(SKIP) - found), [],
                         "SKIP names samples that no longer exist")

    def test_every_runnable_sample_passes(self):
        runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
        failed = []
        for test in self.runnable:
            with self.subTest(sample=test.name):
                result = runner.run(test, clear_globs=False)
                if result.failed:
                    failed.append(test.name)
        self.assertEqual(failed, [], "docstring samples do not run:\n  "
                         + "\n  ".join(failed))


if __name__ == '__main__':
    unittest.main()
