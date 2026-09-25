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
that no longer matches a real sample fails the test too. A check compares what
is collected against the samples found in the source, so a sample doctest cannot
see fails loudly instead of being skipped.

A method sample can continue from its class sample (see :data:`CONTINUES`).
Console progress lines are silenced while samples run, so a sample's expected
output is only what the sample itself prints.

``diive.gui`` is out of scope: it needs the optional PySide6 dependency and its
docstrings carry no samples.

Part of the diive library: https://github.com/holukas/diive
"""

import ast
import copy
import doctest
import importlib
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from diive.core.utils.console import VERBOSE_SILENT, get_verbosity, set_verbosity

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

# Samples that continue from another sample, so a method docstring can show
# ``mscr.flag_manualremoval_test(...)`` without repeating a ten-line setup.
# Keys are doctest-name prefixes; values name the sample that does the setup and
# the objects taken from it. The setup sample runs once, and every continuing
# sample gets its own deep copy, so samples cannot see each other's changes.
CONTINUES = {
    'diive.preprocessing.qaqc.meteoscreening.StepwiseMeteoScreeningDb.':
        ('diive.preprocessing.qaqc.meteoscreening.StepwiseMeteoScreeningDb', ['mscr']),
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


def _sample_names():
    """Yield the doctest name of every module, class and method docstring with a sample.

    What :func:`_collect` should find, worked out from the source alone.
    """
    def walk(body, prefix):
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f'{prefix}.{node.name}'
                doc = ast.get_docstring(node)
                if doc and '>>>' in doc:
                    yield name
                if isinstance(node, ast.ClassDef):
                    yield from walk(node.body, name)

    for modname in _modules_with_samples():
        path = PACKAGE_ROOT.joinpath(*modname.split('.')[1:])
        path = path / '__init__.py' if path.is_dir() else path.with_suffix('.py')
        tree = ast.parse(path.read_text(encoding='utf-8'))
        doc = ast.get_docstring(tree)
        if doc and '>>>' in doc:
            yield modname
        yield from walk(tree.body, modname)


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

    def test_every_sample_is_collected(self):
        collected = {t.name for t in self.runnable + self.skipped}
        self.assertEqual(sorted(set(_sample_names()) - collected), [],
                         "docstrings with samples that doctest does not find")

    def test_skip_list_has_no_stale_entries(self):
        found = {t.name for t in self.skipped}
        self.assertEqual(sorted(set(SKIP) - found), [],
                         "SKIP names samples that no longer exist")

    def test_continues_entries_resolve(self):
        names = {t.name for t in self.runnable}
        for prefix, (setup, _) in CONTINUES.items():
            self.assertIn(setup, names, f"CONTINUES setup sample {setup} not found")
            self.assertTrue(any(n.startswith(prefix) for n in names),
                            f"no sample continues from {setup}")

    def _setup_objects(self):
        """Run each CONTINUES setup sample once; return {setup name: {obj name: obj}}."""
        by_name = {t.name: t for t in self.runnable}
        objects = {}
        for setup, names in CONTINUES.values():
            test = copy.copy(by_name[setup])
            test.globs = test.globs.copy()
            # A failing setup sample is reported when it runs in the main loop.
            doctest.DocTestRunner().run(test, out=lambda _: None, clear_globs=False)
            objects[setup] = {n: test.globs[n] for n in names if n in test.globs}
        return objects

    def test_every_runnable_sample_passes(self):
        runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
        failed = []
        # Samples show calls, not console chatter: silence the progress lines
        # (info, detail) that would otherwise count as sample output.
        verbosity = get_verbosity()
        set_verbosity(VERBOSE_SILENT)
        try:
            setup_objects = self._setup_objects()
            for test in self.runnable:
                for prefix, (setup, _) in CONTINUES.items():
                    if test.name.startswith(prefix):
                        test.globs.update(copy.deepcopy(setup_objects[setup]))
                with self.subTest(sample=test.name):
                    result = runner.run(test, clear_globs=False)
                    if result.failed:
                        failed.append(test.name)
        finally:
            set_verbosity(verbosity)
        self.assertEqual(failed, [], "docstring samples do not run:\n  "
                         + "\n  ".join(failed))


if __name__ == '__main__':
    unittest.main()
