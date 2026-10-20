"""
ML.SHAP_IMPORT: IMPORT SHAP WITHOUT PINNING THE CALLER
======================================================

Imports shap and releases what its import keeps alive by accident.

shap records each optional package it could not import (e.g. ``cv2``) together
with the exception, to raise it later if that feature is used. The exception
keeps its traceback, and the traceback keeps every frame that was running when
shap was first imported, with all their local variables. In the GUI that
pinned the first tab opening the ML stack (L313): the tab and its data were
never freed, and its Project settings subscription kept writing into deleted
widgets.

Part of the diive library: https://github.com/holukas/diive
"""
from __future__ import annotations


def import_shap():
    """Import and return shap, dropping the tracebacks of its stored import errors.

    The stored errors keep their type and message, so shap still raises them
    when a feature needs the missing package.
    """
    import shap
    general = getattr(shap.utils, "_general", None)
    for _msg, err in getattr(general, "import_errors", {}).values():
        while err is not None:
            err.__traceback__ = None
            err = err.__context__
    return shap
