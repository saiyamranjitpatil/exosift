"""Streamlit-side runtime helpers: shared model/calibrator, analysis cache, progress."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

import streamlit as st

from calibration import CNNCalibrator
from exosift_pipeline import AnalysisResult, NoDataError, RetrievalError, analyze_target
from predict_real_star import VARIABLE_MODEL_PATH

log = logging.getLogger("exosift.app")
MAX_CACHED_ANALYSES = 6


@st.cache_resource(show_spinner=False)
def get_cnn_model():
    """Variable-length CNN, loaded once per server process."""
    import tensorflow as tf
    return tf.keras.models.load_model(VARIABLE_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_calibrator() -> Optional[CNNCalibrator]:
    """Platt calibrator, or None when the file is missing (handled in the UI)."""
    try:
        return CNNCalibrator(method="platt")
    except FileNotFoundError:
        return None


def _model_stamp() -> int:
    try:
        return VARIABLE_MODEL_PATH.stat().st_mtime_ns
    except OSError:
        return 0


def cache_key(request: tuple) -> tuple:
    """Everything that changes the computed result: target, quarter, scan limit, model file."""
    kic, quarter, max_products = request
    return (int(kic), quarter, int(max_products), _model_stamp())


def _store() -> OrderedDict:
    return st.session_state.setdefault("_analysis_cache", OrderedDict())


def cached_result(request: tuple) -> Optional[AnalysisResult]:
    return _store().get(cache_key(request))


def clear_cached(request: tuple) -> None:
    _store().pop(cache_key(request), None)


def run_analysis(request: tuple) -> Optional[AnalysisResult]:
    """Return the analysis for `request`, computing it (with visible progress) if needed.

    On failure shows a specific error with a next step and returns None.
    """
    hit = cached_result(request)
    if hit is not None:
        return hit

    kic, quarter, max_products = request
    store = _store()
    failure: Optional[tuple[str, str, str]] = None      # (icon, message, next step), shown outside the status box
    result: Optional[AnalysisResult] = None
    with st.status(f"Analyzing KIC {kic}", expanded=True) as status:
        try:
            result = analyze_target(
                kic, quarter, max_products, get_cnn_model(), get_calibrator(),
                progress=lambda msg: status.update(label=msg),
            )
        except NoDataError:
            status.update(label="No data found", state="error", expanded=False)
            failure = (
                ":material/search_off:",
                f"NASA/MAST has no Kepler long-cadence data for KIC {kic}"
                + (f" in quarter Q{quarter:02d}." if quarter is not None else "."),
                "Check the KIC number, or set Quarter to Auto to search every quarter.",
            )
        except RetrievalError as exc:
            status.update(label="Could not reach NASA/MAST", state="error", expanded=False)
            failure = (
                ":material/cloud_off:", f"The archive search failed: {exc}",
                "This is usually a network or service problem. Check your connection and try again.",
            )
        except Exception as exc:  # noqa: BLE001 - last-resort UI boundary
            log.exception("analysis failed")
            status.update(label="Analysis failed", state="error", expanded=False)
            failure = (":material/error:", f"Unexpected error: {type(exc).__name__}: {exc}", "Details were written to the terminal log.")
        else:
            status.update(label=f"Analysis of KIC {kic} complete", state="complete", expanded=False)

    if failure is not None:
        icon, message, next_step = failure
        st.error(message, icon=icon)
        st.caption(next_step)
        return None

    store[cache_key(request)] = result
    while len(store) > MAX_CACHED_ANALYSES:
        store.popitem(last=False)
    return result
