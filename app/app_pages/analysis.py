
import streamlit as st

from ui import runtime, sections, state, style

style.inject()
threshold = sections.sidebar_controls()

request = st.session_state.get("request")
if request is None:
    sections.render_landing()
    st.stop()

result = runtime.run_analysis(request)
if result is None:
    st.stop()

if result.best_segment is None:
    sections.render_insufficient(result, threshold)
    st.stop()

sections.render_header(result, threshold)
st.write("")
sections.render_kpis(result, threshold)
st.write("")

main, side = st.columns([3.1, 1.2], gap="medium")
with main:
    sections.render_lightcurve(result)
    if result.bls is None or not result.bls.candidates or result.top_view is None:
        st.warning(
            f"Period detection did not produce a result: {result.errors.get('periodicity', 'no candidate periods')}",
            icon=":material/warning:",
        )
        view = None
    else:
        left, right = st.columns(2, gap="medium")
        with left:
            sections.render_periodogram(result)
        with right:
            view = sections.render_fold(result)
        sections.render_evidence(result, view)

with side:
    sections.render_interpretation(result, threshold, state.get_view(result, st.session_state.get("fold_mult", "P")) or result.top_view)

if result.bls is not None and result.top_view is not None:
    sections.render_diagnostics(result, view)
sections.render_footer(result, threshold, view)
