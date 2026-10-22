import streamlit as st

from ui import components, style

style.inject()
components.hero("About ExoSift", [("AI-assisted transit screening", False)], status="Research prototype")
st.write("")

with st.container(border=True):
    st.subheader("What it is")
    st.markdown(
        "ExoSift is a learning-oriented project that screens Kepler light curves for transit-like signals. "
        "A planet crossing its star can cause a small, repeating dip in brightness; ExoSift combines a 1D CNN score "
        "with a Box Least Squares period search so you can inspect both, side by side, on real observations."
    )
    st.warning(
        "ExoSift produces machine-learning candidate scores, not confirmed exoplanet detections. "
        "Nothing here has been verified by independent follow-up.",
        icon=":material/warning:",
    )

left, right = st.columns(2, gap="medium")
with left, st.container(border=True):
    st.subheader("Data and attribution")
    st.markdown(
        "- **Observations:** NASA Kepler mission long-cadence light curves, retrieved through the Mikulski Archive for "
        "Space Telescopes (MAST) with the Lightkurve package.\n"
        "- **Training data:** the Kaggle Kepler exoplanet search dataset (37 planet and 5,050 non-planet training stars).\n"
        "- **Software:** Streamlit, Plotly, TensorFlow/Keras, scikit-learn, astropy and Lightkurve.\n\n"
        "ExoSift is an independent project. It is not affiliated with or endorsed by NASA."
    )
with right, st.container(border=True):
    st.subheader("Limitations that matter")
    st.markdown(
        "- Only 37 planets in training and 5 in the test set: all metrics are noisy.\n"
        "- The CNN's behaviour on inputs shorter than 3,197 points was checked on only 7 validation planets.\n"
        "- The headline score is the maximum over the segments scored, which biases it upward.\n"
        "- BLS peaks rank periods; they are not significance tests, and stellar variability can mimic them.\n"
        "- Very deep transits are removed by the BLS preprocessing.\n"
        "- Window evidence, the calibrated value and the composite score are experimental."
    )

with st.container(border=True):
    st.subheader("Run it locally")
    st.code("source .venv/bin/activate\nstreamlit run app/streamlit_app.py", language="bash")
    st.caption("See the Methodology page for how every number is produced, and docs/scientific_audit.md for the audit.")
