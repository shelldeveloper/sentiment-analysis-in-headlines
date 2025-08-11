"""This script builds a Streamlit web interface to allow clients to use the web service."""

import streamlit as st
import requests

st.markdown("# News Headlines Analyzer")
st.write("#### Enter news headlines to analyze their sentiment.")

if "headlines" not in st.session_state:
    st.session_state.headlines = [""]

# Add headline button
if st.button("Add Another Headline"):
    st.session_state.headlines.append("")

# Create text boxes for users to input headlines
for i, headline in enumerate(st.session_state.headlines):
    st.session_state.headlines[i] = st.text_input(f"Headline {i+1}", value=headline, key=f"headline_{i}")

# Run FastAPI model
if st.button("Get Sentiment"):
    # Display the submitted headlines
    st.markdown("#### Submitted Headlines:")
    filtered_headlines = [h.strip() for h in st.session_state.headlines if h.strip()]
    if len(filtered_headlines) == 0:
        st.write("Please enter a news headline.")
        st.stop()
    else:
        for i, headline in enumerate(filtered_headlines):
            st.write(f"{i+1}. {headline}")

    # Run sentiment analysis model on filtered headlines
    st.markdown("#### Sentiment Analysis:")
    if filtered_headlines:
        response = requests.post(url="http://localhost:8021/score_headlines", json={"headlines": filtered_headlines})
        results = response.json()["labels"]
        for h, label in zip(filtered_headlines, results):
                color = "green" if label == "positive" else "red" if label == "negative" else "gray"
                st.markdown(f"**{h}** — <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)