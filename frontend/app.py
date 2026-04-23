import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="DevDocs", layout="wide")
st.title("DevDocs")
st.caption("ask questions about your documents")


with st.sidebar:
    try:
        health = requests.get(f"{API_URL}/docs", timeout=2)
        st.success("api connected")
    except:
        st.error("api unreachable")

    st.header("settings")
    show_sources = st.checkbox("show source ref", value=True)


query = st.text_input(
    "ask a question", placeholder="probably something about your documents"
)

if st.button("ask", type="primary") and query:
    with st.spinner("getting context and answer"):
        try:
            resp = requests.post(f"{API_URL}/ask", json={"prompt": query}, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()
        except requests.Timeout:
            st.error("request timed out")
            st.stop()
        except requests.RequestException as e:
            st.error(f"api error: {e}")
            st.stop()
