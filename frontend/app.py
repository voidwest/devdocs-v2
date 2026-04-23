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
