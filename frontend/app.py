import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="DevDocs", layout="wide")
st.title("DevDocs")
st.caption("ask questions about your documents")
