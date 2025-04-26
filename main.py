# main.py
import streamlit as st
import importlib

st.set_page_config(page_title="FinSight AI", layout="wide")

# --- Initialize session state ---
if "page" not in st.session_state:
    st.session_state.page = "Home"

# --- Custom tab-like sidebar buttons ---
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "Home"
    if st.button("💬 AI Chatbot", use_container_width=True):
        st.session_state.page = "AI Chatbot"
    if st.button("💼 Financial Advisor", use_container_width=True):
        st.session_state.page = "Financial Advisor"
    if st.button("📄 PDF Analyzer", use_container_width=True):
        st.session_state.page = "PDF Analyzer"

# --- Module Mapping ---
page_modules = {
    "Home": "home",
    "AI Chatbot": "rag_app",
    "Financial Advisor": "financial_advisor",
    "PDF Analyzer": "doc_processor_app"
}

# --- Load Selected Page ---
module = importlib.import_module(page_modules[st.session_state.page])
module.main()
