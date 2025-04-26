# home.py
import streamlit as st

def main():
    # --- Header ---
    st.title("🏠 Welcome to FinSight AI")
    st.markdown("""
    FinSight AI is a suite of intelligent financial tools designed to simplify investment decisions, analyze documents, and provide tailored financial guidance using the latest in AI technology.
    """)

    # --- Highlights ---
    st.markdown("### 🚀 What You Can Do")
    st.markdown("""
    - 🤖 **AI Chatbot**: Ask finance-related questions sourced from YMCA's internal documents.
    - 💼 **Financial Advisor**: Receive personalized investment advice based on your goals and preferences.
    - 📄 **PDF Analyzer**: Upload and analyze financial PDFs for insights and summaries.
    """)

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        "🔒 All tools are private, secure, and powered by advanced retrieval-augmented generation (RAG) and Google Generative AI.")
