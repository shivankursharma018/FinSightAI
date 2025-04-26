# app.py
import streamlit as st
import requests

def main():
    # --- Configuration ---
    API_URL = "http://localhost:8005/ask"  # Adjust if deployed remotely

    # st.set_page_config(page_title="YMCA Financial Advisor Assistant", layout="centered")

    # --- Header ---
    st.title("📊 Financial Advisor Assistant")
    st.markdown("""
    Welcome to the **FinSightAI's financial advisor**.  
    Ask questions based on internal financial documents.  
    The assistant responds using *only* trusted legal materials.
    """)

    st.markdown("---")

    # --- Question Input ---
    st.markdown("### 💬 Ask a Financial Question")
    question = st.text_input(
        "What would you like to know?",
        placeholder="e.g. What are the risks of this investment option?"
    )

    # --- Submit Button ---
    if st.button("🔍 Get Answer"):
        if len(question.strip()) < 3:
            st.warning("⚠️ Please enter a question with at least 3 characters.")
        else:
            with st.spinner("Consulting FinSightAI's knowledge base..."):
                try:
                    response = requests.post(API_URL, json={"question": question.strip()})
                    if response.status_code == 200:
                        answer = response.json().get("answer", "No response received.")
                        st.success("✅ Advisor Response")
                        st.markdown("```markdown\n" + answer + "\n```")
                    else:
                        st.error(f"❌ Error {response.status_code}: {response.json().get('detail')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"⚠️ Could not connect to the advisor service.\n\n{e}")

    # --- Footer ---
    st.markdown("---")
    st.markdown("💡 Powered by a RAG system & Google Generative AI.")

