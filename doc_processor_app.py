import streamlit as st
import requests

def main():
    # st.set_page_config(page_title="Document Analyzer", layout="centered")

    st.title("📄 Document Classifier & Extractor")
    st.markdown("Upload a **financial or legal PDF**, and get structured information extracted.")

    # Backend endpoint
    API_URL = "http://localhost:8010/analyze/"  # Change this if you're deploying to a server

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        if st.button("Analyze Document"):
            with st.spinner("Analyzing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(API_URL, files=files)

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Document Type: **{result['document_type']}**")

                        st.subheader("📋 Extracted Information:")
                        extracted = result.get("extracted_information", {})
                        if extracted:
                            for key, value in extracted.items():
                                st.markdown(f"**{key}:** {value}")
                        else:
                            st.info("No structured information extracted.")

                    else:
                        st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")

                except Exception as e:
                    st.error(f"🚨 Failed to contact API: {str(e)}")
