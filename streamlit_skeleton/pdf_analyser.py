import streamlit as st
import pdfplumber

def main():    
    # st.set_page_config(page_title="PDF Analyzer", page_icon="📄")

    st.title("📄 Financial Document Analyzer")
    st.markdown("Upload your financial document to get insights!")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # open the PDF and extract text
        with pdfplumber.open(uploaded_file) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text()
            
        st.subheader("Extracted Text (Preview):")
        st.text_area("Extracted Content", value=full_text[:1000], height=200)

        # todo: replace with ML model for real suggestions
        if "expense" in full_text.lower():
            advice = "We noticed recurring expenses in your document. You might want to focus on budgeting strategies."
        else:
            advice = "The document seems to contain useful financial data. Consider reviewing the expenses and investments section for better strategies."

        st.subheader("💡 Financial Advice Based on Document:")
        st.write(advice)


