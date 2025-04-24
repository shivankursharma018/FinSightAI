# import streamlit as st

# # Set up page
# st.set_page_config(page_title="FinSightAI", page_icon="💰", layout="wide")

# # Sidebar Navigation
# st.sidebar.title("🧭 Navigation")
# page = st.sidebar.radio("Go to", ["AI Chatbot", "Financial Advisor", "PDF Analyzer"])

# # --- AI Chatbot Page ---
# if page == "AI Chatbot":
#     st.title("💬 AI Chatbot")
#     st.markdown("Ask anything related to finance or get help with concepts.")
    
#     user_input = st.text_input("Type your question here")
#     if st.button("Ask"):
#         # Replace this with your actual chatbot logic
#         st.success("🤖 *Here's a dummy response until model is hooked in...*")

# # --- Financial Advisor Page ---
# elif page == "Financial Advisor":
#     st.title("📊 Financial Advisor")
#     st.markdown("Provide your investment profile for personalized suggestions.")
    
#     gender = st.selectbox("Gender", ["Male", "Female", "Other"])
#     age = st.number_input("Age", min_value=18, max_value=100, value=30)
#     objective = st.text_input("What is your investment objective?")
#     duration = st.selectbox("Investment Duration", ["<1 year", "1-3 years", "3-5 years", "5+ years"])
#     expect = st.text_input("Expected returns (in %)?")
    
#     if st.button("Get Advice"):
#         # Plug in your financial advisor model logic here
#         st.success("📈 *This is where your advice will be shown.*")

# # --- PDF Analyzer Page ---
# elif page == "PDF Analyzer":
#     import pdfplumber
    
#     st.title("📄 PDF Financial Analyzer")
#     st.markdown("Upload financial PDFs to extract and analyze content.")
    
#     uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
#     if uploaded_file:
#         with pdfplumber.open(uploaded_file) as pdf:
#             text = ""
#             for page in pdf.pages:
#                 text += page.extract_text() + "\n"
#         st.subheader("Extracted Text")
#         st.text_area("Text from PDF", text, height=300)

#         if "investment" in text.lower():
#             st.info("💡 Document talks about investments. Consider checking your asset allocation.")
#         else:
#             st.warning("🤷‍♂️ No strong financial topics detected.")


# import streamlit as st
# import importlib

# # Set page config
# st.set_page_config(page_title="FinSightAI", page_icon="💸", layout="wide")

# # Sidebar for navigation
# st.sidebar.title("🧭 Navigation")
# page = st.sidebar.radio("Choose a module", ["AI Chatbot", "Financial Advisor", "PDF Analyzer"])

# # Map page to module
# page_modules = {
#     "AI Chatbot": "chatbot_app",
#     "Financial Advisor": "financial_advisor",
#     "PDF Analyzer": "pdf_analyser"
# }

# # Dynamic import and execution
# module_name = page_modules[page]
# module = importlib.import_module(module_name)
# module.app()


import streamlit as st
import importlib

st.set_page_config(page_title="FinSight AI", layout="wide")

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["AI Chatbot", "Financial Advisor", "PDF Analyzer"])

page_modules = {
    "AI Chatbot": "chatbot_app",
    "Financial Advisor": "financial_advisor",
    "PDF Analyzer": "pdf_analyser"
}

module = importlib.import_module(page_modules[page])
module.main()
