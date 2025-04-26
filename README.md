# 🧠 FinSightAI - Finance Advisor

Your AI-powered assistant for smarter financial planning, document analysis, and portfolio recommendations.

----------

## 🚀 Overview (TL;DR)

A web app that:
-   Answers general financial questions using a RAG-powered AI chatbot 💬
-   Analyzes uploaded documents like invoices/contracts and flags financial risks 📄
-   Provides personalized financial strategy based on user profile 💸
    

----------

## 🎯 Problem Statement

Financial literacy is low, and most people struggle with:

-   Understanding complex documents like balance sheets and contracts
    
-   Knowing how to start investing based on their personal income/expenses
    
-   Getting trustworthy, AI-driven financial guidance
    

----------

## 💡 Our Solution

Personal Finance Advisor is a 3-in-1 tool that:

1.  Acts as a financial chatbot using RAG (Retrieval-Augmented Generation)
    
2.  Classifies and analyzes uploaded documents with a risk flag system
    
3.  Offers investment strategies tailored to your income, goals, and risk appetite
    

----------

## 🔍 Features

### 📚 Module 1: RAG Chat for Financial Queries

-   GPT-4 based AI chatbot
    
-   Trained on curated financial knowledge base
    
-   Explains terms like SIP, tax-saving options, loans, etc.
    

### 📄 Module 2: Financial Document Inspector

-   Upload PDFs, DOCX, or images
    
-   Auto-detects type (invoice, contract, balance sheet)
    
-   Extracts key terms: amount, date, interest rate, penalties
    
-   Flags risks (hidden fees, lock-ins)
    
-   Generates a structured summary + risk report
    

### 📊 Module 3: Personal Strategy Generator

-   Takes user input (age, income, expenses, loans, risk level)
    
-   Suggests diversified portfolios
    
-   Gives actionable financial planning advice
    

----------

## 🗂️ Project Directory Structure

```bash
project/
│
├── main.py                    # Unified Streamlit app integrating all 3 modules
├── advisor_app.py             # Standalone Streamlit UI for financial advisor
├── rag_chatbot_app.py         # Standalone Streamlit UI for RAG chatbot
├── pdf_ai_app.py              # Standalone Streamlit UI for document analysis
│
├── financial_advisor_api.py   # Backend logic for financial strategy module
├── rag_chatbot_api.py         # Backend logic for RAG chatbot
├── pdf_ai_api.py              # Backend logic for document analysis
│
├── advisor_model.pt           # Saved PyTorch model for strategy module
├── rag_model.pt               # Saved model for RAG chatbot
├── pdf_model.pt               # Saved model for PDF understanding
│
├── requirements.txt           # Python dependencies

```

----------

## 🛠️ Tech Stack

-   Backend: FastAPI (Python)
    
-   Frontend: Streamlit
    
-   AI: OpenAI GPT-4 + LangChain

-  ML: Transformers + scikit-learn + PyTorch
    
-   Document Embedding: FAISS
    
-   PDF Parsing: PyMuPDF
    
-   OCR (optional): Tesseract
    

----------

## 📸 Sample Workflow

1.  User launches main.py and sees integrated tabs: Ask Chat, Upload Document, Get Advice
    
2.  Uploads a contract → system flags lock-in clauses + suggests negotiation tips
    
3.  Asks chat: "How to save tax under 80C?" → AI responds with examples
    
4.  Inputs age/income/expenses → gets personalized portfolio suggestion
    

----------

## 🧪 How to Run Locally

```bash
git clone https://github.com/shivankursharma018/FinSightAI.git
cd FinSightAI

# Backend
python -m venv venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
uvicorn app:app --reload

# Integrated Streamlit App
streamlit run main.py
```

----------

## 🧠 Sample Inputs & Outputs

-   Upload: "Invoice_ABC.pdf" → Output: Document type = invoice, Total = ₹22,500, Risk = Late fee
    
-   Chat: "What is a SIP?" → Output: "A Systematic Investment Plan (SIP) allows investors to invest a fixed amount..."
    
-   Advice Tool: Age 27, ₹50k income, ₹20k expenses, medium risk → Output: "30% large cap funds, 20% FDs, 10% gold, 40% emergency + debt"
    

----------

## 👨‍💻 Team
- [Lavya Parasher](https://github.com/Lavyadev) - leader
- [Mokksh Kapur](https://github.com/Mokkshking)
- [Shivankur Sharma](https://github.com/shivankursharma018)
- [Zubair](https://github.com/zubair-iqubal)

----------

## 📜 License

MIT License

----------

## 🔮 Future Scope

-   Add OCR for image-based invoices
    
-   Enable live market integration for portfolio suggestions
    
-   Export advice and reports as downloadable PDFs
        

----------

## 🙌 Acknowledgments

-   OpenAI, LangChain, Streamlit
    
-   HuggingFace (FinBERT)
    
-   Hackathon organizers + mentors

