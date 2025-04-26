# main.py
import os
import textwrap
import logging
import sys
from pathlib import Path
from typing import List, Optional, Union # Added Union

# --- FastAPI and Pydantic Imports ---
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
# --- CORS Middleware Import ---
from fastapi.middleware.cors import CORSMiddleware # <<<<<<<< ADD THIS IMPORT

# --- LangChain & Google Imports ---
import google.generativeai as genai
# Note: FAISS might require specific C++ libs installed on your system
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    logging.error("FAISS not found. Ensure 'faiss-cpu' or 'faiss-gpu' is installed (`pip install faiss-cpu`).")
    sys.exit("FAISS dependency missing.")
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
try:
    # Assuming config.py defines a Config class
    from config import Config
except ImportError:
    logging.error("Error: config.py not found or Config class not defined.")
    sys.exit("config.py is required.")

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Core Service Classes (KnowledgeBaseLoader, FinancialAdvisor) ---
# (Keep the KnowledgeBaseLoader and FinancialAdvisor classes exactly as they were in the previous version)
class KnowledgeBaseLoader:
    """Loads the pre-built FAISS knowledge base"""
    def __init__(self, config: Config):
        self.config = config
        self.vector_db: Optional[FAISS] = None
        logger.info(f"Initializing embeddings with model: {config.embedding_model_name}")
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=config.embedding_model_name,
                google_api_key=config.api_key
            )
            logger.info("Embeddings initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize GoogleGenerativeAIEmbeddings: {e}", exc_info=True)
            raise RuntimeError("Could not initialize embeddings. Check API key and model name.") from e

    def load(self) -> FAISS:
        """Load existing knowledge base or raise error"""
        kb_path = Path(self.config.knowledge_base_dir)
        # FAISS typically saves two files: index.faiss and index.pkl
        faiss_index_path = kb_path / "index.faiss"
        faiss_pkl_path = kb_path / "index.pkl"

        logger.info(f"Attempting to load knowledge base from: {kb_path.resolve()}")
        if not kb_path.exists() or not faiss_index_path.exists() or not faiss_pkl_path.exists():
            logger.error(f"Knowledge base files not found at {kb_path.resolve()}. Expected 'index.faiss' and 'index.pkl'.")
            raise FileNotFoundError(
                f"Knowledge base not found at {kb_path.resolve()}. "
                f"Please run the ingestion script first to build it."
            )
        try:
            self.vector_db = FAISS.load_local(
                folder_path=str(kb_path),
                embeddings=self.embeddings,
                index_name="index",
                allow_dangerous_deserialization=True # Required by FAISS for default pickle loading
            )
            logger.info("Knowledge base loaded successfully.")
            return self.vector_db
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}", exc_info=True)
            raise RuntimeError(f"Could not load knowledge base from {kb_path.resolve()}") from e

class FinancialAdvisor:
    """Main RAG system for financial advice using pre-loaded KB"""
    def __init__(self, config: Config, vector_db: FAISS):
        self.config = config
        self.vector_db = vector_db
        logger.info(f"Setting up retriever with k={config.max_retrieval_docs}")
        self.retriever = self.vector_db.as_retriever(
            search_kwargs={'k': config.max_retrieval_docs}
        )
        logger.info(f"Initializing LLM: {config.llm_model_name}")
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=config.llm_model_name,
                google_api_key=config.api_key,
                temperature=0.7,
                convert_system_message_to_human=True
            )
            logger.info("LLM initialized.")
        except Exception as e:
             logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}", exc_info=True)
             raise RuntimeError("Could not initialize LLM. Check API key and model name.") from e

        logger.info("Setting up prompt template and RAG chain.")
        self.response_prompt_template = ChatPromptTemplate.from_template(
            """You are a helpful and professional financial advisor assistant representing YMCA.
            Your goal is to answer the user's question accurately based *only* on the provided context documents.
            If the context doesn't contain the answer, state that clearly. Do not make up information.

            Follow these instructions precisely:
            1.  Carefully review the following context extracted from YMCA documents.
            2.  Answer the user's QUESTION based *solely* on this CONTEXT.
            3.  Structure your response clearly. If applicable, mention key information, actionable advice (if present in context), potential risks (if mentioned), and cite the source document(s) for each piece of information using the format [Source: filename.ext].
            4.  If the CONTEXT does not provide enough information to answer the QUESTION, respond with: "Based on the available YMCA documents, I cannot provide a specific answer to your question."
            5.  Do not add any external knowledge or information not present in the context.
            6.  Always include the standard disclosures at the end of your response.

            CONTEXT:
            {context}

            QUESTION: {question}

            Helpful Answer:"""
        )
        self.rag_chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.response_prompt_template
            | self.llm
            | StrOutputParser()
        )
        logger.info("RAG chain configured.")

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """Formats retrieved documents for the prompt context."""
        if not docs:
            return "No relevant documents found."
        formatted_docs = []
        for doc in docs:
             source_name = Path(doc.metadata.get('source', 'Unknown')).name
             content = doc.page_content.replace('\n', ' ').strip()
             formatted_docs.append(f"Source: {source_name}\nContent:\n{content}")
        return "\n\n".join(formatted_docs)

    def _add_disclosures(self, response: str) -> str:
        """Ensure required disclosures are present"""
        disclosures = [
            "DISCLAIMER: This is general information based on YMCA documents and not personalized financial advice.",
            "Investing involves risks, including the potential loss of principal.",
            "Past performance is not indicative of future results.",
            "Please consult with a qualified financial professional before making any financial decisions."
        ]
        clean_response = response.strip() if response else ""
        if clean_response:
             clean_response += '\n\n'
        clean_response += "---\n"
        clean_response += "\n".join(disclosures)
        return clean_response

    def generate_response(self, query: str) -> str:
        """Generate response using the RAG chain. Handles internal errors."""
        if not query:
            return self._add_disclosures("Please provide a question.")
        try:
            logger.info(f"Processing query: '{query[:50]}...'")
            llm_response = self.rag_chain.invoke(query)
            logger.info("Successfully invoked RAG chain.")
            final_response = self._add_disclosures(llm_response)
            return final_response
        except Exception as e:
            logger.error(f"An error occurred during RAG chain invocation for query '{query[:50]}...': {e}", exc_info=True)
            return self._add_disclosures("I encountered an technical issue while processing your request. Please try again later or contact support if the problem persists.")

# --- FastAPI Application Setup ---
app = FastAPI(
    title="YMCA Financial Advisor Assistant API",
    description="API providing RAG-based answers to financial questions based on YMCA documents.",
    version="1.0.0"
)

# --- CORS Middleware Configuration --- START ---
# Define the list of origins that are allowed to make requests.
# '*' allows all origins, which is convenient for development but potentially insecure for production.
# For production, replace '*' with the specific origin(s) of your frontend application.
origins = [
    "http://localhost",         # Common origin for local development
    "http://localhost:3000",    # Common React dev port
    "http://localhost:8080",    # Common Vue/other dev ports
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    # Example for production: "https://your-frontend-app.com",
]

# Add the CORS middleware to the application instance
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # List of allowed origins
    allow_credentials=True,     # Allow cookies to be included in requests
    allow_methods=["*"],        # Allow all standard HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],        # Allow all headers
)
# --- CORS Middleware Configuration --- END ---


# --- Global Variables for Singleton Services ---
cfg: Optional[Config] = None
kb_loader: Optional[KnowledgeBaseLoader] = None
advisor: Optional[FinancialAdvisor] = None

# --- Application Startup Logic ---
@app.on_event("startup")
async def startup_event():
    """Initialize services when the FastAPI application starts."""
    global cfg, kb_loader, advisor
    logger.info("--- FastAPI Application Starting Up ---")
    try:
        cfg = Config()
        logger.info("Configuration loaded.")
        kb_loader = KnowledgeBaseLoader(cfg)
        vector_db = kb_loader.load()
        advisor = FinancialAdvisor(cfg, vector_db)
        logger.info("--- FinancialAdvisor Service Initialized Successfully ---")
    except (FileNotFoundError, RuntimeError, Exception) as e:
        logger.critical(f"!!! CRITICAL ERROR DURING STARTUP: {e}", exc_info=True)
        # advisor remains None, health check will fail.

# --- API Request/Response Models ---
class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The financial question to ask the advisor.",
        examples=["What are the risks of this investment option?", "Explain the retirement plan details."]
    )

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The generated answer from the advisor, including disclaimers.")

# --- API Endpoints ---

@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health Check"])
async def health_check():
    """Checks if the core FinancialAdvisor service is initialized and ready."""
    if advisor is not None:
        logger.debug("Health check status: OK")
        return {"status": "ok", "message": "Financial Advisor service is running."}
    else:
        logger.warning("Health check status: Service Unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Financial Advisor service is not initialized. Check startup logs.",
        )

@app.post(
    "/ask",
    response_model=QueryResponse,
    summary="Ask the Financial Advisor a Question",
    description="Submits a question to the RAG pipeline. The advisor will use YMCA documents to generate an answer.",
    tags=["Advisor"]
)
async def ask_question(request: QueryRequest):
    """Endpoint to handle user questions."""
    logger.info(f"Received request for /ask: '{request.question[:50]}...'")
    if advisor is None:
        logger.error("/ask request failed: Advisor service not initialized.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Financial Advisor service is not available. Please check server status or logs.",
        )
    try:
        response_text = advisor.generate_response(request.question)
        logger.info("Generated response successfully.")
        return QueryResponse(answer=response_text)
    except Exception as e:
        logger.error(f"Unexpected error in /ask endpoint for query '{request.question[:50]}...': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while processing your question.",
        )

# --- Uvicorn Runner (for direct execution) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server directly from main.py...")
    # Remember to adjust host/port/reload as needed. Port 8005 used based on previous code.
    uvicorn.run(app, host="0.0.0.0", port=8005) # Changed port to 8005