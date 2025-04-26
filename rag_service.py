import logging
import os
import google.generativeai as genai
# Deprecation Warning comment: Consider migrating to langchain-chroma
from langchain_community.vectorstores import Chroma
# Deprecation Warning comment: Consider migrating to langchain-huggingface
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document # Import Document
import config

# Setup logger for this module
logger = logging.getLogger(__name__)

class RAGService:
    """Encapsulates the RAG logic using LangChain, Gemini, and ChromaDB."""
    def __init__(self):
        logger.info("--- RAGService.__init__ started ---")
        try:
            # Step 1: Configure Gemini API Key (Low-level, mainly for reference)
            logger.info("Step 1: Configuring Gemini low-level access...")
            # genai.configure(api_key=config.GOOGLE_API_KEY) # Keep if using genai directly
            logger.info("Step 1: Gemini configured (Note: Key passed directly to ChatGoogleGenerativeAI).")

            # Step 2: Initialize Embedding Model (Matches indexing script)
            logger.info(f"Step 2: Initializing embedding model: {config.EMBEDDING_MODEL} on device 'cpu'")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Step 2: Embedding model initialized.")

            # Step 3: Load Persistent Vector Store
            vector_db_path = config.VECTOR_DB_PATH
            logger.info(f"Step 3: Loading vector store from: {vector_db_path}")
            if not os.path.exists(vector_db_path):
                 logger.error(f"Vector store path does not exist: {vector_db_path}. Run indexing_script.py first.")
                 raise FileNotFoundError(f"Vector store path not found: {vector_db_path}")

            self.vectorstore = Chroma(
                persist_directory=vector_db_path,
                embedding_function=self.embeddings
            )
            # Create the retriever component
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity", # Standard similarity search
                search_kwargs={"k": config.RETRIEVER_K} # Retrieve top K chunks
            )
            logger.info(f"Step 3: Vector store loaded successfully. Retriever configured for k={config.RETRIEVER_K}.")

            # Step 4: Initialize Gemini LLM via LangChain
            logger.info(f"Step 4: Initializing Gemini LLM: {config.GEMINI_MODEL}")
            self.llm = ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL,
                google_api_key=config.GOOGLE_API_KEY, # Crucial Fix: Pass API key here
                temperature=0.1, # Lower temperature for more factual responses
                convert_system_message_to_human=True, # Format prompts for Gemini
            )
            logger.info("Step 4: Gemini LLM initialized.")

            # Step 5: Define the Prompt Template for RAG
            logger.info("Step 5: Defining RAG prompt template...")
            # Prompt emphasizes grounding and handling missing information
            rag_prompt_template = """
            SYSTEM: You are an AI assistant providing information based ONLY on the following retrieved context from internal documents.
            Your task is to answer the user's QUESTION using the CONTEXT below.
            - Be concise and directly answer the question.
            - If the context contains the answer, synthesize it clearly.
            - If the context does NOT contain information relevant to the question, you MUST explicitly state: "Based on the provided documents, I cannot answer this question."
            - Do NOT use any prior knowledge or information outside the provided context.
            - Do NOT make assumptions or speculate.

            CONTEXT:
            {context}

            QUESTION:
            {question}

            ASSISTANT ANSWER:
            """
            self.prompt = PromptTemplate(
                template=rag_prompt_template, input_variables=["context", "question"]
            )
            logger.info("Step 5: RAG prompt template defined.")

            # Step 6: Define the RAG Chain using LangChain Expression Language (LCEL)
            logger.info("Step 6: Defining RAG chain...")
            self.rag_chain = (
                {
                    "context": self.retriever | self._format_docs, # Retrieve docs, format them
                    "question": RunnablePassthrough() # Pass the original question through
                }
                | self.prompt # Construct the full prompt
                | self.llm # Call the LLM
                | StrOutputParser() # Parse the LLM output message to a string
            )
            logger.info("Step 6: RAG Chain configured successfully.")

            logger.info("--- RAGService.__init__ finished successfully ---")

        # Catch *any* exception during init for clear logging
        except BaseException as e:
             logger.error(f"!!! Error occurred within RAGService.__init__.", exc_info=True)
             # Re-raise the exception so the FastAPI app knows initialization failed
             raise

    def _format_docs(self, docs: list[Document]) -> str:
        """Helper function to format retrieved documents into a single string for the prompt."""
        if not docs:
            return "No relevant context found in the documents."
        # Join document contents, separated clearly. Include source if available.
        formatted_docs = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown Source')
            content = doc.page_content.replace('\n', ' ').strip() # Basic cleaning
            # Limit content length per doc slightly if needed
            # content = content[:800] + '...' if len(content) > 800 else content
            formatted_docs.append(f"--- Context Document {i+1} (Source: {source}) ---\n{content}")
        return "\n\n".join(formatted_docs)


    async def process_query(self, query: str) -> str:
        """
        Processes a user query asynchronously using the configured RAG chain.
        """
        logger.info(f"Processing query in RAGService: '{query}'")
        if not self.rag_chain:
            logger.error("RAG chain is not available.")
            return "Error: RAG system not initialized properly."
        try:
            # Asynchronously invoke the chain
            # Add timeout handling? (More advanced)
            response = await self.rag_chain.ainvoke(query)
            logger.info(f"Received response from RAG chain for query: '{query}'")
            return response
        except Exception as e:
            logger.error(f"Error during RAG chain invocation for query '{query}': {e}", exc_info=True)
            # Return a user-friendly error message
            return "Sorry, I encountered an error while processing your request. Please try again later."