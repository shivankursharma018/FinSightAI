import os
import textwrap
from pathlib import Path
from typing import List, Optional, Dict
from dotenv import load_dotenv

import google.generativeai as genai
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate


# Configuration Manager
class Config:
    def __init__(self):
        load_dotenv()
        self.api_key = self._get_api_key()
        self.data_dir = "YMCA_data"
        self.knowledge_base_dir = "financial_knowledge_base"
        self.max_retrieval_docs = 5
        self.user_agent = "YMCA-Financial-Advisor/1.0"

        # Validate paths
        if not Path(self.data_dir).exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        Path(self.knowledge_base_dir).mkdir(exist_ok=True)

    def _get_api_key(self) -> str:
        """Retrieve and validate API key"""
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("API key not found. Set GOOGLE_API_KEY environment variable")
        return key


# Document Processor
class DocumentProcessor:
    """Handles all document loading and processing"""

    @staticmethod
    def get_loader(file_path: str):
        """Return appropriate loader based on file extension"""
        ext = Path(file_path).suffix.lower()
        return {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.csv': CSVLoader,
            '.md': UnstructuredMarkdownLoader,
            '.xlsx': UnstructuredExcelLoader,
        }.get(ext)

    def load_documents(self, data_dir: str) -> List[Document]:
        """Load all supported documents from directory"""
        documents = []
        supported_extensions = {'.pdf', '.docx', '.txt', '.csv', '.md', '.xlsx'}

        for root, _, files in os.walk(data_dir):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in supported_extensions:
                    try:
                        loader_class = self.get_loader(str(file_path))
                        if loader_class:
                            loader = loader_class(str(file_path))
                            docs = loader.load()
                            for doc in docs:
                                doc.metadata["source"] = str(file_path)
                            documents.extend(docs)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

        if not documents:
            raise ValueError(f"No supported documents found in {data_dir}")

        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into manageable chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)


# Knowledge Base Manager
class KnowledgeBase:
    """Manages the vector store and document retrieval"""

    def __init__(self, config: Config):
        self.config = config
        self.vector_db = None
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=config.api_key
        )

    def initialize(self, documents: List[Document], rebuild: bool = False):
        """Initialize or rebuild the knowledge base"""
        if not rebuild and self._load_existing_knowledge_base():
            return

        processor = DocumentProcessor()
        chunked_docs = processor.chunk_documents(documents)

        self.vector_db = FAISS.from_documents(
            documents=chunked_docs,
            embedding=self.embeddings
        )
        self.vector_db.save_local(self.config.knowledge_base_dir)
        print(f"Knowledge base built with {len(chunked_docs)} chunks")

    def _load_existing_knowledge_base(self) -> bool:
        """Attempt to load existing knowledge base"""
        try:
            self.vector_db = FAISS.load_local(
                self.config.knowledge_base_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return True
        except Exception as e:
            print(f"Could not load existing knowledge base: {e}")
            return False

    def retrieve_relevant_docs(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query"""
        if not self.vector_db:
            raise ValueError("Knowledge base not initialized")

        return self.vector_db.similarity_search(
            query,
            k=self.config.max_retrieval_docs
        )


# Financial Advisor Core
class FinancialAdvisor:
    """Main RAG system for financial advice"""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=config.api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        self.knowledge_base = KnowledgeBase(config)

        # Load documents and initialize knowledge base
        processor = DocumentProcessor()
        documents = processor.load_documents(config.data_dir)
        self.knowledge_base.initialize(documents)

        # Setup prompt templates
        self.response_prompt = ChatPromptTemplate.from_template(
            """You are a professional financial advisor for YMCA. Use the following context to answer the question.

            Context:
            {context}

            Question: {question}

            Provide a detailed response with:
            1. Key information from our documents
            2. Actionable advice
            3. Potential risks
            4. Required disclosures

            Always cite sources using the format: [Source: filename]"""
        )

    def generate_response(self, query: str) -> str:
        """Generate response to financial query"""
        try:
            # Retrieve relevant documents
            docs = self.knowledge_base.retrieve_relevant_docs(query)
            if not docs:
                return "I couldn't find relevant information in our documents."

            # Prepare context with sources
            context = "\n\n".join(
                f"Document {i + 1} [Source: {Path(doc.metadata['source']).name}]:\n{doc.page_content}"
                for i, doc in enumerate(docs)
            )

            # Generate response
            chain = self.response_prompt | self.llm
            response = chain.invoke({"context": context, "question": query})

            # Add standard disclosures if missing
            return self._add_disclosures(response.content)

        except Exception as e:
            return f"An error occurred: {str(e)}"

    def _add_disclosures(self, response: str) -> str:
        """Ensure required disclosures are present"""
        disclosures = [
            "DISCLAIMER: This is general information, not personalized advice.",
            "Investing involves risks including potential loss of principal.",
            "Past performance does not guarantee future results.",
            "Consult a qualified financial professional before making decisions."
        ]

        for disclosure in disclosures:
            if disclosure not in response:
                response += f"\n\n{disclosure}"
        return response


# User Interface
class FinancialAdvisorApp:
    """Command-line interface for the financial advisor"""

    @staticmethod
    def display_response(question: str, response: str):
        """Format and display the response"""
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("-" * 80)
        print(textwrap.fill(response, width=80))
        print("=" * 80 + "\n")

    def run(self):
        """Main application loop"""
        try:
            config = Config()
            advisor = FinancialAdvisor(config)

            print("YMCA Financial Advisor RAG System")
            print("Type 'exit' to quit\n")

            while True:
                try:
                    question = input("How can I help with your financial questions? ").strip()
                    if question.lower() in ['exit', 'quit']:
                        break
                    if not question:
                        continue

                    response = advisor.generate_response(question)
                    self.display_response(question, response)

                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    print(f"Error: {e}")

        except Exception as e:
            print(f"Failed to start advisor: {e}")


if __name__ == "__main__":
    FinancialAdvisorApp().run()