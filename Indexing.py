# ingest.py
import os
from pathlib import Path
from typing import List
import sys

# Add project root to sys.path if necessary, depending on your structure
# sys.path.append(str(Path(__file__).parent))

from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader,
    UnstructuredMarkdownLoader, UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Assuming config.py is in the same directory or Python path
from config import Config

class DocumentProcessor:
    """Handles all document loading and processing"""
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.csv', '.md', '.xlsx'}
    LOADER_MAPPING = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.txt': TextLoader,
        '.csv': CSVLoader,
        '.md': UnstructuredMarkdownLoader,
        '.xlsx': UnstructuredExcelLoader,
    }

    def load_documents(self, data_dir: str) -> List[Document]:
        """Load all supported documents from directory"""
        documents = []
        data_path = Path(data_dir)
        print(f"Scanning directory: {data_path.resolve()}")

        if not data_path.is_dir():
             print(f"Error: {data_dir} is not a valid directory.")
             return []

        loaded_files = 0
        skipped_files = 0
        for item in data_path.rglob('*'): # rglob searches recursively
            if item.is_file() and item.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                loader_class = self.LOADER_MAPPING.get(item.suffix.lower())
                if loader_class:
                    try:
                        print(f"Loading: {item.name}...")
                        loader = loader_class(str(item))
                        docs = loader.load()
                        # Add source metadata
                        for doc in docs:
                             # Use relative path for cleaner source display
                            relative_path = item.relative_to(data_path)
                            doc.metadata["source"] = str(relative_path)
                        documents.extend(docs)
                        loaded_files += 1
                    except Exception as e:
                        print(f"  Error loading {item.name}: {e}")
                        skipped_files += 1
                else:
                     print(f"  Skipping {item.name}: No suitable loader found (this shouldn't happen with current mapping).")
                     skipped_files += 1

        print("-" * 50)
        print(f"Finished loading.")
        print(f"Successfully loaded {loaded_files} files.")
        if skipped_files > 0:
            print(f"Skipped {skipped_files} files due to errors or unsupported types within supported extensions.")

        if not documents:
            print(f"Warning: No supported documents found or loaded in {data_dir}")

        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into manageable chunks"""
        if not documents:
            return []
        print("Chunking documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        chunked_docs = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunked_docs)} chunks.")
        return chunked_docs

def build_knowledge_base():
    """Loads data, processes it, and builds the FAISS vector store."""
    try:
        config = Config()
        processor = DocumentProcessor()

        # 1. Load Documents
        documents = processor.load_documents(config.data_dir)
        if not documents:
             print("No documents loaded. Exiting knowledge base build.")
             return

        # 2. Chunk Documents
        chunked_docs = processor.chunk_documents(documents)
        if not chunked_docs:
             print("No chunks created. Exiting knowledge base build.")
             return

        # 3. Initialize Embeddings
        print("Initializing embedding model...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model=config.embedding_model_name,
            google_api_key=config.api_key
        )

        # 4. Create and Save FAISS Index
        print("Building FAISS index (this may take a while)...")
        vector_db = FAISS.from_documents(
            documents=chunked_docs,
            embedding=embeddings
        )
        vector_db.save_local(config.knowledge_base_dir)
        print("-" * 50)
        print(f"FAISS index built successfully with {len(chunked_docs)} chunks.")
        print(f"Knowledge base saved to: {Path(config.knowledge_base_dir).resolve()}")
        print("-" * 50)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data directory specified in config.py exists.")
    except ValueError as e:
         print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during ingestion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    build_knowledge_base()