# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

class Config:
    """Manages application configuration"""
    def __init__(self):
        load_dotenv()
        self.api_key = self._get_api_key()
        self.data_dir = "YMCA_data"
        self.knowledge_base_dir = "financial_knowledge_base"
        self.embedding_model_name = "models/embedding-001"
        # --- CORRECTED LINE ---
        self.llm_model_name = "gemini-2.0-flash"
        # --- /CORRECTED LINE ---
        self.max_retrieval_docs = 5
        self.user_agent = "YMCA-Financial-Advisor/1.0"

        # Validate data path and create knowledge base path
        data_path = Path(self.data_dir)
        if not data_path.exists() or not data_path.is_dir():
            raise FileNotFoundError(f"Data directory not found or is not a directory: {self.data_dir}")
        Path(self.knowledge_base_dir).mkdir(parents=True, exist_ok=True)

    def _get_api_key(self) -> str:
        """Retrieve and validate Google API key"""
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        return key

# Instantiate config globally if desired, or pass it around
# config = Config()