# app.py
import os
import textwrap
from pathlib import Path
from typing import List, Optional
import sys

# Add project root to sys.path if necessary
# sys.path.append(str(Path(__file__).parent))

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Assuming config.py is in the same directory or Python path
from config import Config

class KnowledgeBaseLoader:
    """Loads the pre-built knowledge base"""

    def __init__(self, config: Config):
        self.config = config
        self.vector_db: Optional[FAISS] = None
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.embedding_model_name,
            google_api_key=config.api_key
        )

    def load(self) -> FAISS:
        """Load existing knowledge base or raise error"""
        kb_path = Path(self.config.knowledge_base_dir)
        if not kb_path.exists() or not (kb_path / "index.faiss").exists():
             raise FileNotFoundError(
                 f"Knowledge base not found at {kb_path.resolve()}. "
                 f"Please run ingest.py first to build it."
             )

        try:
            print(f"Loading knowledge base from: {kb_path.resolve()}...")
            self.vector_db = FAISS.load_local(
                self.config.knowledge_base_dir,
                self.embeddings,
                allow_dangerous_deserialization=True # Be cautious if index comes from untrusted source
            )
            print("Knowledge base loaded successfully.")
            return self.vector_db
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            raise RuntimeError(f"Could not load knowledge base from {kb_path.resolve()}") from e


class FinancialAdvisor:
    """Main RAG system for financial advice using pre-loaded KB"""

    def __init__(self, config: Config, vector_db: FAISS):
        self.config = config
        self.vector_db = vector_db
        self.retriever = self.vector_db.as_retriever(
            search_kwargs={'k': config.max_retrieval_docs}
        )
        self.llm = ChatGoogleGenerativeAI(
            model=config.llm_model_name,
            google_api_key=config.api_key,
            temperature=0.7,
            convert_system_message_to_human=True # Good practice for Gemini
        )

        # Setup prompt template
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

        # Define the RAG chain using LangChain Expression Language (LCEL)
        self.rag_chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.response_prompt_template
            | self.llm
            | StrOutputParser() # Parses the LLM output message to a string
        )

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """Formats retrieved documents for the prompt context."""
        if not docs:
            return "No relevant documents found."
        # Use Path().name to get just the filename from the source metadata
        return "\n\n".join(
            f"Source: {Path(doc.metadata.get('source', 'Unknown')).name}\nContent:\n{doc.page_content}"
            for doc in docs
        )

    def _add_disclosures(self, response: str) -> str:
        """Ensure required disclosures are present"""
        disclosures = [
            "DISCLAIMER: This is general information based on YMCA documents and not personalized financial advice.",
            "Investing involves risks, including the potential loss of principal.",
            "Past performance is not indicative of future results.",
            "Please consult with a qualified financial professional before making any financial decisions."
        ]
        # Add a separator if needed
        if response and not response.endswith('\n\n'):
             response += '\n\n'

        response += "---\n" # Separator for clarity
        response += "\n".join(disclosures)
        return response

    def generate_response(self, query: str) -> str:
        """Generate response using the RAG chain"""
        if not query:
            return "Please ask a question."
        try:
            print("Retrieving relevant documents and generating response...")
            llm_response = self.rag_chain.invoke(query)

            # Add disclosures to the final response
            final_response = self._add_disclosures(llm_response)
            return final_response

        except Exception as e:
            print(f"An error occurred during response generation: {e}")
            import traceback
            traceback.print_exc()
            # Provide a safe response to the user
            return self._add_disclosures("I encountered an technical issue while processing your request. Please try again later.")


class FinancialAdvisorApp:
    """Command-line interface for the financial advisor"""

    @staticmethod
    def display_response(question: str, response: str):
        """Format and display the response"""
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("-" * 80)
        print("ASSISTANT RESPONSE:")
        # Use textwrap for better readability in the terminal
        wrapped_response = textwrap.fill(response, width=80, replace_whitespace=False)
        print(wrapped_response)
        print("=" * 80 + "\n")

    def run(self):
        """Main application loop"""
        try:
            config = Config()
            kb_loader = KnowledgeBaseLoader(config)
            vector_db = kb_loader.load() # Load the pre-built KB
            advisor = FinancialAdvisor(config, vector_db) # Pass the loaded DB

            print("\n--- YMCA Financial Advisor Assistant ---")
            print("Based on information from YMCA documents.")
            print("Type 'exit' or 'quit' to end the session.\n")

            while True:
                try:
                    question = input("Your financial question: ").strip()
                    if question.lower() in ['exit', 'quit']:
                        print("Goodbye!")
                        break
                    if not question:
                        continue

                    response = advisor.generate_response(question)
                    self.display_response(question, response)

                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"\nAn internal error occurred: {e}")
                    # Optionally add more robust error handling or logging here

        except FileNotFoundError as e:
             print(f"Error: {e}")
             print("Please ensure you have run 'python ingest.py' successfully first.")
        except RuntimeError as e:
             print(f"Error: {e}")
             print("Failed to initialize the financial advisor.")
        except Exception as e:
            print(f"A critical error occurred on startup: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    app_instance = FinancialAdvisorApp()
    app_instance.run()