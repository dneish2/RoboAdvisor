# backend/llama_integration.py
import os
import logging
import shutil
from typing import List, Optional, Union
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Document,
    Settings,
    StorageContext, 
    load_index_from_storage
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from openai import RateLimitError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
import hashlib
from .data_fetcher import DataFetcher

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLamaQuery:
    def __init__(
        self,
        docs_dir: str = 'data/company_reports',
        persist_dir: str = 'storage',
        max_documents: Optional[int] = None,
        embedding_model: str = "text-embedding-3-small",
        force_reload: bool = False
    ):
        """
        Initialize LLamaQuery with configurable parameters.
        
        Args:
            docs_dir: Directory containing documents to index
            persist_dir: Directory for storing the index
            max_documents: Maximum number of documents to process
            embedding_model: OpenAI embedding model to use
            force_reload: If True, forces reloading of documents and rebuilding index
        """
        self.docs_dir = docs_dir
        self.persist_dir = persist_dir
        self.max_documents = max_documents
        self.embedding_model = embedding_model
        self.force_reload = force_reload
        
        # Set up embedding model globally
        Settings.embed_model = OpenAIEmbedding(model=embedding_model)
        
        # Ensure directories exist
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize hash file path
        self.hash_file = os.path.join(self.persist_dir, 'documents_hash.txt')
        
        try:
            self.index = self._initialize_llama()
            # Create query engine with more specific parameters
            self.query_engine = self.index.as_query_engine(
                response_mode="compact",
                streaming=False,
                similarity_top_k=5  # Retrieve more context
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLamaIndex: {e}")
            raise

    def _calculate_documents_hash(self) -> str:
        """Calculate a hash of all documents in the docs directory."""
        logger.info("Calculating documents hash...")
        
        hash_list = []
        for root, _, files in os.walk(self.docs_dir):
            for file in sorted(files):  # Sort for consistency
                file_path = os.path.join(root, file)
                if file.endswith(('.pdf', '.txt')):  # Add more extensions if needed
                    try:
                        with open(file_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                            hash_list.append(f"{file}:{file_hash}")
                    except Exception as e:
                        logger.error(f"Error hashing file {file}: {e}")
        
        return hashlib.md5("|".join(hash_list).encode()).hexdigest()

    def _should_rebuild_index(self) -> bool:
        """Determine if the index should be rebuilt based on document changes."""
        if self.force_reload:
            logger.info("Force reload requested")
            return True
            
        current_hash = self._calculate_documents_hash()
        
        try:
            if os.path.exists(self.hash_file):
                with open(self.hash_file, 'r') as f:
                    stored_hash = f.read().strip()
                if stored_hash != current_hash:
                    logger.info("Documents have changed, rebuilding index")
                    return True
            else:
                logger.info("No previous hash found, rebuilding index")
                return True
        except Exception as e:
            logger.error(f"Error checking document hash: {e}")
            return True
            
        return False

    def _load_documents(self) -> List[Document]:
        """Load and optionally limit the number of documents."""
        try:
            if not os.path.exists(self.docs_dir):
                raise FileNotFoundError(f"Documents directory not found: {self.docs_dir}")
            
            if not any(os.scandir(self.docs_dir)):
                raise FileNotFoundError(f"No documents found in directory: {self.docs_dir}")
            
            # Configure the document reader with specific parameters
            reader = SimpleDirectoryReader(
                self.docs_dir,
                filename_as_id=True,  # Use filename as document ID
                recursive=True,  # Search subdirectories
                required_exts=[".pdf", ".txt"],  # Specify allowed extensions
                num_files_limit=self.max_documents
            )
            
            documents = reader.load_data()
            
            # Use sentence splitter for better chunking
            parser = SentenceSplitter(
                chunk_size=1024,
                chunk_overlap=200
            )
            
            for doc in documents:
                doc.text = doc.text.replace('\x00', '')  # Remove null bytes
                nodes = parser.get_nodes_from_documents([doc])
                logger.info(f"Split document {doc.id_} into {len(nodes)} chunks")
            
            logger.info(f"Loaded {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=60),
        retry=retry_if_exception_type(RateLimitError)
    )
    def _initialize_llama(self) -> VectorStoreIndex:
        """Initialize LLamaIndex."""
        try:
            if not self._should_rebuild_index():
                logger.info("Loading existing index from storage...")
                storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
                return load_index_from_storage(storage_context)
            
            # If we need to rebuild, clear the existing storage
            if os.path.exists(self.persist_dir):
                shutil.rmtree(self.persist_dir)
                os.makedirs(self.persist_dir)
            
            logger.info("Loading and indexing documents...")
            documents = self._load_documents()
            
            if not documents:
                raise ValueError("No documents were loaded")
            
            logger.info("Creating new index...")
            index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True  # Show progress bar during indexing
            )
            
            # Save the new document hash
            current_hash = self._calculate_documents_hash()
            with open(self.hash_file, 'w') as f:
                f.write(current_hash)
            
            logger.info("Persisting index...")
            index.storage_context.persist(persist_dir=self.persist_dir)
            
            return index

        except Exception as e:
            logger.error(f"Error initializing LLamaIndex: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=60),
        retry=retry_if_exception_type(RateLimitError)
    )
    def query(self, user_query: str) -> Union[str, None]:
        """Query the index with retry mechanism for rate limits."""
        if not user_query.strip():
            logger.warning("Empty query received")
            return "Please provide a valid query."
            
        try:
            logger.info(f"Querying index with: {user_query}")
            
            # Create a more specific query with additional context
            response = self.query_engine.query(
                f"""Based on the provided documents, {user_query}
                Please provide specific details and cite relevant sections if possible."""
            )
            
            logger.info("Query successful")
            return str(response)
            
        except RateLimitError:
            logger.error("Rate limit exceeded during query")
            return "Rate limit exceeded. Please try again later."
        except Exception as e:
            logger.error(f"Error during query: {e}")
            return f"An error occurred: {str(e)}"

    def force_rebuild(self):
        """Force rebuild the index."""
        logger.info("Force rebuilding index...")
        self.force_reload = True
        self.index = self._initialize_llama()
        self.query_engine = self.index.as_query_engine(
            response_mode="compact",
            streaming=False,
            similarity_top_k=5
        )
        logger.info("Index rebuilt successfully")

    def get_status(self) -> dict:
        """Get the current status of the LLamaQuery instance."""
        try:
            doc_files = [f for f in os.listdir(self.docs_dir) if f.endswith(('.pdf', '.txt'))]
            return {
                "docs_dir_exists": os.path.exists(self.docs_dir),
                "persist_dir_exists": os.path.exists(self.persist_dir),
                "docs_dir_path": os.path.abspath(self.docs_dir),
                "persist_dir_path": os.path.abspath(self.persist_dir),
                "max_documents": self.max_documents,
                "index_initialized": hasattr(self, 'index') and self.index is not None,
                "document_files": doc_files,
                "total_documents": len(doc_files),
                "hash_file_exists": os.path.exists(self.hash_file)
            }
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"error": str(e)}

class LlamaHandler:
    def __init__(self, data_fetcher: DataFetcher, llama_query: LLamaQuery):
        self.data_fetcher = data_fetcher
        self.llama_query = llama_query

    def handle_user_query(self, user_query: str) -> Union[str, None]:
        """
        Handle user queries for mortgage rates, Bitcoin plotting, and existing functionalities.

        Parameters:
            user_query (str): The user's query.

        Returns:
            str: The response to the user's query.
        """
        user_query_lower = user_query.lower()

        if "mortgage interest rates" in user_query_lower:
            current_rate = self.data_fetcher.get_current_mortgage_rates()
            if current_rate:
                response = f"The current 10-year Treasury bond yield, which serves as a proxy for mortgage interest rates, is {current_rate:.2f}%."
                return response
            else:
                return "I'm sorry, I couldn't retrieve the current mortgage interest rates at this time."

        elif "bitcoin price" in user_query_lower or "btc price" in user_query_lower:
            # Indicate that a Bitcoin price chart is available
            response = "Here is the Bitcoin price chart for the past year."
            return response

        elif any(company in user_query_lower for company in ["aapl", "msft", "googl", "amzn", "meta", "nvda", "qqq"]):
            # Handle queries about specific companies
            response = "I can provide information about the performance of the requested company. Please specify your request."
            return response

        else:
            # Default handling for other queries
            response = "I'm sorry, I can only provide information on mortgage interest rates, Bitcoin prices, and specific company information at the moment."
            return response