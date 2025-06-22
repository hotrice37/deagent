# src/config.py
# Centralized configuration and environment variable loading

import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# --- Pinecone Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") # Production index name for dataset metadata
PINECONE_APPROVED_TASKS_INDEX_NAME = os.getenv("PINECONE_APPROVED_TASKS_INDEX_NAME") # New index for approved ETL tasks
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws") # Default to aws if not set
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1") # Default to us-east-1 if not set

# --- Dataset Paths ---
# Corrected DATA_DIRECTORY path to go up one level from 'src'
DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "home-credit-default-risk")
COLUMNS_DESCRIPTION_FILENAME = "HomeCredit_columns_description.csv"
COLUMNS_DESCRIPTION_CSV = os.path.join(DATA_DIRECTORY, COLUMNS_DESCRIPTION_FILENAME)

# --- Ollama LLM and Embedding Models ---
OLLAMA_LLM_MODEL_NAME = "llama3"
OLLAMA_EMBEDDING_MODEL_NAME = "nomic-embed-text"

# --- Debug Flag ---
# Set to True for debugging LLM hangs (will print raw LLM outputs). Set to False for normal operation.
DEBUG_LLM_CALL = False

def validate_environment_variables():
    """
    Validates that essential environment variables and paths are configured.
    Prints warnings/errors if configuration is incomplete.
    """
    if not os.path.isdir(DATA_DIRECTORY):
        print(f"ERROR: Data directory not found at expected path: {DATA_DIRECTORY}")
        print("Please ensure your 'home-credit-default-risk' dataset directory with all CSVs is inside the 'data' directory next to your script.")
        # Program will try to proceed, but data-dependent operations will fail.

    if not PINECONE_API_KEY or not PINECONE_CLOUD or not PINECONE_REGION:
        print("WARNING: One or more Pinecone configuration variables not set.")
        print("Please set PINECONE_API_KEY, PINECONE_CLOUD, and PINECONE_REGION ")
        print("(e.g., in a .env file) to connect to Pinecone.")
        # Program will try to proceed, but Pinecone operations will likely fail.

    # Optional: Check if Ollama is running and models are available (more involved, might add later)
    # For now, rely on OllamaLLM/OllamaEmbeddings to raise errors.
