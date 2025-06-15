# main.py
# Main entry point for the ETL Parser System. Handles configuration,
# initialization of agents and databases, and orchestrates the workflow.

import os
from dotenv import load_dotenv
import json
from typing import List
from langchain_core.documents import Document

# Import modular components
from vector_db_manager import VectorDBManager
from parser_agent import ParserAgent
from hitl_manager import initiate_hitl_review
from utils import ingest_dataset_metadata, ingest_approved_etl_task # Import only what's needed from utils

load_dotenv() # Load environment variables from .env file

# --- Configuration ---
# Pinecone Configuration for Serverless Index
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "etl-backlog-production-index" # Production index name for dataset metadata
PINECONE_APPROVED_TASKS_INDEX_NAME = "etl-approved-tasks-index" # New index for approved ETL tasks
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws") # Default to aws if not set
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1") # Default to us-east-1 if not set

# Path to the dataset columns description CSV (e.g., from Home Credit competition)
COLUMNS_DESCRIPTION_CSV = os.path.join(os.path.dirname(__file__), "home-credit-default-risk", "HomeCredit_columns_description.csv")

# Ensure the CSV file exists for initial metadata loading
if not os.path.exists(COLUMNS_DESCRIPTION_CSV):
    print(f"ERROR: Column description CSV not found at expected path: {COLUMNS_DESCRIPTION_CSV}")
    print("Please ensure 'HomeCredit_columns_description.csv' is inside a 'home-credit-default-risk' subdirectory next to your script.")
    exit("Exiting due to critical CSV loading error.")


# Check if API Key and Cloud/Region are set for Pinecone
if not PINECONE_API_KEY or not PINECONE_CLOUD or not PINECONE_REGION:
    print("WARNING: One or more Pinecone configuration variables not set.")
    print("Please set PINECONE_API_KEY, PINECONE_CLOUD, and PINECONE_REGION ")
    print("(e.g., in a .env file) to connect to Pinecone.")
    print("Example .env content for serverless:")
    print("PINECONE_API_KEY=YOUR_KEY")
    print("PINECONE_CLOUD=aws")
    print("PINECONE_REGION=us-east-1")
    exit("Exiting: Pinecone configuration is incomplete.")


# Configuration for Ollama LLM and Embedding Models
OLLAMA_LLM_MODEL_NAME = "llama3"
OLLAMA_EMBEDDING_MODEL_NAME = "nomic-embed-text"

# --- DEBUG FLAG ---
# Set to True for debugging LLM hangs. Set to False for normal operation.
# When True, it will print raw LLM output before parsing and use minimal context for initial parse.
DEBUG_LLM_CALL = False


# --- Helper Function for Processing ETL Requests (Moved from utils) ---
def process_etl_request(user_request: str, db_manager_metadata: VectorDBManager, db_manager_approved_tasks: VectorDBManager, parser_agent: ParserAgent, debug_mode: bool) -> dict:
    """
    Processes a single natural language ETL request: parses it and initiates HITL review.
    :param user_request: The natural language request for an ETL pipeline.
    :param db_manager_metadata: An initialized VectorDBManager instance for dataset metadata.
    :param db_manager_approved_tasks: An initialized VectorDBManager instance for approved tasks.
    :param parser_agent: An initialized ParserAgent instance.
    :param debug_mode: A boolean flag to enable/disable debug logging.
    :return: A dictionary containing the status ('approved', 'denied', 'error') and the task definition.
    """
    print(f"\n{'='*50}\nProcessing Incoming Request:\nUser Request: '{user_request}'\n{'='*50}")

    # Query both indexes for relevant context
    print("DEBUG_FLOW: Initiating context retrieval for main request.")
    # Fetching docs here and passing them to parse_request
    similar_metadata_docs = db_manager_metadata.query_similar_documents(user_request, k=5)
    similar_approved_tasks_docs = db_manager_approved_tasks.query_similar_documents(user_request, k=3)
    
    # Combine and format context for the LLM
    context_string_full = ""
    if similar_metadata_docs:
        context_string_full += "Relevant Dataset Metadata:\n"
        for i, doc in enumerate(similar_metadata_docs):
            context_string_full += f"  - Document {i+1} (Dataset): '{doc.page_content}'\n"
            if doc.metadata:
                context_string_full += f"    Metadata: {json.dumps(doc.metadata, indent=2)}\n"
    
    if similar_approved_tasks_docs:
        context_string_full += "\nRelevant Approved ETL Tasks (Past Examples):\n"
        for i, doc in enumerate(similar_approved_tasks_docs):
            context_string_full += f"  - Document {i+1} (Approved Task): '{doc.page_content}'\n"
            if doc.metadata:
                context_string_full += f"    Metadata: {json.dumps(doc.metadata, indent=2)}\n"

    if not context_string_full:
        context_string_full = "No highly relevant context found in any vector database."

    parsed_task_definition = parser_agent.parse_request(user_request, context_string_full)
    print("DEBUG_FLOW: Context retrieval and initial parse completed.")


    if "error" in parsed_task_definition:
        print("\nParsing failed. Please review the error details and the LLM's raw output.")
        print(json.dumps(parsed_task_definition, indent=2))
        return {"status": "error", "task": parsed_task_definition}
    else:
        print("\nGenerated Task Definition:")
        print(json.dumps(parsed_task_definition, indent=2))
        print("\n--- Initiating Human-in-the-Loop Review (Type 'approve', 'deny', or feedback for modification) ---")

        # Pass the combined context documents to the review function
        is_approved = initiate_hitl_review(
            parsed_task_definition,
            parser_agent,
            similar_metadata_docs + similar_approved_tasks_docs,
            user_request,
            db_manager_approved_tasks,
            ingest_approved_etl_task, # Pass the function reference here
            debug_mode # Pass the debug flag
        )

        if is_approved:
            print("\nTask definition successfully approved.")
            return {"status": "approved", "task": parsed_task_definition}
        else:
            print("\nTask definition denied. Please refine the user request or review the parsing logic and context.")
            return {"status": "denied", "task": parsed_task_definition}


# --- Main Execution Workflow (Production-Ready Entry Point) ---

if __name__ == "__main__":
    print("Initializing ETL Parser System for Production...")

    # Initialize two separate vector database managers with Pinecone details
    db_manager_metadata = VectorDBManager(
        index_name=PINECONE_INDEX_NAME,
        cloud=PINECONE_CLOUD,
        region=PINECONE_REGION,
        api_key=PINECONE_API_KEY,
        embedding_model_name=OLLAMA_EMBEDDING_MODEL_NAME
    )

    db_manager_approved_tasks = VectorDBManager(
        index_name=PINECONE_APPROVED_TASKS_INDEX_NAME,
        cloud=PINECONE_CLOUD,
        region=PINECONE_REGION,
        api_key=PINECONE_API_KEY,
        embedding_model_name=OLLAMA_EMBEDDING_MODEL_NAME
    )

    # Ingest static dataset metadata (e.g., from your data catalog CSV)
    ingest_dataset_metadata(db_manager_metadata, COLUMNS_DESCRIPTION_CSV)

    # Initialize the Parser Agent with both DB managers and the debug mode
    parser_agent_instance = ParserAgent(
        vector_db_manager_metadata=db_manager_metadata,
        vector_db_manager_approved_tasks=db_manager_approved_tasks,
        llm_model_name=OLLAMA_LLM_MODEL_NAME,
        debug_mode=DEBUG_LLM_CALL # Pass debug_mode here
    )

    print("\nETL Parser System initialized. Ready to process requests.")
    print("-------------------------------------------------------")

    # --- Example of processing a single incoming request (simulating a real production scenario) ---
    sample_incoming_request = "Generate an ETL pipeline to predict loan default on the main application table, joining with previous credit bureau data and ensuring all missing values are handled. I need the output saved as parquet in s3://my-data-lake/processed/loan_defaults."

    # Process the request using the helper function now defined in main.py
    final_task_status = process_etl_request(
        user_request=sample_incoming_request,
        db_manager_metadata=db_manager_metadata,
        db_manager_approved_tasks=db_manager_approved_tasks,
        parser_agent=parser_agent_instance,
        debug_mode=DEBUG_LLM_CALL # Pass debug_mode here
    )

    print(f"\nFinal status for sample request: {final_task_status['status']}")
    if final_task_status['status'] == "approved":
        print("Approved task details (truncated for display):")
        print(json.dumps(final_task_status['task'], indent=2)[:500] + "...")

    print("\n-------------------------------------------------------")
    print("System is ready for further requests.")
