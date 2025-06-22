"""
main.py
Main entry point for the ETL Parser System. Handles configuration,
initialization of agents and databases, and orchestrates the workflow.
"""

import os
from dotenv import load_dotenv
import json
from pyspark.sql import SparkSession

# Import modular components
from src.core.vector_db_manager import VectorDBManager
from src.agents.parser_agent import ParserAgent
from src.agents.planner_agent import PlannerAgent
from src.hitl.hitl_manager import initiate_hitl_review
from src.utils.utils import ingest_dataset_metadata, ingest_approved_etl_task # Import only what's needed from utils

load_dotenv() # Load environment variables from .env file

# --- Configuration ---
# Pinecone Configuration for Serverless Index
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") # Production index name for dataset metadata
PINECONE_APPROVED_TASKS_INDEX_NAME = os.getenv("PINECONE_APPROVED_TASKS_INDEX_NAME") # New index for approved ETL tasks
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws") # Default to aws if not set
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1") # Default to us-east-1 if not set

# Path to the dataset
DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "data", "home-credit-default-risk")
COLUMNS_DESCRIPTION_FILENAME = "HomeCredit_columns_description.csv"
# Updated path for the specific description CSV file
COLUMNS_DESCRIPTION_CSV = os.path.join(DATA_DIRECTORY, COLUMNS_DESCRIPTION_FILENAME)

# Check for existence of the DATA_DIRECTORY (where all CSVs should be)
if not os.path.isdir(DATA_DIRECTORY):
    print(f"ERROR: Data directory not found at expected path: {DATA_DIRECTORY}")
    print("Please ensure your 'home-credit-default-risk' dataset directory with all CSVs is inside the 'data' directory next to your script.")
    # We will no longer exit here, but the functions below might fail without data.
    # It's better to let them handle the missing data gracefully if possible,
    # or print an error and continue only if it's explicitly allowed by the user flow.
    # For now, keeping this print, but removing the exit.

# Check if API Key and Cloud/Region are set for Pinecone
if not PINECONE_API_KEY or not PINECONE_CLOUD or not PINECONE_REGION:
    print("WARNING: One or more Pinecone configuration variables not set.")
    print("Please set PINECONE_API_KEY, PINECONE_CLOUD, and PINECONE_REGION ")
    print("(e.g., in a .env file) to connect to Pinecone.")
    # Similar to DATA_DIRECTORY, we'll let the system attempt to proceed,
    # but Pinecone operations will likely fail without these.
    # Removing the exit here.


# Configuration for Ollama LLM and Embedding Models
OLLAMA_LLM_MODEL_NAME = "llama3"
OLLAMA_EMBEDDING_MODEL_NAME = "nomic-embed-text"

# --- DEBUG FLAG ---
# Set to True for debugging LLM hangs. Set to False for normal operation.
# When True, it will print raw LLM output before parsing and use minimal context for initial parse.
DEBUG_LLM_CALL = False


# --- Helper Function for Processing ETL Requests ---
def process_etl_request(user_request: str, db_manager_metadata: VectorDBManager, db_manager_approved_tasks: VectorDBManager, parser_agent: ParserAgent, planner_agent: PlannerAgent, dataset_schema_map: dict, debug_mode: bool) -> dict:
    """
    Processes a single natural language ETL request: parses it and initiates HITL review.
    :param user_request: The natural language request for an ETL pipeline.
    :param db_manager_metadata: An initialized VectorDBManager instance for dataset metadata.
    :param db_manager_approved_tasks: An initialized VectorDBManager instance for approved tasks.
    :param parser_agent: An initialized ParserAgent instance.
    :param debug_mode: A boolean flag to enable/disable debug logging.
    :param planner_agent: An initialized PlannerAgent instance for further validation.
    :param dataset_schema_map: A structured dictionary representing the dataset schema.
    :return: A dictionary containing the status ('approved', 'denied', 'error') and the task definition.
    """
    print(f"\n{'='*50}\nProcessing Incoming Request:\nUser Request: '{user_request}'\n{'='*50}")

    # Query both indexes for relevant context
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

    # --- Step 1: Natural Language Parsing and Task Definition ---
    parsed_task_definition = parser_agent.parse_request(user_request, context_string_full)

    if "error" in parsed_task_definition:
        print("\nParsing failed. Please review the error details and the LLM's raw output.")
        print(f"Status : {parsed_task_definition.get('error', 'Unknown error')}")
        if "details" in parsed_task_definition:
            print(f"Details: {parsed_task_definition['details']}")
        if "raw_llm_output" in parsed_task_definition:
            print(f"Raw LLM Output:\n{parsed_task_definition['raw_llm_output']}")
        return {"status": "error", "task": parsed_task_definition}
    else:
        # print("\nInitial Parsed Task Definition (from Parser Agent):") # Commented out as requested
        # print(json.dumps(parsed_task_definition, indent=2)) # Commented out as requested

        # --- Step 2: Objective Definition with HITL (Planner Agent) ---
        print("\n--- Initiating Planner Agent for Schema Validation and Refinement ---")
        refined_task_definition = planner_agent.validate_and_refine_task(parsed_task_definition, dataset_schema_map)

        if "error" in refined_task_definition:
            print("\nPlanning failed. Please review the error details.")
            print(f"Status : {refined_task_definition.get('error', 'Unknown error')}")
            if "details" in refined_task_definition:
                print(f"Details: {refined_task_definition['details']}")
            if "raw_llm_output" in refined_task_definition:
                print(f"Raw LLM Output:\n{refined_task_definition['raw_llm_output']}")
            return {"status": "error", "task": refined_task_definition}

        # print("\nRefined and Validated Task Definition (from Planner Agent):") # Commented out as requested
        # print(json.dumps(refined_task_definition, indent=2)) # Commented out as requested

        print("\n--- Initiating Human-in-the-Loop Review (Type 'approve', 'deny', or feedback for modification) ---")

        is_approved = initiate_hitl_review(
            refined_task_definition,
            parser_agent,
            similar_metadata_docs + similar_approved_tasks_docs,
            user_request,
            db_manager_approved_tasks,
            ingest_approved_etl_task,
            debug_mode
        )

        if is_approved:
            print("\nTask definition successfully approved.")
            return {"status": "approved", "task": refined_task_definition}
        else:
            print("\nTask definition denied. Please refine the user request or review the parsing logic and context.")
            return {"status": "denied", "task": refined_task_definition}


# --- Main Execution Workflow (Production-Ready Entry Point) ---

if __name__ == "__main__":
    print("Initializing ETL Parser System...")

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

    print("\nInitializing SparkSession...")
    spark_session = SparkSession.builder \
        .appName("ETLSchemaInference") \
        .master("local[*]") \
        .config("spark.driver.memory", "6g") \
        .getOrCreate()
    print("SparkSession initialized.")

    dataset_schema = ingest_dataset_metadata(db_manager_metadata, spark_session, DATA_DIRECTORY, COLUMNS_DESCRIPTION_CSV)

    if not dataset_schema:
        print("\nWARNING: Dataset metadata ingestion skipped or failed. Proceeding without full dataset schema information.")
    
    parser_agent_instance = ParserAgent(
        vector_db_manager_metadata=db_manager_metadata,
        vector_db_manager_approved_tasks=db_manager_approved_tasks,
        llm_model_name=OLLAMA_LLM_MODEL_NAME,
        debug_mode=DEBUG_LLM_CALL
    )

    planner_agent_instance = PlannerAgent(
        llm_model_name=OLLAMA_LLM_MODEL_NAME,
        debug_mode=DEBUG_LLM_CALL
    )

    print("\nETL Parser System initialized. Ready to process requests.")
    print("-------------------------------------------------------")

    sample_incoming_request = "Generate an ETL pipeline to predict loan default on the main application table, joining with previous credit bureau data and ensuring all missing values are handled. I need the output saved as parquet in s3://my-data-lake/processed/loan_defaults."

    final_task_status = process_etl_request(
        user_request=sample_incoming_request,
        db_manager_metadata=db_manager_metadata,
        db_manager_approved_tasks=db_manager_approved_tasks,
        parser_agent=parser_agent_instance,
        planner_agent=planner_agent_instance,
        dataset_schema_map=dataset_schema,
        debug_mode=DEBUG_LLM_CALL
    )

    print(f"\nFinal status for sample request: {final_task_status['status']}")
    if final_task_status['status'] == "approved":
        print("Approved task details (truncated for display):")
        print(json.dumps(final_task_status['task'], indent=2)[:500] + "...")

    spark_session.stop()
    print("\nSparkSession stopped.")

    print("\n-------------------------------------------------------")
    print("System is ready for further requests.")
