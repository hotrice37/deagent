"""
main.py
Main entry point for the ETL Parser System. Handles configuration,
initialization of agents and databases, and orchestrates the workflow.
"""

import os
import json
from pyspark.sql import SparkSession

# Import modular components
from src.core.vector_db_manager import VectorDBManager
from src.agents.parser_agent import ParserAgent
from src.agents.planner_agent import PlannerAgent
from src.orchestrator import ETLOrchestrator
from src.utils.data_ingestion_utils import ingest_dataset_metadata # From new data_ingestion_utils
from src.utils.etl_task_utils import ingest_approved_etl_task # From new etl_task_utils
from src.config import (
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_APPROVED_TASKS_INDEX_NAME,
    PINECONE_CLOUD, PINECONE_REGION, DATA_DIRECTORY, COLUMNS_DESCRIPTION_CSV,
    OLLAMA_LLM_MODEL_NAME, OLLAMA_EMBEDDING_MODEL_NAME, DEBUG_LLM_CALL,
    validate_environment_variables
)


if __name__ == "__main__":
    print("Initializing ETL Parser System...")

    # Validate environment variables and configuration
    validate_environment_variables()

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

    # Initialize SparkSession for dynamic schema inference
    print("\nInitializing SparkSession...")
    spark_session = SparkSession.builder \
        .appName("ETLSchemaInference") \
        .master("local[*]") \
        .config("spark.driver.memory", "6g") \
        .getOrCreate()
    print("SparkSession initialized.")

    # Ingest dataset metadata (potentially skipped by user or due to errors)
    dataset_schema = ingest_dataset_metadata(db_manager_metadata, spark_session, DATA_DIRECTORY, COLUMNS_DESCRIPTION_CSV)

    if not dataset_schema:
        print("\nWARNING: Dataset metadata ingestion skipped or failed. Proceeding without full dataset schema information.")
    
    # Initialize the Parser Agent
    parser_agent_instance = ParserAgent(
        vector_db_manager_metadata=db_manager_metadata,
        vector_db_manager_approved_tasks=db_manager_approved_tasks,
        llm_model_name=OLLAMA_LLM_MODEL_NAME,
        debug_mode=DEBUG_LLM_CALL
    )

    # Initialize the Planner Agent
    planner_agent_instance = PlannerAgent(
        llm_model_name=OLLAMA_LLM_MODEL_NAME,
        debug_mode=DEBUG_LLM_CALL
    )

    # Initialize the ETL Orchestrator
    etl_orchestrator = ETLOrchestrator(
        db_manager_metadata=db_manager_metadata,
        db_manager_approved_tasks=db_manager_approved_tasks,
        parser_agent=parser_agent_instance,
        planner_agent=planner_agent_instance,
        dataset_schema_map=dataset_schema,
        debug_mode=DEBUG_LLM_CALL,
        ingest_approved_etl_task_func=ingest_approved_etl_task
    )

    print("\nETL Parser System initialized. Ready to process requests.")
    print("-------------------------------------------------------")

    # --- Example of processing a single incoming request (simulating a real production scenario) ---
    sample_incoming_request = "Generate an ETL pipeline to predict loan default on the main application table, joining with previous credit bureau data and ensuring all missing values are handled. I need the output saved as parquet in s3://my-data-lake/processed/loan_defaults."

    # Process the request using the orchestrator
    final_task_status = etl_orchestrator.process_etl_request(sample_incoming_request)

    print(f"\nFinal status for sample request: {final_task_status['status']}")
    if final_task_status['status'] == "approved":
        print("Approved task details (truncated for display):")
        print(json.dumps(final_task_status['task'], indent=2)[:500] + "...")

    # Stop SparkSession
    spark_session.stop()
    print("\nSparkSession stopped.")

    print("\n-------------------------------------------------------")
    print("System is ready for further requests.")
