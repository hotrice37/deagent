"""
utils.py
Contains utility functions used across different modules, such as
dataset metadata ingestion and approved task ingestion, and JSON extraction.
"""

# General Imports
import pandas as pd
import hashlib
import uuid
import re
from typing import List, Optional

# Project Imports- for vector database management
from src.core.vector_db_manager import VectorDBManager


# Helper function for robust JSON extraction from LLM output
def extract_json_from_llm_output(llm_output: str) -> str:
    """
    Extracts a JSON string from LLM output, robustly handling markdown code fences.
    Assumes the JSON is the primary content and attempts to remove surrounding markdown.
    """
    cleaned_output = llm_output.strip()
    # Check for and remove common markdown code block delimiters
    # Using re.DOTALL to allow . to match newlines for multi-line JSON
    match = re.search(r"```json\s*\n(.*)```", cleaned_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback for cases where 'json' might not be specified in the fence, or no fence exists
    match = re.search(r"```\s*\n(.*)```", cleaned_output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no markdown fence is found, assume the entire output is meant to be JSON
    return cleaned_output


def ingest_dataset_metadata(db_manager: VectorDBManager, csv_path: str):
    """
    Ingests dataset column descriptions from a CSV file into the vector database.
    This function is intended to be run once as part of initial setup or a data catalog sync.
    """
    print(f"Ingesting dataset metadata from {csv_path} into vector database...")
    try:
        columns_df = pd.read_csv(csv_path, encoding='latin-1')
        print(f"Successfully loaded column descriptions from {csv_path} with latin-1 encoding.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load CSV '{csv_path}': {e}")
        print("Please ensure the file exists at the correct path and has the correct encoding (e.g., 'latin-1' or 'cp1252').")
        exit("Exiting due to critical CSV loading error.")

    texts_to_add = []
    metadatas_to_add = []
    ids_to_add = [] # List for deterministic IDs

    # CSV columns are 'Table', 'Row', 'Description', 'Special' ('Special' is ignored)
    for index, row in columns_df.iterrows():
        table_name = str(row['Table']).strip() if pd.notna(row['Table']) else None
        column_name = str(row['Row']).strip() if pd.notna(row['Row']) else None
        description = str(row['Description']).strip() if pd.notna(row['Description']) else ""

        # Basic logic to identify target/ID columns based on common names
        is_target = (column_name == 'TARGET') and (table_name == 'application_train')
        is_id = (column_name in ['SK_ID_CURR', 'SK_ID_BUREAU'])

        # Create a more descriptive text for embedding
        if column_name:
            text_for_embedding = f"Table: {table_name}, Column: {column_name}, Description: {description}"
            # Generate a stable ID for column metadata
            stable_id_content = f"dataset-meta-{table_name}-{column_name}-{description}"
        else: # This is a table-level description
            text_for_embedding = f"Table: {table_name}, Description: {description}"
            # Generate a stable ID for table-level metadata
            stable_id_content = f"dataset-meta-{table_name}-{description}"
        
        # Define metadata dictionary
        metadata = {
            "source": "dataset_metadata", # Categorize as dataset metadata
            "table_name": table_name,
            "column_name": column_name,
            "description": description,
            "is_target": is_target,
            "is_id": is_id
        }

        # Hash the content to create a consistent and unique ID
        generated_id = hashlib.sha256(stable_id_content.encode('utf-8')).hexdigest()
        
        texts_to_add.append(text_for_embedding)
        metadatas_to_add.append(metadata)
        ids_to_add.append(generated_id)

    if texts_to_add: # Check if there's any data to add
        db_manager.add_documents_batch(texts_to_add, metadatas_to_add, doc_ids=ids_to_add)
    print("Dataset metadata ingestion complete.")


def ingest_approved_etl_task(db_manager: VectorDBManager, task_definition: dict, original_request_text: str, modification_feedback_history: Optional[List[str]] = None):
    """
    Ingests a newly approved ETL task definition into the dedicated approved tasks vector database.
    This function creates a robust text representation and adds relevant metadata for future retrieval.
    :param db_manager: The VectorDBManager instance for the approved tasks index.
    :param task_definition: The dictionary representing the approved ETLTaskDefinition.
    :param original_request_text: The original natural language request that led to this task.
    :param modification_feedback_history: A list of strings, each representing a piece of human modification feedback.
    """
    if not task_definition:
        print("No task definition provided for ingestion.")
        return

    print(f"Ingesting approved ETL task '{task_definition.get('pipeline_name', 'Unnamed Pipeline')}' into approved tasks index...")

    # Create a concise text representation of the task for embedding
    # Include key details like main goal, initial tables, and scoring model info.
    task_text = (
        f"Approved ETL Pipeline: {task_definition.get('pipeline_name', 'Unnamed Pipeline')}. "
        f"Main Goal: {task_definition.get('main_goal', 'Not specified')}. "
        f"Initial Tables: {', '.join(task_definition.get('initial_tables', []))}. "
    )
    if task_definition.get('scoring_model'):
        scoring_model_info = task_definition['scoring_model']
        task_text += (
            f"Scoring Model: {scoring_model_info.get('name', 'Generic')}, "
            f"Objective: {scoring_model_info.get('objective', 'Not specified')}, "
            f"Target: {scoring_model_info.get('target_column', 'Not specified')}. "
        )
        if scoring_model_info.get('features'):
            task_text += f"Features: {', '.join(scoring_model_info['features'])}."
    
    # Add key details from the task definition as metadata for filtering/context
    task_metadata = {
        "source": "approved_etl_task", # Categorize as approved ETL task
        "pipeline_name": task_definition.get('pipeline_name'),
        "main_goal": task_definition.get('main_goal'),
        "initial_tables": task_definition.get('initial_tables'),
        "join_operations_summary": [
            f"{op.get('left_table', '')} JOIN {op.get('right_table', '')} ON {','.join(op.get('on_columns', []))}"
            for op in task_definition.get('join_operations', [])
        ],
        "data_cleaning_types": [step.get('type', '') for step in task_definition.get('data_cleaning_steps', [])],
        "scoring_model_name": task_definition.get('scoring_model', {}).get('name'),
        "target_column": task_definition.get('scoring_model', {}).get('target_column'),
        "original_request_text": original_request_text # Store the original user request
    }

    # Add modification history to metadata if provided
    if modification_feedback_history:
        task_metadata["modification_feedback_history"] = modification_feedback_history

    # Generate a unique ID for the approved task. UUID is suitable here as these are dynamic additions.
    task_id = str(uuid.uuid4())
    
    db_manager.add_documents_batch(
        texts=[task_text],
        metadatas=[task_metadata],
        doc_ids=[task_id]
    )
    print(f"Approved ETL task '{task_definition.get('pipeline_name', 'Unnamed Pipeline')}' ingested successfully.")
