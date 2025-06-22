# src/utils/etl_task_utils.py
# Contains functions related to managing approved ETL tasks.

import uuid
import json
from typing import List, Optional

from src.core.vector_db_manager import VectorDBManager


def ingest_approved_etl_task(db_manager: VectorDBManager, task_definition: dict, original_request_text: str, modification_feedback_history: Optional[List[str]] = None):
    """
    Ingests a newly approved ETL task definition into the dedicated approved tasks vector database.
    """
    if not task_definition:
        print("No task definition provided for ingestion.")
        return

    print(f"Ingesting approved ETL task '{task_definition.get('pipeline_name', 'Unnamed Pipeline')}' into approved tasks index...")

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
    
    task_metadata = {
        "source": "approved_etl_task",
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
        "original_request_text": original_request_text
    }

    if modification_feedback_history:
        task_metadata["modification_feedback_history"] = modification_feedback_history

    task_id = str(uuid.uuid4())
    
    user_consent = input(f"\nAbout to upsert approved ETL task '{task_definition.get('pipeline_name', 'Unnamed Pipeline')}' to Pinecone index '{db_manager.index_name}'. Do you want to proceed? (yes/no): ").lower().strip()
    if user_consent == 'yes':
        db_manager.add_documents_batch(
            texts=[task_text],
            metadatas=[task_metadata],
            doc_ids=[task_id]
        )
        print(f"Approved ETL task '{task_definition.get('pipeline_name', 'Unnamed Pipeline')}' ingested successfully.")
    else:
        print(f"Upsert of approved ETL task '{task_definition.get('pipeline_name', 'Unnamed Pipeline')}' skipped by user request.")

