"""
src/utils/utils.py
Contains utility functions used across different modules, such as
dataset metadata ingestion and approved task ingestion, and JSON extraction.
"""

# General Imports
import os
import pandas as pd
import hashlib
import uuid
import re
from typing import List, Optional, Dict, Any, Tuple
from pyspark.sql import SparkSession

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

def _get_spark_schema_from_csv(spark: SparkSession, csv_file_path: str, table_name: str) -> Dict[str, Dict[str, str]]:
    """
    Infers schema from a single CSV file using PySpark and returns it as a dictionary.
    :param spark: The active SparkSession.
    :param csv_file_path: Full path to the CSV file.
    :param table_name: The name to assign to this table (e.g., 'application_train').
    :return: A dictionary of column names to their inferred data types and basic metadata.
             Format: {column_name: {'data_type': 'string', 'inferred_from': 'spark'}}
    """

    print(f"DEBUG: Inferring schema for '{table_name}' from '{csv_file_path}' using Spark...")
    try:
        # Read only a small sample to infer schema efficiently
        df = spark.read.csv(csv_file_path, header=True, inferSchema=True, mode="DROPMALFORMED").limit(100)
        spark_schema = {}
        for field in df.schema:
            spark_schema[field.name] = {
                'data_type': str(field.dataType),
                'inferred_from': 'spark_dynamic_inference'
            }
        print(f"DEBUG: Successfully inferred schema for '{table_name}'. Found {len(spark_schema)} columns.")
        return spark_schema
    except Exception as e:
        print(f"WARNING: Could not infer Spark schema for {table_name} from {csv_file_path}: {e}")
        return {}


def ingest_dataset_metadata(db_manager: VectorDBManager, spark: SparkSession, data_dir:str, description_csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Ingests dataset metadata by dynamically inferring schema from data CSVs using Spark,
    and enriching it with descriptions/flags from a static description CSV.
    Then, it adds this combined metadata to the vector database.

    :param spark: The active SparkSession.
    :param data_dir: Path to the directory containing all data CSVs (e.g., 'application_train.csv', 'bureau.csv').
    :param description_csv_path: Path to the HomeCredit_columns_description.csv file.
    :return: A structured dictionary mapping table names to their columns and their enriched metadata.
             Format: {table_name: {column_name: {description, data_type, is_target, is_id, inferred_from}}}
    """
    print(f"Ingesting dataset metadata from {data_dir} using Spark and '{description_csv_path}'...")
    try:
        columns_df = pd.read_csv(description_csv_path, encoding='latin-1')
        print(f"Successfully loaded column descriptions from {description_csv_path} with latin-1 encoding.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load CSV '{description_csv_path}': {e}")
        print("Please ensure the file exists at the correct path and has the correct encoding (e.g., 'latin-1' or 'cp1252').")
        exit("Exiting due to critical CSV loading error.")

    texts_to_add = []
    metadatas_to_add = []
    ids_to_add = [] # List for deterministic IDs
    dataset_schema_map: Dict[str, Dict[str, Any]] = {}

    global_unique_columns: Dict[str, Dict[str, Any]] = {} # To store unique column definitions across all tables for embedding
    global_id_metadata: Dict[str, Dict[str, Any]] = {} # To store universal metadata for ID columns
    
    # 1. Load static descriptions for enrichment
    description_df = pd.DataFrame()
    if os.path.exists(description_csv_path):
        try:
            description_df = pd.read_csv(description_csv_path, encoding='latin-1')
            description_df.columns = [col.strip() for col in description_df.columns] # Clean column names
            print(f"DEBUG: Description CSV loaded. Total rows: {len(description_df)}")
            print(f"DEBUG: First 5 rows of description_df:\n{description_df.head().to_string()}")

            print(f"DEBUG: Successfully loaded column descriptions from {description_csv_path}.")
        except Exception as e:
            print(f"WARNING: Could not load {description_csv_path} for enrichment: {e}")
    else:
        print(f"WARNING: Description CSV not found at '{description_csv_path}'. Only dynamically inferred schema will be used.")

    # Group descriptions by table for easier lookup
    print("\nDEBUG: Building description lookup map...")
    desc_lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not description_df.empty:
        for index, row in description_df.iterrows():
            table_name = str(row.get('Table', '')).strip() if pd.notna(row.get('Table')) else None
            raw_table_pattern = str(row.get('Table', '')).strip() if pd.notna(row.get('Table')) else None
            column_name = str(row.get('Row', '')).strip() if pd.notna(row.get('Row')) else None
            description = str(row.get('Description', '')).strip() if pd.notna(row.get('Description')) else ""

            target_table_names: List[str] = []
            if raw_table_pattern == "application_{train|test}.csv":
                target_table_names = ["application_train", "application_test"]
            elif raw_table_pattern: # For all other patterns, simply standardize as before
                # Standardize: lowercase and remove '.csv' if present
                standardized_name = re.sub(r'\.csv$', '', raw_table_pattern.lower())
                if standardized_name: # Ensure it's not empty
                    target_table_names.append(standardized_name)
            
            for table_name_to_add in target_table_names:
                if table_name_to_add not in desc_lookup:
                    desc_lookup[table_name_to_add] = {}

                if column_name: # It's a column description
                    desc_lookup[table_name_to_add][column_name.lower()] = {
                        'description': description,
                        'is_target': (column_name.lower() == 'target' and table_name_to_add == 'application_train'),
                        'is_id': (column_name.lower() in ['sk_id_curr', 'sk_id_bureau', 'sk_id_prev', 'sk_id_cc', 'sk_id_dpd', 'sk_id_instal'])
                    }
                else: # It's a table-level description (e.g., 'HomeCredit_columns_description.csv' row with no 'Row' value)
                    desc_lookup[table_name_to_add]['_table_description'] = {'description': description}

    # Collect all unique column names found in the description_df (after standardization and pattern handling)
    described_column_names_set = set() # NEW
    for table_name_in_lookup, columns_in_table in desc_lookup.items(): # NEW
        for col_name_in_lookup, col_data_in_lookup in columns_in_table.items(): # NEW
            if col_name_in_lookup != '_table_description': # NEW
                described_column_names_set.add(col_name_in_lookup) # NEW



    print(f"DEBUG: Description lookup map built. Tables found: {list(desc_lookup.keys())}")
    desc_column_count = sum(len([col_name for col_name, col_data in cols.items() if col_name != '_table_description']) for cols in desc_lookup.values())
    print(f"DEBUG: Total column descriptions in lookup (excluding table descriptions): {desc_column_count}")


    # 2. Iterate through data CSVs in the specified directory and infer schemas
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv') and filename != os.path.basename(description_csv_path): # Exclude the description CSV itself
            csv_file_path = os.path.join(data_dir, filename)
            # Use filename without extension as table name (e.g., 'application_train')
            table_name = os.path.splitext(filename)[0].lower() # Keep this as filename without extension, lowercased
            
            inferred_column_names_for_comparison_set = set() # NEW: For tracking columns actually found by Spark

            print(f"\nDEBUG: Processing data file: '{filename}' (Mapped to table: '{table_name}')")

            # Infer schema using Spark
            spark_inferred_schema = _get_spark_schema_from_csv(spark, csv_file_path, table_name)
            
            if table_name not in dataset_schema_map:
                dataset_schema_map[table_name] = {}
            
            for column_name, spark_meta in spark_inferred_schema.items():
                print(f"DEBUG:   Found column '{column_name}' in '{table_name}' (Spark inferred type: {spark_meta.get('data_type')})")
                # Initialize metadata for this column
                combined_metadata = {
                    "source": "dataset_metadata",
                    "table_name": table_name,
                    "column_name": column_name,
                    "data_type": spark_meta.get('data_type'),
                    "inferred_from": spark_meta.get('inferred_from'),

                    "description": "", # Default description
                    "is_target": False, # Default
                    "is_id": False # Default
                }


                # Track inferred column names for final comparison
                inferred_column_names_for_comparison_set.add(column_name.lower()) # NEW

                description_found = False # NEW: Flag to track if a description was found

                
                # Overlay/enrich with static description data
                if table_name in desc_lookup and column_name.lower() in desc_lookup[table_name]:
                    print(f"DEBUG:     Enriching '{column_name}' with static description.")
                    static_desc = desc_lookup[table_name][column_name.lower()]
                    combined_metadata['description'] = static_desc.get('description', '')
                    combined_metadata['is_target'] = static_desc.get('is_target', False)
                    combined_metadata['is_id'] = static_desc.get('is_id', False)
                    description_found = True # NEW: Mark that we found a description
                

                # Try to enrich with global ID metadata, only if no table-specific description found yet
                # This comes *after* table-specific to prioritize explicit table descriptions
                if not description_found and column_name.lower() in global_id_metadata: # MODIFY THIS LINE: Add `not description_found`
                    print(f"DEBUG:     Enriching '{column_name}' with global ID description.")
                    id_meta = global_id_metadata[column_name.lower()]
                    combined_metadata['description'] = id_meta.get('description', combined_metadata['description'])
                    combined_metadata['is_id'] = id_meta.get('is_id', combined_metadata['is_id'])
                    description_found = True # NEW

                if not description_found:
                    print(f"DEBUG:     WARNING: No static description found for '{column_name}' in '{table_name}'.")
                    if table_name not in desc_lookup:
                        print(f"DEBUG:       (Reason: Table '{table_name}' not found in description lookup.)")
                    elif column_name.lower() not in desc_lookup[table_name] and column_name.lower() not in global_id_metadata:
                         print(f"DEBUG:       (Reason: Column '{column_name}' not found for table '{table_name}' or globally as an ID.)")

                # Add to dataset_schema_map
                dataset_schema_map[table_name][column_name] = combined_metadata

                # Aggregate for global unique column embedding
                if column_name not in global_unique_columns:
                    global_unique_columns[column_name] = {
                        'column_name': column_name,
                        'data_type': combined_metadata['data_type'],
                        'description': combined_metadata['description'],
                        'is_target': combined_metadata['is_target'],
                        'is_id': combined_metadata['is_id'],
                        'tables_present_in': [table_name],
                        'source': 'dataset_metadata_global'
                    }
                else: # column_name is already in global_unique_columns
                    existing_meta = global_unique_columns[column_name]

                    # Append table if not already present
                    if table_name not in existing_meta['tables_present_in']:
                        existing_meta['tables_present_in'].append(table_name)

                    # Aggregate data types: store as a list of unique types
                    current_data_types = existing_meta['data_type'] if isinstance(existing_meta['data_type'], list) else [existing_meta['data_type']]
                    if combined_metadata['data_type'] not in current_data_types:
                        current_data_types.append(combined_metadata['data_type'])
                    existing_meta['data_type'] = sorted(list(set(current_data_types))) # Ensure unique and sorted

                    # Aggregate descriptions: concatenate if different and not empty
                    if combined_metadata['description'] and combined_metadata['description'] not in existing_meta['description']:
                        existing_meta['description'] = f"{existing_meta['description']} | {combined_metadata['description']}" if existing_meta['description'] else combined_metadata['description']

                    # Logical OR for boolean flags
                    existing_meta['is_target'] = existing_meta['is_target'] or combined_metadata['is_target']
                    existing_meta['is_id'] = existing_meta['is_id'] or combined_metadata['is_id']

    # 3. Prepare documents for Vector DB from global_unique_columns
    print(f"\nDEBUG: Final global unique columns aggregated: {len(global_unique_columns)} records.")
    print("DEBUG: Preparing unique global column definitions for vector database...")
    for col_name, col_meta in global_unique_columns.items():
        tables_str = ", ".join(sorted(col_meta['tables_present_in']))
        text_for_embedding = (
            # Format data_type if it's a list
            f"Column: {col_name}, Data Type(s): {', '.join(col_meta['data_type']) if isinstance(col_meta['data_type'], list) else col_meta['data_type']}, " # MODIFY THIS LINE
   
            f"Description: {col_meta['description']}. Present in tables: {tables_str}."
        )
        stable_id_content = f"global-col-meta-{col_name}-{tables_str}-{col_meta['data_type']}-{col_meta['description']}"
        generated_id = hashlib.sha256(stable_id_content.encode('utf-8')).hexdigest()
        
        # Use the aggregated metadata for embedding
        texts_to_add.append(text_for_embedding)
        metadatas_to_add.append(col_meta) # Use col_meta directly for metadata
        ids_to_add.append(generated_id)

    if texts_to_add: # Ensure there's data to upsert
        print(f"DEBUG: Upserting {len(texts_to_add)} schema entries to vector database.")
        try:
            db_manager.add_documents_batch(texts_to_add, metadatas_to_add, doc_ids=ids_to_add)
            # Verify count after upsert, if possible with Pinecone client (describe_index or fetch)
            # This is hard to do without direct index access here and is more for visual confirmation.
            print(f"DEBUG: Successfully upserted {len(texts_to_add)} records to vector database.")
        except Exception as e:
            print(f"ERROR: Failed to upsert documents to vector database: {e}")
            print("Please check your Pinecone connection and API key.")

    # NEW: Final comparison and reporting of missing descriptions
    missing_described_columns = described_column_names_set - inferred_column_names_for_comparison_set
    if missing_described_columns:
        print(f"\nWARNING: {len(missing_described_columns)} columns described in '{os.path.basename(description_csv_path)}' were NOT found in any actual data CSVs:") # NEW
        for col in sorted(list(missing_described_columns)): # NEW
            print(f"  - {col}") # NEW
    else: # NEW
        print("\nDEBUG: All columns described in the description CSV were successfully matched with inferred schemas.") # NEW


    print("\nDataset metadata ingestion complete.")

    return dataset_schema_map # Return the structured schema map for further use


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
