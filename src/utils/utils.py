# src/utils/utils.py
# Contains utility functions used across different modules, such as
# dataset metadata ingestion and approved task ingestion, and JSON extraction.

# General Imports
import os
import pandas as pd
import hashlib
import uuid
import re
import json # Explicitly imported now for json.dumps in embedding text
from typing import List, Optional, Dict, Any, Tuple
from pyspark.sql import SparkSession

# Project Imports
from src.core.schemas import DatasetMetadata # Still needed for Pydantic (though not directly used here)
from src.core.vector_db_manager import VectorDBManager # For semantic search


# Helper function for robust JSON extraction from LLM output
def extract_json_from_llm_output(llm_output: str) -> str:
    """
    Extracts a JSON string from LLM output, robustly handling markdown code fences
    and attempting to find JSON even if not perfectly formatted.
    """
    cleaned_output = llm_output.strip()

    # 1. Try to find JSON within a markdown code block (```json or ```)
    match = re.search(r"```json\s*\n(.*?)```", cleaned_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    match = re.search(r"```\s*\n(.*?)```", cleaned_output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. If no markdown fence, try to find the outermost JSON object by curly braces
    # This is a heuristic and assumes the JSON is the dominant or intended part of the output.
    first_brace = cleaned_output.find('{')
    last_brace = cleaned_output.rfind('}')

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        # Extract substring that is potentially the JSON
        potential_json_string = cleaned_output[first_brace : last_brace + 1]
        try:
            # Attempt to parse it to confirm it's valid JSON
            json.loads(potential_json_string)
            return potential_json_string.strip()
        except json.JSONDecodeError:
            # If it's not valid JSON, fall through to returning original cleaned_output
            pass

    # 3. If all else fails, return the original cleaned output (may still be invalid JSON)
    return cleaned_output

def _normalize_column_name(column_name: str) -> str:
    """
    Normalizes column names for consistent lookup primarily by converting to uppercase.
    This function now focuses on consistent casing and basic cleanup.
    """
    # Convert to uppercase and strip any leading/trailing spaces or underscores
    normalized = column_name.upper().strip('_ ').strip() 
    return normalized

def _get_spark_schema_from_csv(spark: SparkSession, csv_file_path: str, table_name: str) -> Dict[str, Dict[str, str]]:
    """
    Infers schema from a single CSV file using PySpark and returns it as a dictionary.
    :param spark: The active SparkSession.
    :param csv_file_path: Full path to the CSV file.
    :param table_name: The name to assign to this table (e.g., 'application_train').
    :return: A dictionary of column names to their inferred data types and basic metadata.
             Format: {column_name: {'data_type': 'string', 'inferred_from': 'spark'}}
    """
    # print(f"DEBUG: Inferring schema for '{table_name}' from '{csv_file_path}' using Spark...")
    try:
        # Read only a small sample to infer schema efficiently
        df = spark.read.csv(csv_file_path, header=True, inferSchema=True, mode="DROPMALFORMED").limit(100)
        spark_schema = {}
        for field in df.schema:
            spark_schema[field.name] = {
                'data_type': str(field.dataType),
                'inferred_from': 'spark_dynamic_inference'
            }
        # print(f"DEBUG: Successfully inferred schema for '{table_name}'. Found {len(spark_schema)} columns.")
        return spark_schema
    except Exception as e:
        print(f"WARNING: Could not infer Spark schema for {table_name} from {csv_file_path}: {e}")
        return {}


def ingest_dataset_metadata(db_manager: VectorDBManager, spark: SparkSession, data_dir: str, description_csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Ingests dataset metadata by dynamically inferring schema from data CSVs using Spark,
    and then semantically enriching it with descriptions/flags from a static description CSV
    via vector similarity search.
    Finally, it adds this combined metadata to the vector database.

    :param db_manager: The VectorDBManager instance for the metadata index.
    :param spark: The active SparkSession.
    :param data_dir: Path to the directory containing all data CSVs (e.g., 'application_train.csv', 'bureau.csv').
    :param description_csv_path: Path to the HomeCredit_columns_description.csv file.
    :return: A structured dictionary mapping table names to their columns and their enriched metadata.
             Format: {table_name: {column_name: {description, data_type, is_target, is_id, inferred_from}}}
    """
    print(f"Ingesting dataset metadata from '{data_dir}' (Spark) and enriching from '{description_csv_path}' (Semantic Search)...")

    # This is the final structured schema map for the Planner Agent
    dataset_schema_map: Dict[str, Dict[str, Any]] = {}
    # This will hold aggregated unique columns for vector DB storage
    global_unique_columns: Dict[str, Dict[str, Any]] = {}

    # --- Phase 1: Ingest Static Descriptions for Semantic Search Lookup ---
    # These documents are TEMPORARILY upserted to the DB for semantic enrichment in Phase 2,
    # and then deleted after Phase 3.
    # print("\nDEBUG: Phase 1 - Ingesting static column descriptions for semantic search lookup (TEMPORARILY upserting to DB)...") # Commented out as requested
    description_df = pd.DataFrame()
    static_description_texts_to_add = []
    static_description_metadatas_to_add = []
    static_description_ids_to_add = []

    # Use a specific prefix and source type for these temporary documents
    DESCRIPTION_DOC_ID_PREFIX = "static_desc_" 
    DESCRIPTION_SOURCE_TYPE = "static_description_source"

    if os.path.exists(description_csv_path):
        try:
            description_df = pd.read_csv(description_csv_path, encoding='latin-1')
            # description_df.columns = [col.strip() for col in description_df.columns]
            # print(f"DEBUG: Description CSV loaded. Total rows: {len(description_df)}")
            # print(f"DEBUG: First 5 rows of description_df:\n{description_df.head().to_string()}")

            # print(f"DEBUG: Successfully loaded column descriptions from {description_csv_path}.")
        except Exception as e:
            print(f"WARNING: Could not load or ingest {description_csv_path} for semantic enrichment: {e}")
            print("         Proceeding with only dynamically inferred schema, descriptions will be empty.")
    else:
        print(f"WARNING: Description CSV not found at '{description_csv_path}'. Descriptions will be empty.")

    # Group descriptions by table for easier lookup
    # print("\nDEBUG: Building description lookup map from static file for Phase 1 ingestion...")
    if not description_df.empty:
        for index, row in description_df.iterrows():
            raw_table_pattern = str(row.get('Table', '')).strip() if pd.notna(row.get('Table')) else None
            column_name_desc_csv = str(row.get('Row', '')).strip() if pd.notna(row.get('Row')) else None
            description_text = str(row.get('Description', '')).strip() if pd.notna(row.get('Description')) else ""

            if not raw_table_pattern or not column_name_desc_csv:
                # print(f"DEBUG: Skipping row in description CSV: Table '{raw_table_pattern}', Column '{column_name_desc_csv}' (Likely table-level description).")
                continue
            
            normalized_col_name_desc = _normalize_column_name(column_name_desc_csv)
            
            target_table_names_for_desc: List[str] = []
            if raw_table_pattern == "application_{train|test}.csv":
                target_table_names_for_desc = ["application_train", "application_test"]
            else:
                standardized_name = re.sub(r'\.csv$', '', raw_table_pattern.lower())
                if standardized_name:
                    target_table_names_for_desc.append(standardized_name)

            for table_name_for_desc in target_table_names_for_desc:
                is_target_flag = (column_name_desc_csv.upper() == 'TARGET' and table_name_for_desc == 'application_train')
                is_id_flag = ('SK_ID' in column_name_desc_csv.upper() or 'SK_BUREAU_ID' in column_name_desc_csv.upper())
                
                embedding_text = (
                    f"Description for table '{table_name_for_desc}', column '{column_name_desc_csv}': "
                    f"{description_text}"
                )
                
                meta = {
                    "source": DESCRIPTION_SOURCE_TYPE,
                    "original_table_name": table_name_for_desc,
                    "original_column_name": column_name_desc_csv,
                    "normalized_column_name": normalized_col_name_desc,
                    "description_text": description_text,
                    "is_target": is_target_flag,
                    "is_id": is_id_flag
                }
                stable_id = hashlib.sha256(f"{DESCRIPTION_DOC_ID_PREFIX}{table_name_for_desc}-{column_name_desc_csv}-{description_text}".encode('utf-8')).hexdigest()
                
                static_description_texts_to_add.append(embedding_text)
                static_description_metadatas_to_add.append(meta)
                static_description_ids_to_add.append(stable_id)
            
    if static_description_texts_to_add:
        user_consent_static = input(f"\nAbout to upsert {len(static_description_texts_to_add)} temporary static description entries for semantic lookup. Do you want to proceed with Phase 1 ingestion? (yes/no): ").lower().strip()
        if user_consent_static == 'yes':
            # print(f"DEBUG: Upserting {len(static_description_texts_to_add)} static descriptions to vector DB (source: {DESCRIPTION_SOURCE_TYPE}).")
            try:
                db_manager.add_documents_batch(static_description_texts_to_add, static_description_metadatas_to_add, doc_ids=static_description_ids_to_add)
                # print(f"DEBUG: Finished Phase 1. Static descriptions ingested (temporarily).")
            except Exception as e:
                print(f"ERROR: Failed to upsert static descriptions to vector database: {e}")
                print("Please check your Pinecone connection and API key. Aborting dataset metadata ingestion.")
                return dataset_schema_map
        else:
            print(f"Upsert of static descriptions skipped by user request. Aborting dataset metadata ingestion to save write units.")
            return dataset_schema_map
    else:
        # print("DEBUG: No static column descriptions found to process in Phase 1.")
        print("Aborting dataset metadata ingestion as no static descriptions were found or user denied.")
        return dataset_schema_map


    # --- Phase 2: Dynamically Infer Schemas and Semantically Enrich ---
    # print("\nDEBUG: Phase 2 - Inferring schemas from data CSVs and enriching metadata...")
    inferred_and_enriched_column_names_set = set() 
    described_column_names_set = set()
    if not description_df.empty:
        for index, row in description_df.iterrows():
            raw_table_pattern = str(row.get('Table', '')).strip() if pd.notna(row.get('Table')) else None
            column_name_desc_csv = str(row.get('Row', '')).strip() if pd.notna(row.get('Row')) else None
            
            if not raw_table_pattern or not column_name_desc_csv:
                continue
            described_column_names_set.add(_normalize_column_name(column_name_desc_csv))


    for filename in os.listdir(data_dir):
        if filename.endswith('.csv') and filename != os.path.basename(description_csv_path):
            csv_file_path = os.path.join(data_dir, filename)
            table_name = os.path.splitext(filename)[0].lower()
            
            # print(f"\nDEBUG: Processing data file: '{filename}' (Mapped to table: '{table_name}')")

            spark_inferred_schema = _get_spark_schema_from_csv(spark, csv_file_path, table_name)
            
            if table_name not in dataset_schema_map:
                dataset_schema_map[table_name] = {}
            
            for column_name, spark_meta in spark_inferred_schema.items():
                normalized_col_name = _normalize_column_name(column_name)
                # print(f"DEBUG:   Found column '{column_name}' (Normalized: '{normalized_col_name}') in '{table_name}' (Spark inferred type: {spark_meta.get('data_type')})")
                
                combined_metadata = {
                    "source": "spark_inferred_schema",
                    "table_name": table_name,
                    "column_name": column_name,
                    "normalized_column_name": normalized_col_name,
                    "data_type": spark_meta.get('data_type'),
                    "inferred_from": spark_meta.get('inferred_from'),
                    "description": "",
                    "is_target": False,
                    "is_id": False
                }
                
                inferred_and_enriched_column_names_set.add(normalized_col_name)

                query_text = f"description for table {table_name} column {column_name}"
                search_results = db_manager.query_similar_documents(query_text, k=3, filter={"source": DESCRIPTION_SOURCE_TYPE})
                
                best_match_meta: Optional[Dict[str, Any]] = None
                best_score = -1.0
                
                for doc in search_results:
                    if doc.metadata.get("source") != DESCRIPTION_SOURCE_TYPE: 
                        continue

                    current_score = doc.metadata.get("score", 0.0)
                    
                    is_perfect_match = (
                        doc.metadata.get("original_table_name") == table_name and
                        doc.metadata.get("normalized_column_name") == normalized_col_name
                    )

                    if is_perfect_match:
                        # print(f"DEBUG:     Semantic search found PERFECT match for '{column_name}' in '{table_name}'. Score: {current_score}")
                        best_match_meta = doc.metadata
                        break
                    elif current_score > best_score:
                        best_score = current_score
                        best_match_meta = doc.metadata
                        # print(f"DEBUG:     Semantic search found BEST non-perfect match for '{column_name}' in '{table_name}'. Score: {current_score}")
                
                if best_match_meta:
                    # print(f"DEBUG:     Enriching '{column_name}' with semantic description from best match.")
                    combined_metadata['description'] = best_match_meta.get('description_text', combined_metadata['description'])
                    combined_metadata['is_target'] = best_match_meta.get('is_target', combined_metadata['is_target'])
                    combined_metadata['is_id'] = best_match_meta.get('is_id', combined_metadata['is_id'])
                else:
                    # print(f"DEBUG:     No strong semantic description match found for '{column_name}' in '{table_name}'.")
                    pass

                dataset_schema_map[table_name][column_name] = combined_metadata

                if normalized_col_name not in global_unique_columns:
                    global_unique_columns[normalized_col_name] = {
                        'column_name': column_name,
                        'normalized_column_name': normalized_col_name,
                        'data_type': combined_metadata['data_type'],
                        'description': combined_metadata['description'],
                        'is_target': combined_metadata['is_target'],
                        'is_id': combined_metadata['is_id'],
                        'tables_present_in': [table_name],
                        'source': 'dataset_metadata_global_semantic_enriched'
                    }
                else:
                    existing_meta = global_unique_columns[normalized_col_name]

                    if table_name not in existing_meta['tables_present_in']:
                        existing_meta['tables_present_in'].append(table_name)

                    current_data_types = existing_meta['data_type'] if isinstance(existing_meta['data_type'], list) else [existing_meta['data_type']]
                    if combined_metadata['data_type'] not in current_data_types:
                        current_data_types.append(combined_metadata['data_type'])
                    existing_meta['data_type'] = sorted(list(set(current_data_types)))

                    if combined_metadata['description'] and combined_metadata['description'] not in existing_meta['description']:
                        existing_meta['description'] = f"{existing_meta['description']} | {combined_metadata['description']}" if existing_meta['description'] else combined_metadata['description']

                    existing_meta['is_target'] = existing_meta['is_target'] or combined_metadata['is_target']
                    existing_meta['is_id'] = existing_meta['is_id'] or combined_metadata['is_id']

    # 3. Prepare documents for Vector DB from global_unique_columns (Final Documents for Planner's search)
    # print(f"\nDEBUG: Final global unique columns aggregated: {len(global_unique_columns)} records.")
    # print("DEBUG: Preparing unique global column definitions for vector database...")
    
    texts_to_add = []
    metadatas_to_add = []
    ids_to_add = []

    for normalized_col_name, col_meta in global_unique_columns.items():
        tables_str = ", ".join(sorted(col_meta['tables_present_in']))
        data_types_str = ', '.join(col_meta['data_type']) if isinstance(col_meta['data_type'], list) else col_meta['data_type']

        text_for_embedding = (
            f"Column: {col_meta.get('column_name', normalized_col_name)}, Data Type(s): {data_types_str}, "
            f"Description: {col_meta['description']}. Present in tables: {tables_str}."
        )
        stable_id_content = f"global-col-meta-{normalized_col_name}-{tables_str}-{data_types_str}-{col_meta['description']}"
        generated_id = hashlib.sha256(stable_id_content.encode('utf-8')).hexdigest()
        
        texts_to_add.append(text_for_embedding)
        metadatas_to_add.append(col_meta)
        ids_to_add.append(generated_id)

    if texts_to_add:
        user_consent_global = input(f"\nAbout to upsert {len(texts_to_add)} inferred global dataset metadata entries to Pinecone index '{db_manager.index_name}'. Do you want to proceed? (yes/no): ").lower().strip()
        if user_consent_global == 'yes':
            # print(f"DEBUG: Upserting {len(texts_to_add)} semantically enriched schema entries to vector database (source: spark_inferred_schema).")
            try:
                db_manager.add_documents_batch(texts_to_add, metadatas_to_add, doc_ids=ids_to_add)
                # print(f"DEBUG: Successfully upserted {len(texts_to_add)} records to vector database.")
            except Exception as e:
                print(f"ERROR: Failed to upsert semantically enriched documents to vector database: {e}")
                print("Please check your Pinecone connection and API key.")
        else:
            print(f"Upsert of {len(texts_to_add)} inferred global dataset metadata entries skipped by user request.")
    else:
        # print("DEBUG: No semantically enriched schema entries to upsert for global metadata.")
        pass

    # --- NEW: Cleanup Phase ---
    # Delete the temporary static_description_source documents from the index.
    # print(f"\nDEBUG: Cleanup Phase - Deleting temporary '{DESCRIPTION_SOURCE_TYPE}' documents from the index.")
    try:
        db_manager.index.delete(filter={"source": DESCRIPTION_SOURCE_TYPE})
        # print(f"DEBUG: Successfully deleted all documents with source '{DESCRIPTION_SOURCE_TYPE}'.")
    except Exception as e:
        print(f"ERROR: Failed to delete temporary '{DESCRIPTION_SOURCE_TYPE}' documents: {e}")
        print("Please check your Pinecone connection and permissions.")

    # Final comparison and reporting of missing descriptions
    missing_described_columns = described_column_names_set - inferred_and_enriched_column_names_set
    if missing_described_columns:
        print(f"\nWARNING: {len(missing_described_columns)} columns described in '{os.path.basename(description_csv_path)}' were NOT found in any actual data CSVs (or their normalized name didn't match):")
        for col in sorted(list(missing_described_columns)):
            print(f"  - {col}")
    else:
        # print("\nDEBUG: All columns described in the description CSV were successfully matched with inferred schemas (or their normalized name matched).")
        pass

    print("\nDataset metadata ingestion complete.")
    return dataset_schema_map


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

