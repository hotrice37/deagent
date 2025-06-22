# src/utils/data_ingestion_utils.py
# Contains functions related to dataset metadata ingestion.

import os
import pandas as pd
import hashlib
import re
from typing import List, Optional, Dict, Any, Tuple
from pyspark.sql import SparkSession

from src.core.schemas import DatasetMetadata
from src.core.vector_db_manager import VectorDBManager


def _normalize_column_name(column_name: str) -> str:
    """
    Normalizes column names for consistent lookup primarily by converting to uppercase.
    """
    normalized = column_name.upper().strip('_ ').strip() 
    return normalized

def _get_spark_schema_from_csv(spark: SparkSession, csv_file_path: str, table_name: str) -> Dict[str, Dict[str, str]]:
    """
    Infers schema from a single CSV file using PySpark and returns it as a dictionary.
    """
    try:
        df = spark.read.csv(csv_file_path, header=True, inferSchema=True, mode="DROPMALFORMED").limit(100)
        spark_schema = {}
        for field in df.schema:
            spark_schema[field.name] = {
                'data_type': str(field.dataType),
                'inferred_from': 'spark_dynamic_inference'
            }
        return spark_schema
    except Exception as e:
        print(f"WARNING: Could not infer Spark schema for {table_name} from {csv_file_path}: {e}")
        return {}

def _load_and_process_description_csv(description_csv_path: str) -> pd.DataFrame:
    """Loads and preprocesses the HomeCredit_columns_description.csv."""
    if os.path.exists(description_csv_path):
        try:
            description_df = pd.read_csv(description_csv_path, encoding='latin-1')
            description_df.columns = [col.strip() for col in description_df.columns]
            return description_df
        except Exception as e:
            print(f"WARNING: Could not load {description_csv_path}: {e}")
            return pd.DataFrame()
    else:
        print(f"WARNING: Description CSV not found at '{description_csv_path}'. Descriptions will be empty.")
        return pd.DataFrame()

def _prepare_static_descriptions_for_ingestion(description_df: pd.DataFrame, description_doc_id_prefix: str, description_source_type: str) -> Tuple[List[str], List[dict], List[str]]:
    """
    Prepares static column descriptions from the DataFrame for batch ingestion into the vector DB.
    """
    static_description_texts_to_add = []
    static_description_metadatas_to_add = []
    static_description_ids_to_add = []

    if not description_df.empty:
        for index, row in description_df.iterrows():
            raw_table_pattern = str(row.get('Table', '')).strip() if pd.notna(row.get('Table')) else None
            column_name_desc_csv = str(row.get('Row', '')).strip() if pd.notna(row.get('Row')) else None
            description_text = str(row.get('Description', '')).strip() if pd.notna(row.get('Description')) else ""

            if not raw_table_pattern or not column_name_desc_csv:
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
                    "source": description_source_type,
                    "original_table_name": table_name_for_desc,
                    "original_column_name": column_name_desc_csv,
                    "normalized_column_name": normalized_col_name_desc,
                    "description_text": description_text,
                    "is_target": is_target_flag,
                    "is_id": is_id_flag
                }
                stable_id = hashlib.sha256(f"{description_doc_id_prefix}{table_name_for_desc}-{column_name_desc_csv}-{description_text}".encode('utf-8')).hexdigest()
                
                static_description_texts_to_add.append(embedding_text)
                static_description_metadatas_to_add.append(meta)
                static_description_ids_to_add.append(stable_id)
    return static_description_texts_to_add, static_description_metadatas_to_add, static_description_ids_to_add

def _infer_and_enrich_data_schemas(
    spark: SparkSession,
    data_dir: str,
    description_csv_path: str,
    db_manager: VectorDBManager,
    description_source_type: str
) -> Tuple[Dict[str, Dict[str, Any]], set, set, Dict[str, Any]]:
    """
    Infers schemas from data CSVs and enriches them semantically.
    Returns the dataset_schema_map, inferred_and_enriched_column_names_set,
    described_column_names_set, and global_unique_columns.
    """
    dataset_schema_map: Dict[str, Dict[str, Any]] = {}
    global_unique_columns: Dict[str, Dict[str, Any]] = {}
    inferred_and_enriched_column_names_set = set()
    described_column_names_set = set()

    description_df = _load_and_process_description_csv(description_csv_path)
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
            
            spark_inferred_schema = _get_spark_schema_from_csv(spark, csv_file_path, table_name)
            
            if table_name not in dataset_schema_map:
                dataset_schema_map[table_name] = {}
            
            for column_name, spark_meta in spark_inferred_schema.items():
                normalized_col_name = _normalize_column_name(column_name)
                
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
                search_results = db_manager.query_similar_documents(query_text, k=3, filter={"source": description_source_type})
                
                best_match_meta: Optional[Dict[str, Any]] = None
                best_score = -1.0
                
                for doc in search_results:
                    if doc.metadata.get("source") != description_source_type: 
                        continue

                    current_score = doc.metadata.get("score", 0.0)
                    
                    is_perfect_match = (
                        doc.metadata.get("original_table_name") == table_name and
                        doc.metadata.get("normalized_column_name") == normalized_col_name
                    )

                    if is_perfect_match:
                        best_match_meta = doc.metadata
                        break
                    elif current_score > best_score:
                        best_score = current_score
                        best_match_meta = doc.metadata
                
                if best_match_meta:
                    combined_metadata['description'] = best_match_meta.get('description_text', combined_metadata['description'])
                    combined_metadata['is_target'] = best_match_meta.get('is_target', combined_metadata['is_target'])
                    combined_metadata['is_id'] = best_match_meta.get('is_id', combined_metadata['is_id'])

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
    
    return dataset_schema_map, inferred_and_enriched_column_names_set, described_column_names_set, global_unique_columns

def _prepare_global_column_metadata_for_ingestion(global_unique_columns: Dict[str, Any]) -> Tuple[List[str], List[dict], List[str]]:
    """Prepares the aggregated global column definitions for final vector DB ingestion."""
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
    return texts_to_add, metadatas_to_add, ids_to_add

def ingest_dataset_metadata(db_manager: VectorDBManager, spark: SparkSession, data_dir: str, description_csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Ingests dataset metadata by dynamically inferring schema from data CSVs using Spark,
    and then semantically enriching it with descriptions/flags from a static description CSV
    via vector similarity search.
    Finally, it adds this combined metadata to the vector database.
    """
    print(f"Ingesting dataset metadata from '{data_dir}' (Spark) and enriching from '{description_csv_path}' (Semantic Search)...")

    DESCRIPTION_DOC_ID_PREFIX = "static_desc_" 
    DESCRIPTION_SOURCE_TYPE = "static_description_source"

    description_df = _load_and_process_description_csv(description_csv_path)
    static_texts, static_metadatas, static_ids = _prepare_static_descriptions_for_ingestion(
        description_df, DESCRIPTION_DOC_ID_PREFIX, DESCRIPTION_SOURCE_TYPE
    )
            
    if static_texts:
        user_consent_static = input(f"\nAbout to upsert {len(static_texts)} temporary static description entries for semantic lookup. Do you want to proceed with Phase 1 ingestion? (yes/no): ").lower().strip()
        if user_consent_static == 'yes':
            try:
                db_manager.add_documents_batch(static_texts, static_metadatas, doc_ids=static_ids)
            except Exception as e:
                print(f"ERROR: Failed to upsert static descriptions to vector database: {e}")
                print("Please check your Pinecone connection and API key. Aborting dataset metadata ingestion.")
                return {}
        else:
            print(f"Upsert of static descriptions skipped by user request. Aborting dataset metadata ingestion to save write units.")
            return {}
    else:
        print("Aborting dataset metadata ingestion as no static descriptions were found or user denied.")
        return {}

    dataset_schema_map, inferred_and_enriched_column_names_set, described_column_names_set, global_unique_columns = \
        _infer_and_enrich_data_schemas(spark, data_dir, description_csv_path, db_manager, DESCRIPTION_SOURCE_TYPE)
    
    texts_to_add, metadatas_to_add, ids_to_add = _prepare_global_column_metadata_for_ingestion(global_unique_columns)

    if texts_to_add:
        user_consent_global = input(f"\nAbout to upsert {len(texts_to_add)} inferred global dataset metadata entries to Pinecone index '{db_manager.index_name}'. Do you want to proceed? (yes/no): ").lower().strip()
        if user_consent_global == 'yes':
            try:
                db_manager.add_documents_batch(texts_to_add, metadatas_to_add, doc_ids=ids_to_add)
            except Exception as e:
                print(f"ERROR: Failed to upsert semantically enriched documents to vector database: {e}")
                print("Please check your Pinecone connection and API key.")
        else:
            print(f"Upsert of {len(texts_to_add)} inferred global dataset metadata entries skipped by user request.")
    else:
        pass

    try:
        db_manager.index.delete(filter={"source": DESCRIPTION_SOURCE_TYPE})
    except Exception as e:
        print(f"ERROR: Failed to delete temporary '{DESCRIPTION_SOURCE_TYPE}' documents: {e}")
        print("Please check your Pinecone connection and permissions.")

    missing_described_columns = described_column_names_set - inferred_and_enriched_column_names_set
    if missing_described_columns:
        print(f"\nWARNING: {len(missing_described_columns)} columns described in '{os.path.basename(description_csv_path)}' were NOT found in any actual data CSVs (or their normalized name didn't match):")
        for col in sorted(list(missing_described_columns)):
            print(f"  - {col}")
    else:
        pass

    print("\nDataset metadata ingestion complete.")
    return dataset_schema_map

