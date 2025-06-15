import os
from dotenv import load_dotenv
import json
import uuid
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import time # Import time for waiting on index readiness
import pandas as pd # Import pandas to read CSV
import hashlib # Import hashlib for consistent ID generation
import re # Import re for regular expressions

# LangChain Imports
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser # Import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
# Pinecone Import - Using the new Pinecone client
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeApiException # Specific exception from pinecone.exceptions
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException

# AutoGen Imports
from autogen import UserProxyAgent, AssistantAgent, ConversableAgent # Import ConversableAgent

load_dotenv() # Load environment variables from .env file

# --- Configuration ---
# Pinecone Configuration for Serverless Index
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "etl-backlog-production-index" # Production index name for dataset metadata
PINECONE_APPROVED_TASKS_INDEX_NAME = "etl-approved-tasks-index" # New index for approved ETL tasks
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws") # Default to aws if not set
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1") # Default to us-east-1 if not set

# Path to the dataset columns description CSV (e.g., from Home Credit competition)
# In a real production environment, this CSV might be part of your data lake/warehouse
# or accessed via a dedicated data catalog service.
COLUMNS_DESCRIPTION_CSV = os.path.join(os.path.dirname(__file__), "home-credit-default-risk", "HomeCredit_columns_description.csv")

# Ensure the CSV file exists for initial metadata loading
if not os.path.exists(COLUMNS_DESCRIPTION_CSV):
    print(f"ERROR: Column description CSV not found at expected path: {COLUMNS_DESCRIPTION_CSV}")
    print("Please ensure 'HomeCredit_columns_description.csv' is inside a 'home-credit-default-risk' subdirectory next to your script.")
    exit("Exiting: HomeCredit_columns_description.csv is missing or in wrong directory.")


# Check if API Key and Cloud/Region are set for Pinecone
if not PINECONE_API_KEY or not PINECONE_CLOUD or not PINECONE_REGION:
    print("WARNING: One or more Pinecone configuration variables not set.")
    print("Please set PINECONE_API_KEY, PINECONE_CLOUD, and PINECONE_REGION ")
    print("(e.env., in a .env file) to connect to Pinecone.")
    print("Example .env content for serverless:")
    print("PINECONE_API_KEY=YOUR_KEY")
    print("PINECONE_CLOUD=aws")
    print("PINECONE_REGION=us-east-1")
    exit("Exiting: Pinecone configuration is incomplete.")


# Configuration for Ollama LLM and Embedding Models
# Ensure Ollama server is running locally and these models are pulled.
# e.env., `ollama run llama3`, `ollama run nomic-embed-text`
OLLAMA_LLM_MODEL_NAME = "llama3" # Updated to llama3
OLLAMA_EMBEDDING_MODEL_NAME = "nomic-embed-text"

# --- DEBUG FLAG ---
# Set to True for debugging LLM hangs. Set to False for normal operation.
# When True, it will print raw LLM output before parsing and use minimal context for initial parse.
DEBUG_LLM_CALL = False # Set to False for normal operation


# --- 1. Pydantic Schemas for Structured Output ---

class DataCleaningStep(BaseModel):
    """Defines a single data cleaning or preprocessing operation."""
    type: str = Field(..., description="Type of cleaning operation (e.g., 'imputation', 'outlier_removal', 'feature_engineering')")
    details: dict = Field(..., description="Specific parameters for the cleaning operation (e.g., {'strategy': 'mean', 'columns': ['age']})")

class ScoringModel(BaseModel):
    """Defines the details of a scoring or prediction model."""
    name: str = Field(..., description="Name of the scoring model (e.g., 'Logistic Regression', 'XGBoost', 'Generic ML Model')")
    objective: str = Field(..., description="The objective of the scoring (e.g., 'predict repayment likelihood', 'classify loan default')")
    target_column: str = Field(..., description="The target column for the scoring model (e.g., 'TARGET', 'LOAN_STATUS')")
    features: Optional[List[str]] = Field(None, description="Optional: List of features to use for the model. If not specified, all relevant features after ETL will be considered.")

class JoinOperation(BaseModel):
    """Defines a data join operation between two table_name."""
    left_table: str = Field(..., description="Name of the left table for the join (e.g., 'application_train')")
    right_table: str = Field(..., description="Name of the right table for the join (e.g., 'bureau')")
    join_type: Literal["inner", "left", "right", "outer"] = Field(..., description="Type of join (e.g., 'left', 'inner'). Defaults to 'left' if not specified.")
    on_columns: List[str] = Field(..., description="List of columns to join on (e.g., ['SK_ID_CURR'])")
    description: Optional[str] = Field(None, description="A brief description of the join operation.")

class ETLTaskDefinition(BaseModel):
    """The overarching schema for a complete ETL pipeline task."""
    pipeline_name: str = Field(..., description="A descriptive name for the ETL pipeline (e.g., 'HomeCredit_ApplicantScoring_Pipeline')")
    main_goal: str = Field(..., description="The overarching objective of the ETL pipeline (e.g., 'predict repayment likelihood for loan applicants')")
    initial_tables: List[str] = Field([], description="List of initial tables required from the dataset (e.g., ['application_train', 'bureau', 'bureau_balance'])")
    join_operations: List[JoinOperation] = Field([], description="List of join operations to perform in sequence.")
    data_cleaning_steps: List[DataCleaningStep] = Field([], description="List of data cleaning and preprocessing steps to apply.")
    scoring_model: Optional[ScoringModel] = Field(None, description="Details about the scoring model to be applied, if any.")
    output_format: Optional[str] = Field("dataframe", description="Desired output format after ETL (e.g., 'dataframe', 'csv', 'parquet').")
    output_location: Optional[str] = Field(None, description="Where to save the final output (e.g., 's3://my-bucket/processed_data/').")

class DatasetMetadata(BaseModel):
    """Schema for individual pieces of dataset metadata, useful for vector embeddings."""
    table_name: Optional[str] = None
    column_name: Optional[str] = None
    description: str
    data_type: Optional[str] = None
    example_values: Optional[List[str]] = None
    is_target: bool = False
    is_id: bool = False


# --- 2. Vector Database Manager (Pinecone Serverless) ---

class VectorDBManager:
    """Manages the interaction with the Pinecone serverless vector database."""
    def __init__(self, index_name: str, cloud: str, region: str, api_key: str, embedding_model_name: str):
        """
        Initializes the VectorDBManager with Pinecone.
        :param index_name: The name of the Pinecone index.
        :param cloud: The cloud provider for the serverless index (e.g., 'aws', 'gcp').
        :param region: The specific region for the serverless index (e.g., 'us-east-1').
        :param api_key: Your Pinecone API key.
        :param embedding_model_name: The Ollama model name for embeddings.
        """
        self.embeddings_model = OllamaEmbeddings(model=embedding_model_name)
        self.pinecone = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.cloud = cloud
        self.region = region

        # Determine embedding dimension dynamically
        try:
            temp_ollama_embeddings = OllamaEmbeddings(model=embedding_model_name)
            sample_embedding = temp_ollama_embeddings.embed_query("test")
            self.embedding_dimension = len(sample_embedding)
            print(f"Embedding dimension detected for {index_name}: {self.embedding_dimension}")
        except Exception as e:
            raise RuntimeError(f"Failed to get embedding dimension from Ollama: {e}. Ensure Ollama is running and model '{embedding_model_name}' is pulled.")

        # Index creation logic with try-except for graceful handling
        if index_name not in self.pinecone.list_indexes():
            print(f"Attempting to create Pinecone serverless index '{index_name}' in {cloud} {region}...")
            try:
                self.pinecone.create_index(
                    name=index_name,
                    dimension=self.embedding_dimension,
                    metric='cosine',
                    spec=ServerlessSpec(cloud=cloud, region=region)
                )
                print(f"Pinecone serverless index '{index_name}' created successfully.")
            except PineconeApiException as e:
                # Check for "already exists" message or 409 status code
                if e.status == 409 or "already exists" in str(e).lower() or "ALREADY_EXISTS" in str(e):
                    print(f"Pinecone index '{index_name}' already exists (caught during creation attempt).")
                else:
                    raise e # Re-raise if it's another type of error
            except Exception as e:
                print(f"An unexpected error occurred during index creation: {e}")
                raise e
        else:
            print(f"Pinecone index '{index_name}' already exists, skipping creation.")

        # Always wait for the index to be ready, whether newly created or pre-existing
        print(f"Waiting for index '{index_name}' to be ready...")
        # Add a timeout for robustness in production
        timeout_seconds = 300 # 5 minutes
        start_time = time.time()
        while not self.pinecone.describe_index(index_name).status['ready']:
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"Pinecone index '{index_name}' did not become ready within {timeout_seconds} seconds.")
            time.sleep(1)
        print(f"Index '{index_name}' is ready.")

        self.index = self.pinecone.Index(index_name)

    def add_documents_batch(self, texts: List[str], metadatas: List[dict] = None, doc_ids: Optional[List[str]] = None) -> List[str]:
        """
        Adds a batch of documents to the Pinecone index.
        :param texts: The list of text contents to embed and upsert.
        :param metadatas: A list of metadata dictionaries, corresponding to `texts`.
        :param doc_ids: Optional list of specific IDs to use for the documents. If provided, must match length of texts.
                        If not provided, UUIDs will be generated.
        :return: A list of IDs of the upserted vectors.
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)

        if doc_ids and len(doc_ids) != len(texts):
            raise ValueError("If 'doc_ids' are provided, their length must match the length of 'texts'.")

        embeddings = self.embeddings_model.embed_documents(texts)

        vectors_to_upsert = []
        ids = []
        for i, (text_content, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            # Use provided ID if available, otherwise generate UUID
            current_id = doc_ids[i] if doc_ids else str(uuid.uuid4())
            ids.append(current_id)
            combined_metadata = {"text_content": text_content, **metadata}
            vectors_to_upsert.append((current_id, embedding, combined_metadata))

        # Upsert vectors to Pinecone in batches to handle large datasets efficiently
        batch_size = 100 # Adjust batch size based on Pinecone limits and network performance
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch)
        print(f"Upserted {len(vectors_to_upsert)} documents to Pinecone index '{self.index_name}'.")
        return ids

    def query_similar_documents(self, query_text: str, k: int = 3) -> List[Document]:
        """
        Queries the Pinecone index for documents semantically similar to the query text.
        :param query_text: The text to query for.
        :param k: The number of top similar documents to retrieve.
        :return: A list of LangChain Document objects, each containing page_content and metadata.
        """
        print(f"DEBUG_VDB: Querying Pinecone index '{self.index_name}' for relevant context for: '{query_text}'...")
        # Add debug print before embedding call
        print(f"DEBUG_VDB: Generating embedding for query text from '{self.index_name}'...")
        query_embedding = self.embeddings_model.embed_query(query_text)
        print(f"DEBUG_VDB: Embedding generated for query text from '{self.index_name}'.")

        response = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )

        results = []
        for match in response.matches:
            page_content = match.metadata.get("text_content", "")
            metadata = {k: v for k, v in match.metadata.items() if k != "text_content"}
            results.append(Document(page_content=page_content, metadata=metadata))

        print(f"DEBUG_VDB: Found {len(results)} similar documents in index '{self.index_name}'.")
        return results

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

    # Assuming CSV columns are 'Table', 'Row', 'Description', 'Special' (if 'Special' exists, it's ignored)
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


# --- Helper function for robust JSON extraction from LLM output ---
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


# --- 3. Parser Agent ---

class ParserAgent:
    """
    An AI agent that parses natural language requests into structured ETL task definitions,
    leveraging context from a vector database.
    """
    def __init__(self, vector_db_manager_metadata: VectorDBManager, vector_db_manager_approved_tasks: VectorDBManager, llm_model_name=OLLAMA_LLM_MODEL_NAME, temperature=0):
        """
        Initializes the ParserAgent.
        :param vector_db_manager_metadata: An instance of VectorDBManager for dataset metadata.
        :param vector_db_manager_approved_tasks: An instance of VectorDBManager for approved tasks.
        :param llm_model_name: The name of the Ollama model to use for text generation (e.g., 'llama2', 'mistral').
        :param temperature: The creativity temperature for the LLM. 0 for deterministic.
        """
        self.llm = OllamaLLM(model=llm_model_name, temperature=temperature, request_timeout=300.0, base_url="http://localhost:11434", verbose=True) # Increased timeout
        self.parser = PydanticOutputParser(pydantic_object=ETLTaskDefinition)
        self.str_parser = StrOutputParser() # Used when DEBUG_LLM_CALL is True
        self.vector_db_manager_metadata = vector_db_manager_metadata
        self.vector_db_manager_approved_tasks = vector_db_manager_approved_tasks

        self.prompt_template = PromptTemplate(
            template="""You are an AI assistant specialized in parsing natural language requests for ETL pipelines and converting them into a structured JSON format.
Your task is to analyze the user's request and any provided historical context or relevant dataset metadata, and then output a JSON object that strictly adheres to the provided Pydantic schema.

**Instructions:**
- Identify the main goal of the pipeline.
- List all initial tables mentioned or clearly implied. Leverage the context for correct table names (e.g., 'application_train', 'bureau') and for inferring relevant tables not explicitly named.
- Deconstruct any join operations, specifying `left_table`, `right_table`, `join_type` (default to 'left' if not specified), and `on_columns`. Use context for correct column names if possible.
- Infer data cleaning or feature engineering steps if described (e.g., "cleaning the data", "impute missing values"). Provide a generic `type` and `details` dictionary for these.
- If a scoring or prediction objective is mentioned, specify the `scoring_model`'s `name` (e.g., 'Logistic Regression', 'XGBoost', or 'Generic ML Model' if unspecified), its `objective`, and `target_column`. Use context to accurately identify the target column (e.e.g., 'TARGET').
- Ensure all fields in the JSON adhere to the types and constraints defined in the schema.
- Do NOT include any explanations or conversational text outside the JSON block.

**Pydantic Schema:**
{format_instructions}

**Relevant Context (from similar past tasks and dataset metadata):**
{context}

**User Request:**
{request}

**JSON Output (strict JSON, no markdown code block or extra text):**
""",
            input_variables=["request", "context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def parse_request(self, user_request: str, context_string: str) -> dict: # Modified: accepts context_string
        """
        Parses a natural language user request into a structured ETL task definition,
        using provided context.
        :param user_request: The natural language backlog item.
        :param context_string: The combined and formatted context from vector databases.
        :return: A dictionary representing the parsed ETLTaskDefinition, or an error dictionary.
        """
        print("DEBUG_FLOW: Starting parse_request method (now accepts context).")
        
        # Use the provided context string
        current_context_for_llm = context_string
        
        if DEBUG_LLM_CALL: # This block will only execute if DEBUG_LLM_CALL is True
            print("DEBUG_LLM_CALL is True: Using provided context string for LLM invocation.")
            print(f"DEBUG_LLM_CALL: Context snippet: '{current_context_for_llm[:500]}...'") # Print a snippet if very long
        else: # This block executes if DEBUG_LLM_CALL is False
            print("DEBUG_LLM_CALL is False: Using full context for LLM invocation.")
            print(f"DEBUG_FLOW: Context string length for LLM: {len(current_context_for_llm)} characters.")


        # Step 2: Formulate the prompt with injected context and invoke the LLM
        chain = self.prompt_template | self.llm # The LLM will return a raw string
        
        print("\nDEBUG: Attempting to invoke LLM chain for initial parse...")
        try:
            # The raw_llm_string_output will be a string from OllamaLLM
            raw_llm_string_output = chain.invoke({"request": user_request, "context": current_context_for_llm}, config={"timeout": 300.0}) # Increased timeout
            print("DEBUG: LLM chain invocation for initial parse completed.")
            
            # Extract clean JSON string from LLM's raw output
            cleaned_json_string = extract_json_from_llm_output(raw_llm_string_output)
            
            if DEBUG_LLM_CALL:
                print("\n--- DEBUG: Raw LLM Output (after stripping fences) ---")
                print(cleaned_json_string)
                print("----------------------------------------------------------")

            # Now, attempt to parse the cleaned string with PydanticOutputParser
            parsed_output_pydantic = self.parser.parse(cleaned_json_string)
            print("DEBUG_FLOW: Pydantic parsing successful.")
            return parsed_output_pydantic.model_dump()
            
        except TimeoutError:
            print(f"Error: LLM parsing timed out after 300 seconds.") # Updated timeout message
            return {"error": "LLM parsing timed out", "details": "The model took too long to generate a response."}
        except OutputParserException as e:
            print(f"Error parsing LLM output: {e}")
            return {"error": "Failed to parse LLM output according to schema", "details": str(e), "raw_llm_output": cleaned_json_string if 'cleaned_json_string' in locals() else "Not available"}
        except json.JSONDecodeError as e: # This handles cases where cleaned_json_string is not valid JSON
            print(f"Error: Cleaned LLM output is not valid JSON: {e}")
            return {"error": "Cleaned LLM output is not valid JSON, cannot proceed.", "details": str(e), "raw_llm_output": cleaned_json_string if 'cleaned_json_string' in locals() else "Not available"}
        except Exception as e:
            print(f"An unexpected error occurred during parsing: {e}")
            return {"error": "An unexpected error occurred", "details": str(e)}


# --- 4. Human-in-the-Loop (HITL) with AutoGen ---

def initiate_hitl_review(task_json: dict, parser_agent: 'ParserAgent', full_context_docs: List[Document], original_request: str, db_manager_approved_tasks: VectorDBManager) -> bool:
    """
    Manages the Human-in-the-Loop (HITL) review process with iterative feedback and modification.
    :param task_json: The structured ETL task definition (dictionary) to be reviewed.
    :param parser_agent: The ParserAgent instance, used to access its LLM for modifications.
    :param full_context_docs: List of LangChain Document objects containing combined dataset and approved task metadata for LLM grounding.
    :param original_request: The original natural language request for storing with approved tasks.
    :param db_manager_approved_tasks: The VectorDBManager instance for the approved tasks index.
    :return: True if the task is approved, False otherwise.
    """
    human_data_engineer = UserProxyAgent(
        name="Data_Engineer",
        human_input_mode="ALWAYS", # Always prompt for human input
        max_consecutive_auto_reply=0, # Never auto-reply for this agent in this setup
        is_termination_msg=lambda x: x.get("content", "").strip().lower() in ["approve", "approved", "deny", "denied"],
        system_message="You are a data engineer responsible for reviewing and approving ETL pipeline task definitions generated by an AI agent. Review the provided JSON carefully. Respond with 'approve', 'deny', or provide feedback for modification.",
        code_execution_config={"use_docker": False}
    )

    # Using AssistantAgent as requested, configured not to auto-reply in this specific chat flow.
    parser_messenger = AssistantAgent(
        name="Parser_Messenger",
        system_message="You present the ETL task definition and facilitate human feedback. You do not generate responses using an LLM in this chat.",
        llm_config=False, # Crucially, this disables its ability to use an LLM for generating its own responses within this specific chat flow.
        max_consecutive_auto_reply=0, # Ensure no auto-replies
    )

    current_task_json = task_json.copy()
    feedback_history = [] # Initialize list to store modification feedback

    # Format the full context for the modification prompt
    full_context_string_for_llm = ""
    if full_context_docs:
        full_context_string_for_llm = "Available Context (Dataset and Approved Task Details):\n"
        for i, doc in enumerate(full_context_docs):
            full_context_string_for_llm += f"  - {doc.page_content}"
            if doc.metadata:
                full_context_string_for_llm += f" (Metadata: {json.dumps(doc.metadata)})"
            full_context_string_for_llm += "\n"
    else:
        full_context_string_for_llm = "No specific context available for features."

    # For modification loop, use full context
    current_context_for_modification_llm = full_context_string_for_llm
    if DEBUG_LLM_CALL: # Still keep this for debug mode if ever re-enabled
        current_context_for_modification_llm = "Minimal context for debugging modification. User feedback: " + original_request[:100] + "..."


    while True: # Loop for iterative feedback
        print("\n--- Current Task Definition for Review ---")
        print(json.dumps(current_task_json, indent=2))
        print("----------------------------------------\n")

        # The parser_messenger initiates the chat with human_data_engineer.
        chat_result = parser_messenger.initiate_chat(
            human_data_engineer,
            message=f"Please review the following ETL task definition. You can '**approve**', '**deny**', or provide **feedback for modification** (e.g., 'add output_location', 'change join type to inner for bureau', 'add features from available dataset context').\n\n```json\n{json.dumps(current_task_json, indent=2)}\n```\n\n{full_context_string_for_llm}\n\nWhat is your decision or feedback?",
        )

        if chat_result.chat_history:
            human_response_raw = chat_result.chat_history[-1].get("content", "")
            human_response = human_response_raw.strip().lower()
        else:
            print("Error: chat_history is empty, cannot retrieve human response. Terminating review.")
            return False


        if human_response in ["approve", "approved"]:
            print("\n--- Task Approved by Data Engineer ---")
            # Ingest the approved task into the dedicated approved tasks index, including feedback history
            ingest_approved_etl_task(db_manager_approved_tasks, current_task_json, original_request, modification_feedback_history=feedback_history)
            return True
        elif human_response in ["deny", "denied"]:
            print("\n--- Task Denied by Data Engineer ---")
            if human_response_raw.strip().lower() not in ["deny", "denied"]:
                print(f"Reason for denial: {human_response_raw.strip()}")
            return False
        else:
            # Human provided feedback for modification - add to history
            feedback_history.append(human_response_raw.strip())
            print(f"\n--- Data Engineer requested modification: '{human_response_raw.strip()}' ---")
            print("AI is updating the task definition based on feedback...")

            # Define the prompt for LLM modification
            modification_prompt = PromptTemplate(
                template="""You are an AI assistant specialized in modifying ETL task definitions.
                You will be provided with the current ETL task definition in JSON format and specific feedback from a data engineer.
                Your goal is to adjust the JSON task definition based on the feedback.
                Crucially, you must **preserve all existing fields** unless the feedback explicitly instructs you to remove or change them.
                If feedback relates to a specific field, modify only that field. If it's about adding information, add it.
                Specifically, when asked to add or modify `features` under `scoring_model`, **select relevant column names ONLY from the provided 'Available Context' (Dataset or Approved Task Examples)**. Do not invent feature names. If no relevant columns are provided in the context, state that you cannot add specific features.

                You must output ONLY the updated JSON, strictly adhering to the Pydantic schema for ETLTaskDefinition.
                Ensure all **required fields** (like `pipeline_name`, `main_goal`) remain present and valid in the output.
                Do NOT include any conversational text, markdown code block (like ```json), or explanations outside the JSON.

                Pydantic Schema:
                {format_instructions}

                Current ETL Task Definition:
                {current_json}

                Data Engineer's Feedback:
                {feedback}

                Available Context:
                {dataset_context}

                Updated JSON Output:
                """,
                input_variables=["current_json", "feedback", "dataset_context"], # Renamed to dataset_context for consistency with prompt
                partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=ETLTaskDefinition).get_format_instructions()},
            )

            # Explicitly use the main parser_agent's LLM for modification
            modification_chain = modification_prompt | parser_agent.llm
            
            print("DEBUG: Attempting to invoke LLM chain for modification...")
            try:
                raw_llm_string_output_mod = modification_chain.invoke({
                    "current_json": json.dumps(current_task_json, indent=2),
                    "feedback": human_response_raw.strip(),
                    "dataset_context": current_context_for_modification_llm # Pass the (possibly minimal) context here
                }, config={"timeout": 300.0}) # Increased timeout
                print("DEBUG: LLM chain invocation for modification completed.")

                # Extract clean JSON string from LLM's raw output for modification
                cleaned_json_string_mod = extract_json_from_llm_output(raw_llm_string_output_mod)

                if DEBUG_LLM_CALL:
                    print("\n--- DEBUG: Raw LLM Output (Modification, after stripping fences) ---")
                    print(cleaned_json_string_mod)
                    print("----------------------------------------------------------------------")
                
                # Now, attempt to parse the cleaned string with PydanticOutputParser
                updated_task_pydantic = parser_agent.parser.parse(cleaned_json_string_mod)
                current_task_json = updated_task_pydantic.model_dump()
                print("Task definition updated. Presenting for re-review.")
            except TimeoutError:
                print(f"Error: LLM modification timed out after 300 seconds.") # Updated timeout message
                print("Failed to apply modification due to timeout. Please try again or provide more precise feedback.")
            except OutputParserException as e:
                print(f"Error parsing LLM output during modification: {e}")
                print(f"Raw LLM output (for debugging): {e.llm_output}")
                print("Failed to apply modification. Please try again or provide more precise feedback.")
            except json.JSONDecodeError as e: # This handles cases where cleaned_json_string is not valid JSON
                print(f"Error: Cleaned LLM modification output is not valid JSON: {e}")
                return {"error": "Cleaned LLM modification output is not valid JSON, cannot proceed.", "details": str(e), "raw_llm_output": cleaned_json_string_mod if 'cleaned_json_string_mod' in locals() else "Not available"}
            except Exception as e:
                print(f"An unexpected error occurred during modification: {e}")
                print("Failed to apply modification. Please try again.")

    return False


# --- Helper Function for Processing ETL Requests ---
def process_etl_request(user_request: str, db_manager_metadata: VectorDBManager, db_manager_approved_tasks: VectorDBManager, parser_agent: ParserAgent) -> dict:
    """
    Processes a single natural language ETL request: parses it and initiates HITL review.
    :param user_request: The natural language request for an ETL pipeline.
    :param db_manager_metadata: An initialized VectorDBManager instance for dataset metadata.
    :param db_manager_approved_tasks: An initialized VectorDBManager instance for approved tasks.
    :param parser_agent: An initialized ParserAgent instance.
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

    parsed_task_definition = parser_agent.parse_request(user_request, context_string_full) # Pass context_string_full
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
        is_approved = initiate_hitl_review(parsed_task_definition, parser_agent, similar_metadata_docs + similar_approved_tasks_docs, user_request, db_manager_approved_tasks)

        if is_approved:
            print("\nTask definition successfully approved.")
            return {"status": "approved", "task": parsed_task_definition}
        else:
            print("\nTask definition denied. Please refine the user request or review the parsing logic and context.")
            return {"status": "denied", "task": parsed_task_definition}


# --- Main Execution Workflow (Production-Ready Entry Point) ---

if __name__ == "__main__":
    # --- Initial Setup for Production Environment ---
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
    # This should typically be run once as part of a data catalog sync job.
    ingest_dataset_metadata(db_manager_metadata, COLUMNS_DESCRIPTION_CSV)

    # Ingest any *existing* approved tasks into the vector DB if you have a historical log
    # For a fresh start, this list can be empty.
    # Example:
    # approved_history_data = [
    #     {"pipeline_name": "PreviousLoanScoring", "main_goal": "Predict previous loan performance", "initial_tables": ["bureau", "bureau_balance"], "scoring_model": {"name": "XGBoost", "objective": "predict repayment", "target_column": "CREDIT_STATUS"}},
    #     {"pipeline_name": "ApplicationDemographics", "main_goal": "Clean and enrich applicant demographic data", "initial_tables": ["application_train"], "data_cleaning_steps": [{"type": "imputation", "details": {"strategy": "median", "columns": ["AMT_INCOME_TOTAL"]}}]}
    # ]
    # for task in approved_history_data:
    #     ingest_approved_etl_task(db_manager_approved_tasks, task, "Historical approved task example")


    # Initialize the Parser Agent with both DB managers
    parser_agent = ParserAgent(
        vector_db_manager_metadata=db_manager_metadata,
        vector_db_manager_approved_tasks=db_manager_approved_tasks,
        llm_model_name=OLLAMA_LLM_MODEL_NAME
    )

    print("\nETL Parser System initialized. Ready to process requests.")
    print("-------------------------------------------------------")

    # --- Example of processing a single incoming request (simulating a real production scenario) ---
    # In a real production system, this would be triggered by an API endpoint,
    # a message queue listener, or a scheduled job.
    sample_incoming_request = "Generate an ETL pipeline to predict loan default on the main application table, joining with previous credit bureau data and ensuring all missing values are handled. I need the output saved as parquet in s3://my-data-lake/processed/loan_defaults."

    # Process the request - pass both db managers
    final_task_status = process_etl_request(
        user_request=sample_incoming_request,
        db_manager_metadata=db_manager_metadata,
        db_manager_approved_tasks=db_manager_approved_tasks,
        parser_agent=parser_agent
    )

    print(f"\nFinal status for sample request: {final_task_status['status']}")
    if final_task_status['status'] == "approved":
        print("Approved task details (truncated for display):")
        print(json.dumps(final_task_status['task'], indent=2)[:500] + "...") # Print partial to avoid huge output

    print("\n-------------------------------------------------------")
    print("System is ready for further requests.")
    # Example of how you might integrate this into a FastAPI app:
    # @app.post("/process_etl/")
    # async def handle_etl_request(request_payload: dict):
    #     user_request = request_payload.get("natural_language_request")
    #     result = process_etl_request(user_request, db_manager_metadata, db_manager_approved_tasks, parser_agent)
    #     return result
