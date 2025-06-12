import os
from dotenv import load_dotenv
import json
import uuid
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import time # Import time for waiting on index readiness
import pandas as pd # Import pandas to read CSV
import hashlib # Import hashlib for consistent ID generation

# LangChain Imports
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
# Pinecone Import - Using the new Pinecone client
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeApiException # Specific exception from pinecone.exceptions
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException

# AutoGen Imports
from autogen import UserProxyAgent, AssistantAgent

load_dotenv() # Load environment variables from .env file

# --- Configuration ---
# Pinecone Configuration for Serverless Index
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "etl-backlog-production-index" # Production index name
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
# e.env., `ollama run gemma3`, `ollama run nomic-embed-text`
OLLAMA_LLM_MODEL_NAME = "gemma3"
OLLAMA_EMBEDDING_MODEL_NAME = "nomic-embed-text"


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
            print(f"Embedding dimension detected: {self.embedding_dimension}")
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
        print(f"Querying Pinecone for relevant context for: '{query_text}'...")
        query_embedding = self.embeddings_model.embed_query(query_text)

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

        print(f"Found {len(results)} similar documents in Pinecone.")
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
        metadatas_to_add.append(metadata) # Now 'metadata' is defined when used
        ids_to_add.append(generated_id)

    if texts_to_add: # Check if there's any data to add
        db_manager.add_documents_batch(texts_to_add, metadatas_to_add, doc_ids=ids_to_add)
    print("Dataset metadata ingestion complete.")


def ingest_approved_tasks_metadata(db_manager: VectorDBManager, approved_tasks_data: List[dict]):
    """
    Ingests previously approved ETL task definitions into the vector database.
    In a production environment, this would be called after a task is approved and finalized.
    """
    if not approved_tasks_data:
        print("No approved tasks data to ingest.")
        return

    print(f"Ingesting {len(approved_tasks_data)} approved ETL tasks into vector database...")
    texts_to_add = []
    metadatas_to_add = []

    for task in approved_tasks_data:
        # Create a concise text representation of the task for embedding
        task_text = f"ETL Pipeline: {task.get('pipeline_name', 'Unnamed Pipeline')}. Goal: {task.get('main_goal', 'No goal specified')}."

        # Add key details from the task definition as metadata
        task_metadata = {
            "source": "approved_etl_task", # Categorize as approved ETL task
            "pipeline_name": task.get('pipeline_name'),
            "main_goal": task.get('main_goal'),
            "initial_tables": task.get('initial_tables'),
            "join_operations_summary": [
                f"{op['left_table']} JOIN {op['right_table']} ON {','.join(op['on_columns'])}"
                for op in task.get('join_operations', [])
            ],
            "data_cleaning_types": [step['type'] for step in task.get('data_cleaning_steps', [])],
            "scoring_model_name": task.get('scoring_model', {}).get('name'),
            "target_column": task.get('scoring_model', {}).get('target_column'),
            "original_request_text": task.get('original_request_text') # If you store the original request
        }
        texts_to_add.append(task_text)
        metadatas_to_add.append(task_metadata)

    if texts_to_add:
        # No doc_ids argument passed, so it will use uuid.uuid4() for these dynamic tasks
        db_manager.add_documents_batch(texts_to_add, metadatas_to_add)
    print("Approved ETL tasks ingestion complete.")


# --- 3. Parser Agent ---

class ParserAgent:
    """
    An AI agent that parses natural language requests into structured ETL task definitions,
    leveraging context from a vector database.
    """
    def __init__(self, vector_db_manager: VectorDBManager, llm_model_name=OLLAMA_LLM_MODEL_NAME, temperature=0):
        """
        Initializes the ParserAgent.
        :param vector_db_manager: An instance of VectorDBManager for context retrieval.
        :param llm_model_name: The name of the Ollama model to use for text generation (e.g., 'llama2', 'mistral').
        :param temperature: The creativity temperature for the LLM. 0 for deterministic.
        """
        self.llm = OllamaLLM(model=llm_model_name, temperature=temperature)
        self.parser = PydanticOutputParser(pydantic_object=ETLTaskDefinition)
        self.vector_db_manager = vector_db_manager

        self.prompt_template = PromptTemplate(
            template="""You are an AI assistant specialized in parsing natural language requests for ETL pipelines and converting them into a structured JSON format.
Your task is to analyze the user's request and any provided historical context or relevant dataset metadata, and then output a JSON object that strictly adheres to the provided Pydantic schema.

**Instructions:**
- Identify the main goal of the pipeline.
- List all initial tables mentioned or clearly implied. Leverage the context for correct table names (e.g., 'application_train', 'bureau') and for inferring relevant tables not explicitly named.
- Deconstruct any join operations, specifying `left_table`, `right_table`, `join_type` (default to 'left' if not specified), and `on_columns`. Use context for correct column names if possible.
- Infer data cleaning or feature engineering steps if described (e.g., "cleaning the data", "impute missing values"). Provide a generic `type` and `details` dictionary for these.
- If a scoring or prediction objective is mentioned, specify the `scoring_model`'s `name` (e.g., 'Logistic Regression', 'XGBoost', or 'Generic ML Model' if unspecified), its `objective`, and `target_column`. Use context to accurately identify the target column (e.g., 'TARGET').
- Ensure all fields in the JSON adhere to the types and constraints defined in the schema.
- If a field is not explicitly mentioned but is optional in the schema, you can omit it or set it to its default/None.
- Do NOT include any explanations or conversational text outside the JSON block.

**Pydantic Schema:**
{format_instructions}

**Relevant Context (from similar past tasks or dataset metadata):**
{context}

**User Request:**
{request}

**JSON Output (strict JSON, no markdown code block or extra text):**
""",
            input_variables=["request", "context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def parse_request(self, user_request: str) -> dict:
        """
        Parses a natural language user request into a structured ETL task definition.
        First queries the vector DB for context, then sends to LLM.
        :param user_request: The natural language backlog item.
        :return: A dictionary representing the parsed ETLTaskDefinition, or an error dictionary.
        """
        # Step 1: Query the vector database for relevant context
        similar_docs = self.vector_db_manager.query_similar_documents(user_request, k=3)

        context_string = ""
        if similar_docs:
            context_string = "Based on similar past tasks or dataset metadata:\n"
            for i, doc in enumerate(similar_docs):
                context_string += f"  - Document {i+1}: '{doc.page_content}'\n"
                if doc.metadata:
                    context_string += f"    Metadata: {json.dumps(doc.metadata, indent=2)}\n"
        else:
            context_string = "No highly relevant context found in the vector database."

        # Step 2: Formulate the prompt with injected context and invoke the LLM
        chain = self.prompt_template | self.llm | self.parser
        try:
            parsed_output = chain.invoke({"request": user_request, "context": context_string})
            return parsed_output.model_dump()
        except OutputParserException as e:
            print(f"Error parsing LLM output: {e}")
            print(f"Raw LLM output (for debugging): {e.llm_output}")
            return {"error": "Failed to parse LLM output according to schema", "details": str(e), "raw_llm_output": e.llm_output}
        except Exception as e:
            print(f"An unexpected error occurred during parsing: {e}")
            return {"error": "An unexpected error occurred", "details": str(e)}


# --- 4. Human-in-the-Loop (HITL) with AutoGen ---

def initiate_hitl_review(task_json: dict) -> bool:
    """
    Simulates the Human-in-the-Loop (HITL) review process using AutoGen agents.
    The 'Data_Engineer' agent (human) approves or denies the generated task.
    :param task_json: The structured ETL task definition (dictionary) to be reviewed.
    :return: True if the task is approved, False otherwise.
    """
    human_data_engineer = UserProxyAgent(
        name="Data_Engineer",
        human_input_mode="ALWAYS",
        max_consecutive_auto_reply=0,
        is_termination_msg=lambda x: x.get("content", "").strip().lower() in ["approve", "approved", "deny", "denied"],
        system_message="You are a data engineer responsible for reviewing and approving ETL pipeline task definitions generated by an AI agent. Review the provided JSON carefully. Respond with 'approve' or 'deny'. If denying, provide specific reasons for the denial.",
        code_execution_config={"use_docker": False}
    )

    parser_bot = AssistantAgent(
        name="Parser_Bot",
        llm_config={"config_list": [{"model": OLLAMA_LLM_MODEL_NAME, "api_type": "ollama","base_url": "http://localhost:11434"}]},
        # Set max_consecutive_auto_reply to 0 to prevent the bot from responding automatically
        # before the human user has a chance to provide input.
        max_consecutive_auto_reply=0,
        system_message="You are an AI assistant that presents the generated ETL task definition to the data engineer for review. You should present the JSON clearly and await approval or denial. Once the human approves or denies, the conversation should end.",
    )

    print("\n--- Presenting Task for Human Review ---")
    print(json.dumps(task_json, indent=2))
    print("----------------------------------------\n")

    # The parser_bot will now initiate the chat with the human_data_engineer.
    # This ensures that after the parser_bot sends its message,
    # the human_data_engineer's 'human_input_mode="ALWAYS"' will take effect
    # and prompt the user for input directly.
    parser_bot.initiate_chat(
        human_data_engineer,
        message=f"Please review the following ETL task definition:\n\n```json\n{json.dumps(task_json, indent=2)}\n```\n\nDo you approve this task definition? Respond with 'approve' or 'deny'.",
    )

    last_message = human_data_engineer.last_message()
    if last_message and last_message.get("content", "").strip().lower() in ["approve", "approved"]:
        print("\n--- Task Approved by Data Engineer ---")
        return True
    else:
        print("\n--- Task Denied by Data Engineer ---")
        if last_message:
            print(f"Reason for denial: {last_message.get('content', 'No reason provided.')}")
        return False

# --- Helper Function for Processing ETL Requests ---
def process_etl_request(user_request: str, db_manager: VectorDBManager, parser_agent: ParserAgent) -> dict:
    """
    Processes a single natural language ETL request: parses it and initiates HITL review.
    :param user_request: The natural language request for an ETL pipeline.
    :param db_manager: An initialized VectorDBManager instance.
    :param parser_agent: An initialized ParserAgent instance.
    :return: A dictionary containing the status ('approved', 'denied', 'error') and the task definition.
    """
    print(f"\n{'='*50}\nProcessing Incoming Request:\nUser Request: '{user_request}'\n{'='*50}")

    parsed_task_definition = parser_agent.parse_request(user_request)

    if "error" in parsed_task_definition:
        print("\nParsing failed. Please review the error details and the LLM's raw output.")
        print(json.dumps(parsed_task_definition, indent=2))
        return {"status": "error", "task": parsed_task_definition}
    else:
        print("\nGenerated Task Definition:")
        print(json.dumps(parsed_task_definition, indent=2))
        print("\n--- Initiating Human-in-the-Loop Review (Type 'approve' or 'deny' in console) ---")

        is_approved = initiate_hitl_review(parsed_task_definition)

        if is_approved:
            print("\nTask definition successfully approved.")
            # In a real production system, you would now trigger the actual ETL pipeline creation
            # and potentially log this approved task in a persistent store.
            # You might also ingest this approved task back into your vector DB for future context.
            # db_manager.add_documents_batch(
            #     texts=[user_request], # Store the original request text
            #     metadatas=[{"source": "approved_task_runtime", **parsed_task_definition}]
            # )
            # print("Approved task added to Pinecone for future context (if enabled).")
            return {"status": "approved", "task": parsed_task_definition}
        else:
            print("\nTask definition denied. Please refine the user request or review the parsing logic and context.")
            return {"status": "denied", "task": parsed_task_definition}


# --- Main Execution Workflow (Production-Ready Entry Point) ---

if __name__ == "__main__":
    # --- Initial Setup for Production Environment ---
    print("Initializing ETL Parser System for Production...")

    # Initialize the vector database manager with Pinecone details
    db_manager = VectorDBManager(
        index_name=PINECONE_INDEX_NAME,
        cloud=PINECONE_CLOUD,
        region=PINECONE_REGION,
        api_key=PINECONE_API_KEY,
        embedding_model_name=OLLAMA_EMBEDDING_MODEL_NAME
    )

    # Ingest static dataset metadata (e.g., from your data catalog CSV)
    # This should typically be run once or as part of a data catalog sync job.
    ingest_dataset_metadata(db_manager, COLUMNS_DESCRIPTION_CSV)

    # Ingest any *existing* approved tasks into the vector DB if you have a historical log
    # For a fresh start, this list can be empty.
    # Example: approved_history = [{"pipeline_name": "Prod Pipeline 1", "main_goal": "Process daily sales", ...}]
    # ingest_approved_tasks_metadata(db_manager, approved_history)


    # Initialize the Parser Agent
    parser_agent = ParserAgent(vector_db_manager=db_manager, llm_model_name=OLLAMA_LLM_MODEL_NAME)

    print("\nETL Parser System initialized. Ready to process requests.")
    print("-------------------------------------------------------")

    # --- Example of processing a single incoming request (simulating a real production scenario) ---
    # In a real production system, this would be triggered by an API endpoint,
    # a message queue listener, or a scheduled job.
    sample_incoming_request = "Generate an ETL pipeline to predict loan default on the main application table, joining with previous credit bureau data and ensuring all missing values are handled."

    # Process the request
    final_task_status = process_etl_request(
        user_request=sample_incoming_request,
        db_manager=db_manager,
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
    #     result = process_etl_request(user_request, db_manager, parser_agent)
    #     return result
