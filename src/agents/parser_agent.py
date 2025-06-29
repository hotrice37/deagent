"""
src/agents/parser_agent.py
Defines the ParserAgent, responsible for converting natural language requests
into structured ETL task definitions.
"""

# General Imports
import json

# LangChain Imports - for LLM and prompt handling
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_core.exceptions import OutputParserException

# Project Imports - for schema and utility functions
from src.core.schemas import ETLTaskDefinition
from src.core.vector_db_manager import VectorDBManager
from src.utils.general_utils import extract_json_from_llm_output # Import from new general_utils


class ParserAgent:
    """
    An AI agent that parses natural language requests into structured ETL task definitions,
    leveraging context from a vector database.
    """
    def __init__(self, vector_db_manager_metadata: VectorDBManager, vector_db_manager_approved_tasks: VectorDBManager, llm_model_name: str, temperature: float = 0, debug_mode: bool = False):
        """
        Initializes the ParserAgent.
        """
        self.llm = OllamaLLM(model=llm_model_name, temperature=temperature, request_timeout=300.0, base_url="http://localhost:11434", verbose=True)
        self.parser = PydanticOutputParser(pydantic_object=ETLTaskDefinition)
        self.vector_db_manager_metadata = vector_db_manager_metadata
        self.vector_db_manager_approved_tasks = vector_db_manager_approved_tasks
        self.debug_mode = debug_mode

        self.prompt_template = PromptTemplate(
            template="""You are an AI assistant specialized in parsing natural language requests for ETL pipelines and converting them into a structured JSON format.
Your task is to analyze the user's request and any provided historical context or relevant dataset metadata, and then output a JSON object that strictly adheres to the provided Pydantic schema.

**Instructions:**
- Identify the main goal of the pipeline.
- List all initial tables mentioned or clearly implied. Leverage the context for correct table names (e.g., 'application_train', 'bureau') and for inferring relevant tables not explicitly named.
- Deconstruct any join operations, specifying `left_table`, `right_table`, `join_type` (default to 'left' if not specified), and `on_columns`. Use context for correct column names if possible.
- Infer data cleaning or feature engineering steps if described (e.g., "cleaning the data", "impute missing values"). Provide a generic `type` and `details` dictionary for these.
- If a scoring or prediction objective is mentioned, specify the `scoring_model`'s `name` (e.g., 'Logistic Regression', 'XGBoost', or 'Generic ML Model' if unspecified), its `objective`, and `target_column`. Use context to accurately identify the target column (e.e.g., 'TARGET').
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

    def parse_request(self, user_request: str, context_string: str) -> dict:
        """
        Parses a natural language user request into a structured ETL task definition,
        using provided context.
        """
        current_context_for_llm = context_string
        
        chain = self.prompt_template | self.llm
        
        try:
            raw_llm_string_output = chain.invoke({"request": user_request, "context": current_context_for_llm}, config={"timeout": 300.0})
            
            cleaned_json_string = extract_json_from_llm_output(raw_llm_string_output)
            
            if self.debug_mode:
                print("\n--- DEBUG: Raw LLM Output (after stripping fences) ---")
                print(cleaned_json_string)
                print("----------------------------------------------------------")

            parsed_output_pydantic = self.parser.parse(cleaned_json_string)
            return parsed_output_pydantic.model_dump()
            
        except TimeoutError:
            print(f"Error: LLM parsing timed out after 300 seconds.")
            return {"error": "LLM parsing timed out", "details": "The model took too long to generate a response."}
        except OutputParserException as e:
            print(f"Error parsing LLM output: {e}")
            return {"error": "Failed to parse LLM output according to schema", "details": str(e), "raw_llm_output": cleaned_json_string if 'cleaned_json_string' in locals() else "Not available"}
        except json.JSONDecodeError as e:
            print(f"Error: Cleaned LLM output is not valid JSON: {e}")
            return {"error": "Cleaned LLM output is not valid JSON, cannot proceed.", "details": str(e), "raw_llm_output": cleaned_json_string if 'cleaned_json_string' in locals() else "Not available"}
        except Exception as e:
            print(f"An unexpected error occurred during parsing: {e}")
            return {"error": "An unexpected error occurred", "details": str(e)}

