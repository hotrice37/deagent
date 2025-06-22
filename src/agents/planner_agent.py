"""
src/agents/planner_agent.py
Defines the PlannerAgent, responsible for refining and validating parsed ETL task definitions
against actual dataset schemas and metadata.
"""

# General Imports
import json
from typing import Dict, Any
from pydantic import ValidationError

# LangChain Imports - for LLM and prompt handling
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_core.exceptions import OutputParserException


# Project Imports - for schema and utility functions
from src.core.schemas import ETLTaskDefinition, DatasetMetadata, JoinOperation, ScoringModel
from src.core.vector_db_manager import VectorDBManager
from src.utils.utils import extract_json_from_llm_output


class PlannerAgent:
    """
    An AI agent that refines and validates a parsed ETL task definition against
    the dataset's schema and metadata. It aims to ensure feasibility and
    correctness of tables, columns, join operations, and target variables.
    """
    def __init__(self, llm_model_name: str, temperature: float = 0, debug_mode: bool = False):
        """
        Initializes the PlannerAgent.
        :param llm_model_name: The name of the Ollama model to use for text generation.
        :param temperature: The creativity temperature for the LLM. 0 for deterministic.
        :param debug_mode: A boolean flag to enable/disable debug logging.
        """
        self.llm = OllamaLLM(model=llm_model_name, temperature=temperature, request_timeout=300.0, base_url="http://localhost:11434", verbose=True)
        self.parser = PydanticOutputParser(pydantic_object=ETLTaskDefinition)
        self.debug_mode = debug_mode

        self.prompt_template = PromptTemplate(
            template="""You are an expert ETL pipeline planner. Your task is to review an initial ETL task definition (in JSON) and refine/validate it based on the provided ACTUAL dataset schema and metadata.
The goal is to produce a refined ETL task definition that is both accurate and executable given the available data.

**Specific Instructions for Refinement and Validation:**
- **Table Validation:** Ensure all `initial_tables` and tables mentioned in `join_operations` exist in the `dataset_schema_info`. If a table is mentioned in the task but not in the schema, highlight it or correct it if an obvious alternative exists (e.g., "application" should be "application_train").
- **Column Validation:** For all tables, ensure columns mentioned (especially in `on_columns` for joins and `target_column` for scoring) actually exist in the corresponding table within the `dataset_schema_info`.
- **Join Feasibility:** For each `join_operation`, verify that the `on_columns` exist in *both* the `left_table` and `right_table` as specified in the `dataset_schema_info`. If a join column is missing from one side, flag it.
- **Target Column Identification:** If a `scoring_model` is specified, confirm that its `target_column` exists in the relevant table (usually the main table for the model) and is marked `is_target: true` in the `dataset_schema_info` if available.
- **Data Types:** Pay attention to data types. If a join is suggested on columns of incompatible types, flag it, but for this step, focus on existence primarily.
- **Refinement:** Adjust the task definition to explicitly name tables and columns if they were vague in the initial parse, using the schema as the source of truth.
- **Strict JSON Output:** Your output MUST be a JSON object that strictly conforms to the ETLTaskDefinition Pydantic schema. Do not include any conversational text or markdown code blocks outside the JSON.

**Pydantic Schema for Output:**
{format_instructions}

**Initial ETL Task Definition (from Parser Agent):**
{initial_task_json}

**Actual Dataset Schema Information (crucial for validation):**
{dataset_schema_info}

**Refined and Validated ETL Task Definition (JSON):**
""",
            input_variables=["initial_task_json", "dataset_schema_info"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def validate_and_refine_task(self, initial_task_json: dict, dataset_schema_map: Dict[str, Dict[str, Any]]) -> dict:
        """
        Validates and refines an ETL task definition against a structured dataset schema map.
        :param initial_task_json: The ETL task definition dictionary from the Parser Agent.
        :param dataset_schema_map: A structured dictionary representing the dataset schema.
                                   Expected format: {table_name: {column_name: {description, data_type, is_target, is_id}}}
        :return: A refined ETL task definition dictionary or an error dictionary.
        """
        print("\nDEBUG_FLOW: Starting PlannerAgent.validate_and_refine_task.")

        # Convert the schema map to a readable string for the LLM
        schema_info_string = json.dumps(dataset_schema_map, indent=2)
        if self.debug_mode:
            print(f"DEBUG_PLANNER: Schema Info Snippet:\n{schema_info_string[:1000]}...")

        # Convert initial_task_json to string for prompt
        initial_task_string = json.dumps(initial_task_json, indent=2)

        chain = self.prompt_template | self.llm

        print("\nDEBUG_PLANNER: Attempting to invoke LLM chain for task refinement...")
        try:
            raw_llm_string_output = chain.invoke({
                "initial_task_json": initial_task_string,
                "dataset_schema_info": schema_info_string
            }, config={"timeout": 300.0})
            print("DEBUG_PLANNER: LLM chain invocation for task refinement completed.")

            cleaned_json_string = extract_json_from_llm_output(raw_llm_string_output)

            if self.debug_mode:
                print("\n--- DEBUG: Raw LLM Output (Planner, after stripping fences) ---")
                print(cleaned_json_string)
                print("------------------------------------------------------------------")

            # Validate the refined output against the schema
            refined_task_pydantic = self.parser.parse(cleaned_json_string)
            print("DEBUG_PLANNER: Pydantic parsing of refined task successful.")
            return refined_task_pydantic.model_dump()

        except TimeoutError:
            print(f"Error: PlannerAgent LLM timed out after 300 seconds.")
            return {"error": "PlannerAgent LLM timed out", "details": "The Planner model took too long to generate a response."}
        except (OutputParserException, ValidationError) as e:
            print(f"Error parsing or validating PlannerAgent LLM output: {e}")
            return {"error": "PlannerAgent failed to produce valid ETLTaskDefinition", "details": str(e), "raw_llm_output": cleaned_json_string if 'cleaned_json_string' in locals() else "Not available"}
        except json.JSONDecodeError as e:
            print(f"Error: PlannerAgent LLM output is not valid JSON: {e}")
            return {"error": "PlannerAgent LLM output is not valid JSON", "details": str(e), "raw_llm_output": cleaned_json_string if 'cleaned_json_string' in locals() else "Not available"}
        except Exception as e:
            print(f"An unexpected error occurred in PlannerAgent: {e}")
            return {"error": "An unexpected error occurred in PlannerAgent", "details": str(e)}