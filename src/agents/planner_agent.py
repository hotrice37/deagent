"""
src/agents/planner_agent.py
Defines the PlannerAgent, responsible for refining and validating parsed ETL task definitions
against actual dataset schemas and metadata.
"""

# General Imports
import json
from typing import Dict, Any, List
from pydantic import ValidationError

# LangChain Imports - for LLM and prompt handling
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_core.exceptions import OutputParserException


# Project Imports - for schema and utility functions
from src.core.schemas import ETLTaskDefinition, DataCleaningStep, FeatureEngineeringStep, JoinOperation, ScoringModel
from src.core.vector_db_manager import VectorDBManager
from src.utils.general_utils import extract_json_from_llm_output, reconstruct_from_etl_tasks_wrapper # Import from general_utils


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

**CRITICAL Output Structure Instructions:**
- **Your output MUST be a single JSON object that directly represents the ETLTaskDefinition.**
- **DO NOT wrap the output in any extra keys like "etl_tasks" or "task_definition".**
- The top-level keys of your JSON MUST include `pipeline_name`, `main_goal`, `initial_tables`, etc., as defined by the Pydantic schema.
- **IMPORTANT**: The `pipeline_name` and `main_goal` fields are *required* and describe the *entire ETL pipeline*, not individual steps. Ensure they are present at the root of your JSON output.
- **DO NOT generate a list of individual steps at the top level.** Your response should be ONE comprehensive pipeline definition.

**Example of REQUIRED Top-Level Output Structure (VERY IMPORTANT):**
```json
{{
  "pipeline_name": "Loan Default Prediction ETL",
  "main_goal": "Generate a dataset for predicting loan default likelihood by joining applicant and credit bureau data.",
  "initial_tables": ["application_train", "bureau"],
  "join_operations": [
    {{
      "left_table": "application_train",
      "right_table": "bureau",
      "join_type": "left",
      "on_columns": ["SK_ID_CURR"]
    }}
  ],
  "data_cleaning_steps": [
    {{
      "type": "imputation",
      "details": {{"strategy": "median", "columns": ["AMT_INCOME_TOTAL", "DAYS_EMPLOYED"]}}
    }}
  ],
  "scoring_model": {{
    "name": "LoanDefaultPredictor",
    "objective": "binary classification",
    "target_column": "TARGET",
    "features": ["AMT_INCOME_TOTAL", "DAYS_EMPLOYED", "CREDIT_INCOME_PERCENT"]
  }},
  "output_format": "parquet",
  "output_location": "s3://my-processed-data/loan_defaults"
}}
```

**General Refinement and Validation Rules:**
- **Table Validation:** Ensure all `initial_tables` and tables mentioned in `join_operations` exist in the `dataset_schema_info`. If a table is mentioned in the task but not in the schema, highlight it or correct it if an obvious alternative exists (e.g., "application" should be "application_train").
- **Column Validation:** For all tables, ensure columns mentioned (especially in `on_columns` for joins and `target_column` for scoring) actually exist in the corresponding table within the `dataset_schema_info`.
- **Join Feasibility:** For each `join_operation`, verify that the `on_columns` exist in *both* the `left_table` and `right_table` as specified in the `dataset_schema_info`. If a join column is missing from one side, flag it.
- **Target Column Identification:** If a `scoring_model` is specified, confirm that its `target_column` exists in the relevant table (usually the main table for the model) and is marked `is_target: true` in the `dataset_schema_info` if available.
- **Data Types:** Pay attention to data types. If a join is suggested on columns of incompatible types, flag it, but for this step, focus on existence primarily.
- **Preserve Valid Fields:** Only modify fields that are incorrect or require refinement based on schema validation. Preserve all other valid fields from the `Initial ETL Task Definition`.
- **No Conversational Text:** Your output MUST be ONLY the JSON object. Do not include any conversational text or markdown code blocks outside the JSON.

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
        """
        schema_info_string = json.dumps(dataset_schema_map, indent=2)
        initial_task_string = json.dumps(initial_task_json, indent=2)

        chain = self.prompt_template | self.llm

        try:
            raw_llm_string_output = chain.invoke({
                "initial_task_json": initial_task_string,
                "dataset_schema_info": schema_info_string
            }, config={"timeout": 300.0})

            cleaned_json_string = extract_json_from_llm_output(raw_llm_string_output)

            if self.debug_mode:
                print("\n--- DEBUG: Raw LLM Output (Planner, after stripping fences) ---")
                print(cleaned_json_string)
                print("------------------------------------------------------------------")

            try:
                refined_task_pydantic = self.parser.parse(cleaned_json_string)
                return refined_task_pydantic.model_dump()
            except ValidationError as ve:
                if self.debug_mode:
                    print(f"DEBUG_PLANNER: Direct Pydantic parsing failed. Attempting fallback for 'etl_tasks' wrapper. Details: {ve}")

                try:
                    parsed_as_dict = json.loads(cleaned_json_string)
                    if isinstance(parsed_as_dict, dict) and "etl_tasks" in parsed_as_dict and isinstance(parsed_as_dict["etl_tasks"], list):
                        # Use the moved function from general_utils
                        reconstructed_task = reconstruct_from_etl_tasks_wrapper(parsed_as_dict, initial_task_json)
                        refined_task_pydantic = self.parser.parse(json.dumps(reconstructed_task))
                        return refined_task_pydantic.model_dump()
                    else:
                        raise ve
                except (json.JSONDecodeError, ValueError) as inner_e:
                    print(f"DEBUG_PLANNER: Fallback parsing failed or unexpected inner structure: {inner_e}")
                    print(f"DEBUG_PLANNER: Original Pydantic validation error: {ve}")
                    raise ve
                except ValidationError as inner_ve:
                    print(f"DEBUG_PLANNER: Pydantic validation failed for reconstructed object: {inner_ve}")
                    raise ve

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

