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
from src.core.schemas import ETLTaskDefinition, DataCleaningStep, FeatureEngineeringStep, JoinOperation, ScoringModel # Import all necessary schemas
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
        :param initial_task_json: The ETL task definition dictionary from the Parser Agent.
        :param dataset_schema_map: A structured dictionary representing the dataset schema.
                                   Expected format: {table_name: {column_name: {description, data_type, is_target, is_id}}}
        :return: A refined ETL task definition dictionary or an error dictionary.
        """
        # print("\nDEBUG_FLOW: Starting PlannerAgent.validate_and_refine_task.")

        # Convert the schema map to a readable string for the LLM
        schema_info_string = json.dumps(dataset_schema_map, indent=2)
        # if self.debug_mode:
        #     print(f"DEBUG_PLANNER: Schema Info Snippet:\n{schema_info_string[:1000]}...")

        # Convert initial_task_json to string for prompt
        initial_task_string = json.dumps(initial_task_json, indent=2)

        chain = self.prompt_template | self.llm

        # print("\nDEBUG_PLANNER: Attempting to invoke LLM chain for task refinement...")
        try:
            raw_llm_string_output = chain.invoke({
                "initial_task_json": initial_task_string,
                "dataset_schema_info": schema_info_string
            }, config={"timeout": 300.0})
            # print("DEBUG_PLANNER: LLM chain invocation for task refinement completed.")

            cleaned_json_string = extract_json_from_llm_output(raw_llm_string_output)

            if self.debug_mode:
                print("\n--- DEBUG: Raw LLM Output (Planner, after stripping fences) ---")
                print(cleaned_json_string)
                print("------------------------------------------------------------------")

            # --- Robust Parsing Logic ---
            try:
                # Attempt direct parsing first
                refined_task_pydantic = self.parser.parse(cleaned_json_string)
                # print("DEBUG_PLANNER: Direct Pydantic parsing of refined task successful.")
                return refined_task_pydantic.model_dump()
            except ValidationError as ve:
                # If direct parsing fails, check if it's due to the 'etl_tasks' wrapping
                if self.debug_mode:
                    print(f"DEBUG_PLANNER: Direct Pydantic parsing failed with ValidationError. Attempting fallback for 'etl_tasks' wrapper. Details: {ve}")

                try:
                    # Attempt to load as a generic JSON object
                    parsed_as_dict = json.loads(cleaned_json_string)

                    # Check for the problematic 'etl_tasks' list structure
                    if isinstance(parsed_as_dict, dict) and "etl_tasks" in parsed_as_dict and isinstance(parsed_as_dict["etl_tasks"], list):
                        # Extract components from the list and re-assemble a proper ETLTaskDefinition
                        reconstructed_task = {
                            "pipeline_name": initial_task_json.get("pipeline_name", "Reconstructed Pipeline"),
                            "main_goal": initial_task_json.get("main_goal", "Reconstructed Goal"),
                            "initial_tables": initial_task_json.get("initial_tables", []),
                            "join_operations": [],
                            "data_cleaning_steps": [],
                            "feature_engineering_steps": [],
                            "scoring_model": None,
                            "output_format": initial_task_json.get("output_format", "dataframe"),
                            "output_location": initial_task_json.get("output_location", None),
                            "data_quality_checks": initial_task_json.get("data_quality_checks", None),
                            "version_control_repo": initial_task_json.get("version_control_repo", None),
                            "orchestration_tool": initial_task_json.get("orchestration_tool", None),
                            "human_approval_required": initial_task_json.get("human_approval_required", True),
                        }

                        for sub_task in parsed_as_dict["etl_tasks"]:
                            # Attempt to extract known sub-schema elements from each "sub_task" dictionary
                            if "join_type" in sub_task and "on_columns" in sub_task:
                                try:
                                    reconstructed_task["join_operations"].append(JoinOperation(**sub_task).model_dump())
                                except ValidationError as e:
                                    # print(f"DEBUG_PLANNER: Could not parse sub-task as JoinOperation: {e}")
                                    pass
                            elif "type" in sub_task and "details" in sub_task and any(k in sub_task["type"].lower() for k in ["imputation", "cleaning", "outlier", "deduplication", "conversion"]):
                                try:
                                    reconstructed_task["data_cleaning_steps"].append(DataCleaningStep(**sub_task).model_dump())
                                except ValidationError as e:
                                    # print(f"DEBUG_PLANNER: Could not parse sub-task as DataCleaningStep: {e}")
                                    pass
                            elif "type" in sub_task and "details" in sub_task and any(k in sub_task["type"].lower() for k in ["feature_engineering", "aggregation", "encoding", "scaling"]):
                                try:
                                    reconstructed_task["feature_engineering_steps"].append(FeatureEngineeringStep(**sub_task).model_dump())
                                except ValidationError as e:
                                    # print(f"DEBUG_PLANNER: Could not parse sub-task as FeatureEngineeringStep: {e}")
                                    pass
                            elif "target_column" in sub_task:
                                try:
                                    reconstructed_task["scoring_model"] = ScoringModel(**sub_task).model_dump()
                                except ValidationError as e:
                                    # print(f"DEBUG_PLANNER: Could not parse sub-task as ScoringModel: {e}")
                                    pass

                            if "table_name" in sub_task and sub_task["table_name"] not in reconstructed_task["initial_tables"]:
                                reconstructed_task["initial_tables"].append(sub_task["table_name"])

                        # print("DEBUG_PLANNER: Reconstructed task from 'etl_tasks' wrapper. Attempting Pydantic parse.")
                        refined_task_pydantic = self.parser.parse(json.dumps(reconstructed_task))
                        # print("DEBUG_PLANNER: Pydantic parsing of reconstructed task successful via fallback.")
                        return refined_task_pydantic.model_dump()
                    else:
                        # print("DEBUG_PLANNER: 'etl_tasks' wrapper not found or not in expected list format. Re-raising original validation error.")
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

