"""
src/agents/etl_generator_agent.py
Defines the ETLGeneratorAgent, responsible for generating PySpark code
based on a validated ETL task definition and dataset schema.
"""

import json
from typing import Dict, Any, List

from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.exceptions import OutputParserException

# Importing the main ETLTaskDefinition schema to reference its structure
from src.core.schemas import ETLTaskDefinition
from src.utils.general_utils import extract_json_from_llm_output


class ETLGeneratorAgent:
    """
    An AI agent that generates a complete PySpark ETL script given a structured
    ETL task definition and comprehensive dataset schema information.
    The generated script will perform Extract, Transform, and Load (to a file) operations.
    It WILL NOT include Great Expectations logic.
    """
    def __init__(self, llm_model_name: str, temperature: float = 0, debug_mode: bool = False):
        """
        Initializes the ETLGeneratorAgent.
        :param llm_model_name: The name of the Ollama model to use for text generation.
        :param temperature: The creativity temperature for the LLM. 0 for deterministic.
        :param debug_mode: A boolean flag to enable/disable debug logging.
        """
        self.llm = OllamaLLM(model=llm_model_name, temperature=temperature, request_timeout=600.0, base_url="http://localhost:11434", verbose=True) # Increased timeout for code gen
        self.debug_mode = debug_mode

        self.prompt_template = PromptTemplate(
            template="""You are an expert ETL (Extract, Transform, Load) PySpark code generator.
Your task is to generate a complete, executable PySpark script based on the provided ETL task definition (in JSON)
and detailed dataset schema information.

**CRITICAL INSTRUCTIONS FOR PYSPARK SCRIPT GENERATION:**

1.  **Strictly Adhere to Task Definition:** Implement ALL `initial_tables`, `join_operations`, `data_cleaning_steps`, `feature_engineering_steps`, `scoring_model` (target and features), `output_format`, and `output_location` as specified in the `etl_task_definition`.
2.  **Dynamic File Paths:** Assume all input CSV files (e.g., `application_train.csv`, `bureau.csv`) are located in a `data_base_path` variable that will be provided at runtime. Do NOT hardcode absolute paths.
3.  **SparkSession Initialization:** The script should *not* initialize a SparkSession, as one will be provided externally. Assume `spark` object is available in scope.
4.  **Extract (Reading Data):**
    * Read all necessary CSV files into Spark DataFrames. Infer schema and use header.
    * Example for reading: `spark.read.csv(f"{data_base_path}/application_train.csv", header=True, inferSchema=True)`
5.  **Transformations:**
    * **Joins:** Implement all `join_operations`. Ensure `on_columns` are correctly used. Pay attention to `join_type`.
        * **Important Note on Bureau Data:** For the 'bureau' and 'bureau_balance' datasets, if mentioned:
            * First, aggregate `bureau_balance.csv` to create features for each `SK_ID_BUREAU`. Common aggregations include mean, sum, count, min, max on relevant numerical columns (e.g., `MONTHS_BALANCE`). Also, consider counting occurrences of specific `STATUS` values (e.g., 'X', 'C', '0', '1', '2', '3', '4', '5').
            * Then, join this aggregated `bureau_balance` data with `bureau.csv` on `SK_ID_BUREAU`.
            * Finally, aggregate this enhanced `bureau` data to create features for each `SK_ID_CURR` before joining with `application_train.csv`.
    * **Data Cleaning:** Implement `data_cleaning_steps`. For `imputation`, use strategies like mean for numerical columns or mode for categorical. For `outlier_removal`, if specified, apply a simple method like Z-score (cap/floor or filter).
    * **Feature Engineering:** Implement `feature_engineering_steps`. Use PySpark functions for aggregations, one-hot encoding, or other transformations.
    * **Feature Selection:** If `scoring_model.features` are specified, select only these features and the `target_column` for the final DataFrame. If features are null, include all relevant columns.
6.  **Data Quality Validation:**
    * **DO NOT** include any Great Expectations setup, expectation definition, or checkpoint execution code in this script. This script is solely for ETL.
7.  **Load (Saving Data):**
    * Save the final transformed DataFrame (`final_df`) to the `output_location` in the specified `output_format`.
    * If `output_format` is "delta", use `final_df.write.format("delta").mode("overwrite").save(output_location)`.
    * If `output_format` is "parquet", use `final_df.write.parquet(output_location, mode="overwrite")`.
    * Assume `output_location` is a local path or a compatible URI (e.g., S3). The variable `output_location` and `output_format` will be provided directly in the script's scope.
8.  **Output Format:** Your response MUST be ONLY the PySpark code, enclosed in a single markdown code block (` ```python ... ```). Do NOT include any conversational text outside this block.

**Input:**
`etl_task_definition`: {etl_task_definition_json}
`dataset_schema_map`: {dataset_schema_map_json}
`data_base_path`: (This will be a string variable in the generated code, e.g., `'/path/to/your/data'`)

**Generated PySpark Script:**
""",
            input_variables=["etl_task_definition_json", "dataset_schema_map_json", "data_base_path"], # Added data_base_path
        )

    def generate_etl_script(
        self,
        etl_task_definition: Dict[str, Any],
        dataset_schema_map: Dict[str, Dict[str, Any]],
        data_base_path: str # data_base_path is now mandatory in input
    ) -> str:
        """
        Generates the PySpark ETL script based on the task definition and schema.
        :param etl_task_definition: The refined ETL task definition (JSON).
        :param dataset_schema_map: The structured dataset schema.
        :param data_base_path: The base path where input CSVs are located.
        :return: A string containing the generated PySpark script.
        """
        etl_task_definition_str = json.dumps(etl_task_definition, indent=2)
        dataset_schema_map_str = json.dumps(dataset_schema_map, indent=2)

        chain = self.prompt_template | self.llm

        try:
            raw_llm_output = chain.invoke({
                "etl_task_definition_json": etl_task_definition_str,
                "dataset_schema_map_json": dataset_schema_map_str,
                "data_base_path": data_base_path, # Passed data_base_path to invoke
            }, config={"timeout": 600.0})

            pyspark_script = extract_json_from_llm_output(raw_llm_output)
            
            return pyspark_script

        except TimeoutError:
            print("Error: ETLGeneratorAgent LLM timed out during script generation.")
            return "# Error: Script generation timed out."
        except OutputParserException as e:
            print(f"Error parsing LLM output (expected PySpark script): {e}")
            return f"# Error: Failed to parse generated script. Details: {str(e)}"
        except Exception as e:
            print(f"An unexpected error occurred in ETLGeneratorAgent: {e}")
            return f"# Error: An unexpected error occurred during script generation: {str(e)}"

