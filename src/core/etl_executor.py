"""
src/core/etl_executor.py
Manages the execution of PySpark ETL scripts. Great Expectations integration
has been removed as per user request.
"""

import os
import json
import shutil
from typing import Dict, Any, Tuple, Optional
from pyspark.sql import SparkSession, DataFrame


class ETLExecutor:
    """
    Executes PySpark ETL scripts. No Great Expectations integration.
    """

    def __init__(self, spark: SparkSession, data_base_path: str):
        """
        Initializes the ETLExecutor.
        :param spark: The active SparkSession.
        :param data_base_path: The base path for input/output data (e.g., '/home/user/data').
        """
        self.spark = spark
        self.data_base_path = data_base_path

    def execute_pyspark_script(self, script_file_path: str) -> Optional[str]: # Changed input from content to path
        """
        Executes the PySpark script from the given file path within the current SparkSession.
        The script is expected to save a final Spark DataFrame named 'final_df'
        to a specified output location and set an 'output_location' variable.
        :param script_file_path: The absolute path to the PySpark script file.
        :return: The path to the saved output file/directory, or None if execution fails.
        """
        if not os.path.exists(script_file_path):
            print(f"ERROR: Script file not found at: {script_file_path}")
            return None

        # Read the script content from the file
        with open(script_file_path, "r") as f:
            pyspark_script_content = f.read()

        # Inject data_base_path into the script's scope.
        # The script is expected to set 'output_location' as a variable.
        injected_script = (
            f"data_base_path = '{self.data_base_path}'\n"
            f"{pyspark_script_content}\n" # Use content read from file
            f"exec_output_location = output_location # Capture the output_location from the script\n"
        )

        local_scope = {
            "spark": self.spark,
            # No need to pass ge_data_context to the script itself if it's not defining GE checks
            # The script will define `output_location` and `output_format`
        }
        
        try:
            print(f"\n--- Executing PySpark Script from file: {script_file_path} ---")
            exec(injected_script, globals(), local_scope)
            print("--- PySpark Script Execution Complete ---")

            exec_output_location = local_scope.get("exec_output_location")
            if not exec_output_location:
                print("WARNING: PySpark script execution completed but 'output_location' variable was not set or found in script's scope.")
                return None
            
            # Optional: Verify if the output location actually exists.
            if not os.path.exists(exec_output_location):
                print(f"WARNING: Output location '{exec_output_location}' does not exist after script execution.")
                return None

            return exec_output_location
        except Exception as e:
            print(f"ERROR: Failed to execute PySpark script from {script_file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_dataframe(self, df: DataFrame, output_location: str, output_format: str) -> bool:
        """
        Saves the Spark DataFrame to the specified output location and format.
        This function is now only a helper and is intended to be called by the generated script
        if it chooses to call a helper function instead of direct write.
        """
        print(f"\n--- Saving DataFrame to: {output_location} (Format: {output_format}) ---")
        try:
            os.makedirs(os.path.dirname(output_location), exist_ok=True)
            if output_format.lower() == "parquet":
                df.write.parquet(output_location, mode="overwrite")
            elif output_format.lower() == "delta":
                df.write.format("delta").mode("overwrite").save(output_location)
            else:
                print(f"ERROR: Unsupported output format: {output_format}")
                return False
            print("DataFrame saved successfully.")
            return True
        except Exception as e:
            print(f"ERROR: Failed to save DataFrame: {e}")
            return False

