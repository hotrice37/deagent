# src/orchestrator.py
# Orchestrates the end-to-end ETL pipeline generation workflow.

import json
import os
from typing import Dict, Any, List, Callable
from pyspark.sql import SparkSession

from langchain_core.documents import Document

from src.core.vector_db_manager import VectorDBManager
from src.agents.parser_agent import ParserAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.etl_generator_agent import ETLGeneratorAgent
from src.core.etl_executor import ETLExecutor
from src.hitl.hitl_manager import initiate_hitl_review # Now uses AutoGen


class ETLOrchestrator:
    """
    Orchestrates the entire ETL pipeline generation process,
    from initial request parsing to human-in-the-loop review.
    Great Expectations validation has been removed from this flow.
    """
    def __init__(
        self,
        spark_session: SparkSession,
        db_manager_metadata: VectorDBManager,
        db_manager_approved_tasks: VectorDBManager,
        parser_agent: ParserAgent,
        planner_agent: PlannerAgent,
        etl_generator_agent: ETLGeneratorAgent,
        dataset_schema_map: Dict[str, Dict[str, Any]],
        data_base_path: str,
        debug_mode: bool,
        ingest_approved_etl_task_func: Callable,
        ollama_llm_model_name: str # Added for AutoGen HITL
    ):
        """
        Initializes the ETLOrchestrator with all necessary components.
        """
        self.spark = spark_session
        self.db_manager_metadata = db_manager_metadata
        self.db_manager_approved_tasks = db_manager_approved_tasks
        self.parser_agent = parser_agent
        self.planner_agent = planner_agent
        self.etl_generator_agent = etl_generator_agent
        self.dataset_schema_map = dataset_schema_map
        self.data_base_path = data_base_path
        self.debug_mode = debug_mode
        self.ingest_approved_etl_task_func = ingest_approved_etl_task_func
        self.ollama_llm_model_name = ollama_llm_model_name # Stored for HITL

        # Initialize the ETLExecutor (no GE context needed here anymore)
        self.etl_executor = ETLExecutor(self.spark, self.data_base_path)

    def _save_script_to_file(self, script_content: str, pipeline_name: str) -> str:
        """
        Saves the generated PySpark script to a file, removing markdown fences.
        :param script_content: The PySpark script as a string (potentially with markdown fences).
        :param pipeline_name: The name of the pipeline, used for filename.
        :return: The absolute path to the saved script file.
        """
        # Define the target directory: coding/spark from the project root
        project_root = os.getcwd() # Assumes execution from project root
        script_dir = os.path.join(project_root, "coding", "spark")
        os.makedirs(script_dir, exist_ok=True)
        
        # Remove markdown fences from the script content
        if script_content.startswith("```python\n") and script_content.endswith("\n```"):
            script_content = script_content[len("```python\n"):-len("\n```")]
        
        # Create a sanitized filename
        sanitized_pipeline_name = "".join(c for c in pipeline_name if c.isalnum() or c in (' ', '.', '_')).rstrip()
        sanitized_pipeline_name = sanitized_pipeline_name.replace(" ", "_")
        
        script_filename = f"{sanitized_pipeline_name}_etl_script.py"
        script_file_path = os.path.join(script_dir, script_filename)

        with open(script_file_path, "w") as f:
            f.write(script_content)
        
        print(f"Generated PySpark script saved to: {script_file_path}")
        return os.path.abspath(script_file_path) # Return absolute path


    def process_etl_request(self, user_request: str) -> dict:
        """
        Processes a single natural language ETL request: parses it, plans, generates code,
        executes (which saves data), and initiates HITL review.
        The process includes a two-step approval: first for the parsed task, then for script execution.
        It also allows for prompt-based modifications via AutoGen.
        :param user_request: The natural language request for an ETL pipeline.
        :return: A dictionary containing the status ('approved', 'denied', 'error', 'quit_session') and the task definition.
        """
        print(f"\n{'='*50}\nProcessing Incoming Request:\nUser Request: '{user_request}'\n{'='*50}")

        # Query both indexes for relevant context
        similar_metadata_docs = self.db_manager_metadata.query_similar_documents(user_request, k=5)
        similar_approved_tasks_docs = self.db_manager_approved_tasks.query_similar_documents(user_request, k=3)
        
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

        # Initialize these outside the loop
        parsed_task_definition = {}
        refined_task_definition = {}
        generated_pyspark_script = ""
        script_file_path = ""
        final_output_path = None
        
        # --- Outer loop for full process (allows JSON modification and re-planning/re-generation) ---
        full_process_loop_active = True
        while full_process_loop_active:
            print("\n--- Starting New Full ETL Generation Cycle ---")

            # --- Step 1: Natural Language Parsing and Task Definition ---
            # Re-parse if this is a fresh start or after a previous JSON modification
            parsed_task_definition = self.parser_agent.parse_request(user_request, context_string_full)

            if "error" in parsed_task_definition:
                print("\nParsing failed. Please review the error details and the LLM's raw output.")
                print(f"Status : {parsed_task_definition.get('error', 'Unknown error')}")
                if "details" in parsed_task_definition:
                    print(f"Details: {parsed_task_definition['details']}")
                if "raw_llm_output" in parsed_task_definition:
                    print(f"Raw LLM Output:\n{parsed_task_definition['raw_llm_output']}")
                return {"status": "error", "task": parsed_task_definition}
            
            # --- Step 2: Objective Definition with HITL (Planner Agent) ---
            print("\n--- Initiating Planner Agent for Schema Validation and Refinement ---")
            refined_task_definition = self.planner_agent.validate_and_refine_task(parsed_task_definition, self.dataset_schema_map)

            if "error" in refined_task_definition:
                print("\nPlanning failed. Please review the error details.")
                print(f"Status : {refined_task_definition.get('error', 'Unknown error')}")
                if "details" in refined_task_definition:
                    print(f"Details: {refined_task_definition['details']}")
                if "raw_llm_output" in refined_task_definition:
                    print(f"Raw LLM Output:\n{refined_task_definition['raw_llm_output']}")
                return {"status": "error", "task": refined_task_definition}

            # --- FIRST HITL: Approve Refined Task Definition (from Planner Agent) ---
            print("\n--- Initiating Human-in-the-Loop Review for Refined ETL Task Definition ---")
            task_review_decision = initiate_hitl_review(
                task_json=refined_task_definition, # Pass the refined task definition here
                parser_agent_instance=self.parser_agent, # Potentially useful for internal AutoGen logic
                full_context_docs=similar_metadata_docs + similar_approved_tasks_docs,
                original_request=user_request,
                db_manager_approved_tasks=self.db_manager_approved_tasks,
                ingest_approved_etl_task_func=self.ingest_approved_etl_task_func,
                debug_mode=self.debug_mode,
                ollama_llm_model_name=self.ollama_llm_model_name, # Pass LLM model name
                review_type="task_definition_review",
                dataset_schema_map=self.dataset_schema_map # Pass schema for AI reviewer context
            )

            if task_review_decision.get("quit_session"):
                print("\nUser opted to quit the session during task definition review.")
                return {"status": "quit_session", "task": refined_task_definition}
            
            if task_review_decision.get("status") == "denied":
                print("\nRefined ETL task definition denied by human. Aborting process.")
                return {"status": "denied", "task": refined_task_definition}

            if task_review_decision.get("status") == "modified_json":
                # User provided modified JSON, update refined_task_definition and restart full process loop
                refined_task_definition = task_review_decision.get("modified_content", refined_task_definition)
                print("\nRefined Task Definition modified by human. Restarting full ETL generation cycle.")
                continue # Restart the full_process_loop_active loop
            
            if task_review_decision.get("status") != "approved":
                print(f"Unexpected status '{task_review_decision.get('status')}' during task definition review. Aborting.")
                return {"status": "error", "task": refined_task_definition, "message": "Unexpected task review status."}
            
            print("\nRefined ETL task definition approved by human. Proceeding to code generation.")

            # --- Step 3: ETL Code Generation (ETLGeneratorAgent) ---
            print("\n--- Initiating ETL Generator Agent for PySpark Code Generation ---")
            generated_pyspark_script = self.etl_generator_agent.generate_etl_script(
                etl_task_definition=refined_task_definition,
                dataset_schema_map=self.dataset_schema_map,
                data_base_path=self.data_base_path
            )

            if not generated_pyspark_script or "Error" in generated_pyspark_script:
                print(f"\nCode generation failed: {generated_pyspark_script}")
                return {"status": "error", "task": refined_task_definition, "code_generation_error": generated_pyspark_script}
            
            print("\nPySpark script generated successfully.")

            # --- Step 4: Save the generated script to a file ---
            script_file_path = self._save_script_to_file(generated_pyspark_script, refined_task_definition.get('pipeline_name', 'untitled_pipeline'))
            
            execution_failed_previously = False
            
            # --- Inner loop for script execution review (allows code modification and re-execution) ---
            script_execution_loop_active = True
            while script_execution_loop_active:
                print("\n--- Entering Human-in-the-Loop Review Cycle for Script Execution ---")
                if execution_failed_previously:
                    print("\nATTENTION: Previous execution failed. Please review the script file, make necessary corrections, and approve re-execution.")

                script_review_decision = initiate_hitl_review(
                    task_json=refined_task_definition, # Still pass the task json
                    parser_agent_instance=self.parser_agent,
                    full_context_docs=similar_metadata_docs + similar_approved_tasks_docs,
                    original_request=user_request,
                    db_manager_approved_tasks=self.db_manager_approved_tasks,
                    ingest_approved_etl_task_func=self.ingest_approved_etl_task_func,
                    debug_mode=self.debug_mode,
                    ollama_llm_model_name=self.ollama_llm_model_name, # Pass LLM model name
                    generated_script=generated_pyspark_script, # Original generated content for reference (for AI Reviewer's context)
                    script_file_path=script_file_path, # Path to the file user can edit
                    execution_failed_previously=execution_failed_previously, # Inform HITL about prior failure
                    review_type="script_execution_review",
                    dataset_schema_map=self.dataset_schema_map # Pass schema for AI reviewer context
                )

                final_status = script_review_decision.get("status")
                execute_approved = script_review_decision.get("execute_approved", False)
                quit_session = script_review_decision.get("quit_session", False)

                if quit_session:
                    print("\nUser opted to quit the session during script execution review.")
                    return {"status": "quit_session", "task": refined_task_definition}
                
                if final_status == "denied":
                    print("\nETL pipeline generation process denied by human during script review. Exiting loop.")
                    return {"status": "denied", "task": refined_task_definition}

                if final_status == "modified_code":
                    # User provided modified code, overwrite file and restart script execution loop
                    modified_script_content = script_review_decision.get("modified_content", "")
                    if modified_script_content:
                        with open(script_file_path, "w") as f: # Overwrite the file
                            f.write(modified_script_content)
                        print(f"\nPySpark script file '{script_file_path}' updated with human/AI modifications. Retrying execution.")
                        execution_failed_previously = False # Reset flag for next attempt
                        continue # Restart the script_execution_loop_active loop
                    else:
                        print("WARNING: User/AI proposed code modification, but content was empty. Please check. Retrying review.")
                        execution_failed_previously = True # Keep error flag if content is empty
                        continue # Loop continues
                
                if final_status == "approved" and execute_approved:
                    print("\nETL pipeline generation and execution approved by human. Attempting execution...")
                    saved_output_location = self.etl_executor.execute_pyspark_script(script_file_path)

                    if saved_output_location is None:
                        print("ERROR: PySpark script execution failed. Returning to HITL for review.")
                        execution_failed_previously = True # Set flag for next iteration
                        continue # Restart the script_execution_loop_active loop
                    else:
                        print(f"PySpark script executed and data saved to: {saved_output_location}")
                        final_output_path = saved_output_location
                        print("\nETL pipeline successfully completed and data saved.")
                        return {"status": "approved", "task": refined_task_definition, "output_path": final_output_path}
                elif final_status == "approved" and not execute_approved:
                    print("\nETL pipeline generation approved, but execution denied by human. Exiting inner loop.")
                    script_execution_loop_active = False # Break inner loop
                    full_process_loop_active = False # Also break outer loop
                    return {"status": "approved_no_execution", "task": refined_task_definition, "script_file_path": script_file_path}
                else:
                    # Should not happen if hitl_decision logic is robust, but as a fallback
                    print(f"Unexpected HITL decision status: {final_status}. Exiting inner loop.")
                    script_execution_loop_active = False # Break inner loop
                    full_process_loop_active = False # Also break outer loop
                    return {"status": "error", "task": refined_task_definition, "message": "Unexpected HITL decision status."}

            # This point is reached if the inner loop completes without success/quit/deny
            # (e.g., if code was modified and then approved for execution but failed repeatedly)
            # or if the user denies execution, which then breaks both loops.
            break # Exit the full_process_loop_active if the inner loop results in a final state
