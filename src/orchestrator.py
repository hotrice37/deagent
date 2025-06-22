# src/orchestrator.py
# Orchestrates the end-to-end ETL pipeline generation workflow.

import json
from typing import Dict, Any, List, Callable

from langchain_core.documents import Document

from src.core.vector_db_manager import VectorDBManager
from src.agents.parser_agent import ParserAgent
from src.agents.planner_agent import PlannerAgent
from src.hitl.hitl_manager import initiate_hitl_review


class ETLOrchestrator:
    """
    Orchestrates the entire ETL pipeline generation process,
    from initial request parsing to human-in-the-loop review.
    """
    def __init__(
        self,
        db_manager_metadata: VectorDBManager,
        db_manager_approved_tasks: VectorDBManager,
        parser_agent: ParserAgent,
        planner_agent: PlannerAgent,
        dataset_schema_map: Dict[str, Dict[str, Any]],
        debug_mode: bool,
        ingest_approved_etl_task_func: Callable
    ):
        """
        Initializes the ETLOrchestrator with all necessary components.
        """
        self.db_manager_metadata = db_manager_metadata
        self.db_manager_approved_tasks = db_manager_approved_tasks
        self.parser_agent = parser_agent
        self.planner_agent = planner_agent
        self.dataset_schema_map = dataset_schema_map
        self.debug_mode = debug_mode
        self.ingest_approved_etl_task_func = ingest_approved_etl_task_func

    def process_etl_request(self, user_request: str) -> dict:
        """
        Processes a single natural language ETL request: parses it and initiates HITL review.
        :param user_request: The natural language request for an ETL pipeline.
        :return: A dictionary containing the status ('approved', 'denied', 'error') and the task definition.
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

        # --- Step 1: Natural Language Parsing and Task Definition ---
        parsed_task_definition = self.parser_agent.parse_request(user_request, context_string_full)

        if "error" in parsed_task_definition:
            print("\nParsing failed. Please review the error details and the LLM's raw output.")
            print(f"Status : {parsed_task_definition.get('error', 'Unknown error')}")
            if "details" in parsed_task_definition:
                print(f"Details: {parsed_task_definition['details']}")
            if "raw_llm_output" in parsed_task_definition:
                print(f"Raw LLM Output:\n{parsed_task_definition['raw_llm_output']}")
            return {"status": "error", "task": parsed_task_definition}
        # else: # Commented out as requested
            # print("\nInitial Parsed Task Definition (from Parser Agent):")
            # print(json.dumps(parsed_task_definition, indent=2))

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
        # else: # Commented out as requested
            # print("\nRefined and Validated Task Definition (from Planner Agent):")
            # print(json.dumps(refined_task_definition, indent=2))

        print("\n--- Initiating Human-in-the-Loop Review (Type 'approve', 'deny', or feedback for modification) ---")

        is_approved = initiate_hitl_review(
            refined_task_definition,
            self.parser_agent,
            similar_metadata_docs + similar_approved_tasks_docs,
            user_request,
            self.db_manager_approved_tasks,
            self.ingest_approved_etl_task_func,
            self.debug_mode
        )

        if is_approved:
            print("\nTask definition successfully approved.")
            return {"status": "approved", "task": refined_task_definition}
        else:
            print("\nTask definition denied. Please refine the user request or review the parsing logic and context.")
            return {"status": "denied", "task": refined_task_definition}

