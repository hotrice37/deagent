# src/hitl/hitl_manager.py
# Manages the Human-in-the-Loop (HITL) review process using AutoGen.

import json
from typing import Dict, Any, List, Callable, Optional 
import re

from langchain_core.documents import Document

# AutoGen imports
import autogen
from autogen import UserProxyAgent, AssistantAgent


def initiate_hitl_review(
    task_json: Dict[str, Any],
    parser_agent_instance: Any, 
    full_context_docs: List[Document],
    original_request: str,
    db_manager_approved_tasks: Any, 
    ingest_approved_etl_task_func: Callable,
    debug_mode: bool,
    ollama_llm_model_name: str, 
    # Optional parameters for script review
    generated_script: Optional[str] = None, 
    script_file_path: Optional[str] = None,
    execution_failed_previously: bool = False,
    review_type: str = "task_definition_review",
    dataset_schema_map: Optional[Dict[str, Dict[str, Any]]] = None 
) -> Dict[str, Any]:
    """
    Initiates a Human-in-the-Loop (HITL) review process using AutoGen.
    Allows for prompt-based modifications to the reviewed content (JSON or code).
    
    :param task_json: The generated ETL task definition in JSON format (can be modified).
    :param parser_agent_instance: The ParserAgent instance (if needed for re-parsing).
    :param full_context_docs: All context documents used by agents.
    :param original_request: The original natural language request.
    :param db_manager_approved_tasks: VectorDBManager for approved tasks.
    :param ingest_approved_etl_task_func: Function to ingest an approved task.
    :param debug_mode: A boolean flag to enable/disable debug logging.
    :param ollama_llm_model_name: The Ollama LLM model name to use for AutoGen agents.
    :param generated_script: The content of the initially generated PySpark script (for initial display).
    :param script_file_path: The filesystem path where the generated script is saved (for user reference and execution).
    :param execution_failed_previously: True if a previous script execution attempt failed, False otherwise.
    :param review_type: Specifies the type of review: "task_definition_review" or "script_execution_review".
    :param dataset_schema_map: The structured dataset schema (for context to the AI reviewer).
    :return: A dictionary containing the human's decision ('approved', 'denied', 'quit'),
             an 'execute_approved' flag if approved for execution, and the modified JSON/code.
    """
    print("\n" + "="*80)
    print("                 HUMAN-IN-THE-LOOP REVIEW REQUIRED (AutoGen)                 ")
    print("="*80)

    # AutoGen LLM configuration
    config_list_ollama = [
        {
            "model": ollama_llm_model_name,
            "base_url": "http://localhost:11434",
            "api_type": "ollama", 
        }
    ]

    # Prepare context strings for the AI Reviewer to be injected into its system message
    context_docs_str = ""
    if full_context_docs:
        context_docs_str += "\n--- Relevant Context Documents (e.g., related ETL tasks, dataset descriptions) ---\n"
        for i, doc in enumerate(full_context_docs):
            context_docs_str += f"  - Document {i+1}: Content='{doc.page_content}' Metadata={json.dumps(doc.metadata)}\n"
    else:
        context_docs_str = "\n--- Relevant Context Documents ---\n  No relevant context documents found.\n"

    dataset_schema_str = ""
    if dataset_schema_map:
        dataset_schema_str += "\n--- Dataset Schema Information ---\n"
        dataset_schema_str += json.dumps(dataset_schema_map, indent=2)
        dataset_schema_str += "\n"
    else:
        dataset_schema_str = "\n--- Dataset Schema Information ---\n  No dataset schema provided.\n"

    # User Proxy Agent: Represents the human reviewer
    user_proxy = autogen.UserProxyAgent(
        name="Human_Reviewer",
        system_message="You are the human data engineer. You will provide feedback "
                        "on the ETL task definition or PySpark code presented by the AI_Reviewer. "
                        "Provide your feedback or suggest modifications in natural language. "
                        "The AI_Reviewer will interpret your comments. "
                        "If you wish to terminate the entire process, explicitly state 'QUIT'.", 
        code_execution_config={"last_n_messages": 1, "work_dir": "coding", "use_docker": False}, 
        human_input_mode="ALWAYS",  
        llm_config={"config_list": config_list_ollama},
    )

    # AI Assistant Agent: The reviewer/generator that proposes changes
    # This template will be filled conditionally based on review_type
    reviewer_agent_base_system_message = (
        "You are an expert AI assistant for ETL tasks. Your goal is to assist the Human_Reviewer. "
        "Your responses should be concise and to the point. "
        "When the Human_Reviewer provides feedback, interpret their intent (approve, deny, modify). "
        "If you propose modifications, your output MUST be the ENTIRE MODIFIED CONTENT "
        "in the correct format (JSON for task definition, Python code block for script)."
        "If no modifications are needed, explicitly state 'NO MODIFICATIONS NEEDED'. "
        "Do NOT explain your changes unless explicitly asked; just provide the modified content. "
        "Upon interpreting 'approve', 'deny', or 'quit' from the Human_Reviewer, you MUST respond by acknowledging their decision "
        "by STARTING your message with one of these exact keywords: 'APPROVED', 'DENIED', or 'QUITTING'. " 
        "For example, if the human approves, you would respond: 'APPROVED'. If denying: 'DENIED'. If quitting: 'QUITTING'. " 
        "Do NOT add any other text before these keywords for termination."
        "\n\n--- IMPORTANT CONTEXT FOR YOUR REVIEW (DO NOT REPEAT THIS TO THE USER) ---"
        f"\nRelevant Context Documents (e.g., related ETL tasks, dataset descriptions):{context_docs_str}"
        f"\n\nDataset Schema Information:{dataset_schema_str}"
    )
    
    # Conditional message content for AI_Reviewer's first turn and final system message
    initial_ai_message_content = "" 
    reviewer_agent_final_system_message = "" 
    
    if review_type == "task_definition_review":
        reviewer_agent_final_system_message = (
            "You are an expert AI assistant for ETL tasks. Your goal is to assist the Human_Reviewer. "
            "Your responses should be concise and to the point. "
            "You have been provided with an ETL Task Definition (JSON format) from the planning stage. "
            "You MUST NOT generate initial ETL task definitions from scratch, but review and modify the provided one. "
            "When the Human_Reviewer provides feedback, interpret their intent (approve, deny, modify). "
            "If you propose modifications, your output MUST be the ENTIRE MODIFIED CONTENT "
            "in the correct format (JSON for task definition). "
            "If no modifications are needed, explicitly state 'NO MODIFICATIONS NEEDED'. "
            "Do NOT explain your changes unless explicitly asked; just provide the modified content. "
            "Upon interpreting 'approve', 'deny', or 'quit' from the Human_Reviewer, you MUST respond by acknowledging their decision "
            "by STARTING your message with one of these exact keywords: 'APPROVED', 'DENIED', or 'QUITTING'. " 
            "For example, if the human approves, you would respond: 'APPROVED'. If denying: 'DENIED'. If quitting: 'QUITTING'. " 
            "Do NOT add any other text before these keywords for termination."
            "\n\n--- IMPORTANT CONTEXT FOR YOUR REVIEW (DO NOT REPEAT THIS TO THE USER) ---"
            f"\nRelevant Context Documents (e.g., related ETL tasks, dataset descriptions):{context_docs_str}"
            f"\n\nDataset Schema Information:{dataset_schema_str}"
        )
        initial_ai_message_content = f"Hello Human_Reviewer, I have the refined ETL Task Definition ready. Please review it and provide your feedback or modifications.\n\n" \
                                      f"```json\n{json.dumps(task_json, indent=2)}\n```"

    elif review_type == "script_execution_review":
        reviewer_agent_final_system_message = reviewer_agent_base_system_message
        
        initial_ai_message_content = f"Hello Human_Reviewer, I have the generated PySpark script for review. " \
                                     f"It has been saved to: {script_file_path}\n" \
                                     "Please review this file directly. You can modify it manually. " \
                                     "Then, return here to tell me your feedback or if you wish to execute it."
        
        if execution_failed_previously:
            initial_ai_message_content += "\n*** WARNING: PREVIOUS EXECUTION FAILED! ***\n" \
                                          f"Please examine the error message in the console, fix the script at:\n" \
                                          f"--> {script_file_path} <--\n" \
                                          "Then, provide feedback or signal approval for re-execution."
        else:
            initial_ai_message_content += "\nProvide feedback for modifications by stating them, or signal approval to execute the script."

        if generated_script:
            cleaned_generated_script = generated_script.replace("```python", "").replace("```", "").strip()
            initial_ai_message_content += f"\n\n--- Original Generated Script Content ---\n```python\n{cleaned_generated_script}\n```\n"

    reviewer_agent = autogen.AssistantAgent(
        name="AI_Reviewer",
        system_message=reviewer_agent_final_system_message, 
        llm_config={"config_list": config_list_ollama},
        is_termination_msg=lambda msg: msg.get("content", "").upper().strip().startswith(("APPROVED", "DENIED", "QUITTING", "NO MODIFICATIONS NEEDED", "APPROVE", "DENY", "QUIT")),
    )

    modified_content = None
    final_decision = {"status": "denied", "execute_approved": False, "quit_session": False}
    
    chat_res = reviewer_agent.initiate_chat( 
        user_proxy,
        message=initial_ai_message_content, 
        clear_history=True, 
        silent=False 
    )

    # Retrieve the last message from the Human_Reviewer to determine the final status
    last_message_content = ""
    for msg in reversed(chat_res.chat_history):
        if msg.get("name") == user_proxy.name: # Changed condition to user_proxy.name as requested
            last_message_content = msg.get("content", "").upper().strip()
            break
    
    # Process the Human_Reviewer's last message to determine the final decision
    if "APPROVED" in last_message_content or "APPROVE" in last_message_content: 
        final_decision["status"] = "approved"
        if review_type == "script_execution_review":
            final_decision["execute_approved"] = True
            ingest_approved_etl_task_func(db_manager_approved_tasks, task_json) 
        print("AutoGen Chat Outcome: APPROVED")
    elif "DENIED" in last_message_content or "DENY" in last_message_content: 
        final_decision["status"] = "denied"
        print("AutoGen Chat Outcome: DENIED")
    elif "QUITTING" in last_message_content or "QUIT" in last_message_content: 
        final_decision["status"] = "quit"
        final_decision["quit_session"] = True
        print("AutoGen Chat Outcome: QUIT")
    elif "NO MODIFICATIONS NEEDED" in last_message_content: 
        final_decision["status"] = "approved" 
        print("AutoGen Chat Outcome: NO MODIFICATIONS NEEDED (TREATED AS APPROVED)")
    else:
        # If none of the explicit termination keywords are found anywhere in the Human_Reviewer's final message,
        # it implies a modification was proposed or an error occurred.
        if review_type == "task_definition_review":
            json_match = re.search(r"```json\n(.*?)\n```", last_message_content, re.DOTALL)
            if json_match:
                try:
                    original_case_last_message = ""
                    for msg in reversed(chat_res.chat_history):
                        if msg.get("name") == user_proxy.name: # Changed condition to user_proxy.name
                            original_case_last_message = msg.get("content", "")
                            break
                    
                    json_match_original_case = re.search(r"```json\n(.*?)\n```", original_case_last_message, re.DOTALL)
                    if json_match_original_case:
                        modified_content = json.loads(json_match_original_case.group(1))
                        print("AutoGen Chat Outcome: MODIFICATION PROPOSED (JSON)")
                        final_decision["status"] = "modified_json"
                        final_decision["modified_content"] = modified_content
                    else:
                        print("WARNING: AutoGen chat ended without a clear AI action (approve/deny/quit/no mods) or valid JSON modification found. Treating as error.")
                        final_decision["status"] = "error"
                except json.JSONDecodeError:
                    print("WARNING: AI_Reviewer returned malformed JSON for task definition modification.")
                    final_decision["status"] = "error" 
            else:
                print("WARNING: AutoGen chat ended without a clear AI action (approve/deny/quit/no mods) or valid modification for JSON. Treating as error.")
                final_decision["status"] = "error" 
        elif review_type == "script_execution_review":
            python_match = re.search(r"```python\n(.*?)\n```", last_message_content, re.DOTALL)
            if python_match:
                original_case_last_message = ""
                for msg in reversed(chat_res.chat_history):
                    if msg.get("name") == user_proxy.name: # Changed condition to user_proxy.name
                        original_case_last_message = msg.get("content", "")
                        break
                
                python_match_original_case = re.search(r"```python\n(.*?)\n```", original_case_last_message, re.DOTALL)
                if python_match_original_case:
                    modified_content = python_match_original_case.group(1)
                    print("AutoGen Chat Outcome: MODIFICATION PROPOSED (CODE)")
                    final_decision["status"] = "modified_code"
                    final_decision["modified_content"] = modified_content
                else:
                    print("WARNING: AutoGen chat ended without a clear AI action (approve/deny/quit/no mods) or valid code modification found. Treating as error.")
                    final_decision["status"] = "error"
            else:
                print("WARNING: AutoGen chat ended without a clear AI action (approve/deny/quit/no mods) or valid modification for CODE. Treating as error.")
                final_decision["status"] = "error" 
        else:
            print("WARNING: AutoGen chat ended without a clear AI action (approve/deny/quit/no mods) or valid modification. Treating as error.")
            final_decision["status"] = "error"

    return final_decision
