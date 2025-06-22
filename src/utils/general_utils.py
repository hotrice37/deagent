# src/utils/general_utils.py
# Contains general utility functions not specific to a particular domain.

import re
import json
from typing import Dict, Any, List
from pydantic import ValidationError

# Project Imports - for schema
from src.core.schemas import DataCleaningStep, FeatureEngineeringStep, JoinOperation, ScoringModel


def extract_json_from_llm_output(llm_output: str) -> str:
    """
    Extracts a JSON string from LLM output, robustly handling markdown code fences
    and attempting to find JSON even if not perfectly formatted.
    """
    cleaned_output = llm_output.strip()

    # 1. Try to find JSON within a markdown code block (```json or ```)
    match = re.search(r"```json\s*\n(.*?)```", cleaned_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    match = re.search(r"```\s*\n(.*?)```", cleaned_output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. If no markdown fence, try to find the outermost JSON object by curly braces
    first_brace = cleaned_output.find('{')
    last_brace = cleaned_output.rfind('}')

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        potential_json_string = cleaned_output[first_brace : last_brace + 1]
        try:
            json.loads(potential_json_string)
            return potential_json_string.strip()
        except json.JSONDecodeError:
            pass

    # 3. If all else fails, return the original cleaned output
    return cleaned_output

def reconstruct_from_etl_tasks_wrapper(parsed_as_dict: Dict[str, Any], initial_task_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstructs a valid ETLTaskDefinition from an LLM output
    that incorrectly wraps the content in an 'etl_tasks' list.
    Moved from planner_agent.py to general_utils.py.
    """
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
        if "join_type" in sub_task and "on_columns" in sub_task:
            try:
                reconstructed_task["join_operations"].append(JoinOperation(**sub_task).model_dump())
            except ValidationError:
                pass
        elif "type" in sub_task and "details" in sub_task and any(k in sub_task["type"].lower() for k in ["imputation", "cleaning", "outlier", "deduplication", "conversion"]):
            try:
                reconstructed_task["data_cleaning_steps"].append(DataCleaningStep(**sub_task).model_dump())
            except ValidationError:
                pass
        elif "type" in sub_task and "details" in sub_task and any(k in sub_task["type"].lower() for k in ["feature_engineering", "aggregation", "encoding", "scaling"]):
            try:
                reconstructed_task["feature_engineering_steps"].append(FeatureEngineeringStep(**sub_task).model_dump())
            except ValidationError:
                pass
        elif "target_column" in sub_task:
            try:
                reconstructed_task["scoring_model"] = ScoringModel(**sub_task).model_dump()
            except ValidationError:
                pass

        if "table_name" in sub_task and sub_task["table_name"] not in reconstructed_task["initial_tables"]:
            reconstructed_task["initial_tables"].append(sub_task["table_name"])
    
    return reconstructed_task

