# src/hitl/hitl_utils.py
# Contains helper functions for the Human-in-the-Loop (HITL) manager.

import json
from typing import List, Dict, Any

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from pydantic import ValidationError

from src.core.schemas import ETLTaskDefinition
from src.utils.general_utils import extract_json_from_llm_output # Assuming general_utils now holds this


def get_formatted_context_string(full_context_docs: List[Document]) -> str:
    """Helper to format context documents for the LLM."""
    context_string_for_llm = ""
    if full_context_docs:
        context_string_for_llm = "Available Context (Dataset and Approved Task Details):\n"
        for i, doc in enumerate(full_context_docs):
            context_string_for_llm += f"  - {doc.page_content}"
            if doc.metadata:
                context_string_for_llm += f" (Metadata: {json.dumps(doc.metadata)})"
            context_string_for_llm += "\n"
    else:
        context_string_for_llm = "No specific context available for features."
    return context_string_for_llm

def apply_modification_to_task(
    current_task_json: Dict[str, Any],
    human_feedback: str,
    parser_agent_llm: Any, # Pass the LLM directly from parser_agent to avoid circular dependency
    dataset_context: str,
    debug_mode: bool
) -> Dict[str, Any]:
    """Applies human feedback to modify the ETL task definition using an LLM."""
    modification_prompt = PromptTemplate(
        template="""You are an AI assistant specialized in modifying ETL task definitions.
        You will be provided with the current ETL task definition in JSON format and specific feedback from a data engineer.
        Your goal is to adjust the JSON task definition based on the feedback.
        Crucially, you must **preserve all existing fields** unless the feedback explicitly instructs you to remove or change them.
        If feedback relates to a specific field, modify only that field. If it's about adding information, add it.
        Specifically, when asked to add or modify `features` under `scoring_model`, **select relevant column names ONLY from the provided 'Available Context' (Dataset or Approved Task Examples)**. Do not invent feature names. If no relevant columns are provided in the context, state that you cannot add specific features.

        You must output ONLY the updated JSON, strictly adhering to the Pydantic schema for ETLTaskDefinition.
        Ensure all **required fields** (like `pipeline_name`, `main_goal`) remain present and valid in the output.
        Do NOT include any conversational text, markdown code block (like ```json), or explanations outside the JSON.

        Pydantic Schema:
        {format_instructions}

        Current ETL Task Definition:
        {current_json}

        Data Engineer's Feedback:
        {feedback}

        Available Context:
        {dataset_context}

        Updated JSON Output:
        """,
        input_variables=["current_json", "feedback", "dataset_context"],
        partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=ETLTaskDefinition).get_format_instructions()},
    )

    modification_chain = modification_prompt | parser_agent_llm # Use the passed LLM
    
    try:
        raw_llm_string_output_mod = modification_chain.invoke({
            "current_json": json.dumps(current_task_json, indent=2),
            "feedback": human_feedback.strip(),
            "dataset_context": dataset_context
        }, config={"timeout": 300.0})

        cleaned_json_string_mod = extract_json_from_llm_output(raw_llm_string_output_mod)

        if debug_mode:
            print("\n--- DEBUG: Raw LLM Output (Modification, after stripping fences) ---")
            print(cleaned_json_string_mod)
            print("----------------------------------------------------------------------")
        
        # We need the Pydantic parser from ParserAgent, but cannot import it directly due to circular dependency.
        # Instead, we will parse it generically and then validate it with ETLTaskDefinition.
        # This assumes the LLM adheres to the schema reasonably well for modifications.
        updated_task_pydantic = ETLTaskDefinition.model_validate_json(cleaned_json_string_mod)
        return updated_task_pydantic.model_dump()
    except TimeoutError:
        print(f"Error: LLM modification timed out after 300 seconds.")
        print("Failed to apply modification due to timeout. Please try again or provide more precise feedback.")
        return {"error": "Modification timeout"}
    except (OutputParserException, ValidationError) as e:
        print(f"Error parsing LLM output during modification: {e}")
        print(f"Raw LLM output (for debugging): {cleaned_json_string_mod if 'cleaned_json_string_mod' in locals() else 'Not available'}")
        print("Failed to apply modification. Please try again or provide more precise feedback.")
        return {"error": "Modification parsing failed"}
    except json.JSONDecodeError as e:
        print(f"Error: Cleaned LLM modification output is not valid JSON: {e}")
        print("Failed to apply modification. Cleaned LLM output is not valid JSON, cannot proceed.")
        return {"error": "Modification JSON invalid"}
    except Exception as e:
        print(f"An unexpected error occurred during modification: {e}")
        print("Failed to apply modification. Please try again.")
        return {"error": "Modification unexpected error"}

