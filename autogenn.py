from pathlib import Path

from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor

# Define the configuration for the language model
config_list = [
    {
        "model": "gemma3",
        "api_type": "ollama",
        "stream": False,
        "client_host": "http://localhost:11434",
        "temperature": 0,
    }
]

# Set up the code executor with a working directory
workdir = Path("coding")
workdir.mkdir(exist_ok=True) # Create the directory if it doesn't exist
code_executor = LocalCommandLineCodeExecutor(work_dir=workdir)

# Initialize the UserProxyAgent, responsible for executing code
user_proxy_agent = UserProxyAgent(
    name="User",
    code_execution_config={"executor": code_executor},
    is_termination_msg=lambda msg: "FINISH" in msg.get("content", ""), # Terminate chat when "FINISH" is received
    human_input_mode="NEVER", # Set to NEVER for automated execution
    max_consecutive_auto_reply=10, # Allow multiple auto-replies
)

# System message for the AssistantAgent
system_message = """
You are a helpful AI assistant. Your task is to analyze and break down software product backlog items into structured components.
Given a backlog item, you **MUST** respond with **ONLY ONE** message containing a Python code block. This Python code block will:
1. Define the parsed backlog item as a Python dictionary named `backlog_item_data`.
2. Save this dictionary to a file named `backlog_item.json` using the `json` module.
3. Print "File backlog_item.json saved successfully." to the console.

The structure for the Python code block **MUST** be exactly like this, with your parsed JSON data filling the `backlog_item_data` dictionary. You must not include any other text or code blocks outside of this single Python block in your first response.

```python
import json

# Replace the content below with the actual parsed backlog item data
backlog_item_data = {
  "task_type": "ETL",
  "sources": [
    {"name": "application_train", "alias": "app", "is_primary": True},
    {"name": "bureau", "alias": "b"}
  ],
  "join_conditions": [
    {"type": "left", "left_table_alias": "app", "right_table_alias": "b", "on_columns": ["SK_ID_CURR"]}
  ],
  "aggregations_on_joined": [
    {
      "table_alias": "b",
      "group_by_columns": ["SK_ID_CURR"],
      "agg_functions": [
        {"function": "count", "column": "*", "alias": "PREVIOUS_CREDITS_COUNT"}
      ]
    }
  ],
  "filters": [
    {"table_alias": "app", "condition": "AMT_INCOME_TOTAL >= 25000"}
  ],
  "transformations": [
    {
      "new_column": "CREDIT_INCOME_PERCENT",
      "formula_template": "{app.AMT_CREDIT} / {app.AMT_INCOME_TOTAL}",
      "involved_aliases": ["app"]
    }
  ],
  "target": {"name": "applications_enriched_v1"},
  "business_rules_summary": "Enrich application data with count of previous bureau credits, filter by income, calculate credit to income ratio."
}

# Save the dictionary to a JSON file
with open('backlog_item.json', 'w') as f:
    json.dump(backlog_item_data, f, indent=2)

print("File backlog_item.json saved successfully.")
```

If any field does not apply to the given backlog item, set its value to `null` within the `backlog_item_data` dictionary inside the Python code block.

**After you have sent this Python code block message, you MUST send a second, separate message that contains ONLY the word `FINISH`.**
"""

# Initialize the AssistantAgent with the updated system message
assistant_agent = AssistantAgent(
    name="Ollama Assistant",
    system_message=system_message,
    llm_config={
        "config_list": config_list,
    },
)

# Start the chat with the sample backlog item
# The instruction to save the file is now part of the assistant's system message.
chat_result = user_proxy_agent.initiate_chat(
    assistant_agent,
    message="""Create an ETL pipeline for new loan applications.\
      Start with application_train.csv.
      Join with bureau.csv on SK_ID_CURR to get a count of previous credits for each applicant.
      Filter out applications where AMT_INCOME_TOTAL is less than $25,000.
      Calculate the CREDIT_INCOME_PERCENT as AMT_CREDIT / AMT_INCOME_TOTAL.
      The final output should be a table named applications_enriched_v1.""",
)

# print(chat_result)
