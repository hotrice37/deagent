from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """You are a helpful AI assistant. Your task is to analyze and break down software product backlog items into structured components.
Given a backlog item, parse it into the following fields and output in **JSON format**.

Use the following JSON as an example format:

```json
{json}
```

If any field does not apply to the given backlog item, set its value to `null`.

After outputting the JSON, print `FINISH` on a new line.

Do not add any other text outside the JSON and `FINISH`.

The backlog item is: {message}"""


json = """{
  "task_type": "ETL",
  "sources": [
    {"name": "application_train", "alias": "app", "is_primary": true},
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
}"""


message = """Create an ETL pipeline for new loan applications.
Start with application_train.csv (or application_test.csv for new data).
Join with bureau.csv on SK_ID_CURR to get a count of previous credits for each applicant.
Filter out applications where AMT_INCOME_TOTAL is less than $25,000.
Calculate the CREDIT_INCOME_PERCENT as AMT_CREDIT / AMT_INCOME_TOTAL.
The final output should be a table named applications_enriched_v1."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="gemma3")

chain = prompt | model

result = chain.invoke({"message": message, "json": json})

print(result)