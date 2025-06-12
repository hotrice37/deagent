Below is an updated prompt that integrates specific details about which parts of the **Kaggle Home Credit Default Risk Dataset** (e.g., `application_train.csv`, `bureau.csv`, `bureau_balance.csv`) to use at each step, and whether AI agents need to learn about the dataset (including what parts and how, such as schema analysis or metadata extraction). The prompt retains all previous requirements, including designing an AI agent to automate data engineering tasks by interpreting product backlog items, generating ETL pipelines, applying transformations, delivering analytical outputs (scoring loan repayment likelihood), and incorporating human-in-the-loop (HITL) validation, natural language parsing, modularity, and the specified technology stack. It ensures seamless integration, uses the latest tools and frameworks as of June 2025, and provides clear, logical steps with practical examples and code artifacts.

---

**Updated Prompt:**

I want a detailed, step-by-step guide on how to design and implement an AI agent that automates data engineering tasks by interpreting product backlog items, generating ETL pipelines, applying transformations, and delivering analytical outputs, tailored for a data engineer with intermediate experience. The guide should use the **Kaggle Home Credit Default Risk Dataset** (available at https://www.kaggle.com/competitions/home-credit-default-risk/data) to automate ETL processes and train a machine learning model to score loan applicants on their likelihood of repaying a loan (ranging from 0 to 100, based on the `TARGET` column, where 1 indicates default and 0 indicates repayment). The workflow must begin with parsing natural language backlog items and incorporate **human-in-the-loop (HITL)** validation at key stages. Each step must specify which parts of the dataset (e.g., `application_train.csv`, `bureau.csv`, `bureau_balance.csv`) are used and whether agents need to learn about the dataset (e.g., schema, relationships, or metadata) and how (e.g., via schema analysis, statistical profiling, or metadata extraction). Steps must be clearly explained, logically connected, and include practical examples or use cases. The guide must address the following requirements and steps:

### Key Requirements:
- **Backlog Interpretation**:
  - Parse natural language backlog items (e.g., "Generate ETL pipeline for Home Credit data, join application and bureau tables, score applicants for repayment") to extract requirements (source tables, transformations, business rules).
  - Use LangChain and AutoGen for NLP task breakdown.
- **Code Generation**:
  - Auto-generate Spark SQL or PySpark code for ETL pipelines, including optimal partition handling.
  - Suggest joins, filters, and aggregations based on dataset structure.
  - Create dbt models or Airflow DAGs for orchestration if applicable.
- **Data Quality & Validation**:
  - Integrate Great Expectations for automated checks (e.g., missing values, schema validation).
  - Flag anomalies for HITL approval.
- **Model Integration**:
  - Apply an AutoML-generated scoring model (e.g., H2O, PyCaret, or MLflow) to produce loan repayment scores (0–100).
  - Output results with metadata (e.g., confidence scores, validation logs).
- **Deployment & Version Control**:
  - Auto-push code to a Git repository or orchestrate via Airflow.
  - Include HITL approval hooks before production.
- **Modularity**:
  - Break tasks into sub-agents (e.g., backlog parser, SQL generator, validator, model scorer) with rule-based fallbacks for cost or LLM limitations.
- **RLHF**:
  - Incorporate reinforcement learning from human feedback (RLHF) to fine-tune agent performance based on HITL inputs.
- **Dataset Learning**:
  - Specify when and how agents learn about the dataset (e.g., schema analysis for `application_train.csv`, relationship mapping for `bureau.csv` and `bureau_balance.csv`).

### Technology Stack:
- **NLP & Agents**: GPT-4, Claude, LangChain, AutoGen
- **ETL**: PySpark, dbt, Apache Airflow
- **Validation**: Great Expectations
- **AutoML**: H2O, PyCaret, or custom MLflow pipelines
- **Deployment**: GitHub, Airflow, Databricks, AWS Bedrock, or SageMaker
- **Dataset**: Kaggle Home Credit Default Risk Dataset (`application_train.csv`, `bureau.csv`, `bureau_balance.csv`)
- **Other**: PydanticAI, OpenAI Assistants API, n8n for workflow automation

### Submission:
- A GitHub repository containing:
  - Agent code (e.g., Python scripts for sub-agents, Airflow DAGs, dbt models).
  - Sample backlog items (e.g., natural language instructions).
  - Demo outputs (e.g., processed data, model scores, validation logs).
- A short report (Markdown or PDF) explaining design choices, limitations, and how the agent meets the requirements.

### Steps:
1. **Natural Language Parsing**:
   - **Dataset Used**: None directly; metadata or schema descriptions may be referenced.
   - **Agent Learning**: Use LangChain and AutoGen to parse natural language backlog items (e.g., "Join Home Credit application and bureau data, score repayment") into structured JSON tasks (e.g., source tables, joins, transformations).
   - **Learning How**: Agents learn dataset context by analyzing provided metadata or Kaggle dataset descriptions (e.g., via API or documentation) to identify tables like `application_train.csv` and `bureau.csv`.
   - **Example**: Parse query into JSON specifying tables, transformations, and scoring.
   - **HITL**: Save parsed tasks for human review via CLI or UI.
2. **Objective Definition with HITL**:
   - **Dataset Used**: Metadata or schema of `application_train.csv`, `bureau.csv`, `bureau_balance.csv`.
   - **Agent Learning**: Agents analyze dataset schema (e.g., columns like `SK_ID_CURR`, `TARGET`) to validate parsed objectives.
   - **Learning How**: Use PySpark to read schema or Great Expectations to profile metadata.
   - **Task**: Confirm ETL and scoring tasks (e.g., merge tables, predict repayment) with human input.
3. **Technology Stack Selection**:
   - **Dataset Used**: None directly.
   - **Agent Learning**: None required; select tools based on task requirements.
   - **Task**: Recommend tools (e.g., LangChain, PySpark, Great Expectations, PyCaret) for NLP, ETL, validation, and AutoML, explaining their roles.
4. **Data Preparation and Integration**:
   - **Dataset Used**: `application_train.csv` (main applicant data), `bureau.csv` (credit bureau data), `bureau_balance.csv` (monthly balance data).
   - **Agent Learning**: Agents learn table relationships (e.g., `SK_ID_CURR` joins `application_train` to `bureau`, `SK_ID_BUREAU` joins `bureau` to `bureau_balance`) via schema analysis or Kaggle API metadata.
   - **Learning How**: Use PySpark’s `describe()` or Pandas profiling to understand column distributions and relationships.
   - **Task**: Integrate data via Kaggle APIs or cloud storage into platforms like Databricks or Snowflake.
5. **ETL Automation with AI Agents and HITL**:
   - **Dataset Used**: `application_train.csv`, `bureau.csv`, `bureau_balance.csv`.
   - **Agent Learning**: SQL generator sub-agent learns table schemas and relationships to suggest joins and aggregations (e.g., average credit from `bureau.csv`).
   - **Learning How**: Analyze schema using PySpark or dbt, with rule-based fallbacks for common transformations.
   - **Task**: Build sub-agents to automate ETL (Extract: Kaggle APIs; Transform: handle missing values, aggregate `bureau_balance.csv` data; Load: to Snowflake/Databricks).
   - **Validation**: Use Great Expectations for checks (e.g., no nulls in `SK_ID_CURR`).
   - **HITL**: Review transformed data samples and validation results.
6. **Tool and API Integration**:
   - **Dataset Used**: Processed data from ETL.
   - **Agent Learning**: None required; agents use learned schema from Step 5.
   - **Task**: Equip agents with APIs (e.g., Kaggle, Snowflake) for ETL, ensuring seamless integration.
7. **Model Integration with HITL**:
   - **Dataset Used**: Processed dataset (merged `application_train.csv`, `bureau.csv`, `bureau_balance.csv`).
   - **Agent Learning**: Model scorer sub-agent learns feature distributions (e.g., `AMT_CREDIT_SUM` from `bureau.csv`) via statistical profiling.
   - **Learning How**: Use PyCaret’s data profiling or MLflow’s feature logging.
   - **Task**: Train an AutoML model (e.g., PyCaret) to score repayment likelihood (0–100), including feature engineering (e.g., aggregating `bureau_balance.csv` data).
   - **Output**: Scores with metadata (e.g., confidence scores).
   - **HITL**: Review feature importance and sample scores.
8. **Workflow Automation**:
   - **Dataset Used**: Processed data and model outputs.
   - **Agent Learning**: None required; use learned schema from Step 5.
   - **Task**: Orchestrate ETL, validation, and scoring with Airflow DAGs or n8n, integrating sub-agents and HITL checkpoints.
9. **Testing and Validation**:
   - **Dataset Used**: Processed data and model outputs.
   - **Agent Learning**: Validator sub-agent reuses schema knowledge from Step 5.
   - **Task**: Test ETL pipelines (Great Expectations) and model (AUC-ROC, precision, recall).
   - **HITL**: Validate data quality reports and model metrics.
10. **Deployment and Monitoring**:
    - **Dataset Used**: Processed data and model outputs.
    - **Agent Learning**: None required.
    - **Task**: Deploy to Databricks/SageMaker, auto-push code to GitHub, and monitor with logging and HITL alerts.
11. **Scalability and Maintenance**:
    - **Dataset Used**: New or updated Home Credit data.
    - **Agent Learning**: Agents relearn schema for new tables (e.g., `previous_applications.csv`) via schema analysis.
    - **Learning How**: Use PySpark or Great Expectations for dynamic schema updates.
    - **Task**: Scale pipelines and retrain models, with RLHF to fine-tune based on HITL feedback.

Ensure the guide uses the latest tools and frameworks available in 2025, referencing platforms like LangChain, AutoGen, Great Expectations, H2O, Databricks, or others from recent sources (e.g., Medium, Databricks, Kaggle blogs). Highlight how sub-agents and tools integrate seamlessly for a modular pipeline. Provide code snippets or configurations (e.g., Python for ETL/AutoML, YAML for Airflow, JSON for parsed backlog) wrapped in artifact tags. Include a sample GitHub repository structure and a report template. Avoid outdated tools or vague instructions, and prioritize clarity, modularity, and logical flow.


# AI-Driven Data Engineering and Scoring Automation Guide for Home Credit Default Risk with HITL
## Step 1: Natural Language Parsing
...
## Step 2: Objective Definition with HITL
...
## Step 3: Technology Stack Selection
...
## Step 4: Data Preparation and Integration
...
## Step 5: ETL Automation with AI Agents and HITL
...
## Step 6: Tool and API Integration
...
## Step 7: Model Integration with HITL
...
## Step 8: Workflow Automation
...
## Step 9: Testing and Validation
...
## Step 10: Deployment and Monitoring
...
## Step 11: Scalability and Maintenance
...


```python
# Sample Python code for ETL and scoring pipeline for Home Credit Default Risk with HITL
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pycaret.classification import *
from great_expectations.dataset import PandasDataset
import json
import git
import os

# Initialize Spark session
spark = SparkSession.builder.appName("HomeCreditETL").getOrCreate()

# Sub-agent 1: Parse natural language backlog item
def parse_backlog_item(instruction):
    # Use LangChain/AutoGen (simplified example)
    parsed_output = {
        "task": "ETL and Scoring",
        "dataset": "Home Credit Default Risk",
        "source_tables": ["application_train.csv", "bureau.csv", "bureau_balance.csv"],
        "transformations": [
            "handle missing values in application_train",
            "aggregate bureau data by SK_ID_CURR",
            "join bureau_balance with bureau",
            "encode categorical variables"
        ],
        "business_rules": ["score loan repayment likelihood from 0-100"],
        "output": "scored_applications"
    }
    # HITL: Save for review
    with open("parsed_backlog.json", "w") as f:
        json.dump(parsed_output, f)
    return parsed_output

# Sub-agent 2: Learn dataset schema
def learn_dataset_schema():
    app_train = spark.read.csv("application_train.csv", header=True, inferSchema=True)
    bureau = spark.read.csv("bureau.csv", header=True, inferSchema=True)
    bureau_balance = spark.read.csv("bureau_balance.csv", header=True, inferSchema=True)
    schema_info = {
        "application_train": [col for col in app_train.columns],
        "bureau": [col for col in bureau.columns],
        "bureau_balance": [col for col in bureau_balance.columns],
        "relationships": [
            {"table": "application_train", "key": "SK_ID_CURR", "joins_to": "bureau"},
            {"table": "bureau", "key": "SK_ID_BUREAU", "joins_to": "bureau_balance"}
        ]
    }
    with open("schema_info.json", "w") as f:
        json.dump(schema_info, f)
    return schema_info

# Sub-agent 3: Validate parsed backlog with HITL
def validate_backlog():
    with open("parsed_backlog.json", "r") as f:
        tasks = json.load(f)
    with open("schema_info.json", "r") as f:
        schema = json.load(f)
    print("Parsed backlog:", tasks)
    print("Dataset schema:", schema)
    human_approval = input("Approve backlog tasks and schema? (yes/no): ")
    return human_approval.lower() == "yes"

# Sub-agent 4: ETL - Extract
def extract_data():
    app_train = spark.read.csv("application_train.csv", header=True, inferSchema=True)
    bureau = spark.read.csv("bureau.csv", header=True, inferSchema=True)
    bureau_balance = spark.read.csv("bureau_balance.csv", header=True, inferSchema=True)
    return app_train, bureau, bureau_balance

# Sub-agent 5: ETL - Transform with Great Expectations and HITL
def transform_data(app_train, bureau, bureau_balance):
    # Handle missing values
    app_train = app_train.fillna({"AMT_ANNUITY": app_train.agg(F.mean("AMT_ANNUITY")).collect()[0][0]})
    # Aggregate bureau_balance
    bureau_balance_agg = bureau_balance.groupBy("SK_ID_BUREAU").agg(
        F.avg("MONTHS_BALANCE").alias("AVG_MONTHS_BALANCE")
    )
    # Join bureau with bureau_balance
    bureau = bureau.join(bureau_balance_agg, "SK_ID_BUREAU", "left")
    # Aggregate bureau data
    bureau_agg = bureau.groupBy("SK_ID_CURR").agg(
        F.avg("AMT_CREDIT_SUM").alias("BUREAU_AVG_CREDIT"),
        F.count("AMT_CREDIT_SUM").alias("BUREAU_LOAN_COUNT")
    )
    # Join with application_train
    data = app_train.join(bureau_agg, "SK_ID_CURR", "left")
    # Encode categorical variables
    data = data.withColumn("NAME_CONTRACT_TYPE_CASH", F.when(F.col("NAME_CONTRACT_TYPE") == "Cash loans", 1).otherwise(0))
    # Optimize partitioning
    data = data.repartition(100, "SK_ID_CURR")
    # Data quality check with Great Expectations
    data_pandas = data.limit(1000).toPandas()
    gx_dataset = PandasDataset(data_pandas)
    gx_dataset.expect_column_values_to_not_be_null("SK_ID_CURR")
    validation_result = gx_dataset.validate()
    with open("data_validation.json", "w") as f:
        json.dump(validation_result.to_json_dict(), f)
    print("Data validation results saved.")
    human_approval = input("Approve transformed data and validation? (yes/no): ")
    if human_approval.lower() == "yes":
        return data
    else:
        raise ValueError("Transformed data not approved.")

# Sub-agent 6: ETL雷

# ETL - Load
def load_data(data):
    data.write.csv("processed_home_credit", header=True, mode="overwrite")
    print("Data loaded to target.")

# Sub-agent 7: AutoML Scoring with HITL
def score_model(data):
    data_pandas = data.toPandas()
    X = data_pandas.drop(["SK_ID_CURR", "TARGET"], axis=1, errors="ignore")
    y = data_pandas["TARGET"] if "TARGET" in data_pandas.columns else None
    # Learn feature distributions
    feature_summary = X.describe().to_dict()
    with open("feature_summary.json", "w") as f:
        json.dump(feature_summary, f)
    # Setup PyCaret
    clf = setup(data=X, target=y, session_id=42, silent=True)
    best_model = compare_models()
    predictions = predict_model(best_model, data=X)
    # Scale scores to 0-100
    predictions["repayment_score"] = 100 * (1 - predictions["Score"])
    predictions[["SK_ID_CURR", "repayment_score"]].to_csv("scored_applications.csv")
    print("Sample scores:", predictions[["SK_ID_CURR", "repayment_score"]].head())
    human_approval = input("Approve model scores and features? (yes/no): ")
    if human_approval.lower() == "yes":
        return best_model, predictions
    else:
        raise ValueError("Model scores not approved.")

# Sub-agent 8: Deploy to Git
def deploy_to_git():
    repo = git.Repo.init("home_credit_pipeline")
    repo.git.add(".")
    repo.index.commit("Automated ETL and scoring pipeline")
    print("Code pushed to GitHub.")
    human_approval = input("Approve deployment? (yes/no): ")
    return human_approval.lower() == "yes"

# Main pipeline
if __name__ == "__main__":
    instruction = "Generate ETL pipeline for Home Credit data, join application and bureau tables, include bureau balance, score applicants for repayment"
    parsed_tasks = parse_backlog_item(instruction)
    learn_dataset_schema()
    if validate_backlog():
        app_train, bureau, bureau_balance = extract_data()
        transformed_data = transform_data(app_train, bureau, bureau_balance)
        load_data(transformed_data)
        model, predictions = score_model(transformed_data)
        if deploy_to_git():
            print("Pipeline deployed successfully.")

```


# GitHub Repository Structure for Home Credit AI Agent

```
home_credit_pipeline/
├── src/
│   ├── backlog_parser.py       # Sub-agent for NLP parsing with LangChain/AutoGen
│   ├── schema_learner.py      # Sub-agent for dataset schema analysis
│   ├── etl_generator.py       # Sub-agent for PySpark ETL code generation
│   ├── validator.py           # Sub-agent for Great Expectations checks
│   ├── model_scorer.py        # Sub-agent for AutoML scoring with PyCaret
├── dags/
│   ├── home_credit_dag.py     # Airflow DAG for orchestration
├── data/
│   ├── parsed_backlog.json    # Parsed backlog items
│   ├── schema_info.json       # Dataset schema and relationships
│   ├── data_validation.json   # Great Expectations results
│   ├── feature_summary.json   # Feature distribution summary
│   ├── scored_applications.csv # Model scores
├── report/
│   ├── report.md              # Design choices and limitations
├── README.md                  # Project overview
├── requirements.txt           # Dependencies
```




# AI Agent for Home Credit Data Engineering and Scoring

## Design Choices
- **Modularity**: Sub-agents for parsing, schema learning, ETL, validation, and scoring ensure flexibility.
- **NLP**: LangChain and AutoGen parse backlog items into structured JSON, with schema learning for dataset context.
- **ETL**: PySpark with partitioning optimizes large-scale processing of `application_train.csv`, `bureau.csv`, and `bureau_balance.csv`.
- **Validation**: Great Expectations ensures data quality, with HITL for anomaly review.
- **AutoML**: PyCaret automates model selection and scoring, producing 0–100 repayment scores.
- **Orchestration**: Airflow DAGs manage workflows, with Git for version control.
- **HITL**: Human approval at parsing, schema validation, data quality, scoring, and deployment ensures reliability.
- **RLHF**: Human feedback fine-tunes agent performance.

## Limitations
- **LLM Costs**: GPT-4/Claude usage may increase costs; rule-based fallbacks mitigate this.
- **Dataset Complexity**: Joining `bureau_balance.csv` requires careful schema mapping.
- **Scalability**: Large datasets need distributed systems like Databricks.
- **HITL Bottlenecks**: Human reviews may delay automation if not optimized.

## Conclusion
The AI agent automates ETL and scoring for the Home Credit Dataset, leveraging schema learning, modularity, and HITL for robust performance, with scalability and RLHF potential.



---

### Key Updates to the Prompt:
- **Dataset Specification**: Explicitly defined which parts of the Home Credit Default Risk Dataset (`application_train.csv`, `bureau.csv`, `bureau_balance.csv`) are used at each step (e.g., `bureau_balance.csv` in ETL for aggregating monthly balance data).
- **Agent Learning**: Added details on when and how agents learn about the dataset:
  - **Step 1**: Learn dataset context via metadata or Kaggle API descriptions.
  - **Step 2**: Analyze schema of `application_train.csv`, `bureau.csv`, `bureau_balance.csv` using PySpark or Great Expectations.
  - **Step 4**: Learn table relationships (e.g., `SK_ID_CURR`, `SK_ID_BUREAU`) via schema analysis.
  - **Step 5**: Reuse schema knowledge for joins and aggregations.
  - **Step 7**: Learn feature distributions via PyCaret or MLflow profiling.
  - **Step 11**: Relearn schema for new tables (e.g., `previous_applications.csv`).
- **Learning Methods**: Specified schema analysis (PySpark’s `describe()`), statistical profiling (Pandas or PyCaret), and metadata extraction (Kaggle API or documentation).
- **Code Artifact**: Updated to include:
  - A `learn_dataset_schema` sub-agent to analyze and save schema/relationship metadata.
  - ETL transformations for `bureau_balance.csv` (e.g., aggregating `MONTHS_BALANCE`).
  - Optimized partitioning in PySpark for scalability.
- **Modularity**: Reinforced sub-agent design (e.g., schema learner, SQL generator) with rule-based fallbacks.
- **Previous Requirements**: Retained all prior requirements (HITL, LangChain/AutoGen, Great Expectations, PyCaret, Airflow, Git, RLHF) and ensured seamless integration.
- **Submission**: Updated GitHub structure to include `schema_learner.py` and `schema_info.json` for dataset learning.

### Next Steps:
Would you like me to generate the full guide using this updated prompt, or do you have further refinements (e.g., specific learning algorithms for RLHF, additional tables like `previous_applications.csv`, or a focus on a particular sub-agent)? Alternatively, I can provide a detailed implementation for a specific step (e.g., schema learning or ETL automation) or set up the GitHub repository structure with sample files. Let me know your preference!