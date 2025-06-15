# schemas.py
# Defines Pydantic schemas for structured data models used throughout the ETL pipeline system.

from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class DataCleaningStep(BaseModel):
    """Defines a single data cleaning or preprocessing operation."""
    type: str = Field(..., description="Type of cleaning operation (e.g., 'imputation', 'outlier_removal', 'feature_engineering')")
    details: dict = Field(..., description="Specific parameters for the cleaning operation (e.g., {'strategy': 'mean', 'columns': ['age']})")

class ScoringModel(BaseModel):
    """Defines the details of a scoring or prediction model."""
    name: str = Field(..., description="Name of the scoring model (e.g., 'Logistic Regression', 'XGBoost', 'Generic ML Model')")
    objective: str = Field(..., description="The objective of the scoring (e.g., 'predict repayment likelihood', 'classify loan default')")
    target_column: str = Field(..., description="The target column for the scoring model (e.g., 'TARGET', 'LOAN_STATUS')")
    features: Optional[List[str]] = Field(None, description="Optional: List of features to use for the model. If not specified, all relevant features after ETL will be considered.")

class JoinOperation(BaseModel):
    """Defines a data join operation between two table_name."""
    left_table: str = Field(..., description="Name of the left table for the join (e.g., 'application_train')")
    right_table: str = Field(..., description="Name of the right table for the join (e.g., 'bureau')")
    join_type: Literal["inner", "left", "right", "outer"] = Field(..., description="Type of join (e.g., 'left', 'inner'). Defaults to 'left' if not specified.")
    on_columns: List[str] = Field(..., description="List of columns to join on (e.g., ['SK_ID_CURR'])")
    description: Optional[str] = Field(None, description="A brief description of the join operation.")

class ETLTaskDefinition(BaseModel):
    """The overarching schema for a complete ETL pipeline task."""
    pipeline_name: str = Field(..., description="A descriptive name for the ETL pipeline (e.g., 'HomeCredit_ApplicantScoring_Pipeline')")
    main_goal: str = Field(..., description="The overarching objective of the ETL pipeline (e.g., 'predict repayment likelihood for loan applicants')")
    initial_tables: List[str] = Field([], description="List of initial tables required from the dataset (e.g., ['application_train', 'bureau', 'bureau_balance'])")
    join_operations: List[JoinOperation] = Field([], description="List of join operations to perform in sequence.")
    data_cleaning_steps: List[DataCleaningStep] = Field([], description="List of data cleaning and preprocessing steps to apply.")
    scoring_model: Optional[ScoringModel] = Field(None, description="Details about the scoring model to be applied, if any.")
    output_format: Optional[str] = Field("dataframe", description="Desired output format after ETL (e.g., 'dataframe', 'csv', 'parquet').")
    output_location: Optional[str] = Field(None, description="Where to save the final output (e.g., 's3://my-bucket/processed_data/').")

class DatasetMetadata(BaseModel):
    """Schema for individual pieces of dataset metadata, useful for vector embeddings."""
    table_name: Optional[str] = None
    column_name: Optional[str] = None
    description: str
    data_type: Optional[str] = None
    example_values: Optional[List[str]] = None
    is_target: bool = False
    is_id: bool = False
