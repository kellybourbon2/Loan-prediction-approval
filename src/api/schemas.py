from typing import Literal

from pydantic import BaseModel, Field


class LoanApplication(BaseModel):
    person_age: int = Field(ge=18, le=125)
    person_income: float = Field(gt=0)
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: float = Field(ge=0)
    loan_intent: Literal[
        "EDUCATION",
        "MEDICAL",
        "PERSONAL",
        "VENTURE",
        "DEBTCONSOLIDATION",
        "HOMEIMPROVEMENT",
    ]
    loan_amnt: float = Field(gt=0)
    loan_percent_income: float = Field(ge=0)
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "person_age": 28,
                "person_income": 45000,
                "person_home_ownership": "RENT",
                "person_emp_length": 3.0,
                "loan_intent": "PERSONAL",
                "loan_amnt": 10000,
                "loan_percent_income": 0.22,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 4,
            }
        }
    }


class PredictionResponse(BaseModel):
    loan_status: int
    approved: bool
    probability: float


class FeatureContribution(BaseModel):
    feature: str
    label: str
    shap: float


class ExplainResponse(BaseModel):
    base_value: float
    features: list[FeatureContribution]
