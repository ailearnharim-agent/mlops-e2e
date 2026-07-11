"""Pydantic request/response models for the prediction API."""

from typing import Optional

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., gt=0, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=0, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., gt=0, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (1 = yes)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    # slope/ca/thal are frequently unrecorded outside of Cleveland in the training
    # data (see notebooks/01_eda.ipynb); the model was explicitly trained to handle
    # them as missing, so the API accepts them as optional.
    slope: Optional[float] = Field(None, ge=1, le=3, description="Slope of peak exercise ST segment (1-3)")
    ca: Optional[float] = Field(
        None, ge=0, le=3, description="Number of major vessels colored by fluoroscopy (0-3)"
    )
    thal: Optional[float] = Field(
        None, description="Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 1,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 2,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 3,
                "ca": 0,
                "thal": 6,
            }
        }
    }


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    prediction: int = Field(..., description="0 = no disease, 1 = disease present")
    label: str
    confidence: float = Field(
        ..., ge=0, le=1, description="Model's predicted probability of the predicted class"
    )
    model_version: str


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
