from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

# Health Check
class HealthResponse(BaseModel):
    status: str
    database: str
    device: str
    active_model: Optional[str]
    timestamp: datetime

# Top-5 item
class TopClassPrediction(BaseModel):
    class_name: str
    confidence: float = Field(..., description="Probability between 0.0 and 1.0")
    percentage: float = Field(..., description="Percentage formatted to 1 decimal place")

class PredictionResponse(BaseModel):
    id: int
    image_filename: str
    image_url: str
    predicted_class: str
    confidence_score: float
    confidence_percentage: float
    confidence_level: str = Field(default="moderate", description="high, moderate, or low")
    confidence_label: str = Field(default="Moderate Confidence", description="Human-readable confidence grade")
    is_confident: bool = Field(default=True, description="False if top confidence is below threshold")
    confidence_explanation: str = Field(default="", description="Calibrated interpretation of output distribution")
    top_five: List[TopClassPrediction]
    model_version: str
    processing_time_ms: float
    created_at: datetime

# Paginated Predictions
class PaginatedPredictionsResponse(BaseModel):
    items: List[PredictionResponse]
    total: int
    page: int
    page_size: int
    total_pages: int

# Dashboard Statistics
class CategoryDistributionItem(BaseModel):
    category: str
    count: int

class RecentActivityItem(BaseModel):
    date: str
    count: int

class DashboardStatsResponse(BaseModel):
    total_predictions: int
    unique_categories_count: int
    active_model_version: str
    model_status: str
    avg_processing_time_ms: float
    recent_predictions: List[PredictionResponse]
    category_distribution: List[CategoryDistributionItem]
    recent_activity: List[RecentActivityItem]

# Dataset Schemas
class DatasetSplitInfo(BaseModel):
    train: int
    validation: int
    test: int

class DatasetInfoResponse(BaseModel):
    dataset_name: str
    dataset_source: str
    class_count: int
    image_count: int
    splits: Dict[str, int]
    classes: List[str]
    dimensions_summary: Dict[str, Any]
    preprocessing_info: Dict[str, Any]
    is_cached_locally: bool
    status_message: str

class DatasetClassesResponse(BaseModel):
    total: int
    classes: List[str]

class SampleImageItem(BaseModel):
    class_name: str
    image_url: str
    width: int
    height: int

class DatasetSamplesResponse(BaseModel):
    samples: List[SampleImageItem]

# Training Schemas
class TrainingStartRequest(BaseModel):
    model_type: str = Field(default="cnn", description="cnn or ann")
    epochs: int = Field(default=5, ge=1, le=50)
    batch_size: int = Field(default=16, ge=4, le=128)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)
    image_size: int = Field(default=128, ge=64, le=256)
    optimizer_name: str = Field(default="Adam (Mini-Batch)")

class MetricEpochItem(BaseModel):
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float

class TrainingStatusResponse(BaseModel):
    job_id: str
    status: str
    model_type: str
    epochs: int
    batch_size: int
    learning_rate: float
    current_epoch: int
    training_metrics: List[Dict[str, Any]]
    validation_metrics: List[Dict[str, Any]]
    logs: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None

# Model Evaluation & Info
class PerClassMetric(BaseModel):
    class_name: str
    precision: float
    recall: float
    f1_score: float
    support: int

class ModelEvaluationResponse(BaseModel):
    model_version: str
    model_type: str
    test_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1_score: float
    weighted_f1_score: float
    total_test_samples: int
    per_class_metrics: List[PerClassMetric]
    confusion_matrix_labels: List[str]
    confusion_matrix_matrix: List[List[int]]
    loss_curves: List[Dict[str, Any]]
    accuracy_curves: List[Dict[str, Any]]
    evaluation_report_markdown: str

class ModelInfoResponse(BaseModel):
    version_name: str
    model_type: str
    num_classes: int
    image_size: int
    is_active: bool
    checkpoint_exists: bool
    checkpoint_path: str
    created_at: Optional[datetime] = None
    architecture_overview: Dict[str, Any]
    training_hyperparameters: Dict[str, Any]
    educational_concepts: Dict[str, str]

class ModelComparisonItem(BaseModel):
    model_name: str
    model_type: str
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    test_accuracy: float
    macro_f1: float
    parameter_count: int
    avg_inference_ms: float
    notes: str

class ModelComparisonResponse(BaseModel):
    comparisons: List[ModelComparisonItem]
