from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON
from app.database.base import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    image_filename = Column(String(255), nullable=False)
    image_path = Column(String(512), nullable=False)
    predicted_class = Column(String(100), nullable=False, index=True)
    confidence_score = Column(Float, nullable=False)
    top_five = Column(JSON, nullable=False)  # list of {class_name: str, confidence: float}
    model_version = Column(String(50), nullable=False, default="CNN-v1.0")
    processing_time_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    job_id = Column(String(64), unique=True, index=True, nullable=False)
    status = Column(String(30), nullable=False, default="pending", index=True)  # pending, running, completed, failed, cancelled
    model_type = Column(String(20), nullable=False, default="cnn")  # cnn or ann
    epochs = Column(Integer, nullable=False, default=5)
    batch_size = Column(Integer, nullable=False, default=16)
    learning_rate = Column(Float, nullable=False, default=0.001)
    image_size = Column(Integer, nullable=False, default=128)
    optimizer_name = Column(String(50), nullable=False, default="Adam (Mini-Batch)")
    current_epoch = Column(Integer, default=0)
    training_metrics = Column(JSON, default=list)  # [{'epoch': 1, 'loss': 1.2, 'acc': 65.4}]
    validation_metrics = Column(JSON, default=list)  # [{'epoch': 1, 'val_loss': 1.4, 'val_acc': 61.2}]
    logs = Column(Text, default="")
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    version_name = Column(String(100), unique=True, index=True, nullable=False)
    model_type = Column(String(20), nullable=False, default="cnn")
    checkpoint_path = Column(String(512), nullable=False)
    num_classes = Column(Integer, nullable=False, default=101)
    image_size = Column(Integer, nullable=False, default=128)
    dataset_metadata = Column(JSON, default=dict)
    training_config = Column(JSON, default=dict)
    evaluation_metrics = Column(JSON, default=dict)
    is_active = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatasetMetadata(Base):
    __tablename__ = "dataset_metadata"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    dataset_name = Column(String(100), nullable=False)
    dataset_source = Column(String(255), nullable=False)
    class_count = Column(Integer, nullable=False, default=101)
    image_count = Column(Integer, nullable=False, default=0)
    split_info = Column(JSON, default=dict)
    classes_list = Column(JSON, default=list)
    last_inspected_at = Column(DateTime, default=datetime.utcnow)
