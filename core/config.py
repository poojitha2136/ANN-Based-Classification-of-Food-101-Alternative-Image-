import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR.parent / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
CACHE_DIR = DATA_DIR / "dataset_cache"
SAMPLES_DIR = DATA_DIR / "samples"

# Ensure runtime directories exist
for directory in [DATA_DIR, UPLOADS_DIR, CHECKPOINTS_DIR, CACHE_DIR, SAMPLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

class Settings(BaseSettings):
    PROJECT_NAME: str = "ANN-Based Food Image Classification System Using CNN"
    PROJECT_NUMBER: str = "P35"
    STUDENT_NAME: str = "KANCHARLA POOJITHA"
    STUDENT_ROLL: str = "10"
    REGISTRATION_NUMBER: str = "2023002679"
    SYLLABUS_UNIT: str = "Unit 3"
    COURSE_OUTCOME: str = "CO3"
    PROBLEM_TYPE: str = "Multi-class food image classification"
    OPTIMIZATION_TECHNIQUE: str = "Mini-Batch Gradient Descent (Adam/SGD with Momentum)"
    DATASET_NAME: str = "Food-101 Alternative Image Set"
    DATASET_URL: str = "https://huggingface.co/datasets/galilai-group/food101"
    
    API_V1_STR: str = "/api"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True
    
    # SQLite Database
    DATABASE_URL: str = f"sqlite:///{(DATA_DIR / 'food_classifier.db').as_posix()}"
    
    # Storage Paths
    UPLOADS_PATH: Path = UPLOADS_DIR
    CHECKPOINTS_PATH: Path = CHECKPOINTS_DIR
    CACHE_PATH: Path = CACHE_DIR
    SAMPLES_PATH: Path = SAMPLES_DIR
    
    # ML Hyperparameters Default
    DEFAULT_IMAGE_SIZE: int = 128
    DEFAULT_BATCH_SIZE: int = 16
    DEFAULT_LEARNING_RATE: float = 0.001
    DEFAULT_EPOCHS: int = 5
    NUM_CLASSES: int = 101
    
    # Security & Limits
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: list[str] = ["jpg", "jpeg", "png", "webp"]
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    model_config = {"env_file": ".env", "extra": "ignore"}

settings = Settings()
