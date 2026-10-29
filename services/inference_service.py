import time
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image

from app.core.config import settings
from app.ml.cnn_model import FoodCNN, FoodBaselineANN
from app.ml.transforms import preprocess_image
from app.services.dataset_service import FOOD_101_CLASSES, calibrate_confidence

logger = logging.getLogger("food_classifier.inference")

class InferenceService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.model_version: str = "MobileNetV2-Food101"
        self.model_type: str = "cnn"
        self.num_classes: int = len(FOOD_101_CLASSES)
        self.classes: List[str] = FOOD_101_CLASSES
        self.is_loaded: bool = False
        
        # Priority checkpoint path: high-accuracy MobileNetV2 trained on Food-101
        self.checkpoint_path: Path = settings.CHECKPOINTS_PATH / "food_mobilenet_v2.pt"
        self.load_model()

    def load_model(self, checkpoint_path: Optional[str] = None):
        """
        Loads model weights from checkpoint into evaluation mode.
        Dynamically adapts to MobileNetV2, custom FoodCNN, or FoodBaselineANN.
        """
        if checkpoint_path:
            target_path = Path(checkpoint_path)
        else:
            # Check high-accuracy Food-101 MobileNetV2 first, fallback to best_food_cnn.pt
            if self.checkpoint_path.exists():
                target_path = self.checkpoint_path
            else:
                target_path = settings.CHECKPOINTS_PATH / "best_food_cnn.pt"

        self.checkpoint_path = target_path

        if not target_path.exists():
            logger.warning(f"[InferenceService] Checkpoint {target_path} not found. Initializing fallback FoodCNN.")
            self.model = FoodCNN(num_classes=self.num_classes).to(self.device)
            self.model.eval()
            self.model_version = "FoodCNN-Fallback"
            self.model_type = "cnn"
            self.is_loaded = True
            return

        try:
            logger.info(f"[InferenceService] Loading checkpoint from {target_path}...")
            checkpoint = torch.load(target_path, map_location=self.device)

            # Extract state dict
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    raw_sd = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    raw_sd = checkpoint["model_state_dict"]
                else:
                    raw_sd = checkpoint
            else:
                raw_sd = checkpoint

            # Strip possible DataParallel 'module.' prefix
            cleaned_sd = {k.replace("module.", ""): v for k, v in raw_sd.items()}
            first_keys = list(cleaned_sd.keys())[:10]

            # Detect architecture from checkpoint weights
            if any("features.0.0.weight" in k or "classifier.1.weight" in k for k in cleaned_sd.keys()):
                # MobileNetV2
                logger.info("[InferenceService] Detected MobileNetV2 Food-101 architecture.")
                self.model = models.mobilenet_v2(num_classes=self.num_classes)
                self.model.load_state_dict(cleaned_sd)
                self.model_version = "MobileNetV2-Food101"
                self.model_type = "cnn"
            elif any("conv1.weight" in k or "features.0" in k for k in cleaned_sd.keys()):
                # Custom FoodCNN
                logger.info("[InferenceService] Detected FoodCNN architecture.")
                self.model = FoodCNN(num_classes=self.num_classes)
                self.model.load_state_dict(cleaned_sd)
                self.model_version = checkpoint.get("version_name", "FoodCNN-v1.0") if isinstance(checkpoint, dict) else "FoodCNN-v1.0"
                self.model_type = "cnn"
            elif any("fc1.weight" in k for k in cleaned_sd.keys()):
                # Baseline ANN
                logger.info("[InferenceService] Detected FoodBaselineANN architecture.")
                self.model = FoodBaselineANN(num_classes=self.num_classes)
                self.model.load_state_dict(cleaned_sd)
                self.model_version = "BaselineANN-v1.0"
                self.model_type = "ann"
            else:
                # Default attempt with MobileNetV2
                logger.info("[InferenceService] Attempting MobileNetV2 load as standard classifier.")
                self.model = models.mobilenet_v2(num_classes=self.num_classes)
                self.model.load_state_dict(cleaned_sd, strict=False)
                self.model_version = "MobileNetV2-Food101"
                self.model_type = "cnn"

            self.model.to(self.device)
            self.model.eval()  # Set strictly to evaluation mode
            self.is_loaded = True
            logger.info(f"[InferenceService] Model {self.model_version} loaded successfully in eval mode on {self.device}.")

        except Exception as e:
            logger.error(f"[InferenceService] Error loading checkpoint: {e}. Falling back to default FoodCNN.", exc_info=True)
            self.model = FoodCNN(num_classes=self.num_classes).to(self.device)
            self.model.eval()
            self.model_version = "FoodCNN-UntrainedFallback"
            self.model_type = "cnn"
            self.is_loaded = True

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Runs calibrated PyTorch inference on the provided PIL image.
        Returns predicted category, calibrated confidence level, top 5 classes,
        and processing time.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Inference engine is not initialized or model is missing.")

        start_time = time.perf_counter()

        # Ensure RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Standardized preprocessing (256 resize, 224 center crop, ImageNet normalization)
        input_tensor = preprocess_image(image).to(self.device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Real PyTorch inference in eval/inference mode
        with torch.inference_mode():
            logits = self.model(input_tensor)
            # Apply softmax exactly once on raw logits
            probabilities = F.softmax(logits, dim=1).squeeze(0)

        # Validate that probability distribution sums to approximately 1.0
        prob_sum = float(probabilities.sum().item())
        if not (0.98 <= prob_sum <= 1.02):
            logger.warning(f"[InferenceService] Probability sum anomaly: {prob_sum}")

        # Extract top 5 classes sorted in descending order of probability
        top_k = min(5, self.num_classes)
        top_probs, top_indices = torch.topk(probabilities, k=top_k)

        top_five = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            cls_name = self.classes[idx] if idx < len(self.classes) else f"class_{idx}"
            top_five.append({
                "class_name": cls_name,
                "confidence": round(float(prob), 4),
                "percentage": round(float(prob) * 100, 1)
            })

        processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        top_prediction = top_five[0]

        # Calibrated confidence analysis
        calibrated = calibrate_confidence(top_prediction["confidence"])

        return {
            "predicted_class": top_prediction["class_name"],
            "confidence_score": top_prediction["confidence"],
            "confidence_percentage": top_prediction["percentage"],
            "confidence_level": calibrated["confidence_level"],
            "confidence_label": calibrated["confidence_label"],
            "is_confident": calibrated["is_confident"],
            "confidence_explanation": calibrated["confidence_explanation"],
            "top_five": top_five,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "processing_time_ms": processing_time_ms
        }

inference_service = InferenceService()
