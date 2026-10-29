import torch
import torch.nn as nn
from typing import Dict, Any

class ConvBlock(nn.Module):
    """Convolutional Block with Conv2D, Batch Normalization, and ReLU."""
    def __init__(self, in_channels: int, out_channels: int, pool: bool = False, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FoodCNN(nn.Module):
    """
    Convolutional Neural Network for Food-101 Alternative Classification.
    Architecture:
    - 4 Convolutional stages with residual feature extraction & batch normalization
    - Max Pooling (spatial reduction)
    - Adaptive Average Pooling
    - Dense classifier with dropout regularization
    """
    def __init__(self, num_classes: int = 101, dropout: float = 0.4):
        super().__init__()
        self.num_classes = num_classes
        self.model_type = "cnn"
        
        # Feature Extractor
        self.stage1 = ConvBlock(3, 32, pool=True, dropout=0.1)      # /2
        self.stage2 = ConvBlock(32, 64, pool=True, dropout=0.15)    # /4
        self.stage3 = ConvBlock(64, 128, pool=True, dropout=0.2)    # /8
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # /16
            nn.Dropout2d(p=0.25)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "model_name": "FoodCNN",
            "model_type": "cnn",
            "num_classes": self.num_classes,
            "trainable_parameters": self.count_parameters(),
            "stages": [
                {"name": "Stage 1", "in": 3, "out": 32, "kernel": "3x3", "pooling": "MaxPool 2x2"},
                {"name": "Stage 2", "in": 32, "out": 64, "kernel": "3x3", "pooling": "MaxPool 2x2"},
                {"name": "Stage 3", "in": 64, "out": 128, "kernel": "3x3", "pooling": "MaxPool 2x2"},
                {"name": "Stage 4", "in": 128, "out": 256, "kernel": "3x3", "pooling": "MaxPool 2x2"},
                {"name": "Global Pooling", "type": "AdaptiveAvgPool2d (1x1)"},
                {"name": "Classifier", "fc1": "256 -> 512 (BN + ReLU + Dropout)", "fc2": f"512 -> {self.num_classes}"}
            ]
        }


class FoodBaselineANN(nn.Module):
    """
    Baseline Artificial Neural Network (Multi-Layer Perceptron).
    Flattens raw image pixels into 1D vectors and uses dense layers.
    Used for academic comparison against CNN to demonstrate spatial feature preservation.
    """
    def __init__(self, num_classes: int = 101, image_size: int = 128, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.model_type = "ann"
        
        input_dim = 3 * image_size * image_size
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "model_name": "FoodBaselineANN (MLP)",
            "model_type": "ann",
            "num_classes": self.num_classes,
            "image_size": f"{self.image_size}x{self.image_size}",
            "input_dimension": 3 * self.image_size * self.image_size,
            "trainable_parameters": self.count_parameters(),
            "layers": [
                {"name": "Flatten", "output": 3 * self.image_size * self.image_size},
                {"name": "Dense 1", "output": "512 (BatchNorm + ReLU + Dropout 0.3)"},
                {"name": "Dense 2", "output": "256 (BatchNorm + ReLU + Dropout 0.3)"},
                {"name": "Output Layer", "output": f"{self.num_classes} logits"}
            ]
        }
