import torchvision.transforms as T
from PIL import Image
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms(image_size: int = 224) -> T.Compose:
    """Data augmentation transforms for training split."""
    resize_dim = int(image_size * 1.14)
    return T.Compose([
        T.Resize((resize_dim, resize_dim)),
        T.RandomCrop((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_eval_transforms(image_size: int = 224) -> T.Compose:
    """
    Deterministic transforms for validation, testing, and inference.
    Resizes keeping aspect ratio before center crop to prevent food deformation.
    """
    resize_dim = int(image_size * 1.14)
    return T.Compose([
        T.Resize(resize_dim),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Preprocess a single PIL Image into a calibrated (1, 3, H, W) RGB float tensor."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform = get_eval_transforms(image_size)
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Shape: (1, 3, H, W)
