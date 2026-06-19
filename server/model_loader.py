"""
Recycling Lab Tycoon — Model Loader & Inference Engine
=======================================================
Load 5 pre-trained garbage classification models and run inference.
Architectures and preprocessing reproduce exactly what was used during training
in the Kaggle notebook (garbage-classification.ipynb).
"""

import os
import time
import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Class mapping — from ImageFolder (alphabetical order, 12 classes)
# This is the ACTUAL mapping used during training, NOT the CLASSES dict
# ──────────────────────────────────────────────────────────────────────
CLASS_NAMES: List[str] = [
    "battery",       # 0
    "biological",    # 1
    "brown-glass",   # 2
    "cardboard",     # 3
    "clothes",       # 4
    "green-glass",   # 5
    "metal",         # 6
    "paper",         # 7
    "plastic",       # 8
    "shoes",         # 9
    "trash",         # 10
    "white-glass",   # 11
]

NUM_CLASSES: int = len(CLASS_NAMES)  # 12

# ──────────────────────────────────────────────────────────────────────
# Model registry — maps name → weight filename
# ──────────────────────────────────────────────────────────────────────
MODEL_REGISTRY: Dict[str, str] = {
    "resnet50":           "resnet50_best.pth",
    "efficientnet_b0":    "efficientnet_b0_best.pth",
    "efficientnet_v2_s":  "efficientnet_v2_s_best.pth",
    "mobilenet_v3_large": "mobilenet_v3_large_best.pth",
    "convnext_tiny":      "convnext_tiny_best.pth",
}

# ──────────────────────────────────────────────────────────────────────
# Preprocessing — MUST match training pipeline exactly
# NOTE: During training, RandomHorizontalFlip and RandomRotation were
#       used as augmentation.  For inference we only need deterministic
#       transforms: Resize → ToTensor → Normalize.
# ──────────────────────────────────────────────────────────────────────
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _build_model(model_name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Reconstruct the model architecture with the correct classification head.
    This is an exact copy of the ``get_models()`` function from the training
    notebook so that ``load_state_dict`` works without key mismatches.
    """
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


class ModelLoader:
    """
    Manages loading and running inference for all 5 ensemble models.

    Parameters
    ----------
    weights_dir : str
        Path to the directory containing ``.pth`` weight files.
    device : str
        ``"cpu"`` or ``"cuda"`` (auto-detected if not specified).
    """

    def __init__(self, weights_dir: str, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.weights_dir = weights_dir
        self.models: Dict[str, nn.Module] = {}
        self._load_all_models()

    # ── Loading ───────────────────────────────────────────────────────

    def _load_all_models(self) -> None:
        """Load every model listed in MODEL_REGISTRY."""
        logger.info("=" * 60)
        logger.info("Loading ensemble models …")
        logger.info("  Device : %s", self.device)
        logger.info("  Weights: %s", self.weights_dir)
        logger.info("=" * 60)

        for name, weight_file in MODEL_REGISTRY.items():
            path = os.path.join(self.weights_dir, weight_file)
            if not os.path.isfile(path):
                logger.error("Weight file not found: %s — skipping %s", path, name)
                continue

            t0 = time.perf_counter()
            model = _build_model(name)
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            elapsed = time.perf_counter() - t0

            self.models[name] = model
            logger.info("  ✓ %-22s loaded in %.2fs", name, elapsed)

        if not self.models:
            raise RuntimeError("No models were loaded — check weights_dir path.")

        logger.info("Loaded %d / %d models.", len(self.models), len(MODEL_REGISTRY))

    # ── Inference ─────────────────────────────────────────────────────

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Convert an OpenCV BGR frame to a batched tensor ready for inference.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR frame from OpenCV (H, W, 3), dtype uint8.

        Returns
        -------
        torch.Tensor
            Shape ``(1, 3, 244, 244)`` on ``self.device``.
        """
        # OpenCV uses BGR — convert to RGB for PIL / torchvision
        rgb = frame[:, :, ::-1].copy()
        pil_img = Image.fromarray(rgb)
        tensor = INFERENCE_TRANSFORM(pil_img)
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_single(
        self, model_name: str, input_tensor: torch.Tensor
    ) -> Tuple[str, float, np.ndarray]:
        """
        Run inference on a single model.

        Returns
        -------
        (predicted_class, confidence, probabilities)
        """
        model = self.models[model_name]
        logits = model(input_tensor)                        # (1, 12)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]   # (12,)
        idx = int(np.argmax(probs))
        return CLASS_NAMES[idx], float(probs[idx]), probs

    @torch.no_grad()
    def predict_all(
        self, frame: np.ndarray
    ) -> List[Tuple[str, float, np.ndarray]]:
        """
        Run inference on ALL loaded models for the given camera frame.

        Returns
        -------
        List of ``(predicted_class, confidence, probabilities)`` — one per model.
        """
        input_tensor = self.preprocess(frame)
        results = []
        for name in self.models:
            results.append(self.predict_single(name, input_tensor))
        return results
