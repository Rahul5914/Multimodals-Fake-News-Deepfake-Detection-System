"""
utils/image_utils.py
====================
Handles loading of the PyTorch CNN vision model,
image preprocessing, and prediction.
Runs on CPU only — compatible with Streamlit Cloud.
"""

import os
import io
from typing import Tuple, Optional, Any

import numpy as np
from PIL import Image

# ── PyTorch imports wrapped in try/except for clear error messages ───────────
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Model path (relative to repo root) ──────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION_MODEL_PATH = os.path.join(_BASE_DIR, "models", "vision_model.pt")

# ── ImageNet normalisation constants ────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Input size expected by the CNN ──────────────────────────────────────────
_INPUT_SIZE = 224

# ── Label mapping: index → human label ──────────────────────────────────────
LABEL_MAP = {0: "FAKE", 1: "REAL"}


# ── Preprocessing pipeline (standard ImageNet pre-processing) ────────────────
def _build_transform() -> Any:
    """Return a torchvision transform that resizes, crops, and normalises."""
    return T.Compose([
        T.Resize((_INPUT_SIZE, _INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ── Fallback CNN architecture ────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    """
    Lightweight fallback CNN used ONLY when the saved model's class
    definition is not available in the runtime environment.
    Architecture mirrors a common binary-classifier design.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 224→112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 112→56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 56→28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.classifier(self.features(x))


def load_vision_model() -> Tuple[Optional[Any], Optional[str]]:
    """
    Load the PyTorch vision model from disk.

    Strategy:
      1. Try torch.load with weights_only=False (full model or state-dict).
      2. If that fails, instantiate SimpleCNN and load state-dict only.
      3. Return (model, None) on success or (None, error_str) on failure.
    """
    if not TORCH_AVAILABLE:
        return None, "PyTorch is not installed. Add 'torch' to requirements.txt."

    if not os.path.exists(VISION_MODEL_PATH):
        return None, (
            f"vision_model.pt not found at '{VISION_MODEL_PATH}'. "
            "Please upload it to the models/ folder."
        )

    device = torch.device("cpu")  # CPU-only for Streamlit Cloud

    # ── Attempt 1: load full saved model ────────────────────────────────────
    try:
        model = torch.load(VISION_MODEL_PATH, map_location=device, weights_only=False)
        model.eval()
        return model, None
    except Exception as full_load_err:
        pass  # fall through to state-dict strategy

    # ── Attempt 2: load as state-dict into SimpleCNN ────────────────────────
    try:
        state_dict = torch.load(VISION_MODEL_PATH, map_location=device, weights_only=True)
        model = SimpleCNN(num_classes=2)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, None
    except Exception as sd_err:
        return None, (
            f"Could not load vision_model.pt.\n"
            f"Full-model error: {full_load_err}\n"
            f"State-dict error: {sd_err}"
        )


def predict_image(
    image_file: Any,
    model: Any,
) -> Tuple[str, float, Optional[str]]:
    """
    Preprocess an uploaded image file and predict FAKE / REAL.

    Parameters
    ----------
    image_file : file-like object (from st.file_uploader)
    model      : loaded PyTorch model

    Returns
    -------
    (label, confidence, error_message)
    """
    if not TORCH_AVAILABLE:
        return "", 0.0, "PyTorch is not available."

    try:
        # ── Read image ───────────────────────────────────────────────────────
        raw_bytes = image_file.read()
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        # ── Preprocess ───────────────────────────────────────────────────────
        transform = _build_transform()
        tensor = transform(pil_img)            # shape: (3, 224, 224)
        tensor = tensor.unsqueeze(0)           # add batch dim → (1, 3, 224, 224)

        # ── Inference (no gradient needed) ──────────────────────────────────
        device = torch.device("cpu")
        tensor = tensor.to(device)
        model = model.to(device)

        with torch.no_grad():
            logits = model(tensor)             # shape: (1, num_classes)

        # ── Convert logits → probabilities ──────────────────────────────────
        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()

        # ── Determine predicted class and confidence ─────────────────────────
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        label = LABEL_MAP.get(pred_idx, "REAL")

        return label, confidence, None

    except Exception as exc:
        return "", 0.0, str(exc)
