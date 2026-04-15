"""
utils/text_utils.py
===================
Handles loading of the sklearn text model + vectorizer,
text preprocessing, and prediction.
"""

import os
import re
import string
import pickle
from typing import Tuple, Optional, Any

# ── Optional NLTK stopwords (falls back to a hardcoded set if unavailable) ──
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    # Lightweight fallback stopword list
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "is", "was", "are", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "that", "this", "these",
        "those", "it", "its", "he", "she", "they", "we", "you", "i", "my",
        "your", "his", "her", "their", "our", "not", "no", "so", "up", "out",
        "about", "into", "through", "by", "from", "as", "if", "then", "than",
    }

# ── Model paths (relative to repo root) ─────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_MODEL_PATH = os.path.join(_BASE_DIR, "models", "text_model.pkl")
VECTORIZER_PATH = os.path.join(_BASE_DIR, "models", "vectorizer.pkl")

# ── Label map ────────────────────────────────────────────────────────────────
LABEL_MAP = {0: "FAKE", 1: "REAL"}


def load_text_models() -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    """
    Load the sklearn classifier and TF-IDF / CountVectorizer from disk.

    Returns
    -------
    (model, vectorizer, error_message)
    error_message is None on success, a string on failure.
    """
    model, vectorizer = None, None

    # ── Load classifier ──
    if not os.path.exists(TEXT_MODEL_PATH):
        return None, None, (
            f"text_model.pkl not found at '{TEXT_MODEL_PATH}'. "
            "Please upload it to the models/ folder."
        )
    try:
        with open(TEXT_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as exc:
        return None, None, f"Failed to load text_model.pkl: {exc}"

    # ── Load vectorizer ──
    if not os.path.exists(VECTORIZER_PATH):
        return None, None, (
            f"vectorizer.pkl not found at '{VECTORIZER_PATH}'. "
            "Please upload it to the models/ folder."
        )
    try:
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
    except Exception as exc:
        return None, None, f"Failed to load vectorizer.pkl: {exc}"

    return model, vectorizer, None


def preprocess_text(text: str) -> str:
    """
    Clean and normalise raw news text:
      1. Lowercase
      2. Remove URLs
      3. Remove punctuation
      4. Remove digits
      5. Remove stopwords
      6. Collapse whitespace
    """
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove digits
    text = re.sub(r"\d+", " ", text)
    # Tokenise and remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def predict_text(
    raw_text: str,
    model: Any,
    vectorizer: Any,
) -> Tuple[str, float, Optional[str]]:
    """
    Preprocess raw_text, vectorize, and predict with the sklearn model.

    Returns
    -------
    (label, confidence, error_message)
    label        : "FAKE" or "REAL"
    confidence   : float in [0, 1]
    error_message: None on success, string on failure
    """
    try:
        cleaned = preprocess_text(raw_text)
        if not cleaned.strip():
            return "", 0.0, "Text became empty after preprocessing. Try longer input."

        # Vectorize — expects a list of strings
        features = vectorizer.transform([cleaned])

        # Predict label
        raw_pred = model.predict(features)[0]

        # Confidence — use predict_proba if available, else binary flag
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            confidence = float(max(proba))
        else:
            confidence = 1.0  # Hard classifier without probability support

        # Map numeric / string labels to FAKE / REAL
        if isinstance(raw_pred, (int, float)):
            label = LABEL_MAP.get(int(raw_pred), "REAL")
        else:
            label = str(raw_pred).upper()
            if label not in ("FAKE", "REAL"):
                # Attempt numeric conversion
                label = LABEL_MAP.get(int(label), "REAL")

        return label, confidence, None

    except Exception as exc:
        return "", 0.0, str(exc)
