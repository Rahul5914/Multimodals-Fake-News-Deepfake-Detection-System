# 🔍 Fake News Detection App

A production-ready Streamlit web application that detects fake news using:

- **Text analysis** — scikit-learn NLP model + TF-IDF vectorizer
- **Image analysis** — PyTorch deep-learning CNN

---

## 📁 Project Structure

```
fake-news-detector/
├── app.py                  # Main Streamlit entry point
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore
├── models/
│   ├── text_model.pkl      # Trained sklearn classifier
│   ├── vectorizer.pkl      # Fitted TF-IDF / CountVectorizer
│   └── vision_model.pt     # Trained PyTorch CNN
└── utils/
    ├── __init__.py
    ├── text_utils.py       # Text preprocessing & prediction
    └── image_utils.py      # Image preprocessing & prediction
```

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/fake-news-detector.git
cd fake-news-detector
```

### 2. Create & activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your models

Copy your trained files into the `models/` directory:

```
models/text_model.pkl
models/vectorizer.pkl
models/vision_model.pt
```

### 5. Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deploy on Streamlit Cloud

### Step 1 — Push to GitHub

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### Step 2 — Add model files via Git LFS (recommended for large files)

```bash
git lfs install
git lfs track "models/*.pkl" "models/*.pt"
git add .gitattributes models/
git commit -m "Add model files via LFS"
git push origin main
```

> **Alternative**: Host models on Google Drive / S3 and load them at runtime using
> `urllib.request.urlretrieve` or the `gdown` library. Store the download URLs in
> Streamlit Secrets (`st.secrets`).

### Step 3 — Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
2. Click **New app** → select your GitHub repo.
3. Set **Main file path** to `app.py`.
4. Click **Deploy**.

Streamlit Cloud automatically installs `requirements.txt` on first boot.

---

## 🧠 Model Details

| Model | Framework | Input | Output |
|-------|-----------|-------|--------|
| `text_model.pkl` | scikit-learn | TF-IDF vector | 0 = FAKE, 1 = REAL |
| `vectorizer.pkl` | scikit-learn | Raw text | TF-IDF sparse matrix |
| `vision_model.pt` | PyTorch (CPU) | 224×224 RGB image | 0 = FAKE, 1 = REAL |

### Text preprocessing pipeline

1. Lowercase
2. Remove URLs
3. Remove punctuation & digits
4. Remove English stopwords (NLTK, with hardcoded fallback)
5. Transform with fitted vectorizer

### Image preprocessing pipeline

1. Resize to 224×224
2. Convert to tensor
3. Normalise with ImageNet mean/std

---

## ⚠️ Notes

- The app runs **CPU-only** — no GPU required.
- Models are loaded once and cached via `@st.cache_resource`.
- All paths are relative — no hardcoded local paths.
- Graceful error messages are shown if model files are missing or corrupt.

---

## 📄 License

MIT — for research and educational use only.
