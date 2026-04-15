"""
Fake News Detection App
=======================
Main Streamlit application entry point.
Supports two detection modes:
  1. Text-based (NLP + TF-IDF)
  2. Image-based (PyTorch CNN)
"""

import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

from utils.text_utils import load_text_models, predict_text
from utils.image_utils import load_vision_model, predict_image

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* ── Title block ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #f5f5f5 30%, #ff4e4e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 0.95rem;
    color: #7a7f91;
    margin-top: 6px;
    margin-bottom: 2rem;
    letter-spacing: 0.03em;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #161920;
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    color: #7a7f91;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 8px 20px;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: #ff4e4e !important;
    color: #fff !important;
}

/* ── Card wrapper ── */
.card {
    background: #161920;
    border: 1px solid #23272f;
    border-radius: 16px;
    padding: 28px 28px 20px 28px;
    margin-top: 16px;
}

/* ── Result badges ── */
.badge-fake {
    display: inline-block;
    background: linear-gradient(135deg, #ff4e4e, #c62828);
    color: #fff;
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 2px;
    padding: 12px 36px;
    border-radius: 50px;
    text-transform: uppercase;
}
.badge-real {
    display: inline-block;
    background: linear-gradient(135deg, #00e096, #00796b);
    color: #fff;
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 2px;
    padding: 12px 36px;
    border-radius: 50px;
    text-transform: uppercase;
}

/* ── Confidence bar ── */
.conf-label {
    font-size: 0.8rem;
    color: #7a7f91;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 18px;
    margin-bottom: 4px;
}

/* ── Textarea / inputs ── */
textarea, .stTextArea textarea {
    background: #1e2229 !important;
    border: 1px solid #2e333e !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #ff4e4e;
    color: #fff;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 10px 32px;
    letter-spacing: 0.05em;
    transition: opacity 0.2s;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.85;
}

/* ── File uploader ── */
.stFileUploader {
    background: #1e2229 !important;
    border: 1px dashed #2e333e !important;
    border-radius: 12px !important;
}

/* ── Divider ── */
hr { border-color: #23272f; }

/* ── Error / warning ── */
.stAlert {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Hero header ─────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🔍 Fake News<br>Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Powered by NLP & Computer Vision — paste text or upload an image to analyse.</p>', unsafe_allow_html=True)
st.markdown("---")


# ── Load models (cached so they load once per session) ──────────────────────
@st.cache_resource(show_spinner=False)
def get_text_models():
    return load_text_models()


@st.cache_resource(show_spinner=False)
def get_vision_model():
    return load_vision_model()


# ── Helper: render result ────────────────────────────────────────────────────
def render_result(label: str, confidence: float):
    """Render a styled prediction badge and confidence bar."""
    badge_class = "badge-fake" if label == "FAKE" else "badge-real"
    icon = "⚠️" if label == "FAKE" else "✅"

    st.markdown(f"""
    <div style="text-align:center; margin-top:18px;">
        <span class="{badge_class}">{icon} {label}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="conf-label">Confidence</p>', unsafe_allow_html=True)
    bar_color = "#ff4e4e" if label == "FAKE" else "#00e096"
    st.markdown(f"""
    <div style="background:#23272f; border-radius:999px; height:10px; overflow:hidden; margin-bottom:6px;">
        <div style="width:{confidence*100:.1f}%; background:{bar_color}; height:100%;
                    border-radius:999px; transition:width 0.6s ease;"></div>
    </div>
    <p style="font-size:0.95rem; color:#e8eaf0; text-align:right; margin:0;">
        <strong>{confidence*100:.1f}%</strong>
    </p>
    """, unsafe_allow_html=True)


# ── Tabs ────────────────────────────────────────────────────────────────────
tab_text, tab_image = st.tabs(["📝  Text Detection", "🖼  Image Detection"])


# ════════════════════════════════════════════════════
#  TAB 1 — TEXT DETECTION
# ════════════════════════════════════════════════════
with tab_text:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Paste or type your news article")
    st.caption("The model analyses language patterns to detect misinformation.")

    user_text = st.text_area(
        label="News text",
        placeholder="Enter the news headline or body text here…",
        height=200,
        label_visibility="collapsed",
    )

    analyse_text = st.button("Analyse Text", key="btn_text")

    if analyse_text:
        if not user_text.strip():
            st.warning("⚠️ Please enter some text before analysing.")
        else:
            with st.spinner("Analysing text…"):
                model, vectorizer, err = get_text_models()

            if err:
                st.error(f"❌ Model loading failed: {err}")
            else:
                with st.spinner("Running prediction…"):
                    label, confidence, pred_err = predict_text(user_text, model, vectorizer)

                if pred_err:
                    st.error(f"❌ Prediction error: {pred_err}")
                else:
                    st.markdown("---")
                    render_result(label, confidence)

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════
#  TAB 2 — IMAGE DETECTION
# ════════════════════════════════════════════════════
with tab_image:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Upload a news image")
    st.caption("The CNN model scans visual manipulations and deepfake artefacts.")

    uploaded_file = st.file_uploader(
        label="Upload image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    analyse_image = st.button("Analyse Image", key="btn_image")

    if analyse_image:
        if uploaded_file is None:
            st.warning("⚠️ Please upload an image before analysing.")
        else:
            with st.spinner("Loading vision model…"):
                vision_model, v_err = get_vision_model()

            if v_err:
                st.error(f"❌ Vision model loading failed: {v_err}")
            else:
                with st.spinner("Running image prediction…"):
                    label, confidence, pred_err = predict_image(uploaded_file, vision_model)

                if pred_err:
                    st.error(f"❌ Prediction error: {pred_err}")
                else:
                    st.markdown("---")
                    render_result(label, confidence)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#3a3f4d; font-size:0.8rem;">'
    "Fake News Detector · Built with Streamlit · For research & educational use only"
    "</p>",
    unsafe_allow_html=True,
)
