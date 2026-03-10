import streamlit as st
import numpy as np
import cv2
import os
import gdown
import plotly.graph_objects as go
from PIL import Image
import io
import time

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chest X-Ray Analyzer",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark clinical aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0e14;
    color: #e2e8f0;
    font-family: 'DM Mono', monospace;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 10%, #0d1f2d 0%, #0a0e14 60%);
}

h1, h2, h3 { font-family: 'Syne', sans-serif; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* Hero header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid #1e2d3d;
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #e879f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #475569;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.6rem;
}
.hero-tags {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.tag {
    background: #0f1e2e;
    border: 1px solid #1e3a52;
    color: #38bdf8;
    padding: 0.2rem 0.75rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 1px;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    background: #0f1a24 !important;
    border: 2px dashed #1e3a52 !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #38bdf8 !important;
}

/* Section labels */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2d3d;
}

/* Result cards */
.result-card {
    background: #0d1926;
    border: 1px solid #1e3a52;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, transform 0.2s;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.result-card.normal::before  { background: linear-gradient(90deg, #22c55e, #16a34a); }
.result-card.detected::before { background: linear-gradient(90deg, #f59e0b, #ef4444); }
.result-card:hover { border-color: #38bdf8; transform: translateY(-1px); }

.card-disease {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
}
.card-status-normal   { color: #22c55e; font-size: 0.8rem; font-weight: 600; letter-spacing: 1px; }
.card-status-detected { color: #f59e0b; font-size: 0.8rem; font-weight: 600; letter-spacing: 1px; }
.card-prob {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #38bdf8;
}
.card-conf { font-size: 0.7rem; color: #475569; letter-spacing: 1px; text-transform: uppercase; }

/* Summary banner */
.summary-clean {
    background: linear-gradient(135deg, #0d2318, #0d1926);
    border: 1px solid #16a34a;
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.summary-clean .label { font-size: 0.72rem; color: #22c55e; letter-spacing: 3px; text-transform: uppercase; }
.summary-clean .msg { font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 700; color: #dcfce7; margin-top: 0.3rem; }

.summary-flagged {
    background: linear-gradient(135deg, #2d1a06, #1a0d0d);
    border: 1px solid #ef4444;
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.summary-flagged .label { font-size: 0.72rem; color: #f87171; letter-spacing: 3px; text-transform: uppercase; }
.summary-flagged .msg { font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 700; color: #fecaca; margin-top: 0.3rem; }

/* Confidence legend */
.conf-box {
    background: #0d1926;
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
}
.conf-row { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }
.conf-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.conf-text { font-size: 0.78rem; color: #94a3b8; }
.conf-name { font-weight: 600; color: #cbd5e1; }

/* Disclaimer */
.disclaimer {
    background: #0d1421;
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.72rem;
    color: #78716c;
    margin-top: 1.5rem;
    line-height: 1.6;
}

/* Spinner override */
.stSpinner > div { border-top-color: #38bdf8 !important; }

/* Image captions */
.img-caption {
    text-align: center;
    font-size: 0.68rem;
    color: #475569;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* Step instructions */
.step {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 0.9rem;
}
.step-num {
    background: #0f2235;
    border: 1px solid #1e3a52;
    color: #38bdf8;
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 2px;
}
.step-text { font-size: 0.82rem; color: #94a3b8; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — PASTE YOUR GOOGLE DRIVE FILE IDs HERE
# ─────────────────────────────────────────────────────────────────────────────
# How to get a file ID:
#   1. Upload .keras file to Google Drive
#   2. Right-click → Share → Anyone with link
#   3. Copy the ID from the URL:
#      https://drive.google.com/file/d/FILE_ID_HERE/view
MODEL_GDRIVE_IDS = {
    "TB"           : "1vc2DPQc5Zj6MsR-u6OkGeFKdvPkk1bvN",
    "Pneumonia"    : "1_IjuGH67drgtiCp9pMQgMa11nikO-M-_",
    "Infiltration" : "1D-4YcfiX7lfI7yl9m0hyvtoC0rdTCwNR",
    "Fibrosis"     : "1Xp2sgwQe6Jl8sZk5UZI-qH_5sIYZp-N8",
}

DISEASES = ["TB", "Pneumonia", "Infiltration", "Fibrosis"]
THRESHOLD = 0.5
MODEL_DIR = "/tmp/xray_models"

DISEASE_INFO = {
    "TB":            "Tuberculosis — bacterial infection causing lung tissue damage.",
    "Pneumonia":     "Pneumonia — lung inflammation typically caused by infection.",
    "Infiltration":  "Infiltration — abnormal substance (fluid/cells) in lung tissue.",
    "Fibrosis":      "Fibrosis — scarring and thickening of lung tissue over time.",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def confidence_label(prob, threshold=THRESHOLD):
    gap = abs(prob - threshold)
    if gap >= 0.35: return "Very High"
    if gap >= 0.20: return "High"
    if gap >= 0.10: return "Moderate"
    return "Low — borderline"

def confidence_color(label):
    return {"Very High": "#22c55e", "High": "#38bdf8",
            "Moderate": "#f59e0b", "Low — borderline": "#ef4444"}.get(label, "#94a3b8")

def preprocess_image(img_bytes):
    """Exact replica of smart_medical_resize from training."""
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    h, w = img_clahe.shape[:2]
    scale = min(512 / w, 512 / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_clahe, (new_w, new_h),
                         interpolation=cv2.INTER_LANCZOS4 if scale < 1.0 else cv2.INTER_CUBIC)

    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    x_off = (512 - new_w) // 2
    y_off = (512 - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    canvas_norm = canvas_rgb.astype(np.float32) / 255.0

    preview = Image.fromarray(canvas_rgb)
    return np.expand_dims(canvas_norm, axis=0), preview


@st.cache_resource(show_spinner=False)
def load_all_models():
    """Download from Google Drive and load all 4 models. Cached after first load."""
    import tensorflow as tf
    os.makedirs(MODEL_DIR, exist_ok=True)
    models = {}
    for disease, file_id in MODEL_GDRIVE_IDS.items():
        if file_id == f"YOUR_{disease.upper()}_MODEL_FILE_ID":
            continue  # skip placeholder IDs
        dest = os.path.join(MODEL_DIR, f"best_model_{disease}.keras")
        if not os.path.exists(dest):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, dest, quiet=True)
        if os.path.exists(dest):
            models[disease] = tf.keras.models.load_model(dest)
    return models

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">🫁 Chest X-Ray Analyzer</p>
    <p class="hero-sub">DenseNet121 · Binary Classification · 4 Disease Models</p>
    <div class="hero-tags">
        <span class="tag">TB</span>
        <span class="tag">Pneumonia</span>
        <span class="tag">Infiltration</span>
        <span class="tag">Fibrosis</span>
        <span class="tag">512×512 Input</span>
        <span class="tag">CLAHE Enhanced</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHECK IF MODELS ARE CONFIGURED
# ─────────────────────────────────────────────────────────────────────────────
models_configured = any(
    v != f"YOUR_{k.upper()}_MODEL_FILE_ID"
    for k, v in MODEL_GDRIVE_IDS.items()
)

if not models_configured:
    st.markdown("""
    <div style="background:#0d1f2d;border:1px solid #1e3a52;border-radius:14px;padding:1.5rem 2rem;margin-bottom:2rem;">
        <div class="section-label">⚙ Setup Required</div>
        <p style="font-size:0.85rem;color:#94a3b8;margin:0.5rem 0 1rem;">
            Before deploying, paste your Google Drive model file IDs into <code style="color:#38bdf8">app.py</code>:
        </p>
    """, unsafe_allow_html=True)

    for i, (disease, key) in enumerate([
        ("TB",           "YOUR_TB_MODEL_FILE_ID"),
        ("Pneumonia",    "YOUR_PNEUMONIA_MODEL_FILE_ID"),
        ("Infiltration", "YOUR_INFILTRATION_MODEL_FILE_ID"),
        ("Fibrosis",     "YOUR_FIBROSIS_MODEL_FILE_ID"),
    ], 1):
        st.markdown(f"""
        <div class="step">
            <div class="step-num">{i}</div>
            <div class="step-text">
                Upload <strong style="color:#e2e8f0">best_model_{disease}.keras</strong> to Google Drive
                → Share (Anyone with link) → Copy File ID
                → Replace <code style="color:#38bdf8">"{key}"</code> in <code>MODEL_GDRIVE_IDS</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.6], gap="large")

with col_left:
    st.markdown('<div class="section-label">Upload X-Ray Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop your chest X-ray here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        st.markdown('<div class="img-caption">Original Image</div>', unsafe_allow_html=True)
        st.image(uploaded, use_container_width=True)

    # ── Confidence Legend ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Confidence Guide</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="conf-box">
        <div class="conf-row">
            <div class="conf-dot" style="background:#22c55e"></div>
            <div class="conf-text"><span class="conf-name">Very High</span> — probability is far from 50% threshold (≥35% gap)</div>
        </div>
        <div class="conf-row">
            <div class="conf-dot" style="background:#38bdf8"></div>
            <div class="conf-text"><span class="conf-name">High</span> — strong signal from the model (20–35% gap)</div>
        </div>
        <div class="conf-row">
            <div class="conf-dot" style="background:#f59e0b"></div>
            <div class="conf-text"><span class="conf-name">Moderate</span> — reasonably clear prediction (10–20% gap)</div>
        </div>
        <div class="conf-row" style="margin-bottom:0">
            <div class="conf-dot" style="background:#ef4444"></div>
            <div class="conf-text"><span class="conf-name">Low / Borderline</span> — near 50%, treat with caution (&lt;10% gap)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠ <strong>Research use only.</strong> This tool is not a substitute for professional
        radiological diagnosis. Always consult a qualified clinician before making
        any medical decisions based on these results.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN — Results
# ─────────────────────────────────────────────────────────────────────────────
with col_right:
    if uploaded and models_configured:

        img_bytes = uploaded.read()

        with st.spinner("Preprocessing image…"):
            tensor, preview_img = preprocess_image(img_bytes)

        if tensor is None:
            st.error("Could not read image. Please upload a valid JPG or PNG.")
        else:
            # ── Preprocessed preview ──
            st.markdown('<div class="section-label">Preprocessed Preview (CLAHE Enhanced)</div>',
                        unsafe_allow_html=True)
            st.image(preview_img, use_container_width=True,
                     caption="512×512 · CLAHE · Aspect-ratio padded")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Load models & predict ──
            with st.spinner("Loading models & running analysis…"):
                models = load_all_models()

            if not models:
                st.error("No models loaded. Check your Google Drive IDs in app.py.")
            else:
                results = []
                progress = st.progress(0, text="Analyzing…")
                for i, disease in enumerate(DISEASES):
                    if disease in models:
                        prob = float(models[disease].predict(tensor, verbose=0)[0][0])
                        conf = confidence_label(prob)
                        results.append({
                            "disease": disease,
                            "prob": prob,
                            "detected": prob >= THRESHOLD,
                            "conf": conf,
                            "conf_color": confidence_color(conf),
                            "info": DISEASE_INFO[disease],
                        })
                    progress.progress((i + 1) / len(DISEASES),
                                      text=f"Analyzed {disease}…")
                    time.sleep(0.1)
                progress.empty()

                # ── Summary banner ──
                detected_list = [r["disease"] for r in results if r["detected"]]
                if detected_list:
                    st.markdown(f"""
                    <div class="summary-flagged">
                        <div class="label">⚠ Conditions Flagged</div>
                        <div class="msg">{" · ".join(detected_list)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="summary-clean">
                        <div class="label">✓ All Clear</div>
                        <div class="msg">No conditions detected by any model</div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Result Cards ──
                st.markdown('<div class="section-label">Per-Model Results</div>',
                            unsafe_allow_html=True)

                card_col1, card_col2 = st.columns(2)
                for i, r in enumerate(results):
                    col = card_col1 if i % 2 == 0 else card_col2
                    card_class = "detected" if r["detected"] else "normal"
                    status_class = "card-status-detected" if r["detected"] else "card-status-normal"
                    status_text = "⚠ DETECTED" if r["detected"] else "✓ NORMAL"
                    with col:
                        st.markdown(f"""
                        <div class="result-card {card_class}">
                            <div style="display:flex;justify-content:space-between;align-items:flex-start">
                                <div>
                                    <div class="card-disease">{r['disease']}</div>
                                    <div class="{status_class}">{status_text}</div>
                                </div>
                                <div style="text-align:right">
                                    <div class="card-prob">{r['prob']*100:.1f}%</div>
                                    <div class="card-conf" style="color:{r['conf_color']}">{r['conf']}</div>
                                </div>
                            </div>
                            <div style="font-size:0.7rem;color:#475569;margin-top:0.6rem;line-height:1.5">
                                {r['info']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # ── Probability Bar Chart ──
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Probability Chart</div>',
                            unsafe_allow_html=True)

                bar_colors = ["#ef4444" if r["detected"] else "#22c55e" for r in results]
                fig = go.Figure()

                fig.add_shape(type="line", x0=THRESHOLD, x1=THRESHOLD, y0=-0.5,
                              y1=len(results) - 0.5,
                              line=dict(color="#f59e0b", width=2, dash="dot"))

                fig.add_trace(go.Bar(
                    x=[r["prob"] for r in results],
                    y=[r["disease"] for r in results],
                    orientation="h",
                    marker=dict(
                        color=bar_colors,
                        opacity=0.85,
                        line=dict(color="rgba(255,255,255,0.1)", width=1)
                    ),
                    text=[f"{r['prob']*100:.1f}%" for r in results],
                    textposition="outside",
                    textfont=dict(color="#e2e8f0", size=13, family="DM Mono"),
                    hovertemplate="<b>%{y}</b><br>Probability: %{x:.1%}<extra></extra>",
                ))

                fig.add_annotation(x=THRESHOLD, y=len(results) - 0.3,
                                   text="Threshold 50%",
                                   showarrow=False,
                                   font=dict(color="#f59e0b", size=11, family="DM Mono"),
                                   xanchor="left", xshift=8)

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#0d1926",
                    xaxis=dict(range=[0, 1.15], tickformat=".0%",
                               gridcolor="#1e2d3d", color="#475569",
                               tickfont=dict(family="DM Mono", size=11)),
                    yaxis=dict(gridcolor="#1e2d3d", color="#e2e8f0",
                               tickfont=dict(family="Syne", size=13, color="#e2e8f0")),
                    margin=dict(l=10, r=60, t=20, b=10),
                    height=260,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

    elif uploaded and not models_configured:
        st.markdown("""
        <div style="background:#0d1421;border:1px solid #f59e0b;border-radius:12px;
                    padding:1.5rem;text-align:center;margin-top:3rem;">
            <div style="font-size:2rem">⚙</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;
                        color:#fde68a;margin:0.5rem 0">Models Not Configured</div>
            <div style="font-size:0.8rem;color:#78716c">
                Add your Google Drive File IDs to <code style="color:#38bdf8">MODEL_GDRIVE_IDS</code> in app.py
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;height:400px;text-align:center;
                    opacity:0.4;">
            <div style="font-size:4rem;margin-bottom:1rem">🫁</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.2rem;color:#475569">
                Upload an X-ray to begin analysis
            </div>
            <div style="font-size:0.75rem;color:#334155;margin-top:0.5rem;letter-spacing:2px;
                        text-transform:uppercase;">
                JPG · PNG · Any size
            </div>
        </div>
        """, unsafe_allow_html=True)
