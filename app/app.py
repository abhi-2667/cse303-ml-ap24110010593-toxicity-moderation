"""
Streamlit live demo - Intelligent Toxicity Moderation Platform.

Run from the project root:   streamlit run app/app.py

It loads the two artifacts saved by the notebook:
    models/tfidf_union.pkl   (word + character TF-IDF)
    models/best_model.pkl    (best of the 3 compared models)
"""
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import streamlit as st

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Same thresholds as Section 8 of the notebook
PUBLISH_BELOW = 0.40
REMOVE_ABOVE = 0.75
CRITICAL_ABOVE = 0.60  # lower bar for 'threat' and 'severe_toxic'

BADGES = {  # decision -> (colour, explanation)
    "PUBLISH": ("#2e7d32", "Looks fine - goes live immediately."),
    "REVIEW": ("#ef6c00", "Uncertain - flagged for a human moderator."),
    "REMOVE": ("#c62828", "Confidently toxic - blocked automatically."),
}


def clean_text(text):
    """Must match clean_text() in the notebook (Section 4)."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def decide(probs):
    """Same 3-tier rule as the notebook's decide()."""
    max_prob = max(probs.values())
    if max_prob > REMOVE_ABOVE or probs["threat"] > CRITICAL_ABOVE or probs["severe_toxic"] > CRITICAL_ABOVE:
        return "REMOVE"
    if max_prob >= PUBLISH_BELOW:
        return "REVIEW"
    return "PUBLISH"


@st.cache_resource
def load_artifacts():
    union = joblib.load(MODELS_DIR / "tfidf_union.pkl")
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    return union, model


def plot_probabilities(probs):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    labels = list(probs.keys())[::-1]            # reversed so 'toxic' is on top
    values = [probs[l] for l in labels]
    colors = ["#c62828" if v > REMOVE_ABOVE else "#ef6c00" if v >= PUBLISH_BELOW else "#2e7d32" for v in values]
    ax.barh(labels, values, color=colors)
    ax.axvline(PUBLISH_BELOW, color="orange", ls="--", lw=1)
    ax.axvline(REMOVE_ABOVE, color="red", ls="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted probability")
    for i, v in enumerate(values):
        ax.text(min(v + 0.01, 0.93), i, f"{v:.2f}", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="Toxicity Moderation", page_icon="🛡️")
st.title("🛡️ Intelligent Toxicity Moderation")
st.caption("Multi-label toxicity detection (Jigsaw dataset) with a 3-tier moderation engine. "
           "Try obfuscated words such as `1d10t` or `i.d.i.o.t`.")

try:
    tfidf_union, model = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run the notebook "
             "`notebooks/toxicity_moderation_capstone.ipynb` first to create `models/*.pkl`.")
    st.stop()

comment = st.text_area("Type a comment to moderate:", height=120,
                       placeholder="e.g. You are an 1d10t")

if st.button("Analyze Comment", type="primary"):
    if not comment.strip():
        st.warning("Please type a comment first.")
    else:
        features = tfidf_union.transform([clean_text(comment)])
        p = model.predict_proba(features)[0]
        probs = {label: float(v) for label, v in zip(LABELS, p)}
        decision = decide(probs)
        color, explanation = BADGES[decision]

        st.markdown(
            f"<div style='background:{color};color:white;padding:14px 20px;border-radius:10px;"
            f"font-size:26px;font-weight:700;text-align:center'>{decision}</div>",
            unsafe_allow_html=True,
        )
        st.write(f"{explanation}  (highest probability: **{max(probs.values()):.2f}**)")
        st.pyplot(plot_probabilities(probs))
        st.caption("Dashed lines: 0.40 (REVIEW starts) and 0.75 (REMOVE). "
                   "`threat` / `severe_toxic` > 0.60 also triggers REMOVE.")
