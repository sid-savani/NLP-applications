"""
NLP Playground — Interactive NLP Lab.
Main application controller and Streamlit entry point.
"""

import os
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="NLP Playground — Interactive NLP Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), "styles", "custom.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Ensure NLTK resources on initial startup
from modules.nlp_setup import ensure_nltk_resources
ensure_nltk_resources()

# Import view components
from views.home import render_home
from views.sentiment_view import render_sentiment_view
from views.spam_view import render_spam_view
from views.summarizer_view import render_summarizer_view
from views.language_view import render_language_view
from views.keyword_view import render_keyword_view
from views.about_nlp import render_about_nlp

# Navigation options
PAGES = {
    "Home": {
        "label": "🏠  Home",
        "renderer": render_home
    },
    "Sentiment Analysis": {
        "label": "🎭  Sentiment Analysis",
        "renderer": render_sentiment_view
    },
    "Spam Message Detection": {
        "label": "🛡️  Spam Detection",
        "renderer": render_spam_view
    },
    "Text Summarization": {
        "label": "📝  Text Summarization",
        "renderer": render_summarizer_view
    },
    "Language Detection": {
        "label": "🌐  Language Detection",
        "renderer": render_language_view
    },
    "Keyword Extraction": {
        "label": "🔑  Keyword Extraction",
        "renderer": render_keyword_view
    },
    "About NLP": {
        "label": "📖  About NLP Guide",
        "renderer": render_about_nlp
    }
}

# Session state initialization
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Home"

# Build Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.2rem 0; border-bottom: 1px solid rgba(99, 102, 241, 0.2); margin-bottom: 1.2rem;">
        <div style="font-size: 1.35rem; font-weight: 800; background: linear-gradient(135deg, #a5b4fc, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🧠 NLP Playground
        </div>
        <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.2rem; letter-spacing: 0.05em; text-transform: uppercase;">
            Interactive NLP Lab
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Convert current state to index
    page_keys = list(PAGES.keys())
    current_idx = page_keys.index(st.session_state["active_page"]) if st.session_state["active_page"] in page_keys else 0

    selected_label = st.radio(
        "Navigation",
        options=[PAGES[k]["label"] for k in page_keys],
        index=current_idx,
        key="nav_radio"
    )

    # Sync back to page key
    for k, v in PAGES.items():
        if v["label"] == selected_label:
            st.session_state["active_page"] = k
            break

    # Sidebar Footer Information
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.6;">
        <strong style="color: #cbd5e1;">Project Highlights:</strong>
        <ul style="padding-left: 1.2rem; margin-top: 0.4rem; margin-bottom: 0.8rem;">
            <li>Zero API keys / offline</li>
            <li>Real scikit-learn models</li>
            <li>No fabricated metrics</li>
            <li>NLTK & TextBlob powered</li>
        </ul>
        <span class="nlp-badge" style="display:inline-block; margin-top:0.3rem;">v1.0.0 Stable</span>
    </div>
    """, unsafe_allow_html=True)

# Render selected page
current_page = st.session_state["active_page"]
renderer = PAGES.get(current_page, {}).get("renderer", render_home)
renderer()
