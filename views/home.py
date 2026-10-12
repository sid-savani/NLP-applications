"""
Home View — NLP Playground.
Features hero section, interactive tool catalog, and comprehensive introduction to NLP.
"""

import streamlit as st


def render_home():
    # Hero Banner
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">NLP Playground — Interactive NLP Lab</div>
        <div class="hero-subtitle">
            An interactive educational laboratory exploring core Natural Language Processing algorithms. 
            Experiment with working models, examine step-by-step pipelines, and inspect underlying mathematical intuitions without third-party APIs.
        </div>
        <div style="margin-top: 1.2rem;">
            <span class="nlp-badge">⚡ 100% Local Inference</span>
            <span class="nlp-badge">📚 Interactive Pipelines</span>
            <span class="nlp-badge">🔍 Transparent Math & Metrics</span>
            <span class="nlp-badge">🎨 Modern Glassmorphic Dashboard</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Metrics / Overview Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">5</div>
            <div class="metric-label">Interactive NLP Tools</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">0</div>
            <div class="metric-label">API Keys Required</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">100%</div>
            <div class="metric-label">Honest Metrics (No Fakes)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">32+</div>
            <div class="metric-label">Supported Languages</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tool Catalog Cards
    st.subheader("🛠️ Explore the NLP Toolkit")
    st.caption("Select a tool below or use the sidebar navigation to launch the interactive workspace.")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.markdown("""
        <div class="tool-card">
            <div class="card-icon">🎭</div>
            <div class="card-title">1. Sentiment Analysis</div>
            <div class="card-desc">
                Quantify emotional tone and polarity across sentences. Computes subjectivity (fact vs. opinion) and polarity (-1.0 to +1.0) using lexicon-based TextBlob scoring.
            </div>
            <div>
                <span class="nlp-badge">Lexicon Scoring</span>
                <span class="nlp-badge">Polarity</span>
                <span class="nlp-badge">Subjectivity</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Sentiment Analysis →", key="btn_home_sentiment", use_container_width=True):
            st.session_state["active_page"] = "Sentiment Analysis"
            st.rerun()

    with row1_col2:
        st.markdown("""
        <div class="tool-card">
            <div class="card-icon">🛡️</div>
            <div class="card-title">2. Spam Message Detection</div>
            <div class="card-desc">
                Classify legitimate vs. fraudulent text messages using TF-IDF n-gram vectorization and Multinomial Naive Bayes trained on genuine split-tested data.
            </div>
            <div>
                <span class="nlp-badge">TF-IDF</span>
                <span class="nlp-badge">Multinomial Naive Bayes</span>
                <span class="nlp-badge">Stratified Split</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Spam Detection →", key="btn_home_spam", use_container_width=True):
            st.session_state["active_page"] = "Spam Message Detection"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.markdown("""
        <div class="tool-card">
            <div class="card-icon">📝</div>
            <div class="card-title">3. Text Summarization</div>
            <div class="card-desc">
                Condense lengthy articles into salient summaries using extractive sentence tokenization, stop-word filtering, and frequency-weighted scoring algorithms.
            </div>
            <div>
                <span class="nlp-badge">Extractive NLP</span>
                <span class="nlp-badge">Sentence Tokenizer</span>
                <span class="nlp-badge">Word Salience</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Summarizer →", key="btn_home_summary", use_container_width=True):
            st.session_state["active_page"] = "Text Summarization"
            st.rerun()

    with row2_col2:
        st.markdown("""
        <div class="tool-card">
            <div class="card-icon">🌐</div>
            <div class="card-title">4. Language Detection</div>
            <div class="card-desc">
                Identify the language of input text in real-time. Employs n-gram character profile probability distributions with resilient edge-case handling for 30+ languages.
            </div>
            <div>
                <span class="nlp-badge">Character N-Grams</span>
                <span class="nlp-badge">Probability Distribution</span>
                <span class="nlp-badge">ISO 639-1</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Language Detection →", key="btn_home_lang", use_container_width=True):
            st.session_state["active_page"] = "Language Detection"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 5th tool card full width
    col_single, _ = st.columns([1, 1])
    with col_single:
        st.markdown("""
        <div class="tool-card">
            <div class="card-icon">🔑</div>
            <div class="card-title">5. Keyword & Keyphrase Extraction</div>
            <div class="card-desc">
                Extract top keywords and multi-word keyphrases using TF-IDF feature importance with sublinear term-frequency scaling and stop-word elimination.
            </div>
            <div>
                <span class="nlp-badge">TF-IDF Ranking</span>
                <span class="nlp-badge">N-Gram Phrases</span>
                <span class="nlp-badge">Stop-Word Pruning</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Keyword Extraction →", key="btn_home_keyword", use_container_width=True):
            st.session_state["active_page"] = "Keyword Extraction"
            st.rerun()

    st.markdown("---")

    # Brief Explanation of NLP
    st.subheader("🧠 What is Natural Language Processing?")
    st.markdown("""
    **Natural Language Processing (NLP)** sits at the intersection of **computer science**, **linguistics**, and **artificial intelligence**. 
    Its primary purpose is enabling computing systems to interpret, structure, analyze, and generate human language in a meaningful and context-aware manner.
    """)

    col_nlp1, col_nlp2 = st.columns(2)
    with col_nlp1:
        st.markdown("""
        #### Why Human Language is Computationally Hard:
        * **Ambiguity & Polysemy:** The word *"bank"* means a financial institution or the side of a river depending solely on context.
        * **Sarcasm & Nuance:** *"Oh, fantastic, another delay!"* has positive dictionary words but an overtly negative emotional intent.
        * **Varied Syntax:** Different languages, dialects, slang, typos, and shorthand introduce infinite variations.
        * **Implicit Common Knowledge:** Humans omit obvious world knowledge that algorithms must infer from training data.
        """)

    with col_nlp2:
        st.markdown("""
        #### The Computational Paradigm:
        1. **Text Normalization:** Converting raw characters into clean, uniform lowercase tokens without extraneous noise.
        2. **Numerical Vectorization:** Algorithms cannot compute on characters directly; text must be mapped into numerical vectors (Bag-of-Words, TF-IDF, Word2Vec, or Transformer embeddings).
        3. **Inference & Decision:** Mathematical operations (dot products, Bayes theorem, classification boundaries) derive predictions from those vectors.
        """)

    st.info("💡 **Ready to dive deeper?** Check out the **About NLP** tab in the sidebar for a breakdown comparing Rule-Based, Statistical, and Transformer architectures!")
