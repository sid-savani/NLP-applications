"""
Text Summarization View.
Features:
- Try It: Extractive summarization with sentence count slider, sample articles, compression metrics, and sentence score breakdown
- Comprehensive How It Works: Extractive vs Abstractive, mathematical sentence weighting
- Workflow Pipeline
"""

import streamlit as st
import pandas as pd
from modules.summarizer import summarize_text

SAMPLE_ARTICLES = {
    "🤖 The Evolution of AI": (
        "Artificial intelligence has undergone remarkable transformations over the past six decades. "
        "In the early era, symbolic AI relied on hand-crafted expert rules and deductive logic trees. "
        "However, these brittle early systems could not scale to the chaotic complexity of real-world human language and visual perception. "
        "The emergence of statistical machine learning in the late 1990s introduced probabilistic data-driven decision boundaries. "
        "Later, the deep learning revolution in the 2010s demonstrated that multi-layered artificial neural networks could extract hierarchical representations directly from massive datasets. "
        "Today, transformer-based foundation models leverage self-attention mechanisms to achieve unprecedented comprehension across multilingual corpora. "
        "Despite their superhuman fluency, modern generative systems still struggle with factual grounding, hallucinations, and high computational energy costs. "
        "Consequently, hybrid architectures combining neural networks with structured symbolic knowledge graphs represent the exciting frontier of research."
    ),
    "🚀 Deep Space Exploration": (
        "Humanity's journey into deep space relies on advanced robotic observatories and autonomous navigation systems. "
        "The James Webb Space Telescope orbits the second Lagrange point, capturing infrared emissions from galaxies formed over thirteen billion years ago. "
        "These astronomical observations reveal the chemical signatures of early stellar nurseries and atmospheric compositions of distant exoplanets. "
        "Simultaneously, robotic rovers traversing Martian terrain collect mineral core samples to discover ancient biosignatures. "
        "Operating instruments millions of kilometers from Earth requires resilient autonomous protocols because round-trip radio signals incur significant propagation delays. "
        "Future missions plan to return these sealed Martian samples to terrestrial laboratories for high-precision isotopic mass spectrometry. "
        "International cooperation remains indispensable as space agencies collaborate to build sustainable lunar habitats under the Artemis accords."
    ),
    "🌱 Urban Green Infrastructure": (
        "Rapid urbanization has magnified the urban heat island effect, causing metropolitan centers to trap extreme thermal energy. "
        "Concrete pavements and asphalt absorb intense solar radiation throughout the day and radiate heat long after sunset. "
        "To combat this challenge, urban planners are integrating extensive green infrastructure across city centers. "
        "Vegetated green roofs and living vertical walls substantially reduce indoor cooling energy demands by providing natural thermal insulation and evaporative cooling. "
        "Furthermore, urban wetlands and permeable bioswales mitigate severe storm runoff, filtering chemical pollutants before water enters local aquifers. "
        "Recent municipal studies indicate that increasing neighborhood canopy cover by twenty percent yields measurable declines in heat-related illnesses. "
        "Transforming sterile grey infrastructure into biodiverse living corridors improves both community public health and long-term climate resilience."
    )
}


def render_summarizer_view():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="margin-bottom: 0.2rem;">📝 Text Summarization</h1>
        <p style="color: #94a3b8; font-size: 1.05rem;">
            Extractive document summarization using sentence tokenization and normalized word-frequency salience ranking.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_try, tab_how, tab_workflow = st.tabs(["🎮 Try It", "🔬 How It Works", "🔄 Workflow Pipeline"])

    # =========================================================================
    # TAB 1: TRY IT
    # =========================================================================
    with tab_try:
        st.markdown("### 1. Choose or Paste an Article")
        st.caption("Click a sample button to load a full paragraph, or paste your own document:")

        btn_cols = st.columns(len(SAMPLE_ARTICLES))
        for idx, (label, sample_text) in enumerate(SAMPLE_ARTICLES.items()):
            with btn_cols[idx]:
                if st.button(label, key=f"btn_sample_sum_{idx}", use_container_width=True):
                    st.session_state["summary_input"] = sample_text

        default_article = st.session_state.get("summary_input", SAMPLE_ARTICLES["🤖 The Evolution of AI"])
        user_article = st.text_area(
            "Document content to summarize:",
            value=default_article,
            height=160,
            key="summary_textarea",
            placeholder="Paste your article or long paragraph here..."
        )

        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        with col_ctrl1:
            num_sentences = st.slider(
                "Target Summary Sentences:",
                min_value=1,
                max_value=6,
                value=3,
                help="Controls how many of the highest-scoring sentences to retain."
            )

        with col_ctrl2:
            st.markdown("<br>", unsafe_allow_html=True)
            summarize_clicked = st.button("📝 Generate Extractive Summary", type="primary", use_container_width=True)

        if summarize_clicked or user_article:
            if not user_article.strip():
                st.warning("⚠️ Document text is empty. Please enter or paste some text.")
            else:
                res = summarize_text(user_article, num_sentences=num_sentences)
                if "error" in res:
                    st.error(f"❌ {res['error']}")
                else:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader("📑 Generated Extractive Summary")

                    # Metrics row
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #60a5fa;">{res['summary_sentence_count']} / {res['original_sentence_count']}</div>
                            <div class="metric-label">Sentences Selected</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #a78bfa;">{res['summary_word_count']}</div>
                            <div class="metric-label">Summary Words</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #34d399;">{res['compression_ratio']}%</div>
                            <div class="metric-label">Word Compression</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m4:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #fbbf24;">{res['original_word_count']}</div>
                            <div class="metric-label">Original Words</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Summary card
                    st.markdown(f"""
                    <div class="result-card" style="border-left: 4px solid #8b5cf6;">
                        <h4 style="margin-top:0; color:#c7d2fe;">Generated Summary:</h4>
                        <p style="font-size: 1.05rem; line-height: 1.7; color: #f8fafc;">
                            {res['summary']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Top keywords contributing to sentence weight
                    if res.get("top_words"):
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("#### 🔑 Dominant Salient Keywords Driving Sentences")
                        badges_html = " ".join([f'<span class="nlp-badge">{w} ({cnt})</span>' for w, cnt in res["top_words"][:8]])
                        st.markdown(badges_html, unsafe_allow_html=True)

                    # Sentence ranking table
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("#### 📊 Sentence Scoring Diagnostic Table")
                    st.caption("Inspect the algorithmic score assigned to each sentence in the document:")

                    table_rows = []
                    for s in res["sentence_scores"]:
                        table_rows.append({
                            "#": s["index"] + 1,
                            "Included": "✅ YES" if s["selected"] else "❌ No",
                            "Salience Score": f"{s['score']:.4f}",
                            "Words": s["word_count"],
                            "Sentence Content": s["sentence"]
                        })
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: HOW IT WORKS
    # =========================================================================
    with tab_how:
        st.subheader("🔬 How Extractive Summarization Works")
        
        st.markdown("""
        #### 1. Extractive vs. Abstractive Summarization
        * **Extractive Summarization:** Identifies and selects the most informative existing sentences verbatim from the source document, acting like an intelligent digital highlighter.
        * **Abstractive Summarization:** Generates brand new sentences that rephrase the core ideas (similar to how humans summarize or how generative LLMs like GPT produce summaries).
        """)

        st.markdown("""
        #### 2. The Frequency-Weighted Sentence Scoring Algorithm
        The extractive algorithm operates on the hypothesis that **sentences containing words that appear frequently and distinctively across the document are the most central to its core thesis**.
        
        1. **Text Tokenization:** Segment the raw text into distinct sentences $\{S_1, S_2, \dots, S_m\}$ and constituent words.
        2. **Stop-Word Filtering:** Exclude grammatical glue words (*"the"*, *"with"*, *"is"*, *"at"*) that have high frequency but zero informational value.
        3. **Word Frequency Normalization:**
           For each non-stop word $w$:
           $$\\text{NormFreq}(w) = \\frac{\\text{Count}(w)}{\\max_{v} \\text{Count}(v)}$$
        4. **Sentence Salience Scoring:**
           A sentence's raw score is the sum of normalized frequencies of its constituent words, adjusted by a length factor to prevent arbitrarily rewarding run-on sentences:
           $$\\text{Score}(S_i) = \\frac{\\sum_{w \\in S_i} \\text{NormFreq}(w)}{\\sqrt{|S_i|}}$$
        5. **Order-Preserving Assembly:**
           Pick the top $K$ highest-scoring sentences, then sort them by their original position in the text to retain logical flow.
        """)

        st.markdown("""
        #### 3. Worked Step-by-Step Example
        Consider three sentences:
        - $S_1$: *"Neural networks process data."*
        - $S_2$: *"Data is essential for machine learning algorithms."*
        - $S_3$: *"The weather is nice today."*
        
        1. Frequency table (excluding stop words): `'data': 2`, `'neural': 1`, `'networks': 1`, `'process': 1`, `'machine': 1`, `'learning': 1`, `'algorithms': 1`, `'weather': 1`, `'nice': 1`, `'today': 1`.
        2. Max frequency $= 2$ (word `'data'`). Normalized: `'data' = 1.0`, other content words $= 0.5$.
        3. Sentences containing `'data'` ($S_1$ and $S_2$) receive significantly higher salience scores than the unrelated sentence $S_3$.
        4. Ranking correctly selects $S_1$ and $S_2$ as the summary.
        """)

        col_sum_adv, col_sum_lim = st.columns(2)
        with col_sum_adv:
            st.markdown("""
            #### ✅ Advantages
            * **Factually Grounded:** Zero risk of model hallucination since every sentence is copied exactly from the author's original words.
            * **Computationally Lightweight:** Requires minimal CPU memory and finishes in milliseconds.
            * **Domain Agnostic:** Works effectively across medical, scientific, and legal domains without custom fine-tuning.
            """)
        with col_sum_lim:
            st.markdown("""
            #### ⚠️ Limitations
            * **Pronoun Resolution (Anaphora):** An extracted sentence might start with *"He decided to..."* without including the preceding sentence that introduced who *"He"* is.
            * **No Paraphrasing:** Cannot condense a 40-word verbose sentence into a crisp 5-word takeaway.
            """)

    # =========================================================================
    # TAB 3: WORKFLOW PIPELINE
    # =========================================================================
    with tab_workflow:
        st.subheader("🔄 Visual Summarization Pipeline")
        st.caption("How an article is converted into an extractive summary:")

        steps = [
            ("1. Document Ingestion", "Accept full-length text, preserve paragraph breaks, validate length requirements."),
            ("2. Sentence Tokenization", "Use NLTK sent_tokenize to split text along linguistic punctuation boundaries."),
            ("3. Word Tokenization & Cleansing", "Strip punctuation, convert to lowercase, extract alphabetic tokens."),
            ("4. Stop-Word Removal & Frequency Table", "Filter out English stopwords; construct word occurrence frequency dictionary."),
            ("5. Max-Frequency Normalization", "Divide each word's count by max count to scale weights between 0.0 and 1.0."),
            ("6. Sentence Scoring & Normalization", "Sum word weights per sentence divided by length damping factor."),
            ("7. Top-K Selection & Chronological Reordering", "Extract top scoring sentences and restore them to their original sequence.")
        ]

        for title, desc in steps:
            st.markdown(f"""
            <div class="pipeline-step">
                <span class="pipeline-number">STEP</span>
                <strong style="color: #e2e8f0;">{title}</strong>
                <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.25rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
