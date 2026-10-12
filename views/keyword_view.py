"""
Keyword & Keyphrase Extraction View.
Features:
- Try It: Extraction with top-N slider, keyphrase toggle, sample texts, interactive badges, and TF-IDF relevance table
- Detailed How It Works: TF-IDF math, n-gram windows, stop-word pruning
- Workflow Pipeline
"""

import streamlit as st
import pandas as pd
from modules.keyword_extractor import extract_keywords

SAMPLE_TEXTS = {
    "⚛️ Quantum Computing": (
        "Quantum computing harnesses the strange principles of quantum mechanics to solve computational problems "
        "that remain intractable for classical supercomputers. Instead of standard binary bits that exist strictly as "
        "zero or one, quantum processors utilize qubits capable of existing in superposition states. "
        "Through quantum entanglement, qubits demonstrate correlated behavior across spatial distances, enabling "
        "exponential parallelism. Key applications include Shor's algorithm for factoring prime integers, Grover's algorithm "
        "for unstructured database search, quantum cryptography, and complex molecular simulations for drug discovery."
    ),
    "🌍 Climate Policy & Energy": (
        "Transitioning global energy grids away from fossil fuels toward renewable energy infrastructure is essential "
        "for mitigating dangerous climate change. Solar photovoltaic arrays and offshore wind turbines provide scalable clean electricity. "
        "However, integrating intermittent renewable sources requires robust energy storage systems and advanced battery chemistry. "
        "Carbon pricing mechanisms, emissions trading systems, and international climate agreements establish financial incentives "
        "for industrial decarbonization and carbon capture technologies."
    ),
    "🧠 Deep Neural Networks": (
        "Deep learning architectures rely on multi-layer artificial neural networks to learn hierarchical representations "
        "from raw sensory data. Convolutional neural networks specialize in spatial feature extraction for computer vision, "
        "while recurrent networks and attention transformers excel at sequential natural language processing. "
        "During training, stochastic gradient descent optimizes parameter weights by backpropagating loss gradients calculated via the chain rule."
    )
}


def render_keyword_view():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="margin-bottom: 0.2rem;">🔑 Keyword & Keyphrase Extraction</h1>
        <p style="color: #94a3b8; font-size: 1.05rem;">
            Isolate salient topics, terminology, and multi-word phrases using TF-IDF feature importance ranking.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_try, tab_how, tab_workflow = st.tabs(["🎮 Try It", "🔬 How It Works", "🔄 Workflow Pipeline"])

    # =========================================================================
    # TAB 1: TRY IT
    # =========================================================================
    with tab_try:
        st.markdown("### 1. Select Sample or Enter Text")
        st.caption("Click a sample button to load a passage, or input your own content:")

        btn_cols = st.columns(len(SAMPLE_TEXTS))
        for idx, (label, sample_text) in enumerate(SAMPLE_TEXTS.items()):
            with btn_cols[idx]:
                if st.button(label, key=f"btn_sample_kw_{idx}", use_container_width=True):
                    st.session_state["keyword_input"] = sample_text

        default_text = st.session_state.get("keyword_input", SAMPLE_TEXTS["⚛️ Quantum Computing"])
        user_text = st.text_area(
            "Enter text to extract keywords from:",
            value=default_text,
            height=140,
            key="keyword_textarea",
            placeholder="Type or paste paragraphs here..."
        )

        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            top_k = st.slider(
                "Number of Keywords to Extract (Top-N):",
                min_value=3,
                max_value=15,
                value=8,
                help="Adjusts how many top scoring items to return."
            )
        with col_cfg2:
            include_phrases = st.checkbox(
                "Include 2-Word Keyphrases (Bigrams)",
                value=True,
                help="When enabled, extracts both single words and meaningful pairs like 'quantum computing' or 'renewable energy'."
            )

        extract_clicked = st.button("🔑 Extract Keywords", type="primary", use_container_width=True)

        if extract_clicked or user_text:
            if not user_text.strip():
                st.warning("⚠️ Input is empty. Please enter some text to extract keywords.")
            else:
                res = extract_keywords(user_text, top_n=top_k, include_phrases=include_phrases)
                if "error" in res:
                    st.error(f"❌ {res['error']}")
                else:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader("🏷️ Extracted Salient Keywords")

                    # Tag Cloud / Badges
                    st.markdown("#### Topic Tags:")
                    badges_html = " ".join([
                        f'<span class="nlp-badge" style="font-size: 0.95rem; padding: 0.35rem 0.85rem; border-color: #8b5cf6;">'
                        f'#{kw["keyword"]} <small style="color:#a5b4fc">({kw["relevance_pct"]}%)</small>'
                        f'</span>'
                        for kw in res["keywords"]
                    ])
                    st.markdown(f'<div style="margin-bottom: 1.5rem;">{badges_html}</div>', unsafe_allow_html=True)

                    # Diagnostic Table
                    st.markdown("#### 📊 Relevance Score Breakdown")
                    st.caption("Inspect the algorithmic TF-IDF weights and raw frequencies:")

                    kw_df = pd.DataFrame(res["keywords"])
                    kw_df.rename(columns={
                        "rank": "Rank",
                        "keyword": "Extracted Term",
                        "score": "TF-IDF Weight",
                        "relevance_pct": "Relative Salience (%)",
                        "frequency": "Occurrences in Text",
                        "type": "Gram Type"
                    }, inplace=True)
                    st.dataframe(kw_df[["Rank", "Extracted Term", "Gram Type", "Relative Salience (%)", "Occurrences in Text", "TF-IDF Weight"]], use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: HOW IT WORKS
    # =========================================================================
    with tab_how:
        st.subheader("🔬 How TF-IDF Keyword Extraction Works")
        
        st.markdown("""
        #### 1. The NLP Problem Being Solved
        Every day, millions of articles, research papers, and customer reviews are created. 
        Reading full documents is impossible at scale; humans and machines need **salient tags and keywords** that instantly summarize the core subject matter for search indexing, tagging, and recommendation engines.
        """)

        st.markdown("""
        #### 2. Why Simple Frequency is Not Enough
        If you only count word frequencies in a passage, the top words will always be:
        *"the", "is", "at", "which", "on", "and"*
        These are **stop words**—syntactic glue that conveys little semantic meaning.
        
        Even after removing stop words, general words like *"make"*, *"say"*, or *"thing"* may occur frequently without reflecting the distinct subject matter of the text.
        """)

        st.markdown("""
        #### 3. The TF-IDF Solution
        **TF-IDF** (Term Frequency - Inverse Document Frequency) balances local occurrence with global distinctiveness:
        $$\\text{TF-IDF}(t) = \\text{TF}(t) \\times \\text{IDF}(t)$$
        * **Sublinear Term Frequency:** Instead of linear count, we use $1 + \\ln(\\text{TF})$ to prevent a word repeated 10 times from having 10x the importance.
        * **Inverse Document Frequency:** Down-weights words that occur in many sub-clauses or sentences, elevating terms that are uniquely concentrated in specific key ideas.
        * **N-Gram Keyphrase Generation:** By examining sliding 2-word windows, the algorithm captures compound concepts like *"quantum computing"* or *"machine learning"* that carry far more specific meaning than *"quantum"* or *"learning"* individually.
        """)

        st.markdown("""
        #### 4. Worked Step-by-Step Example
        In the passage on Quantum Computing:
        1. Tokenizer extracts words: `qubits`, `quantum`, `computing`, `classical`, `superposition`.
        2. Bigrams generated: `["quantum computing", "quantum mechanics", "quantum processors"]`.
        3. Stop words removed: `["the", "to", "for", "that", "instead", "of"]`.
        4. The phrase `"quantum computing"` appears multiple times and is highly distinctive, earning the highest TF-IDF vector score.
        """)

        col_k_adv, col_k_lim = st.columns(2)
        with col_k_adv:
            st.markdown("""
            #### ✅ Advantages
            * **Unsupervised:** Requires no manual labeling or expensive model training.
            * **Fast & Scalable:** Processes thousands of documents per second.
            * **Transparent:** Every score is a direct product of word frequency mathematics.
            """)
        with col_k_lim:
            st.markdown("""
            #### ⚠️ Limitations
            * **Synonym Blindness:** Treats *"car"* and *"automobile"* as completely unrelated terms.
            * **Syntactic Variations:** Plural forms like *"qubits"* and singular *"qubit"* may appear separately unless stemmed or lemmatized.
            """)

    # =========================================================================
    # TAB 3: WORKFLOW PIPELINE
    # =========================================================================
    with tab_workflow:
        st.subheader("🔄 Visual Keyword Extraction Pipeline")
        st.caption("How text is transformed into ranked keyword tags:")

        steps = [
            ("1. Input Document Segmentation", "Split document into sentences or conceptual clauses to form a pseudo-corpus."),
            ("2. N-Gram Generation", "Extract 1-gram tokens (single words) and 2-gram combinations (compound keyphrases)."),
            ("3. Stop-Word & Punctuation Pruning", "Filter out universal stop words and punctuation symbols."),
            ("4. TF-IDF Matrix Computation", "Calculate sublinear term frequencies and inverse document frequencies across sentences."),
            ("5. Salience Aggregation", "Sum TF-IDF weights for each vocabulary feature across the document."),
            ("6. Normalization & Ranking", "Scale the maximum score to 100% and return the top-N highest scoring keywords.")
        ]

        for title, desc in steps:
            st.markdown(f"""
            <div class="pipeline-step">
                <span class="pipeline-number">STEP</span>
                <strong style="color: #e2e8f0;">{title}</strong>
                <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.25rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
