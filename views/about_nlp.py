"""
About NLP View.
Dedicated educational masterclass explaining:
- The End-to-End NLP Pipeline
- Traditional NLP vs. Transformers & Modern LLMs
- Rule-Based vs. Statistical vs. Machine Learning vs. Deep Learning
"""

import streamlit as st
import pandas as pd


def render_about_nlp():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="margin-bottom: 0.2rem;">📖 Masterclass: Understanding Modern NLP</h1>
        <p style="color: #94a3b8; font-size: 1.05rem;">
            A foundational guide to linguistic pipelines, mathematical representations, and the architectural evolution of language AI.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_pipeline, tab_evolution, tab_paradigms = st.tabs([
        "🔄 The Classical NLP Pipeline",
        "⚡ Traditional NLP vs. Transformers & LLMs",
        "🏛️ Rule-Based vs. Statistical vs. Machine Learning"
    ])

    # =========================================================================
    # TAB 1: THE NLP PIPELINE
    # =========================================================================
    with tab_pipeline:
        st.subheader("🛠️ The End-to-End NLP Lifecycle")
        st.markdown("""
        Before human language can be parsed by any computational model, it must pass through a multi-stage refinement pipeline. 
        Each phase strips noise and transforms unstructured raw text into structured numerical representations.
        """)

        pipeline_stages = [
            ("1. Text Acquisition & Ingestion", "Web scraping, OCR scans, audio-to-text transcripts, or raw database strings.", "Raw HTML or text file containing typos, symbols, and formatting quirks."),
            ("2. Text Cleaning & Normalization", "Case folding (lowercasing), HTML stripping, Unicode normalization, handling contractions (e.g. 'can't' → 'cannot').", "Removes irrelevant character noise so identical words match."),
            ("3. Sentence & Word Tokenization", "Segmenting continuous character streams into discrete sentences and lexical tokens (words, sub-words, punctuation).", "Converts 'I love NLP!' into ['I', 'love', 'NLP', '!']."),
            ("4. Stop-Word Removal", "Filtering high-frequency grammatical functional words ('the', 'is', 'at') that lack domain specificity.", "Reduces vocabulary size and concentrates computational power on salient content words."),
            ("5. Stemming & Lemmatization", "Reducing words to their root forms. Stemming cuts off suffixes heuristically ('running' → 'run'); Lemmatization uses vocabulary and morphological analysis to return valid dictionary lemmas ('better' → 'good').", "Unifies morphological variants into common base representations."),
            ("6. Part-of-Speech (POS) Tagging", "Assigning grammatical categories (noun, verb, adjective, preposition) using syntactic context.", "Differentiates between 'book' as a noun ('read a book') vs 'book' as a verb ('book a flight')."),
            ("7. Vectorization & Feature Extraction", "Transforming text into numerical matrices: One-Hot Encoding, Bag-of-Words, TF-IDF, Word2Vec embeddings, or contextual Transformer tensors.", "Enables linear algebra, dot products, and statistical models to operate on text."),
            ("8. Modeling & Downstream Tasks", "Classification, sentiment scoring, named entity recognition (NER), translation, summarization, or question answering.", "Final business value or prediction output.")
        ]

        for step_title, step_detail, step_why in pipeline_stages:
            st.markdown(f"""
            <div class="pipeline-step" style="border-left-color: #8b5cf6;">
                <strong style="color: #c7d2fe; font-size: 1.05rem;">{step_title}</strong>
                <div style="color: #cbd5e1; margin-top: 0.3rem;"><strong>Action:</strong> {step_detail}</div>
                <div style="color: #94a3b8; font-size: 0.88rem; margin-top: 0.2rem;"><em>Why it matters:</em> {step_why}</div>
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: TRADITIONAL NLP VS TRANSFORMERS
    # =========================================================================
    with tab_evolution:
        st.subheader("⚡ The Paradigm Shift: Traditional NLP vs. Transformers")
        st.markdown("""
        Between 1990 and 2017, NLP relied on handcrafted linguistic features, n-grams, and statistical counters. 
        In 2017, the landmark paper *"Attention Is All You Need"* (Vaswani et al.) introduced the **Transformer architecture**, revolutionizing the entire field.
        """)

        comp_data = {
            "Dimension": [
                "Core Representation",
                "Word Context",
                "Word Order",
                "Polysemy Handling",
                "Computational Footprint",
                "Training Requirement",
                "Interpretability",
                "Typical Tooling"
            ],
            "Traditional NLP (TF-IDF, Naive Bayes)": [
                "Sparse, high-dimensional vectors (Bag-of-Words, TF-IDF)",
                "Fixed, static representations. A word's vector never changes.",
                "Mostly discarded (Bag-of-Words) or limited to local n-grams (bigrams/trigrams)",
                "Fails: 'apple' the fruit and 'Apple' the company have the same representation.",
                "Extremely lightweight (CPU-friendly, runs in milliseconds, kilobytes of RAM)",
                "Trains on small datasets in seconds without special hardware",
                "High: You can inspect individual word log-odds and exact feature weights",
                "NLTK, spaCy, TextBlob, scikit-learn"
            ],
            "Modern Transformers (BERT, GPT, T5)": [
                "Dense, continuous embeddings (768 to 4096 dimensions)",
                "Contextual: A word's vector shifts based on all surrounding words in the sentence.",
                "Positionally encoded using self-attention across the entire sequence",
                "Excels: Self-attention dynamically assigns different meanings based on sentence context.",
                "Resource heavy (Requires modern GPUs, gigabytes of VRAM)",
                "Pre-trained on trillions of tokens across clusters of hundreds of GPUs",
                "Black-box: Billions of neural parameters make exact causal reasoning difficult",
                "PyTorch, Hugging Face, Transformers, vLLM"
            ]
        }

        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        #### Why Traditional NLP Still Matters in Production:
        1. **Zero Latency:** Traditional models execute in sub-millisecond speeds on cheap micro-instances.
        2. **Explainability & Compliance:** Regulated industries (finance, healthcare, legal) often require transparent rules where every decision can be audited.
        3. **Low Maintenance:** No risk of hallucinations, drift, or catastrophic forgetting.
        4. **Ideal Baselines:** Every rigorous machine learning project must evaluate whether a simple TF-IDF + Logistic Regression / Naive Bayes model suffices before paying for expensive LLM infrastructure.
        """)

    # =========================================================================
    # TAB 3: RULE-BASED VS STATISTICAL VS MACHINE LEARNING
    # =========================================================================
    with tab_paradigms:
        st.subheader("🏛️ The Three Historic Eras of NLP")
        st.markdown("""
        Natural Language Processing has evolved across three distinct technological eras:
        """)

        p_col1, p_col2, p_col3 = st.columns(3)

        with p_col1:
            st.markdown("""
            <div class="tool-card">
                <div class="card-icon">📜</div>
                <div class="card-title">1. Rule-Based (1950s–1980s)</div>
                <div class="card-desc">
                    Linguists wrote explicit handwritten grammatical rules, regular expressions, and formal grammar trees.
                </div>
                <hr style="border-color: rgba(99,102,241,0.2);">
                <div style="font-size: 0.85rem; color: #cbd5e1;">
                    <strong>Mechanism:</strong> If-then logic, CFGs (Context-Free Grammars), regex patterns, sentiment dictionaries.<br><br>
                    <strong>Strengths:</strong> 100% deterministic, zero training data needed.<br><br>
                    <strong>Weakness:</strong> Extremely brittle. Human language has too many exceptions to hardcode by hand.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with p_col2:
            st.markdown("""
            <div class="tool-card">
                <div class="card-icon">📈</div>
                <div class="card-title">2. Statistical (1990s–2010s)</div>
                <div class="card-desc">
                    Replaced handwritten rules with probability distributions calculated from large corpora.
                </div>
                <hr style="border-color: rgba(99,102,241,0.2);">
                <div style="font-size: 0.85rem; color: #cbd5e1;">
                    <strong>Mechanism:</strong> Hidden Markov Models (HMMs), n-gram language models, TF-IDF, Naive Bayes.<br><br>
                    <strong>Strengths:</strong> Handles uncertainty probabilistically, robust to typos and rare phrasing.<br><br>
                    <strong>Weakness:</strong> Suffers from sparsity; cannot generalize to synonyms not in training corpus.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with p_col3:
            st.markdown("""
            <div class="tool-card">
                <div class="card-icon">🧠</div>
                <div class="card-title">3. Deep Learning & LLMs (2015+)</div>
                <div class="card-desc">
                    Multi-layer neural networks learn semantic representations and reasoning directly from data.
                </div>
                <hr style="border-color: rgba(99,102,241,0.2);">
                <div style="font-size: 0.85rem; color: #cbd5e1;">
                    <strong>Mechanism:</strong> Recurrent networks (LSTMs), Transformers (Self-Attention), Auto-Regressive LLMs.<br><br>
                    <strong>Strengths:</strong> Human-like fluency, transfer learning across domains, few-shot reasoning.<br><br>
                    <strong>Weakness:</strong> High compute cost, potential hallucinations, lack of explainability.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **Key Takeaway:** Real-world engineering rarely uses just one approach in isolation. The most resilient production systems combine rule-based filters (guardrails), fast statistical classifiers (triage), and deep transformer models (complex synthesis).")
