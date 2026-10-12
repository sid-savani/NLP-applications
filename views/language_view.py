"""
Language Detection View.
Features:
- Try It: Real-time detection with multi-language sample buttons, candidate probability rankings, flag badges, and edge-case validation
- Detailed How It Works: Character n-gram profiles, Markov models, ISO-639 standards
- Workflow Pipeline
"""

import streamlit as st
import pandas as pd
from modules.lang_detector import detect_language

SAMPLE_LANGUAGES = {
    "🇫🇷 French": "Bonjour! Le traitement automatique du langage naturel est une discipline passionnante.",
    "🇪🇸 Spanish": "Hola a todos! La inteligencia artificial y el aprendizaje automático están transformando el mundo.",
    "🇩🇪 German": "Guten Tag! Die Verarbeitung natürlicher Sprache bietet viele faszinierende Möglichkeiten für moderne Software.",
    "🇯🇵 Japanese": "こんにちは！自然言語処理は人工知能の非常に重要な分野です。",
    "🇷🇺 Russian": "Привет! Обработка естественного языка открывает невероятные горизонты для науки и технологий.",
    "🇮🇳 Hindi": "नमस्ते! प्राकृतिक भाषा प्रसंस्करण कंप्यूटर विज्ञान और कृत्रिम बुद्धिमत्ता का एक महत्वपूर्ण क्षेत्र है।",
    "🇸🇦 Arabic": "مرحبا! معالجة اللغة الطبيعية هي أحد أكثر مجالات الذكاء الاصطناعي إثارة للاهتمام.",
    "🇮🇹 Italian": "Buongiorno a tutti! L'apprendimento automatico sta rivoluzionando il modo in cui comunichiamo."
}


def render_language_view():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="margin-bottom: 0.2rem;">🌐 Language Detection</h1>
        <p style="color: #94a3b8; font-size: 1.05rem;">
            Statistical language identification using character n-gram probability profiles across 30+ languages.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_try, tab_how, tab_workflow = st.tabs(["🎮 Try It", "🔬 How It Works", "🔄 Workflow Pipeline"])

    # =========================================================================
    # TAB 1: TRY IT
    # =========================================================================
    with tab_try:
        st.markdown("### 1. Test Different Languages")
        st.caption("Click any international sample below, or type your own text in any language:")

        # 2 rows of 4 buttons
        keys = list(SAMPLE_LANGUAGES.keys())
        r1_cols = st.columns(4)
        for i in range(4):
            with r1_cols[i]:
                if st.button(keys[i], key=f"btn_lang_{i}", use_container_width=True):
                    st.session_state["lang_input"] = SAMPLE_LANGUAGES[keys[i]]

        r2_cols = st.columns(4)
        for i in range(4, 8):
            with r2_cols[i - 4]:
                if st.button(keys[i], key=f"btn_lang_{i}", use_container_width=True):
                    st.session_state["lang_input"] = SAMPLE_LANGUAGES[keys[i]]

        default_text = st.session_state.get("lang_input", SAMPLE_LANGUAGES["🇫🇷 French"])
        user_text = st.text_area(
            "Enter text in any language to detect:",
            value=default_text,
            height=100,
            key="lang_textarea",
            placeholder="Type or paste words in any language..."
        )

        detect_clicked = st.button("🌐 Detect Language", type="primary", use_container_width=True)

        if detect_clicked or user_text:
            if not user_text.strip():
                st.warning("⚠️ Input is empty. Please enter some text to detect its language.")
            else:
                res = detect_language(user_text)
                if "error" in res:
                    st.error(f"❌ {res['error']}")
                else:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader("🎯 Detection Result")

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"""
                        <div class="metric-box" style="border-top: 3px solid #6366f1;">
                            <div class="metric-value">
                                {res['flag']} {res['name']}
                            </div>
                            <div class="metric-label">Identified Language ({res['code'].upper()})</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with c2:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #34d399;">
                                {res['confidence_pct']}%
                            </div>
                            <div class="metric-label">Model Confidence Score</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with c3:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #a78bfa; font-size: 1.3rem;">
                                {res['family']}
                            </div>
                            <div class="metric-label">Linguistic Family</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"**Confidence Gauge ({res['name']}):**")
                    st.progress(float(res["confidence_pct"] / 100.0))

                    # Candidate probabilities
                    if len(res["candidates"]) > 1:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("#### 📊 Candidate Probabilities")
                        st.caption("When text has overlapping roots or short length, multiple languages may share probabilities:")
                        
                        cand_df = pd.DataFrame(res["candidates"])
                        cand_df.rename(columns={
                            "flag": "Flag",
                            "name": "Language",
                            "code": "ISO Code",
                            "family": "Language Family",
                            "probability": "Probability (%)"
                        }, inplace=True)
                        st.dataframe(cand_df[["Flag", "Language", "ISO Code", "Language Family", "Probability (%)"]], use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: HOW IT WORKS
    # =========================================================================
    with tab_how:
        st.subheader("🔬 How Statistical Language Detection Works")
        
        st.markdown("""
        #### 1. The NLP Problem Being Solved
        Modern global platforms receive user-generated content in hundreds of tongues. Before an NLP pipeline can tokenize, stem, parse grammar, or translate text, it must first establish: **what language is this text written in?**
        """)

        st.markdown("""
        #### 2. The Character N-Gram Profile Algorithm
        The `langdetect` library is a Python port of Google's renowned language-detection library. Instead of memorizing giant dictionaries for every language on Earth, it uses **character n-grams**:
        * An **n-gram** is a contiguous sequence of $n$ characters.
        * For example, in the word *"apple"*, character 3-grams (tri-grams) are: `["app", "ppl", "ple"]`.
        * Different human languages possess distinct statistical fingerprints of character tri-grams:
          - English features frequent sequences like `"the"`, `"ing"`, `"ion"`, `"ed "`.
          - French has `"les"`, `"des"`, `"que"`, `"eur"`.
          - German features `"sch"`, `"der"`, `"ung"`, `"ich"`.
        """)

        st.markdown("""
        #### 3. The Probability Model
        1. Pre-computed probability profiles for over 50 languages store the expected distribution of tri-grams derived from massive multilingual corpora (Wikipedia, news articles).
        2. When a user supplies input text $T$, the detector extracts all character tri-grams $t_1, t_2, \dots, t_k$.
        3. A Naive Bayes classifier computes the likelihood $P(T \\mid L)$ across each candidate language $L$:
           $$P(L \\mid T) \\propto P(L) \\prod_{i=1}^k P(t_i \\mid L)$$
        4. The language with the highest posterior probability is returned as the primary classification.
        """)

        st.markdown("""
        #### 4. Worked Step-by-Step Example
        Input: *"Bonjour tout le monde"*
        1. Extract character 3-grams: `["bon", "onj", "njo", "jou", "our", "ur ", " to", "tou", "out", "ut ", " le", "le ", " mo", "mon", "ond", "nde"]`.
        2. Look up the prior probability of each tri-gram in language frequency tables:
           - In French: `"bon"`, `"jou"`, `"tou"`, `" le"` have high frequency.
           - In English: `"bon"`, `"jou"` have virtually zero occurrence.
        3. Result: $P(\\text{French} \\mid T) > 99.9\%$.
        """)

        col_l_adv, col_l_lim = st.columns(2)
        with col_l_adv:
            st.markdown("""
            #### ✅ Advantages
            * **Compact & Fast:** Frequency profiles consume negligible disk and RAM.
            * **Robust to Typos:** Since it evaluates overlapping character chunks, a single typo in a sentence does not break detection.
            * **No Dictionary Lookups:** Can identify languages even for words never seen before.
            """)
        with col_l_lim:
            st.markdown("""
            #### ⚠️ Limitations
            * **Short Text Ambiguity:** A single word like *"no"* exists in English, Spanish, Italian, and Polish.
            * **Code-Switching:** Sentences mixing two languages (e.g., Spanglish) force the model to choose one dominant label.
            * **Non-Alphabetical Input:** Numbers, emojis, or punctuation alone cannot be classified.
            """)

    # =========================================================================
    # TAB 3: WORKFLOW PIPELINE
    # =========================================================================
    with tab_workflow:
        st.subheader("🔄 Visual Language Identification Pipeline")
        st.caption("How raw character streams are mapped to language classifications:")

        steps = [
            ("1. String Sanitization & Validation", "Check that string is non-empty and contains at least 3 alphanumeric characters."),
            ("2. Unicode Normalization", "Normalize accents, diacritics, and non-Latin scripts (Cyrillic, Arabic, Devanagari, Hanzi)."),
            ("3. Character N-Gram Extraction", "Slide a window of lengths 2, 3, and 4 characters across words to extract sub-word tokens."),
            ("4. Probability Profile Lookup", "Match extracted n-grams against pre-trained probabilistic tables for 50+ languages."),
            ("5. Bayesian Posterior Estimation", "Aggregate log probabilities across all n-grams to estimate candidate language posteriors."),
            ("6. Metadata & Flag Mapping", "Map ISO 639-1 two-letter code to human-readable language name, flag emoji, and linguistic family.")
        ]

        for title, desc in steps:
            st.markdown(f"""
            <div class="pipeline-step">
                <span class="pipeline-number">STEP</span>
                <strong style="color: #e2e8f0;">{title}</strong>
                <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.25rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
