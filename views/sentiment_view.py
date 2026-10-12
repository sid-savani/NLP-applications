"""
Sentiment Analysis View.
Includes:
- Interactive Try It with sample buttons, polarity gauge, and sentence breakdown
- Detailed How It Works (lexicons, polarity, subjectivity math)
- Step-by-step Workflow Pipeline
"""

import streamlit as st
import pandas as pd
from modules.sentiment import analyze_sentiment

SAMPLE_TEXTS = {
    "🌟 Rave Product Review": "This noise-canceling headset is absolutely phenomenal! The sound quality is crystal clear, battery life lasts days, and it feels supremely comfortable even during long flights.",
    "😡 Frustrated Customer Complaint": "Terrible customer service and extremely delayed shipping. The package arrived damaged and nobody responded to my support tickets for two weeks. Complete waste of money.",
    "📰 Neutral Factual Report": "The quarterly financial report was published this morning at 9:00 AM Eastern Time. Revenue increased by 2.4 percent compared to the prior fiscal quarter.",
    "⚖️ Mixed Ambivalent Feedback": "The camera on this phone takes breathtaking pictures, but the operating system battery drain is horribly frustrating and sluggish."
}


def render_sentiment_view():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="margin-bottom: 0.2rem;">🎭 Sentiment Analysis</h1>
        <p style="color: #94a3b8; font-size: 1.05rem;">
            Quantify emotional polarity and subjectivity in natural text using pattern-based lexicon scoring.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_try, tab_how, tab_workflow = st.tabs(["🎮 Try It", "🔬 How It Works", "🔄 Workflow Pipeline"])

    # =========================================================================
    # TAB 1: TRY IT
    # =========================================================================
    with tab_try:
        st.markdown("### 1. Choose or Type a Sentence")
        st.caption("Click any sample button below to instantly load an example, or enter your own custom text.")

        # Sample Buttons
        btn_cols = st.columns(len(SAMPLE_TEXTS))
        for idx, (label, sample_text) in enumerate(SAMPLE_TEXTS.items()):
            with btn_cols[idx]:
                if st.button(label, key=f"btn_sample_sent_{idx}", use_container_width=True):
                    st.session_state["sentiment_input"] = sample_text

        default_input = st.session_state.get("sentiment_input", SAMPLE_TEXTS["🌟 Rave Product Review"])
        user_input = st.text_area(
            "Enter text to evaluate sentiment:",
            value=default_input,
            height=120,
            key="sentiment_textarea",
            placeholder="Type your review, comment, or message here..."
        )

        analyze_clicked = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

        if analyze_clicked or user_input:
            if not user_input.strip():
                st.warning("⚠️ Input is empty. Please enter some text or click a sample button above.")
            else:
                result = analyze_sentiment(user_input)
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader("📊 Analysis Results")

                    # Primary Metrics
                    m_col1, m_col2, m_col3 = st.columns(3)
                    
                    with m_col1:
                        st.markdown(f"""
                        <div class="metric-box" style="border-top: 3px solid {result['sentiment_color']};">
                            <div class="metric-value" style="color: {result['sentiment_color']};">
                                {result['sentiment_emoji']} {result['sentiment_label']}
                            </div>
                            <div class="metric-label">Overall Sentiment</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with m_col2:
                        pol_val = result['polarity']
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #60a5fa;">
                                {pol_val:+.3f}
                            </div>
                            <div class="metric-label">Polarity (-1.0 to +1.0)</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with m_col3:
                        sub_val = result['subjectivity']
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #c084fc;">
                                {sub_val:.3f}
                            </div>
                            <div class="metric-label">Subjectivity (0.0 to 1.0)</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Visual Gauges
                    col_gauge1, col_gauge2 = st.columns(2)
                    with col_gauge1:
                        st.markdown("**Polarity Scale (-1.0 Negative ↔ +1.0 Positive)**")
                        # Normalize -1..1 to 0..1 for st.progress
                        normalized_pol = (result["polarity"] + 1.0) / 2.0
                        st.progress(float(normalized_pol))
                        st.caption(f"Score: `{result['polarity']}` (Center 0.0 is completely neutral)")

                    with col_gauge2:
                        st.markdown("**Subjectivity Scale (0.0 Objective ↔ 1.0 Opinion)**")
                        st.progress(float(result["subjectivity"]))
                        st.caption(f"Score: `{result['subjectivity']}` — *{result['subjectivity_label']}*")

                    # Sentence Breakdown if more than 1 sentence
                    if result["sentence_count"] > 1:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("#### 🔍 Sentence-by-Sentence Breakdown")
                        st.caption("Inspect how individual sentences contribute to the overall score:")
                        
                        breakdown_data = []
                        for s in result["sentence_breakdown"]:
                            breakdown_data.append({
                                "#": s["index"],
                                "Sentence": s["text"],
                                "Polarity": f"{s['polarity']:+.2f}",
                                "Subjectivity": f"{s['subjectivity']:.2f}",
                                "Tone": s["label"]
                            })
                        st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: HOW IT WORKS
    # =========================================================================
    with tab_how:
        st.subheader("🔬 How Lexicon-Based Sentiment Analysis Works")
        
        st.markdown("""
        #### 1. The NLP Problem Being Solved
        Human text conveys sentiment: attitudes, opinions, and emotions. Determining whether a piece of writing is positive, negative, or neutral enables organizations to gauge customer satisfaction, detect brand sentiment, and prioritize urgent complaints automatically.
        """)

        st.markdown("""
        #### 2. The TextBlob Algorithm & Core Intuition
        Under the hood, **TextBlob** uses the **Pattern** sentiment lexicon. This lexicon maps individual words, idioms, and adjectives to predefined numerical values for:
        * **Polarity ($P \in [-1.0, +1.0]$):** Measures semantic valence. Words like *"great"*, *"superb"*, and *"delightful"* have positive scores ($+0.8$), while *"awful"*, *"broken"*, and *"horrible"* have negative scores ($-0.7$).
        * **Subjectivity ($S \in [0.0, 1.0]$):** Measures whether a statement is factual ($0.0$) or an opinion ($1.0$). *"The phone weighs 180 grams"* has $S \approx 0$, while *"The phone is gorgeous"* has $S \approx 0.9$.
        """)

        st.markdown("""
        #### 3. Mathematical Intuition: Modifiers & Negations
        TextBlob does not merely sum isolated words; it applies syntactic rules for **intensifiers** and **negators**:
        * **Modifiers (Intensifiers):** Words like *"very"*, *"extremely"*, or *"slightly"* multiply the subsequent adjective's polarity.
          $$\\text{Polarity}(\\text{"very good"}) = 1.3 \\times \\text{Polarity}(\\text{"good"})$$
        * **Negations:** The word *"not"* flips the polarity sign and reduces intensity:
          $$\\text{Polarity}(\\text{"not good"}) = -0.5 \\times \\text{Polarity}(\\text{"good"})$$
        """)

        st.markdown("""
        #### 4. Worked Step-by-Step Example
        Consider the sentence: *"The phone has a great camera but terrible battery."*
        1. **Tokenization:** Breaks the sentence into words: `['the', 'phone', 'has', 'a', 'great', 'camera', 'but', 'terrible', 'battery']`.
        2. **Lexicon Matching:**
           - `'great'`: Polarity $= +0.8$, Subjectivity $= 0.75$
           - `'terrible'`: Polarity $= -1.0$, Subjectivity $= 1.0$
           - Other words (`phone`, `camera`, `battery`): Neutral (0.0)
        3. **Aggregation:** Averaging matched descriptors yields Polarity $= \\frac{0.8 + (-1.0)}{2} = -0.10$ (Slightly Negative).
        """)

        st.markdown("""
        #### 5. Real-World Applications
        * **Customer Support Triage:** Routing angry tickets to senior agents instantly.
        * **E-Commerce Product Reviews:** Aggregating thousands of buyer comments into product sentiment scores.
        * **Social Media Brand Monitoring:** Tracking public perception during marketing campaigns.
        * **Financial Market Signals:** Measuring news sentiment regarding company earnings reports.
        """)

        col_adv, col_lim = st.columns(2)
        with col_adv:
            st.markdown("""
            #### ✅ Advantages
            * **Ultra-fast:** Runs in milliseconds without GPU or heavy neural networks.
            * **Deterministic & Explainable:** You can trace exactly which words caused the score.
            * **Zero Training Data Needed:** Uses curated linguistic dictionaries out-of-the-box.
            """)
        with col_lim:
            st.markdown("""
            #### ⚠️ Limitations
            * **Struggles with Sarcasm:** *"Oh wonderful, another flat tire!"* often scores positive.
            * **Domain Sensitivity:** In medical text, *"cancer positive"* is medically bad, but lexicon may mark *"positive"* as good.
            * **No Global Context:** Lacks deep bidirectional sentence semantics found in Transformers.
            """)

    # =========================================================================
    # TAB 3: WORKFLOW PIPELINE
    # =========================================================================
    with tab_workflow:
        st.subheader("🔄 Visual Sentiment Processing Pipeline")
        st.caption("Here is the exact algorithmic path raw user text follows during analysis:")

        steps = [
            ("1. Input Ingestion & Sanitization", "Trims extraneous whitespace, validates string length, ensures non-empty content."),
            ("2. Sentence Segmentation", "Applies sentence tokenization boundary rules (punctuation, capitalization) to isolate clauses."),
            ("3. Part-of-Speech & Lemmatization", "Identifies grammatical roles (nouns, adjectives, adverbs) to locate modifiers and sentiment carriers."),
            ("4. Lexicon Lookup & Modifier Tuning", "Queries the sentiment dictionary. Multiplies intensity for adverbs ('very') and reverses sign for negations ('not')."),
            ("5. Averaging & Classification", "Calculates mean polarity and subjectivity, categorizes into emotional tiers (Positive/Neutral/Negative)."),
            ("6. Visual Presentation", "Renders progress gauges, badges, and sentence breakdown tables.")
        ]

        for title, desc in steps:
            st.markdown(f"""
            <div class="pipeline-step">
                <span class="pipeline-number">STEP</span>
                <strong style="color: #e2e8f0;">{title}</strong>
                <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.25rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
