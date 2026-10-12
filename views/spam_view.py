"""
Spam Message Detection View.
Features:
- Try It: Live classification with sample buttons, probability bars, and contributing token analysis
- Honest model training telemetry: Real accuracy, precision, recall, F1, and confusion matrix from test split
- Demonstration warning banner
- Comprehensive How It Works with Bayes theorem math and TF-IDF explanation
- Workflow Pipeline
"""

import streamlit as st
import pandas as pd
from modules.spam_classifier import get_spam_classifier

SAMPLE_MESSAGES = {
    "🎁 Prize Lottery Scam (Spam)": "CONGRATULATIONS! You have won a $1,000 Walmart Gift Card. Click here to claim your prize now!",
    "🚨 Phishing Account Alert (Spam)": "URGENT: Your bank account has been suspended due to suspicious activity. Verify your identity immediately at http://secure-bank-login.com",
    "🥪 Casual Meeting (Ham)": "Hey, are we still meeting for lunch at 12:30 today?",
    "💻 Code Review Notice (Ham)": "I pushed the new branch with unit tests for the classifier. Can you please review the pull request when you get a chance?"
}


def render_spam_view():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="margin-bottom: 0.2rem;">🛡️ Spam Message Detection</h1>
        <p style="color: #94a3b8; font-size: 1.05rem;">
            Binary text classification using TF-IDF n-gram vectorization and Multinomial Naive Bayes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Mandatory Demonstration Notice
    st.markdown("""
    <div class="custom-warning">
        ⚠️ <strong>Demonstration Dataset Notice:</strong> This model is trained on a small curated bundled dataset (60 samples) with an authentic 75/25 stratified train/test split. 
        It is intended strictly for interactive educational exploration of NLP classification mechanics and <strong>not for production security enforcement</strong>.
    </div>
    """, unsafe_allow_html=True)

    model = get_spam_classifier()

    tab_try, tab_diagnostics, tab_how, tab_workflow = st.tabs([
        "🎮 Try It",
        "📊 Model Telemetry & Metrics",
        "🔬 How It Works",
        "🔄 Workflow Pipeline"
    ])

    # =========================================================================
    # TAB 1: TRY IT
    # =========================================================================
    with tab_try:
        st.markdown("### 1. Test a Message")
        st.caption("Click a sample button to load a known message, or write your own text below:")

        btn_cols = st.columns(len(SAMPLE_MESSAGES))
        for idx, (label, sample_msg) in enumerate(SAMPLE_MESSAGES.items()):
            with btn_cols[idx]:
                if st.button(label, key=f"btn_sample_spam_{idx}", use_container_width=True):
                    st.session_state["spam_input"] = sample_msg

        default_text = st.session_state.get("spam_input", SAMPLE_MESSAGES["🎁 Prize Lottery Scam (Spam)"])
        user_text = st.text_area(
            "Enter message text to classify:",
            value=default_text,
            height=110,
            key="spam_textarea",
            placeholder="Type or paste any SMS or email text..."
        )

        check_clicked = st.button("🛡️ Classify Message", type="primary", use_container_width=True)

        if check_clicked or user_text:
            if not user_text.strip():
                st.warning("⚠️ Message is empty. Please enter some text to classify.")
            else:
                pred = model.predict(user_text)
                if "error" in pred:
                    st.error(f"❌ {pred['error']}")
                else:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader("🎯 Classification Decision")

                    col_res1, col_res2, col_res3 = st.columns(3)
                    with col_res1:
                        if pred["is_spam"]:
                            badge_html = """
                            <div class="metric-box" style="border-top: 3px solid #ef4444;">
                                <div class="metric-value" style="color: #ef4444;">🚨 SPAM</div>
                                <div class="metric-label">Predicted Class</div>
                            </div>
                            """
                        else:
                            badge_html = """
                            <div class="metric-box" style="border-top: 3px solid #10b981;">
                                <div class="metric-value" style="color: #10b981;">✅ HAM (Legitimate)</div>
                                <div class="metric-label">Predicted Class</div>
                            </div>
                            """
                        st.markdown(badge_html, unsafe_allow_html=True)

                    with col_res2:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #f87171;">
                                {pred['spam_probability'] * 100:.1f}%
                            </div>
                            <div class="metric-label">P(Spam | Message)</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_res3:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #34d399;">
                                {pred['ham_probability'] * 100:.1f}%
                            </div>
                            <div class="metric-label">P(Ham | Message)</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**Probability Distribution:**")
                    st.progress(float(pred["spam_probability"]))
                    st.caption(f"Spam probability: `{pred['spam_probability'] * 100:.2f}%` vs Legitimate probability: `{pred['ham_probability'] * 100:.2f}%`")

                    # Active Token Explanations
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("#### 🔍 Active Vocabulary Tokens Detected")
                    st.caption("These recognized words directly influenced the Naive Bayes likelihood calculations:")

                    if pred["active_tokens"]:
                        tokens_df = pd.DataFrame(pred["active_tokens"])
                        tokens_df.rename(columns={
                            "token": "Vocabulary Word",
                            "tendency": "Model Tendency",
                            "spam_log_prob": "Log-Prob (Spam)",
                            "ham_log_prob": "Log-Prob (Ham)"
                        }, inplace=True)
                        st.dataframe(tokens_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("ℹ️ No words in this input directly matched the training vocabulary (Out-of-Vocabulary words). In this case, classification relied solely on the class prior probability.")

    # =========================================================================
    # TAB 2: AUTHENTIC MODEL TELEMETRY & DIAGNOSTICS
    # =========================================================================
    with tab_diagnostics:
        st.subheader("📊 Authentic Model Training & Evaluation Telemetry")
        st.caption("All metrics shown below are genuinely calculated from the 75/25 stratified split without fabricated values.")

        metrics = model.metrics
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        with t_col1:
            st.metric("Test Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
        with t_col2:
            st.metric("Spam Precision", f"{metrics['precision'] * 100:.1f}%")
        with t_col3:
            st.metric("Spam Recall", f"{metrics['recall'] * 100:.1f}%")
        with t_col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.3f}")

        st.markdown("<br>", unsafe_allow_html=True)

        col_cm, col_split = st.columns(2)
        with col_cm:
            st.markdown("#### 🧩 Confusion Matrix (Test Split)")
            cm = metrics["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm,
                columns=["Pred: Ham", "Pred: Spam"],
                index=["Actual: Ham", "Actual: Spam"]
            )
            st.dataframe(cm_df, use_container_width=True)
            st.caption(f"True Negatives: {cm[0][0]} | False Positives: {cm[0][1]} | False Negatives: {cm[1][0]} | True Positives: {cm[1][1]}")

        with col_split:
            st.markdown("#### 📦 Dataset Distribution")
            st.markdown(f"""
            * **Total Samples:** `{metrics['total_samples']}`
            * **Training Split (75%):** `{metrics['train_samples']}` samples
            * **Testing Split (25%):** `{metrics['test_samples']}` samples
            * **Vocabulary Size:** `{metrics['vocabulary_size']}` unique n-grams
            * **Smoothing Parameter (Alpha):** `0.5` (Laplace / Lidstone)
            """)

        st.markdown("<br>", unsafe_allow_html=True)
        col_kw1, col_kw2 = st.columns(2)
        with col_kw1:
            st.markdown("#### 🚨 Top Spam-Associated Tokens")
            spam_kw_df = pd.DataFrame(metrics["top_spam_keywords"], columns=["Token", "Log-Odds Weight"])
            st.dataframe(spam_kw_df, use_container_width=True, hide_index=True)

        with col_kw2:
            st.markdown("#### ✅ Top Ham-Associated Tokens")
            ham_kw_df = pd.DataFrame(metrics["top_ham_keywords"], columns=["Token", "Log-Odds Weight"])
            st.dataframe(ham_kw_df, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 3: HOW IT WORKS
    # =========================================================================
    with tab_how:
        st.subheader("🔬 How Spam Classification Works")
        
        st.markdown("""
        #### 1. The NLP Problem Being Solved
        Electronic communication is flooded with unsolicited, fraudulent, or malicious messages (phishing, prize scams, credential harvesters). 
        The objective of **Binary Text Classification** is to automatically map arbitrary free-form text $x$ to a discrete label $y \in \{\\text{ham}, \\text{spam}\}$.
        """)

        st.markdown("""
        #### 2. The TF-IDF Feature Representation
        Computers cannot directly compute mathematical products on raw words. We convert text into a high-dimensional vector space using **TF-IDF**:
        $$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\text{IDF}(t, D)$$
        * **Term Frequency (TF):** How frequently term $t$ appears in document $d$.
        * **Inverse Document Frequency (IDF):** Penalizes words that appear across almost all documents (like *"the"*, *"is"*):
          $$\\text{IDF}(t, D) = \\ln\\left(\\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\\right) + 1$$
        """)

        st.markdown("""
        #### 3. Multinomial Naive Bayes & Bayes' Theorem
        Using Bayes' rule, the posterior probability of a class $C$ given feature vector $\mathbf{x} = (x_1, x_2, \dots, x_n)$ is:
        $$P(C \\mid \\mathbf{x}) = \\frac{P(C) \\prod_{i=1}^n P(x_i \\mid C)}{P(\\mathbf{x})}$$
        * **Prior $P(C)$:** The overall frequency of class $C$ in the training data.
        * **Likelihood $P(x_i \\mid C)$:** The probability that word $x_i$ appears given class $C$.
        * **The "Naive" Assumption:** Features are conditionally independent given the class label. Even though words often depend on each other, this simplification works surprisingly well in practice.
        * **Laplace Smoothing ($\\alpha = 0.5$):** Prevents zero-probability multiplication when an unseen word occurs:
          $$\\hat{P}(w_i \\mid C) = \\frac{N_{C, i} + \\alpha}{N_C + \\alpha |V|}$$
        """)

        st.markdown("""
        #### 4. Worked Step-by-Step Example
        Suppose the message contains the word *"urgent"*:
        1. From training data:
           - In Spam: *"urgent"* appeared 8 times out of 100 total spam words.
           - In Ham: *"urgent"* appeared 1 time out of 100 total ham words.
        2. With equal prior $P(\\text{Spam}) = P(\\text{Ham}) = 0.5$:
           $$P(\\text{urgent} \\mid \\text{Spam}) \\approx \\frac{8 + 0.5}{100 + 50} = 0.056$$
           $$P(\\text{urgent} \\mid \\text{Ham}) \\approx \\frac{1 + 0.5}{100 + 50} = 0.010$$
        3. Since $0.056 > 0.010$, the presence of *"urgent"* multiplies the likelihood in favor of Spam by over 5x!
        """)

        col_sp_adv, col_sp_lim = st.columns(2)
        with col_sp_adv:
            st.markdown("""
            #### ✅ Advantages
            * **Blazing Fast Training & Inference:** Linear computational complexity $O(N \\times d)$.
            * **Effective with Small Datasets:** Few parameters compared to neural models.
            * **High Interpretability:** Inspecting feature weights tells you exactly why a message was flagged.
            """)
        with col_sp_lim:
            st.markdown("""
            #### ⚠️ Limitations
            * **Ignores Word Order:** Bag-of-words sees *"not bad, good"* similarly to *"not good, bad"*.
            * **Vulnerable to Adversarial Attacks:** Spammers can bypass filters using misspellings (*"w!n f-r-e-e"*) or appending random legitimate dictionary text (*"good word stuffing"*).
            """)

    # =========================================================================
    # TAB 4: WORKFLOW PIPELINE
    # =========================================================================
    with tab_workflow:
        st.subheader("🔄 Visual Spam Classification Pipeline")
        st.caption("From raw text input to probabilistic classification:")

        steps = [
            ("1. Data Preprocessing & Cleaning", "Strip HTML tags, lowercase all characters, extract words and bi-grams."),
            ("2. Stop-Word Removal", "Eliminate common non-informative English tokens ('the', 'and', 'at') to reduce noise."),
            ("3. TF-IDF Matrix Transformation", "Map token counts to statistical TF-IDF weights using the vocabulary learned during training."),
            ("4. Prior & Likelihood Computation", "Sum log-probabilities for each class using the fitted Multinomial Naive Bayes parameters."),
            ("5. Softmax Normalization", "Exponentiate and normalize log-likelihoods to produce calibrated posterior probabilities."),
            ("6. Threshold & Decision Rule", "If P(Spam | text) >= 0.5, classify as SPAM; otherwise classify as HAM.")
        ]

        for title, desc in steps:
            st.markdown(f"""
            <div class="pipeline-step">
                <span class="pipeline-number">STEP</span>
                <strong style="color: #e2e8f0;">{title}</strong>
                <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.25rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
