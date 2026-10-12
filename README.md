# 🧠 NLP Playground — Interactive NLP Lab

An interactive, local, and educational dashboard exploring core Natural Language Processing (NLP) algorithms. Built with Python and Streamlit, this project provides a hands-on laboratory for testing live inputs, inspecting step-by-step vectorization pipelines, and viewing honest algorithmic evaluation metrics without third-party APIs, API keys, or cloud dependencies.

---

## 📌 1. Project Overview & Purpose

### The Core Problem
Most modern NLP tutorials and demonstrations either rely on opaque cloud APIs (e.g., commercial LLM endpoints) or treat machine learning algorithms as "black boxes." In doing so, they obscure how text is actually normalized, vectorized into numbers, and evaluated mathematically.

### The Purpose of NLP Playground
1. **Educational Transparency:** Demystify classical NLP mechanics by exposing the full lifecycle: from character sanitization and tokenization to TF-IDF matrix generation and Bayesian posterior calculation.
2. **100% Local Execution:** Runs completely offline on commodity CPU hardware with zero API keys, rate limits, or external network latency.
3. **Honest, Unfabricated Metrics:** Evaluates machine learning models (e.g., Spam Classifier) on an authentic empirical 75/25 train/test split, displaying real accuracy, precision, recall, and a confusion matrix.
4. **Three-Tier Exploration Architecture:** Every tool includes:
   - **🎮 Try It / Sandbox:** 1-click preset sample scenarios, text areas, and live visual gauges.
   - **🔬 How It Works:** Deep-dive into the underlying math, formulas, worked examples, advantages, and limitations.
   - **🔄 Workflow Pipeline:** Sequential step-by-step visual cards illustrating data transformation stages.

---

## 🛠️ 2. What Was Made (The 5 NLP Tools + Guide)

| # | Tool | Technique & Algorithm | Functionality & Features |
| :--- | :--- | :--- | :--- |
| **1** | **🎭 Sentiment Analysis** | Pattern Sentiment Lexicon (`TextBlob`) | Computes continuous **Polarity** ($-1.0$ to $+1.0$) and **Subjectivity** ($0.0$ to $1.0$). Handles intensifier modifiers (`very`), negation flipping (`not good`), and granular sentence breakdown. |
| **2** | **🛡️ Spam Message Detection** | TF-IDF + Multinomial Naive Bayes (`scikit-learn`) | Classifies messages into **Spam** or **Ham (Legitimate)** with posterior probabilities, active token attribution, and genuine test split metrics (Accuracy, Precision, Recall, Confusion Matrix). |
| **3** | **📝 Text Summarization** | Extractive Frequency Ranking (`NLTK`) | Extracts high-salience sentences based on normalized word frequencies and length-penalized sentence scoring, achieving variable compression with **0% risk of hallucination**. |
| **4** | **🌐 Language Detection** | Character N-Gram Probability Models (`langdetect`) | Identifies language across **32+ ISO 639-1 languages**, outputting native language names, linguistic families, confidence scores, and multi-candidate probability rankings. |
| **5** | **🔑 Keyword Extraction** | Sublinear TF-IDF Feature Salience (`scikit-learn`) | Extracts unigrams and compound bigram keyphrases (e.g., *"quantum computing"*), complete with relative salience percentages and in-text occurrence counts. |
| **6** | **📖 About NLP Guide** | Pedagogical Architecture Masterclass | Outlines the end-to-end NLP pipeline, compares Traditional NLP vs. Modern Transformers (BERT/GPT), and breaks down the three historical epochs (Rule-Based, Statistical, Deep Learning). |

---

## 💻 3. Technology Stack & Tools Used

### Core Frameworks & Runtime
* **Python 3.10+**: Core programming language.
* **Streamlit (v1.65+)**: Modern reactive web application framework and state management.

### Machine Learning & NLP Stack
* **NLTK (v3.10+)**: Sentence and word tokenization (`punkt`, `punkt_tab`), English stop-word filtering (`stopwords`).
* **Scikit-Learn (v1.7+)**:
  * `TfidfVectorizer`: Sublinear term-frequency scaling, n-gram extraction `(1, 2)`.
  * `MultinomialNB`: Naive Bayes classifier with Laplace smoothing ($\alpha = 0.5$).
  * Model evaluation: `train_test_split`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, and `confusion_matrix`.
* **TextBlob (v0.20+)**: Lexicon-based sentiment analysis, polarity, and subjectivity calculation.
* **langdetect (v1.0.9+)**: Character tri-gram probabilistic language identification.

### Data Manipulation & Styling
* **Pandas & NumPy**: Tabular data manipulation, metrics formatting, and matrix transformations.
* **Vanilla CSS (Glassmorphism & Obsidian Dark System)**: Custom CSS injected into Streamlit to deliver dark theme styling, segmented tab navigation, monospace telemetry tiles, and glowing borders.

---

## 🏗️ 4. System Architecture & Methodology

The codebase is organized modularly to decouple business logic from Streamlit rendering:

```
NLPproject/
├── .streamlit/
│   └── config.toml           # Theme configuration (Obsidian base, cyan accents)
├── data/
│   └── spam_dataset.csv      # Balanced SMS/email sample dataset (60 samples)
├── modules/                  # Reusable Backend Engines (Business Logic)
│   ├── __init__.py
│   ├── nlp_setup.py          # Automatic, silent NLTK corpus dependency downloads
│   ├── sentiment.py          # TextBlob sentiment & sentence parsing logic
│   ├── spam_classifier.py    # TF-IDF + Naive Bayes pipeline & metric calculations
│   ├── summarizer.py         # Frequency-weighted extractive summarization algorithm
│   ├── lang_detector.py      # Resilient language detector & ISO metadata dictionary
│   └── keyword_extractor.py  # TF-IDF keyword & bigram ranking engine
├── views/                    # Frontend UI Renderers (Presentation Layer)
│   ├── __init__.py
│   ├── home.py               # Landing hero banner & algorithm catalog
│   ├── sentiment_view.py     # Sentiment tool tabs & visualization
│   ├── spam_view.py          # Spam detector, telemetry & confusion matrix
│   ├── summarizer_view.py    # Summarizer slider & sentence scoring table
│   ├── language_view.py      # Multilingual presets & candidate distributions
│   ├── keyword_view.py       # Keyphrase extraction tags & feature table
│   └── about_nlp.py          # Classical vs Transformer educational guide
├── styles/
│   └── custom.css            # Dark mode UI styles, badges & custom tabs
├── app.py                    # Main controller, sidebar router & state management
├── requirements.txt          # Reproducible dependencies list
└── README.md                 # Full project documentation & terminal setup guide
```

### Key Engineering Practices:
1. **Graceful Dependency Management:** `modules/nlp_setup.py` automatically checks for and silently downloads required NLTK corpora (`punkt`, `stopwords`) without crashing.
2. **Defensive Input Validation:** All tools validate empty inputs, non-alphanumeric text, and micro-inputs ($< 3$ characters), delivering friendly user guidance.
3. **Deterministic Modeling:** Fixed random seeds (`random_state=42` and `DetectorFactory.seed=0`) guarantee reproducible train/test splits and consistent inferences.
4. **Decoupled Architecture:** Algorithm modules are pure Python and can be imported directly into CLI scripts, FastAPI backends, or Jupyter Notebooks.

---

## 🚀 5. Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.9+ (tested on Python 3.10)
- `pip` package manager

### 2. Navigate to Project Directory
```powershell
cd "c:\Users\Slim 3 5EIN\OneDrive\Desktop\NLPproject"
```

### 3. (Optional) Create & Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 5. Launch the Application
```powershell
streamlit run app.py
```

The application will launch and open your default browser at:
👉 **`http://localhost:8501`**

---

## 📜 6. License & Academic Attribution
MIT License. Open for educational, academic, and research use.
