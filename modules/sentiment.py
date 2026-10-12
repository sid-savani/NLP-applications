"""
Sentiment Analysis Module.
Uses TextBlob to analyze text polarity (positivity/negativity) and subjectivity (fact vs. opinion).
Includes sentence-level breakdown and polarity classifications.
"""

from typing import Dict, Any, List
from textblob import TextBlob
from modules.nlp_setup import ensure_nltk_resources

ensure_nltk_resources()


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyzes polarity and subjectivity of the given text using TextBlob.

    Returns a structured dictionary with:
    - polarity: float in [-1.0, 1.0]
    - subjectivity: float in [0.0, 1.0]
    - sentiment_label: str
    - sentiment_color: str
    - subjectivity_label: str
    - sentence_breakdown: list of dicts with per-sentence metrics
    - word_count: int
    - character_count: int
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return {
            "error": "Input text cannot be empty."
        }

    blob = TextBlob(cleaned_text)
    polarity = round(blob.sentiment.polarity, 4)
    subjectivity = round(blob.sentiment.subjectivity, 4)

    # Determine classification label and theme colors
    if polarity >= 0.4:
        sentiment_label = "Very Positive"
        sentiment_color = "#10b981"  # Emerald green
        sentiment_emoji = "😄"
    elif polarity >= 0.1:
        sentiment_label = "Positive"
        sentiment_color = "#34d399"  # Mint
        sentiment_emoji = "🙂"
    elif polarity <= -0.4:
        sentiment_label = "Very Negative"
        sentiment_color = "#ef4444"  # Red
        sentiment_emoji = "😡"
    elif polarity <= -0.1:
        sentiment_label = "Negative"
        sentiment_color = "#f87171"  # Coral red
        sentiment_emoji = "🙁"
    else:
        sentiment_label = "Neutral"
        sentiment_color = "#94a3b8"  # Slate grey
        sentiment_emoji = "😐"

    # Determine subjectivity category
    if subjectivity < 0.25:
        subjectivity_label = "Objective / Fact-Oriented"
        subjectivity_desc = "Mostly stated as objective facts, reporting, or neutral descriptions."
    elif subjectivity < 0.6:
        subjectivity_label = "Mixed / Moderate"
        subjectivity_desc = "Contains a balance of factual information and personal opinion."
    else:
        subjectivity_label = "Subjective / Opinion-Heavy"
        subjectivity_desc = "Heavily expresses personal sentiments, emotional evaluations, or viewpoints."

    # Sentence-level breakdown
    sentence_breakdown: List[Dict[str, Any]] = []
    for idx, sent in enumerate(blob.sentences, start=1):
        s_pol = round(sent.sentiment.polarity, 3)
        s_sub = round(sent.sentiment.subjectivity, 3)
        if s_pol > 0.1:
            s_label = "Positive"
        elif s_pol < -0.1:
            s_label = "Negative"
        else:
            s_label = "Neutral"

        sentence_breakdown.append({
            "index": idx,
            "text": str(sent),
            "polarity": s_pol,
            "subjectivity": s_sub,
            "label": s_label
        })

    words = blob.words
    word_count = len(words)
    char_count = len(cleaned_text)

    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "sentiment_label": sentiment_label,
        "sentiment_color": sentiment_color,
        "sentiment_emoji": sentiment_emoji,
        "subjectivity_label": subjectivity_label,
        "subjectivity_desc": subjectivity_desc,
        "sentence_breakdown": sentence_breakdown,
        "sentence_count": len(sentence_breakdown),
        "word_count": word_count,
        "character_count": char_count,
    }
