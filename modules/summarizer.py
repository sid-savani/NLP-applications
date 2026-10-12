"""
Text Summarization Module.
Implements Extractive Summarization via NLTK sentence tokenization and normalized word-frequency sentence scoring.
Extracts the highest-scoring sentences while preserving natural reading order.
"""

import re
from typing import Dict, Any, List
import nltk
from nltk.corpus import stopwords
from modules.nlp_setup import ensure_nltk_resources

ensure_nltk_resources()


def summarize_text(text: str, num_sentences: int = 3, min_sentence_len: int = 4) -> Dict[str, Any]:
    """
    Extractive summarization using frequency-weighted sentence ranking.

    Steps:
    1. Sentence Tokenization (nltk.sent_tokenize)
    2. Word Tokenization & Stop Words Removal
    3. Word Frequency Scoring & Max Normalization
    4. Sentence Scoring (sum of normalized word scores / sentence length factor)
    5. Top-K Sentence Selection preserving chronological order
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return {"error": "Input text cannot be empty."}

    # Sentence Tokenization
    try:
        raw_sentences = nltk.sent_tokenize(cleaned_text)
    except Exception:
        # Fallback regex splitter if tokenizer is unavailable
        raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', cleaned_text) if s.strip()]

    if not raw_sentences:
        return {"error": "Could not identify any valid sentences in input."}

    total_sentences = len(raw_sentences)
    if total_sentences <= 1:
        return {
            "summary": cleaned_text,
            "original_sentence_count": 1,
            "summary_sentence_count": 1,
            "original_word_count": len(cleaned_text.split()),
            "summary_word_count": len(cleaned_text.split()),
            "compression_ratio": 0.0,
            "sentence_scores": [{"sentence": cleaned_text, "score": 1.0, "selected": True}],
            "word_frequencies": {}
        }

    # Stop words setup
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
            "is", "was", "are", "were", "it", "this", "that", "by", "from", "as", "be"
        }

    # Word Frequency Calculation
    word_freq: Dict[str, int] = {}
    words = re.findall(r'\b[a-zA-Z]{2,}\b', cleaned_text.lower())
    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1

    if not word_freq:
        # If text consists only of stopwords or non-alphabetic words
        selected = raw_sentences[:min(num_sentences, total_sentences)]
        return {
            "summary": " ".join(selected),
            "original_sentence_count": total_sentences,
            "summary_sentence_count": len(selected),
            "original_word_count": len(cleaned_text.split()),
            "summary_word_count": len(" ".join(selected).split()),
            "compression_ratio": round((1 - len(selected) / total_sentences) * 100, 1),
            "sentence_scores": [],
            "word_frequencies": {}
        }

    # Normalize frequencies by maximum frequency
    max_freq = max(word_freq.values())
    normalized_freq = {w: count / max_freq for w, count in word_freq.items()}

    # Score each sentence
    sentence_scores: List[Dict[str, Any]] = []
    for idx, sent in enumerate(raw_sentences):
        sent_words = re.findall(r'\b[a-zA-Z]{2,}\b', sent.lower())
        meaningful_words = [w for w in sent_words if w in normalized_freq]
        
        # Raw score: sum of normalized frequencies of words in sentence
        raw_score = sum(normalized_freq[w] for w in meaningful_words)
        
        # Length normalization factor (avoids over-penalizing or over-rewarding huge sentences)
        length_factor = max(len(sent_words), 1) ** 0.5
        score = raw_score / length_factor if length_factor > 0 else 0.0

        sentence_scores.append({
            "index": idx,
            "sentence": sent,
            "score": round(score, 4),
            "word_count": len(sent_words),
            "selected": False
        })

    # Pick top K sentences based on score
    k = max(1, min(num_sentences, total_sentences))
    sorted_by_score = sorted(sentence_scores, key=lambda x: x["score"], reverse=True)
    top_k_indices = set(item["index"] for item in sorted_by_score[:k])

    for item in sentence_scores:
        if item["index"] in top_k_indices:
            item["selected"] = True

    # Assemble summary in original chronological order
    summary_sentences = [sent for sent in raw_sentences if sentence_scores[raw_sentences.index(sent)]["selected"]]
    summary_text = " ".join(summary_sentences)

    original_word_count = len(cleaned_text.split())
    summary_word_count = len(summary_text.split())
    compression_ratio = round((1 - (summary_word_count / max(original_word_count, 1))) * 100, 1)

    # Top words contributing to the summary
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "summary": summary_text,
        "original_sentence_count": total_sentences,
        "summary_sentence_count": len(summary_sentences),
        "original_word_count": original_word_count,
        "summary_word_count": summary_word_count,
        "compression_ratio": max(0.0, compression_ratio),
        "sentence_scores": sentence_scores,
        "top_words": top_words,
    }
