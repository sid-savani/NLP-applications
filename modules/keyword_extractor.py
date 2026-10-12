"""
Keyword & Keyphrase Extraction Module.
Extracts salient keywords and 2-gram phrases using TF-IDF feature weighting and stop-word filtering.
Allows configuring top-k keywords, minimum word length, and n-gram type.
"""

import re
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from modules.nlp_setup import ensure_nltk_resources

ensure_nltk_resources()


def extract_keywords(
    text: str,
    top_n: int = 8,
    include_phrases: bool = True,
    min_df: int = 1
) -> Dict[str, Any]:
    """
    Extracts top keywords/keyphrases using TF-IDF scoring over sentence documents.
    Treats each sentence as a mini-document so that TF-IDF reflects salience across the text.
    """
    cleaned = text.strip()
    if not cleaned:
        return {"error": "Input text cannot be empty."}

    # Sentence splitting for pseudo-document corpus
    try:
        sentences = nltk.sent_tokenize(cleaned)
    except Exception:
        sentences = [s.strip() for s in re.split(r'[.!?]+', cleaned) if s.strip()]

    # If only one sentence, split into clauses to allow TF-IDF document frequency
    if len(sentences) <= 1:
        clauses = [c.strip() for c in re.split(r'[,;—\n]+', cleaned) if len(c.strip()) > 3]
        docs = clauses if len(clauses) >= 2 else [cleaned]
    else:
        docs = sentences

    try:
        stop_list = list(stopwords.words("english"))
    except Exception:
        stop_list = "english"

    ngram_range = (1, 2) if include_phrases else (1, 1)

    try:
        vectorizer = TfidfVectorizer(
            stop_words=stop_list,
            ngram_range=ngram_range,
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b',
            sublinear_tf=True,
            min_df=min_df
        )
        tfidf_matrix = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError as e:
        # Handles cases like empty vocabulary after stop words
        return {
            "error": "Could not extract keywords. Text might contain only common stop words or special symbols."
        }

    # Aggregate TF-IDF scores across documents (mean of non-zero entries or max)
    # Sum over documents gives corpus-wide importance
    scores = tfidf_matrix.sum(axis=0).A1
    word_scores = list(zip(feature_names, scores))

    # Sort descending by score
    sorted_keywords = sorted(word_scores, key=lambda x: x[1], reverse=True)

    max_score = sorted_keywords[0][1] if sorted_keywords else 1.0

    results: List[Dict[str, Any]] = []
    for rank, (kw, score) in enumerate(sorted_keywords[:top_n], start=1):
        # Calculate raw occurrences in full text
        pattern = r'\b' + re.escape(kw) + r'\b'
        count = len(re.findall(pattern, cleaned, flags=re.IGNORECASE))
        is_phrase = " " in kw

        normalized_score = round((score / max_score) * 100, 1)

        results.append({
            "rank": rank,
            "keyword": kw,
            "score": round(float(score), 4),
            "relevance_pct": normalized_score,
            "frequency": max(count, 1),
            "type": "Keyphrase" if is_phrase else "Keyword"
        })

    return {
        "keywords": results,
        "total_extracted": len(results),
        "vocabulary_size": len(feature_names),
        "total_words": len(cleaned.split()),
    }
