"""
Spam Message Detection Module.
Trains a Multinomial Naive Bayes classifier on TF-IDF word vectors using scikit-learn.
Uses an authentic train/test split, calculates real metrics, and highlights top informative tokens.
"""

import os
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "spam_dataset.csv")

class SpamClassifierModel:
    """
    Encapsulates dataset loading, TF-IDF feature extraction, model fitting,
    authentic evaluation, and real-time inference.
    """
    def __init__(self, data_path: str = DATASET_PATH, test_size: float = 0.25, random_state: int = 42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )
        self.classifier = MultinomialNB(alpha=0.5)
        self.metrics: Dict[str, Any] = {}
        self.is_trained = False
        self.df: pd.DataFrame = pd.DataFrame()
        self._train()

    def _train(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Spam dataset not found at {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        X = self.df["message"]
        y = self.df["label"]  # 'ham' or 'spam'

        # Stratified train/test split to preserve label distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_vec, y_train)

        # Honest evaluation on unseen test split
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.classifier.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label="spam", zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label="spam", zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label="spam", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])

        # Top predictive words
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        # MultinomialNB log prob: class 0 (ham), class 1 (spam)
        classes = list(self.classifier.classes_)
        spam_idx = classes.index("spam")
        ham_idx = classes.index("ham")

        spam_log_probs = self.classifier.feature_log_prob_[spam_idx]
        ham_log_probs = self.classifier.feature_log_prob_[ham_idx]
        
        # Informativeness ratio
        spam_ratio = spam_log_probs - ham_log_probs
        top_spam_indices = np.argsort(spam_ratio)[-10:][::-1]
        top_ham_indices = np.argsort(spam_ratio)[:10]

        top_spam_keywords = [(feature_names[i], round(float(spam_ratio[i]), 3)) for i in top_spam_indices]
        top_ham_keywords = [(feature_names[i], round(float(abs(spam_ratio[i])), 3)) for i in top_ham_indices]

        self.metrics = {
            "total_samples": len(self.df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
            "classes": classes,
            "top_spam_keywords": top_spam_keywords,
            "top_ham_keywords": top_ham_keywords,
            "vocabulary_size": len(feature_names),
        }
        self.is_trained = True

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classifies incoming user text as 'spam' or 'ham'.
        Returns probability distribution and active vocabulary tokens.
        """
        cleaned = text.strip()
        if not cleaned:
            return {"error": "Input text cannot be empty."}

        if not self.is_trained:
            return {"error": "Model is not trained."}

        vec = self.vectorizer.transform([cleaned])
        probabilities = self.classifier.predict_proba(vec)[0]
        prediction = self.classifier.predict(vec)[0]

        classes = list(self.classifier.classes_)
        spam_prob = round(float(probabilities[classes.index("spam")]), 4)
        ham_prob = round(float(probabilities[classes.index("ham")]), 4)

        # Detect which words in text matched the vocabulary
        words_in_text = cleaned.lower().split()
        active_tokens = []
        vocab = self.vectorizer.vocabulary_
        for w in set(words_in_text):
            if w in vocab:
                idx = vocab[w]
                spam_weight = self.classifier.feature_log_prob_[classes.index("spam")][idx]
                ham_weight = self.classifier.feature_log_prob_[classes.index("ham")][idx]
                tendency = "Spam indicator" if spam_weight > ham_weight else "Legitimate indicator"
                active_tokens.append({
                    "token": w,
                    "tendency": tendency,
                    "spam_log_prob": round(float(spam_weight), 2),
                    "ham_log_prob": round(float(ham_weight), 2)
                })

        return {
            "prediction": prediction,
            "is_spam": (prediction == "spam"),
            "spam_probability": spam_prob,
            "ham_probability": ham_prob,
            "active_tokens": active_tokens,
            "input_text": cleaned
        }


# Global singleton instance for efficient memory usage
_MODEL_INSTANCE = None

def get_spam_classifier() -> SpamClassifierModel:
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = SpamClassifierModel()
    return _MODEL_INSTANCE
