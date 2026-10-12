"""
NLP Environment Setup & Resource Management.
Safely checks and downloads required NLTK corpora and resources without crashing.
"""

import nltk

_DOWNLOADED_RESOURCES = set()

def ensure_nltk_resources():
    """
    Downloads required NLTK resources silently if they are not already installed.
    Handles punkt, punkt_tab (for newer NLTK versions), and stopwords.
    """
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    
    for resource_path, resource_name in resources:
        if resource_name in _DOWNLOADED_RESOURCES:
            continue
        try:
            nltk.data.find(resource_path)
            _DOWNLOADED_RESOURCES.add(resource_name)
        except (LookupError, AttributeError):
            try:
                nltk.download(resource_name, quiet=True)
                _DOWNLOADED_RESOURCES.add(resource_name)
            except Exception as e:
                # Fallback: if network fails or resource unavailable, continue gracefully
                print(f"[Warning] Could not download NLTK resource '{resource_name}': {e}")
