"""
Language Detection Module.
Uses langdetect with ISO-639 language code mapping, probability distribution,
and resilient error handling for short or ambiguous inputs.
"""

from typing import Dict, Any, List
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Enforce deterministic results
DetectorFactory.seed = 0

# Comprehensive ISO 639-1 language dictionary with flag emojis
LANGUAGE_METADATA: Dict[str, Dict[str, str]] = {
    "en": {"name": "English", "native": "English", "flag": "🇬🇧", "family": "Indo-European (Germanic)"},
    "es": {"name": "Spanish", "native": "Español", "flag": "🇪🇸", "family": "Indo-European (Romance)"},
    "fr": {"name": "French", "native": "Français", "flag": "🇫🇷", "family": "Indo-European (Romance)"},
    "de": {"name": "German", "native": "Deutsch", "flag": "🇩🇪", "family": "Indo-European (Germanic)"},
    "it": {"name": "Italian", "native": "Italiano", "flag": "🇮🇹", "family": "Indo-European (Romance)"},
    "pt": {"name": "Portuguese", "native": "Português", "flag": "🇵🇹", "family": "Indo-European (Romance)"},
    "nl": {"name": "Dutch", "native": "Nederlands", "flag": "🇳🇱", "family": "Indo-European (Germanic)"},
    "ru": {"name": "Russian", "native": "Русский", "flag": "🇷🇺", "family": "Indo-European (Slavic)"},
    "zh-cn": {"name": "Chinese (Simplified)", "native": "中文 (简体)", "flag": "🇨🇳", "family": "Sino-Tibetan"},
    "zh-tw": {"name": "Chinese (Traditional)", "native": "中文 (繁體)", "flag": "🇹🇼", "family": "Sino-Tibetan"},
    "ja": {"name": "Japanese", "native": "日本語", "flag": "🇯🇵", "family": "Japonic"},
    "ko": {"name": "Korean", "native": "한국어", "flag": "🇰🇷", "family": "Koreanic"},
    "ar": {"name": "Arabic", "native": "العربية", "flag": "🇸🇦", "family": "Afroasiatic (Semitic)"},
    "hi": {"name": "Hindi", "native": "हिन्दी", "flag": "🇮🇳", "family": "Indo-European (Indo-Aryan)"},
    "bn": {"name": "Bengali", "native": "বাংলা", "flag": "🇧🇩", "family": "Indo-European (Indo-Aryan)"},
    "tr": {"name": "Turkish", "native": "Türkçe", "flag": "🇹🇷", "family": "Turkic"},
    "vi": {"name": "Vietnamese", "native": "Tiếng Việt", "flag": "🇻🇳", "family": "Austroasiatic"},
    "pl": {"name": "Polish", "native": "Polski", "flag": "🇵🇱", "family": "Indo-European (Slavic)"},
    "sv": {"name": "Swedish", "native": "Svenska", "flag": "🇸🇪", "family": "Indo-European (Germanic)"},
    "da": {"name": "Danish", "native": "Dansk", "flag": "🇩🇰", "family": "Indo-European (Germanic)"},
    "fi": {"name": "Finnish", "native": "Suomi", "flag": "🇫🇮", "family": "Uralic"},
    "el": {"name": "Greek", "native": "Ελληνικά", "flag": "🇬🇷", "family": "Indo-European (Hellenic)"},
    "he": {"name": "Hebrew", "native": "עברית", "flag": "🇮🇱", "family": "Afroasiatic (Semitic)"},
    "id": {"name": "Indonesian", "native": "Bahasa Indonesia", "flag": "🇮🇩", "family": "Austronesian"},
    "uk": {"name": "Ukrainian", "native": "Українська", "flag": "🇺🇦", "family": "Indo-European (Slavic)"},
    "cs": {"name": "Czech", "native": "Čeština", "flag": "🇨🇿", "family": "Indo-European (Slavic)"},
    "ro": {"name": "Romanian", "native": "Română", "flag": "🇷🇴", "family": "Indo-European (Romance)"},
    "hu": {"name": "Hungarian", "native": "Magyar", "flag": "🇭🇺", "family": "Uralic"},
    "th": {"name": "Thai", "native": "ไทย", "flag": "🇹🇭", "family": "Kra-Dai"},
    "fa": {"name": "Persian", "native": "فارسی", "flag": "🇮🇷", "family": "Indo-European (Iranian)"},
    "sw": {"name": "Swahili", "native": "Kiswahili", "flag": "🇰🇪", "family": "Niger-Congo (Bantu)"},
    "no": {"name": "Norwegian", "native": "Norsk", "flag": "🇳🇴", "family": "Indo-European (Germanic)"},
}

def detect_language(text: str) -> Dict[str, Any]:
    """
    Detects language using n-gram frequency models from langdetect.
    Returns primary language, confidence, and candidate alternatives.
    """
    cleaned = text.strip()
    if not cleaned:
        return {"error": "Input text cannot be empty."}

    # Pre-validation for very short strings or purely numeric/punct
    alphanumeric_chars = [c for c in cleaned if c.isalnum()]
    if len(alphanumeric_chars) < 3:
        return {
            "error": "Input is too short for reliable detection. Please enter at least 3-4 letters or a complete word/phrase."
        }

    try:
        results = detect_langs(cleaned)
    except LangDetectException:
        return {
            "error": "Could not identify a language. Text might contain only symbols, punctuation, or numbers."
        }
    except Exception as e:
        return {
            "error": f"An unexpected error occurred during language detection: {str(e)}"
        }

    if not results:
        return {"error": "No confident language matches found."}

    primary = results[0]
    primary_code = primary.lang.lower()
    primary_prob = round(primary.prob, 4)

    meta = LANGUAGE_METADATA.get(primary_code, {
        "name": primary_code.upper(),
        "native": primary_code.upper(),
        "flag": "🌐",
        "family": "Unknown / Other"
    })

    candidates: List[Dict[str, Any]] = []
    for r in results:
        code = r.lang.lower()
        cand_meta = LANGUAGE_METADATA.get(code, {
            "name": code.upper(),
            "native": code.upper(),
            "flag": "🌐",
            "family": "Unknown / Other"
        })
        candidates.append({
            "code": code,
            "name": cand_meta["name"],
            "native": cand_meta["native"],
            "flag": cand_meta["flag"],
            "family": cand_meta["family"],
            "probability": round(r.prob * 100, 2)
        })

    is_reliable = (primary_prob >= 0.70)

    return {
        "code": primary_code,
        "name": meta["name"],
        "native": meta["native"],
        "flag": meta["flag"],
        "family": meta["family"],
        "confidence_pct": round(primary_prob * 100, 1),
        "is_reliable": is_reliable,
        "candidates": candidates,
        "char_count": len(cleaned),
        "word_count": len(cleaned.split()),
    }
