import re
from typing import List, Tuple, Dict, Any

STOP_WORDS = {
    "what", "is", "are", "the", "a", "an", "of", "in", "for", "to", "and", "on", "with",
    "how", "can", "i", "do", "should", "my", "why", "does", "during", "about", "be", "me",
    "you", "your", "it", "at", "from", "by", "that", "this", "or", "as", "into"
}


def normalize_query(q: str) -> str:
    q = " ".join(q.lower().strip().split())
    words = [w for w in q.split() if w not in STOP_WORDS]
    return " ".join(words)


def stem_word(w: str) -> str:
    """Lightweight rule-based suffix stemming without external dependencies."""
    for suf in ["ments", "ment", "ing", "ies", "es", "ed", "s"]:
        if len(w) > len(suf) + 3 and w.endswith(suf):
            if suf == "ies":
                return w[:-3] + "y"
            return w[:-len(suf)]
    return w


def tokenize(text: str) -> List[str]:
    """Tokenize and filter stop words."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in clean.split() if w and w not in STOP_WORDS]


def extract_phrases(text: str, max_n: int = 3) -> List[str]:
    """Extract n-grams (bigrams and trigrams) from text."""
    words = tokenize(text)
    phrases = []
    for n in range(2, max_n + 1):
        for i in range(len(words) - n + 1):
            phrases.append(" ".join(words[i:i + n]))
    return phrases


def extract_temporal_entities(text: str) -> List[str]:
    """Extract temporal markers, ages, trimesters, and life stages."""
    patterns = [
        r"\b\d+\s*(?:month|months|week|weeks|day|days|year|years|trimester|trimesters|kg|grams?|hrs?|hours?)\b",
        r"\b(?:first|second|third|1st|2nd|3rd)\s+trimester\b",
        r"\b(?:newborn|infant|toddler|neonate)\b",
    ]
    matches = []
    lowered = text.lower()
    for pat in patterns:
        for m in re.finditer(pat, lowered):
            matches.append(m.group(0).strip())
    return matches


def rescore_candidate(
    query: str,
    query_terms: List[str],
    query_stems: List[str],
    query_phrases: List[str],
    query_temporals: List[str],
    doc_text: str,
    dense_score: float,
    max_boost: float = 0.04
) -> float:
    """
    Computes a dense-anchored hybrid score. The dense score is the primary ranking signal,
    while lexical, phrase, and temporal features provide a bounded positive boost (max 0.04)
    to break ties and promote well-aligned candidates without demoting strong dense matches.
    """
    doc_lower = doc_text.lower()
    doc_tokens = set(tokenize(doc_text))
    doc_stems = set(stem_word(t) for t in doc_tokens)

    # 1. Lexical Term Overlap & Saturation
    if not query_terms:
        term_score = 0.0
    else:
        matched = 0
        freq = 0
        for t, st in zip(query_terms, query_stems):
            if (
                t in doc_tokens
                or st in doc_stems
                or re.search(rf"\b{re.escape(t)}", doc_lower)
                or re.search(rf"\b{re.escape(st)}", doc_lower)
            ):
                matched += 1
                cnt = len(re.findall(rf"\b{re.escape(t)}", doc_lower)) or len(re.findall(rf"\b{re.escape(st)}", doc_lower))
                freq += min(cnt, 3)
        recall = matched / len(query_terms)
        saturation = freq / (len(query_terms) * 3)
        term_score = 0.65 * recall + 0.35 * saturation

    # 2. Phrase Match Score
    phrase_score = 0.0
    if query_phrases:
        matched_phrases = sum(1 for p in query_phrases if p in doc_lower)
        phrase_score = matched_phrases / len(query_phrases)

    # 3. Temporal / Age Specificity Score
    temporal_score = 0.0
    if query_temporals:
        matched_temp = 0
        for temp in query_temporals:
            clean_temp = temp.rstrip("s")
            if temp in doc_lower or clean_temp in doc_lower:
                matched_temp += 1
        temporal_score = matched_temp / len(query_temporals)

    # 4. Composite Bounded Boost
    if query_temporals:
        lexical_combo = 0.60 * term_score + 0.25 * phrase_score + 0.15 * temporal_score
    else:
        lexical_combo = 0.70 * term_score + 0.30 * phrase_score

    boost = min(0.08 * lexical_combo, max_boost)
    return dense_score + boost


def rerank_candidates(
    query: str,
    candidates: List[Tuple[str, Dict[str, Any], float]],
    top_n: int = 5,
    max_boost: float = 0.04
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Reranks candidate passages from dense retrieval using pure-Python hybrid scoring.
    candidates: List of (text_content, metadata, dense_similarity_score)
    Returns: List of top_n (text_content, metadata)
    """
    if not candidates:
        return []

    q_terms = tokenize(query)
    q_stems = [stem_word(t) for t in q_terms]
    q_phrases = extract_phrases(query)
    q_temporals = extract_temporal_entities(query)

    scored = []
    for text, meta, dense_score in candidates:
        final_score = rescore_candidate(
            query=query,
            query_terms=q_terms,
            query_stems=q_stems,
            query_phrases=q_phrases,
            query_temporals=q_temporals,
            doc_text=text,
            dense_score=dense_score,
            max_boost=max_boost
        )
        scored.append((text, meta, final_score))

    # Sort descending by composite score
    scored.sort(key=lambda x: x[2], reverse=True)

    return [(item[0], item[1]) for item in scored[:top_n]]
