def normalize_query(q: str) -> str:
    q = " ".join(q.lower().strip().split())

    stop_words = {"what", "are", "the", "is", "of", "a", "an"}
    words = [w for w in q.split() if w not in stop_words]

    return " ".join(words)
