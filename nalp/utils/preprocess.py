"""Text pre-processing utilities."""

import re


def tokenize(text: str, corpus_type: str) -> list[str]:
    """Lowercase, filter, and tokenize text as characters or words."""

    if corpus_type not in {"char", "word"}:
        raise RuntimeError("Corpus type should be `char` or `word`.")

    text = re.sub(r"[^A-Za-z0-9\s]", "", text.lower())
    return list(text) if corpus_type == "char" else text.split()
