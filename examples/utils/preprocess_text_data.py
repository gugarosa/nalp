from pathlib import Path

import nalp.utils.preprocess as p

# Loads an input .txt file
text = Path("data/text/chapter1_harry.txt").read_text(encoding="utf-8")

# Pre-processing text into character and word tokens
chars_tokens = p.tokenize(text, "char")
words_tokens = p.tokenize(text, "word")

# Printing tokenized characters and words
print(chars_tokens)
print(words_tokens)
