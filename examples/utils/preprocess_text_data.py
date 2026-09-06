# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from pathlib import Path

import nalp.utils.preprocess as p

text = Path("data/text/chapter1_harry.txt").read_text(encoding="utf-8")

chars_tokens = p.tokenize(text, "char")
words_tokens = p.tokenize(text, "word")

print(chars_tokens)
print(words_tokens)
