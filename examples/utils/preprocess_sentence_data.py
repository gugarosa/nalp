# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from pathlib import Path

import nalp.utils.preprocess as p

sentences = Path("data/sentence/coco_image_captions.txt").read_text(encoding="utf-8").splitlines()

chars_tokens = [p.tokenize(sentence, "char") for sentence in sentences]
words_tokens = [p.tokenize(sentence, "word") for sentence in sentences]

print(chars_tokens)
print(words_tokens)
