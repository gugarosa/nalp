from pathlib import Path

import nalp.utils.preprocess as p

# Loads an input .txt file with sentences
sentences = (
    Path("data/sentence/coco_image_captions.txt")
    .read_text(encoding="utf-8")
    .splitlines()
)

# Pre-processing sentences into character and word tokens
chars_tokens = [p.tokenize(sentence, "char") for sentence in sentences]
words_tokens = [p.tokenize(sentence, "word") for sentence in sentences]

# Printing tokenized characters and words
print(chars_tokens)
print(words_tokens)
