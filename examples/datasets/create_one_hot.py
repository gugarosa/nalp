import nalp.stream.preprocess as p
import numpy as np
from nalp.datasets.one_hot import OneHot

# Defines your own input text
input_text = "I have two hippos and three cats"

# Creates a pre-processing pipeline
# This will tokenize the input text into chars
pipe_char = p.pipeline(
    p.tokenize_to_char
)

# And this will tokenize into words
pipe_word = p.pipeline(
    p.tokenize_to_word
)

# Applying pre-processing pipelines to input text
chars = pipe_char(input_text)
words = pipe_word(input_text)

# Creates the dataset (c will be for chars and w for words)
c = OneHot(chars, max_length=3)
w = OneHot(words, max_length=2)

# Acessing properties from OneHot class
# Note that every property can be acessed, please refer to the docs to know all of them
print(f'Char -> Tokens: {c.tokens} | Vocabulary size: {c.vocab_size} | X[0]:  {c.X[0]} | Y[0]: {c.Y[0]}')
print(f'Word -> Tokens: {w.tokens} | Vocabulary size: {w.vocab_size} | X[0]: {w.X[0]} | Y[0]: {w.Y[0]}')
