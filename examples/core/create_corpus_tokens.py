import nalp.utils.loader as l
import nalp.utils.preprocess as p
from nalp.core.corpus import Corpus

# Loads an input .txt file
text = l.load_txt('data/text/chapter1_harry.txt')

# Creates a character pre-processing pipeline
pipe = p.pipeline(p.lower_case, p.valid_char, p.tokenize_to_char)

# Creates a word pre-processing pipeline
# pipe = p.pipeline(p.lower_case, p.valid_char, p.tokenize_to_word)

# Applying character pre-processing pipeline to text
tokens = pipe(text)

# Applying word pre-processing pipeline to text
# tokens = pipe(text)

# Creating a character Corpus from tokens
corpus = Corpus(tokens=tokens)

# Creating a word Corpus from tokens
# corpus = Corpus(tokens=words)

# Accessing Corpus properties
print(corpus.tokens)
print(corpus.vocab, corpus.vocab_size)
print(corpus.vocab_index, corpus.index_vocab)
