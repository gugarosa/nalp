import nalp.utils.preprocess as p
from nalp.core.corpus import Corpus

# Creating a character Corpus from file
corpus = Corpus(from_file='data/text/chapter1_harry.txt', type='char')

# Creating a word Corpus from file
# corpus = Corpus(from_file='data/text/chapter1_harry.txt', type='word')

# Accessing Corpus properties
print(corpus.tokens)
print(corpus.vocab, corpus.vocab_size)
print(corpus.vocab_index, corpus.index_vocab)
