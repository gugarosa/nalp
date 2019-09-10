import nalp.utils.preprocess as p
from nalp.corpus.text import TextCorpus

# Creating a character TextCorpus from file
corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', type='char')

# Creating a word TextCorpus from file
# corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', type='word')

# Accessing TextCorpus properties
print(corpus.tokens)
print(corpus.vocab, corpus.vocab_size)
print(corpus.vocab_index, corpus.index_vocab)
