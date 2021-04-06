from nalp.corpus import TextCorpus

# Creating a character TextCorpus from file
corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', corpus_type='char', min_frequency=1)

# Creating a word TextCorpus from file
# corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', corpus_type='word', min_frequency=1)

# Accessing TextCorpus properties
print(corpus.tokens)
print(corpus.vocab, corpus.vocab_size)
print(corpus.vocab_index, corpus.index_vocab)
