from nalp.corpus import AudioCorpus

# Creating an AudioCorpus from file
corpus = AudioCorpus(from_file='data/audio/sample.mid', min_frequency=1)

# Accessing AudioCorpus properties
print(corpus.tokens)
print(corpus.vocab, corpus.vocab_size)
print(corpus.vocab_index, corpus.index_vocab)
