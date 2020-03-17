import nalp.utils.preprocess as p
from nalp.corpus.audio import AudioCorpus

# Creating an AudioCorpus from file
corpus = AudioCorpus(from_file='data/audio/sample.mid')

# Accessing AudioCorpus properties
print(corpus.tokens)
print(corpus.vocab, corpus.vocab_size)
print(corpus.vocab_index, corpus.index_vocab)
