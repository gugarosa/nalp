from nalp.corpus.document import DocumentCorpus
from nalp.encoders.word2vec import Word2vecEncoder

# Creating a DocumentCorpus from file
corpus = DocumentCorpus(from_file='data/document/chapter1_harry.txt')

# Creating an Word2vecEncoder
encoder = Word2vecEncoder()

# Learns the encoding based on the DocumentCorpus tokens
encoder.learn(corpus.tokens)

# Applies the encoding on same or new data
encoded_tokens = encoder.encode(corpus.tokens)

# Printing encoded tokens
print(encoded_tokens)