from nalp.corpus import TextCorpus
from nalp.encoders import Word2vecEncoder

# Creating a TextCorpus from file
corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', type='word')

# Creating an Word2vecEncoder
encoder = Word2vecEncoder()

# Learns the encoding based on the DocumentCorpus tokens
encoder.learn(corpus.tokens)

# Applies the encoding on same or new data
encoded_tokens = encoder.encode(corpus.tokens)

# Printing encoded tokens
print(encoded_tokens)

# Decodes the encoded tokens
decoded_tokens = encoder.decode(encoded_tokens)

# Printing decoded tokens
print(decoded_tokens)
