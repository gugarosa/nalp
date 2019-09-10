import nalp.utils.preprocess as p
from nalp.corpus.document import DocumentCorpus
from nalp.encoders.count import CountEncoder

# Creating a DocumentCorpus from file
corpus = DocumentCorpus(from_file='data/document/chapter1_harry.txt')

# Creating an CountEncoder
encoder = CountEncoder()

# Learns the encoding based on the DocumentCorpus tokens
encoder.learn(corpus.tokens, top_tokens=100)

# Applies the encoding on same or new data
encoded_tokens = encoder.encode(corpus.tokens)

# Printing encoded tokens
print(encoded_tokens)

# Decoding the encoded tokens
decoded_tokens = encoder.decode(encoded_tokens)

# Printing decoded tokens
print(decoded_tokens)