import nalp.utils.preprocess as p
from nalp.core.corpus import Corpus
from nalp.encoders.count import CountEncoder

# Creating a character Corpus from file
corpus = Corpus(from_file='data/text/chapter1_harry.txt', type='char')

# Creating an CountEncoder
encoder = CountEncoder()

# Learns the encoding based on the Corpus tokens
encoder.learn(corpus.tokens, top_tokens=100)

# Applies the encoding on same or new data
encoded_tokens = encoder.encode(corpus.tokens)

# Printing encoded tokens
print(encoded_tokens)

# Decoding the encoded tokens
decoded_tokens = encoder.decode(encoded_tokens)

# Printing decoded tokens
print(decoded_tokens)