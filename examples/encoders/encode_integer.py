import nalp.utils.preprocess as p
from nalp.core.corpus import Corpus
from nalp.encoders.integer import IntegerEncoder

# Creating a character Corpus from file
corpus = Corpus(from_file='data/text/chapter1_harry.txt', type='char')

# Creating an IntegerEncoder
encoder = IntegerEncoder()

# Learns the encoding based on the Corpus dictionary
encoder.learn(corpus.vocab_index)

# Applies the encoding on new data
encoded_tokens = encoder.encode(corpus.tokens)

# Printing encoded tokens
print(encoded_tokens)