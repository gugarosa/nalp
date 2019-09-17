import nalp.utils.preprocess as p
from nalp.corpus.text import TextCorpus
from nalp.encoders.onehot import OnehotEncoder

# Creating a character TextCorpus from file
corpus = TextCorpus(from_file='data/text/chapter1_harry.txt', type='char')

# Creating an OnehotEncoder
encoder = OnehotEncoder()

# Learns the encoding based on the TextCorpus dictionary, reverse dictionary and vocabulary size
encoder.learn(corpus.vocab_index, corpus.index_vocab, corpus.vocab_size)

# Applies the encoding on new data
encoded_tokens = encoder.encode(corpus.tokens)

# Printing encoded tokens
print(encoded_tokens)

# Decodes the encoded tokens
decoded_tokens = encoder.decode(encoded_tokens)

# Printing decoded tokens
print(decoded_tokens)
