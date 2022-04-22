from nalp.corpus import TextCorpus
from nalp.encoders import IntegerEncoder

# Creating a character TextCorpus from file
corpus = TextCorpus(from_file="data/text/chapter1_harry.txt", corpus_type="char")

# Creating an IntegerEncoder and learning encoding
encoder = IntegerEncoder()
encoder.learn(corpus.vocab_index, corpus.index_vocab)

# Applies the encoding on new data
encoded_tokens = encoder.encode(corpus.tokens)
print(encoded_tokens)

# Decodes the encoded tokens
decoded_tokens = encoder.decode(encoded_tokens)
print(decoded_tokens)
