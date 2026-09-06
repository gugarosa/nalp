# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from nalp.corpus import TextCorpus
from nalp.encoders import IntegerEncoder

corpus = TextCorpus(from_file="data/text/chapter1_harry.txt", corpus_type="char")
encoder = IntegerEncoder()
encoder.learn(corpus.vocab_index, corpus.index_vocab)

encoded_tokens = encoder.encode(corpus.tokens)
print(encoded_tokens)

decoded_tokens = encoder.decode(encoded_tokens)
print(decoded_tokens)
