# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from nalp.corpus import TextCorpus
from nalp.encoders import Word2vecEncoder

corpus = TextCorpus(from_file="data/text/chapter1_harry.txt", corpus_type="word")
encoder = Word2vecEncoder()
encoder.learn(corpus.tokens)

encoded_tokens = encoder.encode(corpus.tokens)
print(encoded_tokens)

decoded_tokens = encoder.decode(encoded_tokens)
print(decoded_tokens)
