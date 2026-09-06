# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from nalp.corpus import AudioCorpus

corpus = AudioCorpus(from_file="data/audio/sample.mid", min_frequency=1)

print(corpus.tokens)
print(corpus.vocab, corpus.vocab_size)
print(corpus.vocab_index, corpus.index_vocab)
