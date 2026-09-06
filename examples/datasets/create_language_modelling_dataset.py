# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from nalp.corpus import TextCorpus
from nalp.datasets import LanguageModelingDataset
from nalp.encoders import IntegerEncoder

corpus = TextCorpus(from_file="data/text/chapter1_harry.txt", corpus_type="char")
encoder = IntegerEncoder()
encoder.learn(corpus.vocab_index, corpus.index_vocab)
encoded_tokens = encoder.encode(corpus.tokens)
dataset = LanguageModelingDataset(encoded_tokens, max_contiguous_pad_length=10, batch_size=1, shuffle=True)

for input_batch, target_batch in dataset.batches.take(1):
    for x, y in zip(input_batch, target_batch):
        print(encoder.decode(x.numpy()), encoder.decode(y.numpy()))
