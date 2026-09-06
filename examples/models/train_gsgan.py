# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import tensorflow as tf

from nalp.corpus import TextCorpus
from nalp.datasets import LanguageModelingDataset
from nalp.encoders import IntegerEncoder
from nalp.models import GSGAN

corpus = TextCorpus(from_file="data/text/chapter1_harry.txt", corpus_type="char")
encoder = IntegerEncoder()
encoder.learn(corpus.vocab_index, corpus.index_vocab)
encoded_tokens = encoder.encode(corpus.tokens)
dataset = LanguageModelingDataset(encoded_tokens, max_contiguous_pad_length=10, batch_size=64)

gsgan = GSGAN(
    encoder=encoder,
    vocab_size=corpus.vocab_size,
    embedding_size=256,
    hidden_size=512,
    tau=5,
)

gsgan.compile(
    pre_optimizer=tf.optimizers.Adam(learning_rate=0.01),
    d_optimizer=tf.optimizers.Adam(learning_rate=0.001),
    g_optimizer=tf.optimizers.Adam(learning_rate=0.001),
)

gsgan.pre_fit(dataset.batches, epochs=100)

gsgan.fit(dataset.batches, epochs=100)

gsgan.save_weights("trained/gsgan", save_format="tf")
