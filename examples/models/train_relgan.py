# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import tensorflow as tf

from nalp.corpus import TextCorpus
from nalp.datasets import LanguageModelingDataset
from nalp.encoders import IntegerEncoder
from nalp.models import RelGAN

corpus = TextCorpus(from_file="data/text/chapter1_harry.txt", corpus_type="char")
encoder = IntegerEncoder()
encoder.learn(corpus.vocab_index, corpus.index_vocab)
encoded_tokens = encoder.encode(corpus.tokens)
dataset = LanguageModelingDataset(encoded_tokens, max_contiguous_pad_length=10, batch_size=64)

relgan = RelGAN(
    encoder=encoder,
    vocab_size=corpus.vocab_size,
    max_length=10,
    embedding_size=256,
    n_slots=5,
    n_heads=5,
    head_size=25,
    n_blocks=1,
    n_layers=3,
    n_filters=(64, 128, 256),
    filters_size=(3, 5, 5),
    dropout_rate=0.25,
    tau=5,
)

relgan.compile(
    pre_optimizer=tf.optimizers.Adam(learning_rate=0.01),
    d_optimizer=tf.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=tf.optimizers.Adam(learning_rate=0.0001),
)

relgan.pre_fit(dataset.batches, epochs=200)

relgan.fit(dataset.batches, epochs=50)

relgan.save_weights("trained/relgan", save_format="tf")
